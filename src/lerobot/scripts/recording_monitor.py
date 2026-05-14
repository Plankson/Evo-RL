"""Dedicated dual-system recording loop for predictor + detector monitor setups."""

from __future__ import annotations

import copy
import logging
import math
import multiprocessing
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.remote_client.modeling_remote_client import batch_to_client_observation
from lerobot.policies.remote_monitor.configuration_remote_monitor import RemoteMonitorEndpointConfig
from lerobot.policies.remote_monitor.modeling_remote_monitor import OpenPiWebsocketClient
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    ACPInferenceConfig,
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    _predict_policy_action_with_runtime_state,
)
from lerobot.scripts.local_detector_runtime import (
    LocalDetectorConfig,
    LocalOpenPIDetectorRuntime,
    extract_episode_frame_for_visualization,
    render_detector_episode_in_process,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import prepare_observation_for_inference
from lerobot.utils.recording_annotations import resolve_collector_policy_id
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data

logger = logging.getLogger(__name__)

INFER_PI0_GRIPPER_CLOSE_THRESHOLD = 0.02
INFER_PI0_GRIPPER_OPEN_THRESHOLD = 0.03
INFER_PI0_GRIPPER_CLOSE_COMMAND = -1
INFER_PI0_GRIPPER_OPEN_COMMAND = 100


def _convert_joint_positions_deg_to_rad(observation: RobotObservation) -> RobotObservation:
    converted = dict(observation)
    for key, value in observation.items():
        if not key.endswith(".pos"):
            continue
        if "joint_" in key:
            converted[key] = math.radians(float(value))
            continue
        if "gripper.pos" in key:
            converted[key] = float(value) / 1000.0
    return converted


def _convert_joint_positions_rad_to_deg(action: RobotAction) -> RobotAction:
    converted = dict(action)
    for key, value in action.items():
        if not key.endswith(".pos"):
            continue
        if "joint_" in key:
            converted[key] = math.degrees(float(value))
            continue
        if "gripper.pos" in key:
            converted[key] = float(value) * 1000.0
    return converted


def _apply_infer_pi0_gripper_logic(
    policy_action: RobotAction,
    robot_action: RobotAction,
) -> RobotAction:
    adjusted = dict(robot_action)
    for key, value in policy_action.items():
        if "gripper.pos" not in key:
            continue
        if float(value) < INFER_PI0_GRIPPER_CLOSE_THRESHOLD:
            adjusted[key] = INFER_PI0_GRIPPER_CLOSE_COMMAND
        elif float(value) > INFER_PI0_GRIPPER_OPEN_THRESHOLD:
            adjusted[key] = INFER_PI0_GRIPPER_OPEN_COMMAND
    return adjusted


def _predict_policy_action_with_acp_inference_from_source(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
) -> PolicyAction:
    if not acp_inference.enable:
        obs = prepare_observation_for_inference(copy.deepcopy(observation_frame), device, task, robot_type)
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        return postprocessor(action)

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        obs = prepare_observation_for_inference(
            copy.deepcopy(observation_frame), device, conditional_task, robot_type
        )
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        return postprocessor(action)

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


def publish_observation_to_detector(
    *,
    detector_source_queue,
    seq: int,
    raw_obs: RobotObservation,
) -> None:
    if detector_source_queue is None:
        return
    payload = {
        "seq": seq,
        "observed_at_s": time.time(),
        "raw_obs": copy.deepcopy(raw_obs),
    }
    try:
        detector_source_queue.put_nowait(payload)
    except queue.Full:
        try:
            detector_source_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            detector_source_queue.put_nowait(payload)
        except queue.Full:
            pass


def handle_failure(signal: dict[str, Any]) -> None:
    # TODO: implement real failure handling here.
    del signal


def detector_process_worker(
    observation_queue,
    detector_endpoint_config: RemoteMonitorEndpointConfig,
    policy_device: str | None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset_features: dict[str, Any],
    task: str | None,
    robot_type: str | None,
    signal_queue,
    stop_event,
    history_size: int,
    ready_event,
    failed_event,
    error_queue,
) -> None:
    logger.info("[DETECTOR] Process started.")
    try:
        detector_client = OpenPiWebsocketClient(
            host=detector_endpoint_config.host,
            port=detector_endpoint_config.port,
            api_key=detector_endpoint_config.api_key,
        )
        history: deque[dict[str, Any]] = deque(maxlen=history_size)
        device = get_safe_torch_device(policy_device)
        ready_event.set()
    except Exception as exc:
        error_queue.put(f"[DETECTOR] Initialization failed: {exc}")
        failed_event.set()
        return

    try:
        while not stop_event.is_set():
            wait_t0 = time.perf_counter()
            try:
                item = observation_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame_wait_ms = (time.perf_counter() - wait_t0) * 1e3

            try:
                infer_t0 = time.perf_counter()
                history.append(item)
                latest = history[-1]
                raw_obs = latest["raw_obs"]
                obs_processed = robot_observation_processor(raw_obs)
                obs_processed = _convert_joint_positions_deg_to_rad(obs_processed)
                observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
                batch = prepare_observation_for_inference(
                    copy.deepcopy(observation_frame),
                    device,
                    task,
                    robot_type,
                )
                detector_obs = batch_to_client_observation(batch, detector_endpoint_config)
                result = detector_client.infer(detector_obs)
                infer_ms = (time.perf_counter() - infer_t0) * 1e3
                logger.info(
                    "[DETECTOR] timing frame_wait=%.2fms infer=%.2fms total=%.2fms",
                    frame_wait_ms,
                    infer_ms,
                    frame_wait_ms + infer_ms,
                )
                signal_queue.put(
                    {
                        "source": "detector",
                        "is_dangerous": result.get("is_dangerous", False),
                        "score": result.get("score", 0.0),
                        "timestamp": time.time(),
                        "source_seq": latest["seq"],
                    }
                )
            except Exception as exc:
                logger.error("[DETECTOR] Error in background loop: %s", exc)
                time.sleep(0.01)
    except Exception as exc:
        logger.error("[DETECTOR] Fatal error: %s", exc)


def detector_local_worker(
    observation_queue,
    local_detector_config: LocalDetectorConfig,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset_features: dict[str, Any],
    task: str | None,
    robot_type: str | None,
    signal_queue,
    stop_event,
    history_size: int,
    episode_index: int,
    ready_event,
    failed_event,
    error_queue,
) -> None:
    logger.info("[DETECTOR-LOCAL] Worker started.")
    try:
        detector_runtime = LocalOpenPIDetectorRuntime(local_detector_config)
        history: deque[dict[str, Any]] = deque(maxlen=history_size)
        device = get_safe_torch_device("cpu")
        episode_records: list[dict[str, Any]] = []
        ready_event.set()
    except Exception as exc:
        error_queue.put(f"[DETECTOR-LOCAL] Initialization failed: {exc}")
        failed_event.set()
        return

    try:
        while not stop_event.is_set():
            wait_t0 = time.perf_counter()
            try:
                item = observation_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame_wait_ms = (time.perf_counter() - wait_t0) * 1e3

            try:
                infer_t0 = time.perf_counter()
                history.append(item)
                latest = history[-1]
                raw_obs = latest["raw_obs"]
                obs_processed = robot_observation_processor(raw_obs)
                obs_processed = _convert_joint_positions_deg_to_rad(obs_processed)
                observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
                batch = prepare_observation_for_inference(
                    copy.deepcopy(observation_frame),
                    device,
                    task,
                    robot_type,
                )
                detector_obs = batch_to_client_observation(batch, local_detector_config)  # type: ignore[arg-type]
                result = detector_runtime.infer(detector_obs)
                infer_ms = (time.perf_counter() - infer_t0) * 1e3
                logger.info(
                    "[DETECTOR-LOCAL] timing frame_wait=%.2fms infer=%.2fms total=%.2fms",
                    frame_wait_ms,
                    infer_ms,
                    frame_wait_ms + infer_ms,
                )
                frame = extract_episode_frame_for_visualization(detector_obs)
                episode_records.append(
                    {
                        "frame": frame,
                        "score": result.get("score", 0.0),
                        "threshold": result.get("threshold", 0.0),
                        "is_dangerous": result.get("is_dangerous", False),
                        "title": (
                            f"detector t={result.get('timestep', 0)} "
                            f"score={float(result.get('score', 0.0)):.4f} "
                            f"band={float(result.get('threshold', 0.0)):.4f}"
                        ),
                    }
                )
                signal_queue.put(
                    {
                        "source": "detector",
                        "is_dangerous": result.get("is_dangerous", False),
                        "score": result.get("score", 0.0),
                        "timestamp": time.time(),
                        "source_seq": latest["seq"],
                    }
                )
            except Exception as exc:
                logger.error("[DETECTOR-LOCAL] Error in background loop: %s", exc)
                time.sleep(0.01)
    except Exception as exc:
        logger.error("[DETECTOR-LOCAL] Fatal error: %s", exc)
    finally:
        try:
            render_proc = render_detector_episode_in_process(episode_records, local_detector_config, episode_index)
            if render_proc is not None:
                logger.info("[DETECTOR-LOCAL] Waiting for episode %d render to finish...", episode_index)
                render_proc.join()
                logger.info("[DETECTOR-LOCAL] Episode %d render completed.", episode_index)
        except Exception as exc:
            logger.error("[DETECTOR-LOCAL] Failed to render episode video: %s", exc)


def alarm_poller_worker(signal_queue, shared_danger, stop_event) -> None:
    logger.info("[ALARM] Poller started.")
    while not stop_event.is_set():
        try:
            signal = signal_queue.get(timeout=0.1)
            if signal["is_dangerous"]:
                logger.error(
                    "!!! ALARM !!! %s detected danger! Score: %.4f",
                    signal["source"].upper(),
                    signal.get("score", 0.0),
                )
                shared_danger.value = 1
                handle_failure(signal)
        except queue.Empty:
            continue
        except Exception as exc:
            logger.error("[ALARM] Poller error: %s", exc)


def record_loop_monitor(
    *,
    robot: Robot,
    events: dict,
    fps: int,
    observation_poll_hz: float,
    observation_pool_size: int,
    detector_queue_size: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: str = "policy",
    collector_policy_id_human: str = "human",
    acp_inference: ACPInferenceConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
    detector_mode: str = "remote",
    local_detector_config: LocalDetectorConfig | None = None,
    local_detector_episode_index: int = 0,
    detector_ready_timeout_s: float = 120.0,
):
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()
    if dataset is None:
        raise ValueError("Dual monitor recording requires a dataset for feature mapping.")
    if policy is None or preprocessor is None or postprocessor is None:
        raise ValueError("Dual monitor recording requires policy, preprocessor, and postprocessor.")

    action_feature_names = dataset.features[ACTION]["names"]
    action_feature_names = list(robot.action_features) if action_feature_names is None else list(action_feature_names)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    has_teleop = isinstance(teleop, (Teleoperator, list))
    intervention_enabled = intervention_state_machine_enabled and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    teleop_fallback_warned = False

    teleop_arm_for_mode_switch: Any | None = teleop if isinstance(teleop, Teleoperator) else teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is None or not hasattr(teleop_arm_for_mode_switch, "set_manual_control"):
            return
        try:
            teleop_arm_for_mode_switch.set_manual_control(enabled)
        except Exception:
            logger.exception("Failed to switch teleop manual-control mode to %s", enabled)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        set_teleop_manual_control(False)

    if observation_poll_hz != fps:
        logger.warning(
            "`observation_poll_hz` is ignored in stable monitor mode; observations are sampled once per "
            "control-loop frame at dataset fps=%s to match the normal HIL camera access pattern.",
            fps,
        )

    use_local_detector = detector_mode == "local"
    if detector_mode not in {"remote", "local"}:
        raise ValueError(f"Unsupported detector_mode={detector_mode}")

    if use_local_detector:
        if local_detector_config is None:
            raise ValueError("local_detector_config must be provided when detector_mode='local'")
        detector_source_queue = queue.Queue(maxsize=max(detector_queue_size, 1))
        signal_queue = queue.Queue()
        stop_event = threading.Event()
        detector_ready_event = threading.Event()
        detector_failed_event = threading.Event()
        detector_error_queue = queue.Queue()

        @dataclass
        class _DangerState:
            value: int = 0

        shared_danger = _DangerState(0)
        detector_proc = threading.Thread(
            target=detector_local_worker,
            args=(
                detector_source_queue,
                local_detector_config,
                robot_observation_processor,
                dataset.features,
                single_task,
                robot.robot_type,
                signal_queue,
                stop_event,
                max(observation_pool_size, 1),
                local_detector_episode_index,
                detector_ready_event,
                detector_failed_event,
                detector_error_queue,
            ),
            daemon=True,
            name="detector-local-thread",
        )
        alarm_proc = threading.Thread(
            target=alarm_poller_worker,
            args=(signal_queue, shared_danger, stop_event),
            daemon=True,
            name="alarm-poller-thread",
        )
    else:
        # Do not fork after robot/camera connections are live. RealSense cameras run
        # background reader threads, and forking a multithreaded process can corrupt
        # the parent-side camera pipeline even when the child never touches the robot.
        if "spawn" in multiprocessing.get_all_start_methods():
            ctx = multiprocessing.get_context("spawn")
        else:
            ctx = multiprocessing
        detector_source_queue = ctx.Queue(maxsize=max(detector_queue_size, 1))
        signal_queue = ctx.Queue()
        stop_event = ctx.Event()
        detector_ready_event = ctx.Event()
        detector_failed_event = ctx.Event()
        detector_error_queue = ctx.Queue()
        shared_danger = ctx.Value("i", 0)
        detector_proc = ctx.Process(
            target=detector_process_worker,
            args=(
                detector_source_queue,
                policy.config.detector_remote,
                policy.config.device,
                robot_observation_processor,
                dataset.features,
                single_task,
                robot.robot_type,
                signal_queue,
                stop_event,
                max(observation_pool_size, 1),
                detector_ready_event,
                detector_failed_event,
                detector_error_queue,
            ),
            daemon=False,
        )
        alarm_proc = ctx.Process(
            target=alarm_poller_worker,
            args=(signal_queue, shared_danger, stop_event),
            daemon=False,
        )

    def run_with_connection_retry(action_name: str, fn):
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logger.warning("%s recovered after %d retries in %.2fs.", action_name, attempts - 1, elapsed_s)
                return result
            except ConnectionError as error:
                logger.warning(
                    "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                    action_name,
                    timeout_s,
                    error,
                )
                if timeout_s <= 0.0:
                    raise
                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise
                time.sleep(min(interval_s if interval_s > 0.0 else remaining_s, remaining_s))

    detector_proc.start()
    alarm_proc.start()

    ready_deadline_t = time.perf_counter() + max(detector_ready_timeout_s, 0.0)
    while not detector_ready_event.is_set():
        if detector_failed_event.is_set():
            try:
                init_error = detector_error_queue.get_nowait()
            except Exception:
                init_error = "Detector initialization failed without detailed error."
            raise RuntimeError(init_error)
        if not use_local_detector and not detector_proc.is_alive():
            raise RuntimeError("Detector process exited before becoming ready.")
        if time.perf_counter() > ready_deadline_t:
            raise TimeoutError(
                f"Detector did not become ready within {detector_ready_timeout_s:.1f}s. "
                "Robot execution is blocked until detector initializes."
            )
        time.sleep(0.05)
    logger.info("Detector is ready. Starting robot execution loop.")

    timestamp = 0.0
    sample_seq = 0
    events["episode_initial_pose"] = None
    # Episode time starts when detector is ready and robot execution begins.
    start_episode_t = time.perf_counter()

    try:
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if events.get("toggle_intervention", False):
                events["toggle_intervention"] = False
                if intervention_enabled:
                    if intervention_state == INTERVENTION_STATE_POLICY:
                        intervention_state = INTERVENTION_STATE_ACTIVE
                        set_teleop_manual_control(True)
                        logger.info("Intervention enabled (S1): teleop actions now override policy execution.")
                    else:
                        intervention_state = INTERVENTION_STATE_RELEASE
                        set_teleop_manual_control(False)
                        policy.reset()
                        preprocessor.reset()
                        postprocessor.reset()
                        if acp_inference.enable and acp_inference.use_cfg:
                            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
                            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
                        logger.info("Policy cache reset on release: next policy action is recomputed.")
                        logger.info("Intervention release requested (S2): returning control to policy.")
                else:
                    logger.info("Intervention toggle ignored because policy+teleop are not both active.")

            raw_obs = robot.get_observation()
            if events.get("episode_initial_pose") is None:
                initial_pose = {}
                for action_key in robot.action_features:
                    if action_key.endswith(".pos") and action_key in raw_obs:
                        initial_pose[action_key] = float(raw_obs[action_key])
                events["episode_initial_pose"] = initial_pose if initial_pose else None
            source_seq = sample_seq
            sample_seq += 1
            publish_observation_to_detector(
                detector_source_queue=detector_source_queue,
                seq=source_seq,
                raw_obs=raw_obs,
            )
            obs_processed = robot_observation_processor(raw_obs)
            obs_processed = _convert_joint_positions_deg_to_rad(obs_processed)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            act_processed_policy: RobotAction | None = None
            act_processed_teleop: RobotAction | None = None

            if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
                policy_action = _predict_policy_action_with_acp_inference_from_source(
                    observation_frame=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                    acp_inference=acp_inference,
                    cond_runtime_state=cond_policy_runtime_state,
                    uncond_runtime_state=uncond_policy_runtime_state,
                )
                act_processed_policy = make_robot_action(policy_action, dataset.features)

            if getattr(policy, "last_predictor_safety", None):
                signal_queue.put({**policy.last_predictor_safety, "timestamp": time.time(), "source_seq": source_seq})
                policy.last_predictor_safety = None

            if isinstance(teleop, Teleoperator):
                act = run_with_connection_retry("teleop.get_action", teleop.get_action)
                act_processed_teleop = teleop_action_processor((act, raw_obs))
                act_processed_teleop = _convert_joint_positions_deg_to_rad(act_processed_teleop)
            elif isinstance(teleop, list):
                arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, raw_obs))
                act_processed_teleop = _convert_joint_positions_deg_to_rad(act_processed_teleop)

            if act_processed_policy is None and act_processed_teleop is None:
                logger.info(
                    "No policy or teleoperator provided, skipping action generation."
                    " The robot won't be at its rest position at the start of the next episode."
                )
                continue

            if act_processed_teleop is not None:
                last_teleop_action = act_processed_teleop
                teleop_fallback_warned = False

            policy_action_for_storage = (
                act_processed_policy if act_processed_policy is not None else zero_policy_action
            )

            is_intervention = 0.0
            if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
                is_intervention = 1.0
                if act_processed_teleop is not None:
                    action_values = act_processed_teleop
                elif last_teleop_action is not None:
                    action_values = last_teleop_action
                    if not teleop_fallback_warned:
                        logger.warning(
                            "Intervention is active but no fresh teleop action is available; reusing last teleop action."
                        )
                        teleop_fallback_warned = True
                elif act_processed_policy is not None:
                    action_values = act_processed_policy
                    if not teleop_fallback_warned:
                        logger.warning(
                            "Intervention is active but teleop action is unavailable; falling back to policy action."
                        )
                        teleop_fallback_warned = True
                else:
                    action_values = zero_policy_action
            else:
                action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

            selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
            action_values_for_robot = _convert_joint_positions_rad_to_deg(action_values)
            if selected_from_policy:
                action_values_for_robot = _apply_infer_pi0_gripper_logic(
                    policy_action=act_processed_policy,
                    robot_action=action_values_for_robot,
                )

            robot_action_to_send = robot_action_processor((action_values_for_robot, raw_obs))

            if policy_sync_executor is not None and selected_from_policy:
                run_with_connection_retry(
                    "policy_sync_executor.send_action",
                    lambda action=robot_action_to_send: policy_sync_executor.send_action(action),
                )
            else:
                run_with_connection_retry(
                    "robot.send_action",
                    lambda action=robot_action_to_send: robot.send_action(action),
                )

            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
            )
            frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

            if "complementary_info.is_intervention" in dataset.features:
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            if "complementary_info.state" in dataset.features:
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
            if "complementary_info.collector_policy_id" in dataset.features:
                frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                    intervention_enabled=intervention_enabled,
                    is_intervention=bool(is_intervention),
                    selected_from_policy=selected_from_policy,
                    policy_id=collector_policy_id_policy,
                    human_id=collector_policy_id_human,
                )
            dataset.add_frame(frame)

            if display_data:
                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )

            if intervention_state == INTERVENTION_STATE_RELEASE:
                intervention_state = INTERVENTION_STATE_POLICY

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))
            timestamp = time.perf_counter() - start_episode_t
    finally:
        stop_event.set()
        if use_local_detector:
            detector_proc.join()
        else:
            detector_proc.join(timeout=2.0)
        alarm_proc.join(timeout=2.0)
        if not use_local_detector and detector_proc.is_alive():
            detector_proc.terminate()
            detector_proc.join(timeout=2.0)
        if not use_local_detector and alarm_proc.is_alive():
            alarm_proc.terminate()
            alarm_proc.join(timeout=2.0)
