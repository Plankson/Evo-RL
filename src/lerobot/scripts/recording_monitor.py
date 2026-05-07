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


@dataclass
class ObservationSample:
    seq: int
    observed_at_s: float
    raw_obs: RobotObservation


class ObservationPool:
    def __init__(self, maxlen: int):
        self._samples: deque[ObservationSample] = deque(maxlen=maxlen)
        self._cond = threading.Condition()

    def publish(self, sample: ObservationSample) -> None:
        with self._cond:
            self._samples.append(sample)
            self._cond.notify_all()

    def latest(self, min_seq: int | None = None, timeout_s: float | None = None) -> ObservationSample:
        with self._cond:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while True:
                if self._samples:
                    sample = self._samples[-1]
                    if min_seq is None or sample.seq > min_seq:
                        return sample

                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if not self._samples:
                            raise TimeoutError("Observation pool did not receive any sample in time.")
                        return self._samples[-1]
                    self._cond.wait(timeout=remaining)


class ObservationPoller:
    def __init__(
        self,
        robot: Robot,
        observation_pool: ObservationPool,
        detector_source_queue,
        poll_hz: float,
    ) -> None:
        self.robot = robot
        self.observation_pool = observation_pool
        self.detector_source_queue = detector_source_queue
        self.poll_hz = poll_hz
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="observation-poller")
        self._seq = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        period_s = 0.0 if self.poll_hz <= 0 else 1.0 / self.poll_hz
        while not self.stop_event.is_set():
            start_t = time.perf_counter()
            try:
                raw_obs = self.robot.get_observation()
                sample = ObservationSample(seq=self._seq, observed_at_s=time.time(), raw_obs=raw_obs)
                self._seq += 1
                self.observation_pool.publish(sample)

                if self.detector_source_queue is not None:
                    payload = {
                        "seq": sample.seq,
                        "observed_at_s": sample.observed_at_s,
                        "raw_obs": copy.deepcopy(raw_obs),
                    }
                    try:
                        self.detector_source_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            self.detector_source_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.detector_source_queue.put_nowait(payload)
                        except queue.Full:
                            pass
            except Exception as exc:
                logger.error("[OBS_POOL] Error while reading robot observation: %s", exc)
                time.sleep(0.01)

            if period_s > 0:
                precise_sleep(max(period_s - (time.perf_counter() - start_t), 0.0))


def handle_failure(signal: dict[str, Any]) -> None:
    # TODO: implement real failure handling here.
    del signal


def detector_process_worker(
    observation_queue,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset_features: dict[str, Any],
    task: str | None,
    robot_type: str | None,
    signal_queue,
    stop_event,
    history_size: int,
) -> None:
    logger.info("[DETECTOR] Process started.")
    policy.reset()
    if preprocessor is not None:
        preprocessor.reset()
    history: deque[dict[str, Any]] = deque(maxlen=history_size)
    device = get_safe_torch_device(policy.config.device)

    try:
        while not stop_event.is_set():
            try:
                item = observation_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
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
                if preprocessor is not None:
                    batch = preprocessor(batch)
                result = policy.infer_detector(batch)
                signal_queue.put({**result, "timestamp": time.time(), "source_seq": latest["seq"]})
            except Exception as exc:
                logger.error("[DETECTOR] Error in background loop: %s", exc)
                time.sleep(0.01)
    except Exception as exc:
        logger.error("[DETECTOR] Fatal error: %s", exc)


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

    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing
    )
    detector_source_queue = ctx.Queue(maxsize=max(detector_queue_size, 1))
    signal_queue = ctx.Queue()
    stop_event = ctx.Event()
    shared_danger = ctx.Value("i", 0)
    observation_pool = ObservationPool(max(observation_pool_size, 1))
    observation_poller = ObservationPoller(
        robot=robot,
        observation_pool=observation_pool,
        detector_source_queue=detector_source_queue,
        poll_hz=observation_poll_hz,
    )
    detector_proc = ctx.Process(
        target=detector_process_worker,
        args=(
            detector_source_queue,
            policy,
            preprocessor,
            robot_observation_processor,
            dataset.features,
            single_task,
            robot.robot_type,
            signal_queue,
            stop_event,
            observation_pool_size,
        ),
        daemon=True,
    )
    alarm_proc = ctx.Process(
        target=alarm_poller_worker,
        args=(signal_queue, shared_danger, stop_event),
        daemon=True,
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

    observation_poller.start()
    detector_proc.start()
    alarm_proc.start()

    timestamp = 0.0
    last_sample_seq = -1
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

            sample = observation_pool.latest(min_seq=last_sample_seq, timeout_s=max(1.0 / max(observation_poll_hz, 1.0), 0.1))
            last_sample_seq = sample.seq
            raw_obs = sample.raw_obs
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
                signal_queue.put({**policy.last_predictor_safety, "timestamp": time.time(), "source_seq": sample.seq})
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
        observation_poller.stop()
        detector_proc.join(timeout=1.0)
        alarm_proc.join(timeout=1.0)
