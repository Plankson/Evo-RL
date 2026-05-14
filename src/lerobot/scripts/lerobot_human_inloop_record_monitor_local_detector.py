#!/usr/bin/env python

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.robots.robot_io import RobotIOClient
from lerobot.scripts.hdf5_episode_recorder import HDF5EpisodeRecorder
from lerobot.scripts.lerobot_human_inloop_record import _HumanInloopFailureResetController
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.local_detector_runtime import LocalDetectorConfig, validate_local_detector_paths
from lerobot.scripts.recording_hil import PolicySyncDualArmExecutor
from lerobot.scripts.recording_monitor import record_loop_monitor
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.recording_annotations import (
    infer_collector_policy_id,
    infer_collector_policy_version,
    normalize_episode_success_label,
    resolve_episode_success_label,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


@dataclass
class MonitorLocalDetectorRecordConfig(RecordConfig):
    observation_poll_hz: float = 30
    observation_pool_size: int = 256
    detector_queue_size: int = 1

    distributed_robot_io: bool = False
    robot_io_obs_address: str = "tcp://127.0.0.1:5555"
    robot_io_action_address: str = "tcp://127.0.0.1:5556"
    robot_io_meta_address: str = "tcp://127.0.0.1:5557"
    robot_io_obs_timeout_s: float = 5.0

    local_detector: LocalDetectorConfig = field(
        default_factory=lambda: LocalDetectorConfig(
            monitor_config="",
            monitor_dir="",
            detector_conformal_path="",
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if self.policy is None or self.policy.type != "remote_monitor":
            raise ValueError(
                "`lerobot-human-inloop-record-monitor-local-detector` requires `--policy.type=remote_monitor`."
            )
        if self.observation_poll_hz <= 0:
            raise ValueError("`observation_poll_hz` must be > 0.")
        if self.observation_pool_size <= 0:
            raise ValueError("`observation_pool_size` must be > 0.")
        if self.detector_queue_size <= 0:
            raise ValueError("`detector_queue_size` must be > 0.")


@parser.wrap()
def human_inloop_record_monitor_local_detector(cfg: MonitorLocalDetectorRecordConfig):
    if cfg.teleop is None:
        raise ValueError("This script requires `teleop` config.")

    validate_local_detector_paths(cfg.local_detector)

    cfg._save_hdf5_episodes = True
    cfg.policy_sync_to_teleop = True
    cfg.intervention_state_machine_enabled = True
    cfg.enable_episode_outcome_labeling = True
    cfg.default_episode_success = None
    cfg.require_episode_success_label = True
    cfg.enable_collector_policy_id = True
    if cfg.collector_policy_id_policy is None:
        cfg.collector_policy_id_policy = infer_collector_policy_version(cfg.policy)

    failure_reset_controller = _HumanInloopFailureResetController(cfg)
    cfg._on_record_connected = failure_reset_controller.on_record_connected
    cfg._on_record_episode_outcome = failure_reset_controller.on_episode_outcome
    cfg._before_record_episode = failure_reset_controller.before_record_episode
    cfg._skip_reset_time_loop = True

    init_logging()
    logging.info(
        "Local-detector monitor recording enabled. ACP inference: enable=%s use_cfg=%s cfg_beta=%.3f. "
        "Observation poll hz=%.1f pool_size=%d detector_queue_size=%d.",
        cfg.acp_inference.enable,
        cfg.acp_inference.use_cfg,
        cfg.acp_inference.cfg_beta,
        cfg.observation_poll_hz,
        cfg.observation_pool_size,
        cfg.detector_queue_size,
    )
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="recording_monitor_local_detector", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    if cfg.distributed_robot_io:
        logging.info("Distributed robot IO enabled for monitor client.")
        robot = RobotIOClient(
            observation_address=cfg.robot_io_obs_address,
            action_address=cfg.robot_io_action_address,
            metadata_address=cfg.robot_io_meta_address,
            observation_timeout_s=cfg.robot_io_obs_timeout_s,
        )
    else:
        logging.info("Distributed robot IO disabled; using local hardware robot.")
        robot = make_robot_from_config(cfg.robot)
    robot_connected_early = False
    if cfg.distributed_robot_io:
        # RobotIOClient gets action/observation feature schemas from server metadata on connect.
        # We must connect before building dataset feature schemas.
        robot.connect()
        robot_connected_early = True

    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    action_names = dataset_features[ACTION]["names"]
    action_names = list(robot.action_features) if action_names is None else list(action_names)
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": action_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }
    dataset_features["complementary_info.collector_policy_id"] = {
        "dtype": "string",
        "shape": (1,),
        "names": ["collector_policy_id"],
    }

    dataset = None
    listener = None
    policy_sync_executor = None
    use_hdf5_episode_recorder = bool(getattr(cfg, "_save_hdf5_episodes", False))

    try:
        if use_hdf5_episode_recorder:
            camera_name_map = getattr(cfg.policy.predictor_remote, "image_key_map", None)
            dataset = HDF5EpisodeRecorder(
                repo_id=cfg.dataset.repo_id,
                fps=cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                camera_name_map=camera_name_map,
            )
        else:
            if cfg.resume:
                dataset = LeRobotDataset(
                    cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                )
                if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                    dataset.start_image_writer(
                        num_processes=cfg.dataset.num_image_writer_processes,
                        num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                    )
                sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
            else:
                sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
                dataset = LeRobotDataset.create(
                    cfg.dataset.repo_id,
                    cfg.dataset.fps,
                    root=cfg.dataset.root,
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=cfg.dataset.video,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                )

        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        collector_policy_id_policy = (
            cfg.collector_policy_id_policy
            if cfg.collector_policy_id_policy is not None
            else infer_collector_policy_id(cfg.policy)
        )
        collector_policy_id_human = cfg.collector_policy_id_human

        if not robot_connected_early:
            robot.connect()
        if teleop is not None:
            teleop.connect()
        if callable(getattr(cfg, "_on_record_connected", None)):
            cfg._on_record_connected(robot, teleop)

        if teleop is None or isinstance(teleop, list):
            raise ValueError("This script requires exactly one teleoperator with send_feedback support.")
        policy_sync_executor = PolicySyncDualArmExecutor(
            robot=robot,
            teleop=teleop,
            parallel_dispatch=cfg.policy_sync_parallel,
        )

        listener, events = init_keyboard_listener(
            intervention_toggle_key=cfg.intervention_toggle_key,
            episode_success_key=cfg.episode_success_key,
            episode_failure_key=cfg.episode_failure_key,
        )

        dataset_context = nullcontext() if use_hdf5_episode_recorder else VideoEncodingManager(dataset)
        with dataset_context:
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                events["episode_outcome"] = None
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                if callable(getattr(cfg, "_before_record_episode", None)):
                    cfg._before_record_episode(robot, teleop, recorded_episodes)

                record_loop_monitor(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    observation_poll_hz=cfg.observation_poll_hz,
                    observation_pool_size=cfg.observation_pool_size,
                    detector_queue_size=cfg.detector_queue_size,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    policy_sync_executor=policy_sync_executor,
                    intervention_state_machine_enabled=cfg.intervention_state_machine_enabled,
                    collector_policy_id_policy=collector_policy_id_policy,
                    collector_policy_id_human=collector_policy_id_human,
                    acp_inference=cfg.acp_inference,
                    communication_retry_timeout_s=cfg.communication_retry_timeout_s,
                    communication_retry_interval_s=cfg.communication_retry_interval_s,
                    detector_mode="local",
                    local_detector_config=cfg.local_detector,
                    local_detector_episode_index=dataset.num_episodes,
                )

                if events.get("stop_recording"):
                    break

                episode_outcome = resolve_episode_success_label(
                    events,
                    default_label=normalize_episode_success_label(cfg.default_episode_success),
                    require_explicit_label=cfg.require_episode_success_label,
                )
                if episode_outcome is None:
                    log_say("Episode label missing, retrying this episode.", cfg.play_sounds)
                    continue
                if callable(getattr(cfg, "_on_record_episode_outcome", None)):
                    cfg._on_record_episode_outcome(robot, teleop, episode_outcome)

                if episode_outcome:
                    dataset.save_episode(task=cfg.dataset.single_task)
                    recorded_episodes += 1
                else:
                    dataset.clear_episode_buffer()

    finally:
        if listener is not None:
            listener.stop()
        if policy_sync_executor is not None:
            policy_sync_executor.shutdown()
        if teleop is not None:
            teleop.disconnect()
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    human_inloop_record_monitor_local_detector()


if __name__ == "__main__":
    main()
