from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _to_monitor_training_schema(
    raw_obs: dict[str, Any], *, merge_state_gripper: bool
) -> dict[str, Any]:
    """Convert the websocket-style local payload to the detector training schema."""
    inputs = dict(raw_obs)
    images = inputs.get("images")
    if isinstance(images, dict):
        aliases = {
            "global_image": "cam_high",
            "left_image": "cam_left_wrist",
            "right_image": "cam_right_wrist",
        }
        for target, source in aliases.items():
            if target not in inputs and source in images:
                inputs[target] = images[source]

    joints = inputs.get("state.joints")
    gripper = inputs.get("state.gripper_w")
    if merge_state_gripper and joints is not None and gripper is not None:
        joints = np.asarray(joints, dtype=np.float32)
        gripper = np.asarray(gripper, dtype=np.float32)
        if joints.ndim == 1 and gripper.ndim == 1 and joints.size == 12 and gripper.size == 2:
            inputs["state.joints"] = np.concatenate(
                [joints[:6], gripper[:1], joints[6:], gripper[1:]], axis=0
            )
    return inputs


@dataclass
class LocalDetectorConfig:
    monitor_config: str
    monitor_dir: str
    detector_conformal_path: str
    task_stats_path: str | None = None
    task_index: int = 0
    task_length_subset: str = "all"
    detector_head_dir: str | None = None
    detector_head_model_name: str = "auto"
    history_len_detection: int = 1
    conformal_timestep_frequency: int = 20
    disable_warmup: bool = False
    payload_format: str = "infer_pi0"
    policy_name: str = "pi0"
    image_key_map: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "right_front": "cam_high",
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        }
    )
    state_joints_indices: list[int] = dataclasses.field(default_factory=list)
    state_gripper_indices: list[int] = dataclasses.field(default_factory=list)
    convert_images_to_uint8: bool = True
    render_episode_video: bool = True
    render_output_dir: str = "outputs/monitor_local_detector_videos"
    render_fps: int = 12


class _MonitorObservationAdapter:
    def __init__(self, transform, *, merge_state_gripper: bool = False):
        self._transform = transform
        self._merge_state_gripper = merge_state_gripper

    def __call__(self, raw_obs: dict[str, Any]):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: x, _to_monitor_training_schema(raw_obs, merge_state_gripper=self._merge_state_gripper))
        inputs = self._transform(inputs)

        def _batch_leaf(x):
            if isinstance(x, str):
                return x
            if isinstance(x, np.ndarray) and x.dtype.kind in {"U", "S", "O"}:
                return x
            if isinstance(x, np.generic) and getattr(x.dtype, "kind", None) in {"U", "S", "O"}:
                return x
            return jnp.asarray(x)[np.newaxis, ...]

        inputs = jax.tree.map(_batch_leaf, inputs)
        return _model.Observation.from_dict(inputs)


def _build_monitor_input_transform(train_cfg):
    from openpi import transforms as _transforms
    from openpi.training import checkpoints as _checkpoints

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    norm_stats = None
    asset_dir = None
    monitor_weight_path = getattr(train_cfg, "monitor_weight_path", None)
    if monitor_weight_path:
        monitor_path = os.path.abspath(str(monitor_weight_path))
        if os.path.isdir(monitor_path):
            candidate = os.path.join(monitor_path, "assets")
        else:
            candidate = os.path.join(os.path.dirname(monitor_path), "assets")
        if os.path.isdir(candidate):
            asset_dir = candidate

    if data_config.asset_id is not None:
        if asset_dir is not None:
            norm_stats = _checkpoints.load_norm_stats(asset_dir, data_config.asset_id)
        else:
            norm_stats = data_config.norm_stats

    return _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ]
    )


def _load_monitor(cfg: LocalDetectorConfig):
    import jax
    import flax.nnx as nnx
    import f_token.utils as f_token_utils
    from openpi.training import config as _config

    train_cfg = _config.get_config(cfg.monitor_config)
    train_cfg = dataclasses.replace(
        train_cfg,
        monitor=dataclasses.replace(train_cfg.monitor, enabled=True),
        monitor_weight_path=cfg.monitor_dir,
    )

    rng = jax.random.key(0)
    detector_head_dir = str(cfg.detector_head_dir or "").strip()
    if detector_head_dir:
        monitor = train_cfg.monitor.build(
            model_config=train_cfg.model,
            rngs=nnx.Rngs(rng),
            weight_path=None,
            encoder_weight_loader=train_cfg.weight_loader,
        )
        print("monitor_dir=", cfg.monitor_dir, "is_file=", Path(cfg.monitor_dir).is_file(), "is_dir=", Path(cfg.monitor_dir).is_dir())
        if str(cfg.monitor_dir).strip():
            monitor = f_token_utils.load_monitor_weights_partial(monitor, str(cfg.monitor_dir))

        detector_head_model_name = str(cfg.detector_head_model_name or "auto").strip() or "auto"
        monitor = f_token_utils.load_safe_head_weights_by_model(
            monitor,
            detector_head_dir,
            target_head_name="detector",
            source_model_name=detector_head_model_name,
        )
        logger.info(
            "Loaded detector head from %s (source_model=%s)", detector_head_dir, detector_head_model_name
        )
    else:
        monitor, _ = f_token_utils.build_monitor(train_cfg, rng)

    monitor = f_token_utils.cast_module_to_bfloat16(monitor)
    obs_adapter = _MonitorObservationAdapter(
        _build_monitor_input_transform(train_cfg),
        merge_state_gripper=cfg.monitor_config in {
            "pi05_real_robot_arrange_flower_evorl_detector",
            "pi05_real_robot_stack_plates_evorl_detector",
            "pi05_real_robot_pick_cubes_evorl_detector",
        },
    )
    return monitor, obs_adapter


def _stack_backbone_history(history: list[Any]) -> Any:
    import jax.numpy as jnp
    if not history:
        raise ValueError("history must not be empty")
    first = jnp.asarray(history[0])
    stacked = [jnp.asarray(x) for x in history]
    for idx, item in enumerate(stacked[1:], start=1):
        if item.shape != first.shape:
            raise ValueError(
                "All buffered backbone features must have the same shape, "
                f"got first.shape={first.shape} and history[{idx}].shape={item.shape}"
            )
    return jnp.stack(stacked, axis=1)


def _pad_backbone_history(history: list[Any], target_len: int) -> Any:
    import jax.numpy as jnp
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    if not history:
        raise ValueError("history must not be empty")

    trimmed = [jnp.asarray(x) for x in history[-target_len:]]
    if len(trimmed) < target_len:
        pad_count = target_len - len(trimmed)
        trimmed = [trimmed[0]] * pad_count + trimmed
    return _stack_backbone_history(trimmed)


def _score_to_scalar(score: Any) -> float:
    score_arr = np.asarray(score)
    if score_arr.ndim == 0:
        return float(score_arr)
    if score_arr.size == 1:
        return float(score_arr.reshape(()))
    raise ValueError(f"detection score must be scalar or size-1, got shape={score_arr.shape}")


class LocalOpenPIDetectorRuntime:
    def __init__(self, cfg: LocalDetectorConfig):
        from f_token.safe.conformal import ConformalSafetyManager
        from openpi.shared import nnx_utils as _nnx_utils

        self.cfg = cfg
        self.monitor, self.obs_adapter = _load_monitor(cfg)
        self.conformal = ConformalSafetyManager.from_json_files(detector_path=cfg.detector_conformal_path)
        self.reference_episode_length = None
        if cfg.task_stats_path:
            payload = json.loads(Path(cfg.task_stats_path).expanduser().read_text())
            task = payload["tasks"][str(int(cfg.task_index))]
            self.reference_episode_length = float(task["length"][cfg.task_length_subset]["mean"])
            if not np.isfinite(self.reference_episode_length) or self.reference_episode_length <= 0:
                raise ValueError(f"Invalid reference episode length: {self.reference_episode_length}")
        self._history_len = max(1, int(cfg.history_len_detection))
        self._feature_buffer: deque[Any] = deque(maxlen=self._history_len)
        self._serve_start_time = time.monotonic()

        self._encode_backbone = _nnx_utils.module_jit(self.monitor.encode_backbone_features)
        self._encode_safe_features = _nnx_utils.module_jit(self.monitor.encode_safe_features_from_backbone)
        self._call_detection = _nnx_utils.module_jit(self.monitor.call_detection_from_features)

        if not cfg.disable_warmup:
            self.warmup()

    def warmup(self) -> None:
        raw_obs = {
            "images": {
                "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            },
            "state.joints": np.zeros((12,), dtype=np.float32),
            "state.gripper_w": np.zeros((2,), dtype=np.float32),
            "state.ee_pos": np.zeros((6,), dtype=np.float32),
            "state.ee_rot": np.zeros((6,), dtype=np.float32),
            "state.ee_pos_cam": np.zeros((6,), dtype=np.float32),
            "state.ee_rot_cam": np.zeros((6,), dtype=np.float32),
            "prompt": "warmup",
        }

        obs = self.obs_adapter(raw_obs)
        backbone = self._encode_backbone(obs)
        history = _pad_backbone_history([backbone], self._history_len)
        safe_features = self._encode_safe_features(history)
        _ = self._call_detection(safe_features)
        logger.info("Local detector warmup complete")

    def infer(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        obs = self.obs_adapter(raw_obs)
        backbone = self._encode_backbone(obs)

        self._feature_buffer.append(backbone)
        history_backbone = _pad_backbone_history(list(self._feature_buffer), self._history_len)

        safe_features = self._encode_safe_features(history_backbone)
        score = self._call_detection(safe_features)

        score_scalar = _score_to_scalar(score)
        elapsed_seconds = max(0.0, time.monotonic() - self._serve_start_time)
        timestep = int(elapsed_seconds * max(1, int(self.cfg.conformal_timestep_frequency)))

        threshold_timestep = timestep
        if self.reference_episode_length is not None:
            relative_time = min(timestep / self.reference_episode_length, 1.0)
            threshold_timestep = int(relative_time * max(1, self.conformal.detection_band.values.size - 1))
        threshold = self.conformal.detection_threshold_at(threshold_timestep)
        is_dangerous = score_scalar > threshold

        return {
            "is_dangerous": bool(is_dangerous),
            "score": score_scalar,
            "threshold": float(threshold),
            "timestep": timestep,
        }


def validate_local_detector_paths(cfg: LocalDetectorConfig) -> None:
    required_paths = {
        "monitor_dir": cfg.monitor_dir,
        "detector_conformal_path": cfg.detector_conformal_path,
    }
    for name, path in required_paths.items():
        if not path:
            raise ValueError(f"`{name}` must be provided for local detector mode")
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    if cfg.task_stats_path and not Path(cfg.task_stats_path).exists():
        raise FileNotFoundError(f"task_stats_path does not exist: {cfg.task_stats_path}")

    if cfg.detector_head_dir and not Path(cfg.detector_head_dir).exists():
        raise FileNotFoundError(f"detector_head_dir does not exist: {cfg.detector_head_dir}")


def extract_episode_frame_for_visualization(raw_obs: dict[str, Any]) -> np.ndarray | None:
    try:
        from f_token.utils.episode_viz import extract_episode_frame

        return extract_episode_frame(raw_obs)
    except Exception:
        images = raw_obs.get("images", {})
        if not images:
            return None
        first_key = sorted(images.keys())[0]
        frame = np.asarray(images[first_key])
        if frame.ndim == 3 and frame.shape[0] in {1, 3}:
            frame = np.transpose(frame, (1, 2, 0))
        return frame


def _render_detector_episode_worker(
    records: list[dict[str, Any]],
    output_dir: str,
    fps: int,
    episode_idx: int,
) -> None:
    from f_token.utils.episode_viz import EpisodeTrajectory, render_episode_video

    episode_trace = EpisodeTrajectory()
    for item in records:
        frame = item.get("frame", None)
        if frame is None:
            continue
        episode_trace.append(
            frame=frame,
            detector_score=float(item["score"]),
            detector_band=float(item["threshold"]),
            failure_flag=bool(item["is_dangerous"]),
            title=str(item["title"]),
        )

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode_{episode_idx:06d}.mp4"
    render_episode_video(
        episode_trace,
        out_path,
        fps=max(1, int(fps)),
        episode_title=f"Local Detector Episode {episode_idx}",
    )


def render_detector_episode_in_process(
    records: list[dict[str, Any]],
    cfg: LocalDetectorConfig,
    episode_idx: int,
) -> multiprocessing.Process | None:
    if not cfg.render_episode_video or len(records) == 0:
        return None
    if "spawn" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("spawn")
    else:
        ctx = multiprocessing
    proc = ctx.Process(
        target=_render_detector_episode_worker,
        args=(records, cfg.render_output_dir, cfg.render_fps, episode_idx),
        daemon=False,
    )
    proc.start()
    return proc
