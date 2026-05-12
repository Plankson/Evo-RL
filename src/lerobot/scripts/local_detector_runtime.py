from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LocalDetectorConfig:
    monitor_config: str
    monitor_dir: str
    detector_conformal_path: str
    detector_head_dir: str | None = None
    detector_head_model_name: str = "auto"
    history_len_detection: int = 1
    conformal_timestep_frequency: int = 30
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


class _MonitorObservationAdapter:
    def __init__(self, transform):
        self._transform = transform

    def __call__(self, raw_obs: dict[str, Any]):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: x, raw_obs)
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
    obs_adapter = _MonitorObservationAdapter(_build_monitor_input_transform(train_cfg))
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

        threshold = self.conformal.detection_threshold_at(timestep)
        is_dangerous = self.conformal.is_dangerous_detection(score_scalar, timestep=timestep)

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

    if cfg.detector_head_dir and not Path(cfg.detector_head_dir).exists():
        raise FileNotFoundError(f"detector_head_dir does not exist: {cfg.detector_head_dir}")
