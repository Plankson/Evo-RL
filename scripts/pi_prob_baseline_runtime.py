"""Runtime adapters for the Pi-Prob/SAFE baseline monitors.

The adapter deliberately mirrors the offline model contracts:
indep/logpZO consume ``feature``; RND consumes ``feature`` plus a flattened
action chunk.  Bands are the JSON artifacts emitted by SAFE's conformal
evaluation and are mapped on the normalized episode-time axis.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_paths() -> None:
    roots = [
        Path(os.environ.get("PI_PROB_ROOT", "/data/users/liujingyuan/workspace/pi_prob")),
        Path(os.environ.get("SAFE_ROOT", "/data/users/liujingyuan/third_party/SAFE")),
    ]
    for root in reversed(roots):
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


_bootstrap_paths()


@dataclass
class JsonRiskBand:
    values: np.ndarray
    reference_length: float
    alpha: float

    @classmethod
    def load(cls, path: str | Path, *, alpha: float, reference_length: float) -> "JsonRiskBand":
        p = Path(path).expanduser()
        if p.is_dir():
            p = p / "json" / "classify_cp_functional__model_bands.json"
        payload = json.loads(p.read_text())
        key = str(float(alpha))
        if key not in payload:
            candidates = sorted((float(k), k) for k in payload)
            if not candidates:
                raise ValueError(f"No alpha bands in {p}")
            key = min(candidates, key=lambda item: abs(item[0] - float(alpha)))[1]
        values = np.asarray(payload[key], dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid band values in {p} alpha={key}")
        return cls(values=values, reference_length=float(reference_length), alpha=float(key))

    def threshold_at(self, timestep: int) -> float:
        rel = min(max(float(timestep), 0.0) / max(self.reference_length, 1.0), 1.0)
        idx = min(int(rel * self.values.size), self.values.size - 1)
        return float(self.values[idx])


class PiProbBaselineRuntime:
    def __init__(
        self,
        *,
        baseline: str,
        config_path: str,
        checkpoint_path: str,
        band_path: str,
        feature_dim: int,
        action_dim: int = 14,
        action_horizon: int = 36,
        device: str = "cuda",
        alpha: float = 0.1,
        reference_length: float = 1000.0,
    ) -> None:
        import torch
        from omegaconf import OmegaConf
        from failure_prob.conf import process_cfg
        from failure_prob.model import get_model
        from pi_prob.utils.load import load_checkpoint

        self.torch = torch
        self.baseline = str(baseline).lower()
        if self.baseline not in {"indep", "logpzo", "rnd"}:
            raise ValueError(f"Unsupported Pi-Prob baseline: {baseline}")
        cfg_file = Path(config_path).expanduser().resolve()
        if not cfg_file.is_file():
            raise FileNotFoundError(cfg_file)
        cfg = process_cfg(OmegaConf.load(cfg_file))

        self.model_cfg = cfg
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        model_input_dim = self.feature_dim
        model = get_model(cfg, model_input_dim)
        device_obj = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        model.to(device_obj)
        load_checkpoint(model, Path(checkpoint_path), device_obj)
        model.eval()
        self.model = model
        self.device = device_obj
        self.band = JsonRiskBand.load(band_path, alpha=alpha, reference_length=reference_length)
        self._mean = None
        self._std = None
        train_dir = Path(checkpoint_path).expanduser().parent
        mean_path, std_path = train_dir / "feat_mean.npy", train_dir / "feat_std.npy"
        if bool(getattr(cfg.dataset, "normalize_hidden_states", False)) and mean_path.is_file() and std_path.is_file():
            self._mean = np.load(mean_path).astype(np.float32)
            self._std = np.maximum(np.load(std_path).astype(np.float32), 1e-6)

    def _feature(self, feature: np.ndarray) -> np.ndarray:
        x = np.asarray(feature, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, None]
        elif x.ndim == 2:
            x = x[:, None]
        elif x.ndim != 3:
            raise ValueError(f"feature must be [D], [B,D], or [B,T,D], got {x.shape}")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"feature dim {x.shape[-1]} != trained dim {self.feature_dim}")
        if self._mean is not None:
            x = (x - self._mean.reshape(1, 1, -1)) / self._std.reshape(1, 1, -1)
        return x

    def infer(self, feature: np.ndarray, *, action: np.ndarray | None = None, timestep: int = 0) -> dict[str, Any]:
        x = self._feature(feature)
        batch: dict[str, Any] = {"feature": self.torch.from_numpy(x).to(self.device)}
        if self.baseline == "rnd":
            if action is None:
                raise ValueError("rnd baseline requires action")
            a = np.asarray(action, dtype=np.float32)
            if a.ndim == 2:
                a = a[None]
            a = a[..., : self.action_dim]
            if a.shape[1] != self.action_horizon:
                raise ValueError(f"rnd action horizon {a.shape[1]} != {self.action_horizon}")
            # RND accepts either key; keep only ``action`` so its legacy
            # ``batch.get('action_vectors') or batch.get('action')`` fallback
            # does not evaluate a multi-element tensor as a boolean.
            batch["action"] = self.torch.from_numpy(a.reshape(a.shape[0], 1, -1)).to(self.device)
        with self.torch.no_grad():
            score = self.model(batch)
        value = float(np.asarray(score.detach().float().cpu()).reshape(-1)[-1])
        threshold = self.band.threshold_at(timestep)
        return {
            "score": value,
            "threshold": threshold,
            "is_dangerous": bool(value > threshold),
            "timestep": int(timestep),
            "baseline": self.baseline,
        }


class AccelBaselineRuntime:
    def __init__(
        self,
        *,
        band_path: str,
        scale_path: str | None = None,
        action_dim: int = 14,
        n_exec: int | None = None,
        prefix_steps: int = 9,
        alpha: float = 0.1,
        reference_length: float = 1000.0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.n_exec = None if n_exec is None else int(n_exec)
        self.prefix_steps = int(prefix_steps)
        self.scale = None
        if scale_path:
            self.scale = np.asarray(np.load(Path(scale_path).expanduser()), dtype=np.float32)
        self.band = JsonRiskBand.load(band_path, alpha=alpha, reference_length=reference_length)

    def infer(self, fm_path: np.ndarray, *, timestep: int = 0) -> dict[str, Any]:
        from pi_prob.training_free import accel_scores

        path = np.asarray(fm_path, dtype=np.float32)
        if path.ndim == 3:
            path = path[None]
        score = accel_scores(
            path,
            scale=self.scale,
            n_exec=self.n_exec,
            action_dim=self.action_dim,
            prefix_steps=self.prefix_steps,
            scale_mode="official_chunk" if self.scale is None else "calibration",
        )
        value = float(np.asarray(score).reshape(-1)[-1])
        threshold = self.band.threshold_at(timestep)
        return {
            "score": value,
            "threshold": threshold,
            "is_dangerous": bool(value > threshold),
            "timestep": int(timestep),
            "baseline": "accel",
        }
