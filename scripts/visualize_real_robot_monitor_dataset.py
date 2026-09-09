#!/usr/bin/env python3
"""Render a LeRobot real-robot monitor episode with camera and risk traces.

The monitor recorder stores predictor/detector values in the same row as the
control observation. This script uses that row index as the synchronization
clock and keeps the prediction timestep in each plot title.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RISK_FIELDS = {
    "predictor": (
        "complementary_info.predictor_risk",
        "complementary_info.predictor_threshold",
        "complementary_info.predictor_timestep",
        "complementary_info.predictor_is_dangerous",
        "complementary_info.predictor_valid",
    ),
    "detector": (
        "complementary_info.detector_risk",
        "complementary_info.detector_threshold",
        "complementary_info.detector_timestep",
        "complementary_info.detector_is_dangerous",
        "complementary_info.detector_valid",
    ),
}


def _number(item: dict, key: str, default: float = np.nan) -> float:
    value = item.get(key)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    values = np.asarray(value).reshape(-1)
    return float(values[0]) if values.size else default


def _image_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if np.issubdtype(image.dtype, np.floating):
        if np.nanmax(image) <= 1.01:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _choose_camera(dataset: LeRobotDataset, requested: str | None) -> str:
    cameras = list(dataset.meta.camera_keys)
    if requested:
        if requested not in cameras:
            raise KeyError(f"Camera '{requested}' is not present. Available cameras: {cameras}")
        return requested
    priorities = ("right_front", "front", "cam_high", "head", "global")
    for token in priorities:
        matches = [key for key in cameras if token in key.lower()]
        if matches:
            return matches[0]
    if not cameras:
        raise RuntimeError("Dataset has no camera/video feature.")
    return cameras[0]


def _limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    pad = max((high - low) * 0.12, 0.05)
    return low - pad, high + pad


def _render_frame(
    image: np.ndarray,
    x: np.ndarray,
    values: dict[str, np.ndarray],
    index: int,
    episode: int,
    frame_index: float,
    camera_key: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), hspace=0.34, wspace=0.18)
    camera_ax = fig.add_subplot(grid[:, 0])
    camera_ax.imshow(image)
    camera_ax.set_title(f"{camera_key} | episode {episode} | frame {frame_index:.0f}")
    camera_ax.axis("off")

    for row, name in enumerate(("predictor", "detector")):
        ax = fig.add_subplot(grid[row, 1])
        risk, band = values[name]["risk"], values[name]["band"]
        ax.plot(x[: index + 1], risk[: index + 1], color="#1769aa", lw=1.8, label="risk")
        ax.plot(x[: index + 1], band[: index + 1], color="#d97706", ls="--", lw=1.5, label="band")
        if np.isfinite(risk[index]):
            color = "#c62828" if values[name]["danger"][index] > 0.5 else "#2e7d32"
            ax.scatter([x[index]], [risk[index]], color=color, s=34, zorder=4)
        ax.axvline(x[index], color="#777", lw=0.8, alpha=0.65)
        ax.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0] + 1))
        ax.set_ylim(*_limits(np.concatenate((risk, band))))
        ax.set_title(
            f"{name.title()}  risk={risk[index]:.4g}  band={band[index]:.4g}  "
            f"pred_t={values[name]['timestep'][index]:.0f}"
        )
        ax.set_xlabel("control timestep")
        ax.set_ylabel("risk")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Real Robot Monitor", fontsize=14)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _write_video(frames: Iterable[np.ndarray], output: Path, fps: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_iterator = iter(frames)
    first_frame = next(frame_iterator)
    height, width = first_frame.shape[:2]
    frame_count = 0
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for image in chain((first_frame,), frame_iterator):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
            frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Dataset name, or the final directory name under --root")
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to render")
    parser.add_argument("--camera-key", default=None, help="Camera feature key; auto-detected when omitted")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, episodes=[args.episode])
    camera_key = _choose_camera(dataset, args.camera_key)
    dataset._ensure_hf_dataset_loaded()
    rows = dataset.hf_dataset
    if len(rows) == 0:
        raise RuntimeError(f"Episode {args.episode} contains no frames")

    x = np.asarray([_number(row, "complementary_info.control_timestep", i) for i, row in enumerate(rows)])
    values = {}
    for name, fields in RISK_FIELDS.items():
        risk_key, band_key, timestep_key, danger_key, valid_key = fields
        risk = np.asarray([_number(row, risk_key) for row in rows])
        band = np.asarray([_number(row, band_key) for row in rows])
        valid = np.asarray([_number(row, valid_key, 1.0) for row in rows])
        risk[valid <= 0] = np.nan
        band[valid <= 0] = np.nan
        values[name] = {
            "risk": risk,
            "band": band,
            "timestep": np.asarray([_number(row, timestep_key) for row in rows]),
            "danger": np.asarray([_number(row, danger_key, 0.0) for row in rows]),
        }

    def rendered_frames():
        for index, row in enumerate(rows):
            sample = dataset[index]
            frame_index = _number(row, "frame_index", index)
            yield _render_frame(
                _image_array(sample[camera_key]), x, values, index, args.episode, frame_index, camera_key
            )
            if (index + 1) % 100 == 0:
                print(f"Rendered {index + 1}/{len(rows)} frames")

    frame_count = _write_video(rendered_frames(), args.output, args.fps)
    print(f"Wrote {frame_count} frames to {args.output} (camera={camera_key})")


if __name__ == "__main__":
    main()
