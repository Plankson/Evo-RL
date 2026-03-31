#!/usr/bin/env python

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGES, OBS_STATE


def _map_camera_name(local_key: str, camera_name_map: dict[str, str] | None) -> str:
    if not camera_name_map:
        return local_key
    full_key = f"{OBS_IMAGES}.{local_key}"
    return camera_name_map.get(full_key, camera_name_map.get(local_key, local_key))


class HDF5EpisodeRecorder:
    """Episode sink that preserves the recording loop API but writes HDF5 files."""

    def __init__(
        self,
        repo_id: str,
        fps: int,
        root: str | Path | None,
        robot_type: str,
        features: dict[str, dict[str, Any]],
        camera_name_map: dict[str, str] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = SimpleNamespace(
            features=features,
            fps=fps,
            stats={},
            robot_type=robot_type,
            root=self.root,
        )
        self._episode_buffer: list[dict[str, Any]] = []
        self._camera_specs = self._build_camera_specs(camera_name_map)
        self._action_dim = int(features[ACTION]["shape"][0])
        self._state_dim = int(features[OBS_STATE]["shape"][0])
        self.num_episodes = self._discover_existing_episode_count()

    def _build_camera_specs(
        self, camera_name_map: dict[str, str] | None
    ) -> list[tuple[str, str]]:
        specs: list[tuple[str, str]] = []
        prefix = f"{OBS_IMAGES}."
        for key, feature in self.features.items():
            if not key.startswith(prefix):
                continue
            if feature.get("dtype") not in {"image", "video"}:
                continue
            local_key = key.removeprefix(prefix)
            specs.append((local_key, _map_camera_name(local_key, camera_name_map)))
        return specs

    def _discover_existing_episode_count(self) -> int:
        existing = sorted(self.root.glob("episode_*.hdf5"))
        episode_indices: list[int] = []
        for path in existing:
            stem = path.stem
            try:
                episode_indices.append(int(stem.removeprefix("episode_")))
            except ValueError:
                continue
        return (max(episode_indices) + 1) if episode_indices else 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        copied: dict[str, Any] = {}
        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                copied[key] = np.array(value, copy=True)
            else:
                copied[key] = value
        self._episode_buffer.append(copied)

    def clear_episode_buffer(self) -> None:
        self._episode_buffer.clear()

    def _stack_vector(self, key: str, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        default = np.zeros((size,), dtype=dtype)
        return np.stack(
            [
                np.asarray(frame.get(key, default), dtype=dtype)
                for frame in self._episode_buffer
            ],
            axis=0,
        )

    def save_episode(self, extra_episode_metadata: dict[str, Any] | None = None) -> Path | None:
        if not self._episode_buffer:
            logging.warning("Episode buffer is empty; skipping HDF5 save.")
            return None

        episode_idx = self.num_episodes
        file_path = self.root / f"episode_{episode_idx}.hdf5"
        qpos = self._stack_vector(OBS_STATE, self._state_dim)
        action = self._stack_vector(ACTION, self._action_dim)
        qvel = np.zeros_like(qpos, dtype=np.float32)
        effort = np.zeros_like(qpos, dtype=np.float32)
        base_action = np.zeros((len(self._episode_buffer), 2), dtype=np.float32)
        ee_pos = np.zeros((len(self._episode_buffer), 6), dtype=np.float32)
        ee_rot = np.zeros((len(self._episode_buffer), 6), dtype=np.float32)

        with h5py.File(file_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["fps"] = self.fps
            root.attrs["repo_id"] = self.repo_id
            root.attrs["robot_type"] = self.meta.robot_type

            task = self._episode_buffer[0].get("task", "")
            if task:
                root.attrs["task"] = task

            if extra_episode_metadata:
                for key, value in extra_episode_metadata.items():
                    if value is not None:
                        root.attrs[key] = value

            obs_group = root.create_group("observations")
            images_group = obs_group.create_group("images")

            for local_key, hdf5_key in self._camera_specs:
                frame_key = f"{OBS_IMAGES}.{local_key}"
                images = np.stack(
                    [np.asarray(frame[frame_key]) for frame in self._episode_buffer],
                    axis=0,
                )
                images_group.create_dataset(
                    hdf5_key,
                    data=images,
                    dtype=images.dtype,
                    chunks=(1, *images.shape[1:]),
                )

            obs_group.create_dataset("qpos", data=qpos, dtype=np.float32)
            obs_group.create_dataset("qvel", data=qvel, dtype=np.float32)
            obs_group.create_dataset("effort", data=effort, dtype=np.float32)
            obs_group.create_dataset("ee_pos", data=ee_pos, dtype=np.float32)
            obs_group.create_dataset("ee_rot", data=ee_rot, dtype=np.float32)
            root.create_dataset("action", data=action, dtype=np.float32)
            root.create_dataset("base_action", data=base_action, dtype=np.float32)

            if "complementary_info.policy_action" in self._episode_buffer[0]:
                policy_action = self._stack_vector("complementary_info.policy_action", self._action_dim)
                root.create_dataset("policy_action", data=policy_action, dtype=np.float32)

        self.num_episodes += 1
        self.clear_episode_buffer()
        logging.info("Saved HDF5 episode to %s", file_path)
        return file_path

    def finalize(self) -> None:
        return

