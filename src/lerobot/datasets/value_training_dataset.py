#!/usr/bin/env python

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Sequence

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import (
    ALIGNED_STATE_NAMES,
    CANONICAL_CAMERA_KEYS,
    OBS_STATE,
    RAW_CAMERA_KEYS,
    STATE_GRIPPER_KEY,
    STATE_JOINTS_KEY,
)
from lerobot.utils.recording_annotations import EPISODE_SUCCESS


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _episode_table(dataset: LeRobotDataset) -> dict[str, list[Any]]:
    table = dataset.meta.episodes.with_format(None)[:]
    if dataset.episodes is None:
        return {key: list(values) for key, values in table.items()}

    selected = set(int(episode) for episode in dataset.episodes)
    episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    keep_mask = np.isin(episode_indices, list(selected))
    filtered: dict[str, list[Any]] = {}
    for key, values in table.items():
        filtered[key] = [value for value, keep in zip(values, keep_mask, strict=False) if bool(keep)]
    return filtered


def align_state(item: dict[str, Any]) -> torch.Tensor:
    _ensure(
        STATE_JOINTS_KEY in item and STATE_GRIPPER_KEY in item,
        f"Missing '{STATE_JOINTS_KEY}' or '{STATE_GRIPPER_KEY}' in value-training sample.",
    )
    joints = torch.as_tensor(item[STATE_JOINTS_KEY], dtype=torch.float32).reshape(-1)
    gripper = torch.as_tensor(item[STATE_GRIPPER_KEY], dtype=torch.float32).reshape(-1)
    _ensure(joints.shape[0] == 12, f"Expected '{STATE_JOINTS_KEY}' to be 12D, got {tuple(joints.shape)}.")
    _ensure(gripper.shape[0] == 2, f"Expected '{STATE_GRIPPER_KEY}' to be 2D, got {tuple(gripper.shape)}.")
    return torch.cat([joints[:6], gripper[:1], joints[6:12], gripper[1:2]], dim=0)


def align_state_stats(stats: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    _ensure(
        STATE_JOINTS_KEY in stats and STATE_GRIPPER_KEY in stats,
        f"Missing '{STATE_JOINTS_KEY}' or '{STATE_GRIPPER_KEY}' in value-training stats.",
    )
    joints_stats = stats[STATE_JOINTS_KEY]
    gripper_stats = stats[STATE_GRIPPER_KEY]
    aligned: dict[str, np.ndarray] = {}
    for stat_name, joints_value in joints_stats.items():
        joints_array = np.asarray(joints_value, dtype=np.float32).reshape(-1)
        gripper_array = np.asarray(gripper_stats[stat_name], dtype=np.float32).reshape(-1)
        if stat_name == "count":
            aligned[stat_name] = joints_array[:1]
            continue
        aligned[stat_name] = np.concatenate(
            [joints_array[:6], gripper_array[:1], joints_array[6:12], gripper_array[1:2]],
            axis=0,
        )
    return aligned


def _state_feature_spec() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (14,),
        "names": ALIGNED_STATE_NAMES,
    }


@dataclass(frozen=True)
class DatasetSlice:
    dataset: LeRobotDataset
    repo_id: str
    sample_start: int
    frame_offset: int
    episode_offset: int


@dataclass
class ValueTrainingDatasetMetadata:
    repo_id: str | list[str]
    info: dict[str, Any]
    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, np.ndarray]]
    tasks: pd.DataFrame
    episodes: datasets.Dataset
    subtasks: None = None

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def camera_keys(self) -> list[str]:
        return list(CANONICAL_CAMERA_KEYS)

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


class ValueTrainingLeRobotDataset(torch.utils.data.Dataset):
    """Concatenate multiple LeRobot datasets for value training with fixed camera/state alignment."""

    def __init__(self, datasets_: Sequence[LeRobotDataset], repo_ids: Sequence[str] | None = None):
        super().__init__()
        _ensure(bool(datasets_), "At least one dataset is required for value training.")

        self._datasets = list(datasets_)
        self.repo_ids = list(repo_ids) if repo_ids is not None else [dataset.repo_id for dataset in self._datasets]
        self.delta_timestamps = None

        reference_meta = self._datasets[0].meta

        task_names: list[str] = []
        seen_tasks: set[str] = set()
        episode_tables: list[datasets.Dataset] = []
        for dataset in tqdm(
            self._datasets,
            desc="Preparing value dataset metadata",
        ):
            episode_table = _episode_table(dataset)
            episode_tables.append(episode_table)

            candidate_tasks: list[str] = [str(name) for name in dataset.meta.tasks.index]
            for tasks in episode_table["tasks"]:
                if isinstance(tasks, list):
                    candidate_tasks.extend(str(task) for task in tasks)
                else:
                    candidate_tasks.append(str(tasks))

            for task_name in candidate_tasks:
                if task_name not in seen_tasks:
                    seen_tasks.add(task_name)
                    task_names.append(task_name)
        task_table = pd.DataFrame({"task_index": range(len(task_names))}, index=task_names)

        frame_rows: dict[str, list[int]] = {
            "episode_index": [],
            "frame_index": [],
            "index": [],
            "task_index": [],
        }
        episode_rows: dict[str, list[Any]] = {
            "episode_index": [],
            "tasks": [],
            "length": [],
            "episode_success": [],
        }

        per_dataset_stats: list[dict[str, dict[str, np.ndarray]]] = []
        dataset_slices: list[DatasetSlice] = []
        sample_starts: list[int] = []
        sample_offset = 0
        frame_offset = 0
        episode_offset = 0

        for dataset, repo_id, episode_table in tqdm(
            list(zip(self._datasets, self.repo_ids, episode_tables, strict=True)),
            desc="Building value dataset",
        ):
            sample_starts.append(sample_offset)
            dataset_slices.append(
                DatasetSlice(
                    dataset=dataset,
                    repo_id=repo_id,
                    sample_start=sample_offset,
                    frame_offset=frame_offset,
                    episode_offset=episode_offset,
                )
            )
            sample_offset += len(dataset)

            raw_frames = dataset.hf_dataset.with_format(None)
            frame_episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
            frame_rows["episode_index"].extend((frame_episode_indices + episode_offset).tolist())
            frame_rows["frame_index"].extend(np.asarray(raw_frames["frame_index"], dtype=np.int64).tolist())
            frame_rows["index"].extend((np.asarray(raw_frames["index"], dtype=np.int64) + frame_offset).tolist())

            episode_task_names: dict[int, str] = {}
            for ep_idx, tasks in zip(episode_table["episode_index"], episode_table["tasks"], strict=True):
                ep_idx_int = int(ep_idx)
                if isinstance(tasks, list):
                    episode_task_names[ep_idx_int] = str(tasks[0])
                else:
                    episode_task_names[ep_idx_int] = str(tasks)
            frame_task_indices = [
                int(task_table.loc[episode_task_names[int(ep_idx)], "task_index"]) for ep_idx in frame_episode_indices
            ]
            frame_rows["task_index"].extend(frame_task_indices)

            row_count = len(episode_table["episode_index"])
            episode_rows["episode_index"].extend(
                int(value) + episode_offset for value in episode_table["episode_index"]
            )
            episode_rows["tasks"].extend(episode_table["tasks"])
            episode_rows["length"].extend(int(value) for value in episode_table["length"])
            episode_rows["episode_success"].extend(
                episode_table.get("episode_success", [EPISODE_SUCCESS] * row_count)
            )

            aligned_stats: dict[str, dict[str, np.ndarray]] = {}
            for raw_camera_key, canonical_camera_key in zip(RAW_CAMERA_KEYS, CANONICAL_CAMERA_KEYS, strict=True):
                aligned_stats[canonical_camera_key] = {
                    stat_name: np.asarray(stat_value)
                    for stat_name, stat_value in dataset.meta.stats[raw_camera_key].items()
                }
            aligned_stats[OBS_STATE] = align_state_stats(dataset.meta.stats)
            per_dataset_stats.append(aligned_stats)

            frame_indices = np.asarray(raw_frames["index"], dtype=np.int64)
            if len(frame_indices) > 0:
                frame_offset += int(frame_indices.max()) + 1

            episode_indices = np.asarray(episode_table["episode_index"], dtype=np.int64)
            if len(episode_indices) > 0:
                episode_offset += int(episode_indices.max()) + 1

        self.dataset_slices = dataset_slices
        self.sample_offsets = sample_starts
        self.task_table = task_table
        self.hf_dataset = datasets.Dataset.from_dict(frame_rows)

        features = {
            canonical_camera_key: dict(reference_meta.features[raw_camera_key])
            for raw_camera_key, canonical_camera_key in zip(RAW_CAMERA_KEYS, CANONICAL_CAMERA_KEYS, strict=True)
        }
        features[OBS_STATE] = _state_feature_spec()

        episodes = datasets.Dataset.from_dict(episode_rows)
        stats = aggregate_stats(per_dataset_stats)
        self.meta = ValueTrainingDatasetMetadata(
            repo_id=self.repo_ids if len(self.repo_ids) > 1 else self.repo_ids[0],
            info={
                **reference_meta.info,
                "features": features,
                "total_frames": len(self.hf_dataset),
                "total_episodes": len(episodes),
                "total_tasks": len(task_table),
            },
            features=features,
            stats=stats,
            tasks=task_table,
            episodes=episodes,
        )

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.meta.episodes)

    def __len__(self) -> int:
        return self.num_frames

    def _locate_dataset(self, index: int) -> tuple[DatasetSlice, int]:
        _ensure(0 <= index < len(self), f"Index {index} out of bounds for dataset of length {len(self)}.")
        dataset_idx = bisect_right(self.sample_offsets, index) - 1
        dataset_slice = self.dataset_slices[dataset_idx]
        return dataset_slice, index - dataset_slice.sample_start

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_slice, local_index = self._locate_dataset(index)
        item = dataset_slice.dataset[local_index]
        sample = {
            "task": str(item["task"]),
            "source_repo_id": dataset_slice.repo_id,
            "index": torch.tensor(int(item["index"].item()) + dataset_slice.frame_offset, dtype=torch.long),
            "episode_index": torch.tensor(
                int(item["episode_index"].item()) + dataset_slice.episode_offset,
                dtype=torch.long,
            ),
            OBS_STATE: align_state(item),
        }
        for raw_camera_key, canonical_camera_key in zip(RAW_CAMERA_KEYS, CANONICAL_CAMERA_KEYS, strict=True):
            sample[canonical_camera_key] = item[raw_camera_key]
        return sample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_ids={self.repo_ids},\n"
            f"  num_frames={self.num_frames},\n"
            f"  num_episodes={self.num_episodes},\n"
            f"  camera_keys={self.meta.camera_keys},\n"
            ")"
        )
