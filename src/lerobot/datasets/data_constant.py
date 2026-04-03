#!/usr/bin/env python

"""Shared dataset path constants for local training and evaluation scripts."""

from __future__ import annotations

from typing import Any

ace_fold_cloth_v2 = [
    "/data/dataset/data_fold_cloth_zql_0203_ck_lerobot",
    "/data/dataset/data_fold_cloth_zql_0204_ck2_lerobot",
    "/data/dataset/data_fold_cloth_wd_0105_1_lerobot",
    "/data/dataset/data_fold_cloth_wd_0105_2_lerobot",
    "/data/dataset/data_fold_cloth_1109_1_trim",
    "/data/dataset/data_fold_cloth_1210_lerobot",
    "/data/dataset/data_fold_cloth_1117_trim_edit",
    "/data/dataset/data_fold_cloth_1110_small_edit1",
    "/data/dataset/data_fold_cloth_1119_move",
    "/data/dataset/data_fold_cloth_zy_1211_lerobot",
    "/data/dataset/data_fold_cloth_1216lerobot",
    "/data/dataset/data_fold_cloth__wd_zd_0112_lerobot",
]

ace_fold_cloth_v2_debug = [
    "/data/dataset/data_fold_cloth__wd_zd_0112_lerobot",
]

__all__ = [
    "ace_fold_cloth_v2",
    "ace_fold_cloth_v2_debug",
    "resolve_dataset_repo_id",
]


def resolve_dataset_repo_id(repo_id: str | list[str]) -> str | list[str]:
    resolved: Any = globals().get(repo_id)
    if isinstance(resolved, list) and all(isinstance(item, str) for item in resolved):
        return list(resolved)
    return repo_id
