"""Serialization helpers for Robot IO server/client IPC.

Security note: pickle is used for trusted local-network IPC only.
Do not expose these sockets to untrusted networks or clients.
"""

from __future__ import annotations

import pickle  # nosec B403
from typing import Any

import numpy as np
import torch


def serialize_message(payload: Any) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_message(payload: bytes) -> Any:
    return pickle.loads(payload)  # nosec B301


def normalize_action_for_transport(action: Any) -> Any:
    """Convert actions to pickle-friendly CPU objects while preserving structure."""
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy()

    if isinstance(action, np.ndarray):
        return action

    if isinstance(action, dict):
        return {key: normalize_action_for_transport(value) for key, value in action.items()}

    if isinstance(action, list):
        return [normalize_action_for_transport(value) for value in action]

    if isinstance(action, tuple):
        return tuple(normalize_action_for_transport(value) for value in action)

    return action
