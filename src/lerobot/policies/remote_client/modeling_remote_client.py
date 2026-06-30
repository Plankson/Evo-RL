#!/usr/bin/env python

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, OBS_STATE

from .configuration_remote_client import RemoteClientConfig


def client_image_key(local_key: str, image_key_map: dict[str, str]) -> str:
    short_key = local_key.removeprefix(f"{OBS_IMAGES}.")
    if local_key in image_key_map:
        return image_key_map[local_key]
    if short_key in image_key_map:
        return image_key_map[short_key]
    return short_key


def _uses_da3_image_list(policy_name: str) -> bool:
    return policy_name.lower() == "da3"


def _is_openvla_oft_policy(policy_name: str | None) -> bool:
    return (policy_name or "").lower().replace("_", "-") in {"openvla-oft", "openvlaoft"}


def duplicate_current_frame_and_stack(frame: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError(f"Expected count >= 1, got {count}.")
    frames = [frame.copy() for _ in range(count)]
    return np.stack(frames, axis=0)


def image_tensor_to_uint8_chw(image: Tensor, convert_images_to_uint8: bool = True) -> np.ndarray:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image tensor, got shape={tuple(image.shape)}")
        image = image.squeeze(0)

    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")

    image = image.detach().cpu()
    if not convert_images_to_uint8:
        return image.numpy()

    if torch.is_floating_point(image):
        image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    return image.numpy()


def _default_state_layout_for_policy_name(policy_name: str) -> str:
    policy_name = policy_name.lower()
    if policy_name in {"pi0", "pi05", "pi0_fast", "infer_pi0", "joints12"}:
        return "pi0"
    if policy_name in {"ace_policy", "qwenvla", "zero_pad"}:
        return "ace_policy"
    if policy_name in {"da3"}:
        return "da3"
    if _is_openvla_oft_policy(policy_name):
        return "openvla_oft"
    raise ValueError(
        f"Unsupported policy_name '{policy_name}' for automatic state layout selection."
    )


def split_state_vector(
    state: Tensor,
    joints_indices: list[int],
    gripper_indices: list[int],
    policy_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for state tensor, got shape={tuple(state.shape)}")
        state = state.squeeze(0)

    if state.ndim != 1:
        raise ValueError(f"Expected 1D state tensor after squeeze, got shape={tuple(state.shape)}")

    state = state.detach().cpu()

    if joints_indices or gripper_indices:
        joints = state[joints_indices].numpy() if joints_indices else np.empty((0,), dtype=np.float32)
        gripper = state[gripper_indices].numpy() if gripper_indices else np.empty((0,), dtype=np.float32)
        return joints, gripper

    if state.shape[0] != 14:
        raise ValueError(
            "Remote infer_pi0 payload requires explicit `state_joints_indices` and "
            "`state_gripper_indices` when observation.state is not 14D."
        )

    gripper = state[[6, 13]].numpy()
    state_layout = _default_state_layout_for_policy_name(policy_name)
    if state_layout in {"pi0", "openvla_oft"}:
        joints = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].numpy()
        if state_layout == "openvla_oft":
            gripper = np.clip(gripper / 0.07, 0.0, 1.0)
    elif state_layout == "ace_policy":
        joints = torch.cat(
            (
                state[:6],
                torch.ones(1, dtype=state.dtype) * -10000,
                state[7:13],
                torch.ones(1, dtype=state.dtype) * -10000,
            )
        ).numpy()
    elif state_layout == "da3":
        joints = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].numpy()
    else:
        raise AssertionError(f"Unexpected state layout '{state_layout}'")
    return joints, gripper


def _as_bool(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(value.detach().cpu().flatten()[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def batch_to_client_observation(
    batch: dict[str, Any],
    config: RemoteClientConfig,
) -> dict[str, Any]:
    if OBS_STATE not in batch:
        raise KeyError(f"Missing required state key '{OBS_STATE}' for remote infer_pi0 payload.")

    use_da3_image_list = _uses_da3_image_list(config.policy_name)
    images = {}
    for key, value in batch.items():
        if not key.startswith(f"{OBS_IMAGES}."):
            continue
        remote_key = client_image_key(key, config.image_key_map)
        frame = image_tensor_to_uint8_chw(
            value,
            convert_images_to_uint8=config.convert_images_to_uint8,
        )
        if use_da3_image_list:
            images[remote_key] = duplicate_current_frame_and_stack(frame, config.n_image_history_steps)
        else:
            images[remote_key] = frame

    print(
        "remote_client images:",
        f"n_image_history_steps={config.n_image_history_steps}",
        {key: value.shape for key, value in images.items()},
    )

    joints, gripper = split_state_vector(
        batch[OBS_STATE],
        joints_indices=config.state_joints_indices,
        gripper_indices=config.state_gripper_indices,
        policy_name=config.policy_name,
    )

    task = batch.get("task", "")
    if not isinstance(task, str):
        raise TypeError(f"Expected task to be a string, got {type(task)}")

    is_pi = _default_state_layout_for_policy_name(config.policy_name)
    if is_pi in {"pi0", "openvla_oft"}:
        padding = np.ones((6,), dtype=np.float32)*-10000
    else:
        padding = np.ones((1,6), dtype=np.float32)*-10000
        joints = joints[None, :]
        gripper = gripper[None, :]
    return {
        "images": images,
        "state.joints": joints,
        "state.gripper_w": gripper,
        #TODO: currently, ee_pos/ee_rot/ee_pos_cam/ee_rot_cam are zero numpy ndarrays,
        "state.ee_pos": padding,
        "state.ee_rot": padding,
        "state.ee_pos_cam": padding,   
        "state.ee_rot_cam": padding,
        "prompt": task,
    }


def current_state_vector(state: Tensor) -> np.ndarray:
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for state tensor, got shape={tuple(state.shape)}")
        state = state.squeeze(0)
    if state.ndim != 1:
        raise ValueError(f"Expected 1D state tensor after squeeze, got shape={tuple(state.shape)}")
    return state.detach().cpu().numpy().astype(np.float32, copy=True)


def smooth_openvla_oft_action_chunk(
    action_chunk: np.ndarray,
    current_state: Tensor,
    config: RemoteClientConfig,
) -> np.ndarray:
    if not _is_openvla_oft_policy(config.policy_name) or not config.openvla_oft_smooth_actions:
        return action_chunk

    current = current_state_vector(current_state)
    if current.shape[-1] != action_chunk.shape[-1]:
        raise ValueError(
            f"Cannot smooth OpenVLA-OFT actions: current state dim {current.shape[-1]} "
            f"!= action dim {action_chunk.shape[-1]}"
        )

    smoothed = action_chunk.astype(np.float32, copy=True)
    n_steps = smoothed.shape[0]
    interpolate_steps = min(int(config.openvla_oft_interpolate_start_steps), n_steps)
    if interpolate_steps > 0:
        denom = max(1, interpolate_steps)
        for i in range(interpolate_steps):
            alpha = i / float(denom)
            smoothed[i] = current * (1.0 - alpha) + smoothed[i] * alpha

    gripper_idx = np.array([6, 13], dtype=np.int64)
    joint_idx = np.array([i for i in range(smoothed.shape[-1]) if i not in {6, 13}], dtype=np.int64)
    max_joint_delta = float(config.openvla_oft_max_joint_delta_rad)
    max_gripper_delta = float(config.openvla_oft_max_gripper_delta)

    prev = current.copy()
    for i in range(n_steps):
        target = smoothed[i]
        limited = target.copy()
        limited[joint_idx] = prev[joint_idx] + np.clip(
            target[joint_idx] - prev[joint_idx],
            -max_joint_delta,
            max_joint_delta,
        )
        limited[gripper_idx] = prev[gripper_idx] + np.clip(
            target[gripper_idx] - prev[gripper_idx],
            -max_gripper_delta,
            max_gripper_delta,
        )
        smoothed[i] = limited
        prev = limited

    return smoothed


def normalize_remote_action_chunk(result: dict[str, Any], expected_action_dim: int | None = None, policy_name: str = None) -> np.ndarray:
    if "actions" not in result:
        raise KeyError("Remote server response is missing required 'actions' field.")

    action_chunk = np.asarray(result["actions"], dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[np.newaxis, :]

    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (T, D), got {action_chunk.shape}")

    if policy_name == 'da3':
        action_chunk = np.concatenate(
            [action_chunk[..., :6], action_chunk[..., 7:8], action_chunk[..., 8:8+6], action_chunk[..., 15:16]], axis=-1
        )
    if _is_openvla_oft_policy(policy_name):
        action_chunk = action_chunk.copy()
        action_chunk[..., [6, 13]] = np.clip(action_chunk[..., [6, 13]], 0.0, 1.0) * 0.07
    if policy_name=='ace_policy':
        action_chunk = np.concatenate(
            [action_chunk[..., :6], action_chunk[..., 7:8], action_chunk[..., 8:8+6], action_chunk[..., 15:16]], axis=-1
        )
    if expected_action_dim is not None and action_chunk.shape[1] != expected_action_dim:
        raise ValueError(
            f"Remote action dim mismatch: expected {expected_action_dim}, got {action_chunk.shape[1]}"
        )
    return action_chunk


class RemoteClientPolicy(PreTrainedPolicy):
    config_class = RemoteClientConfig
    name = "remote_client"

    def __init__(self, config: RemoteClientConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._action_queue: deque[Tensor] = deque()
        self._client = self._make_client()
        print('success load remote client')

    def _make_client(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        return WebsocketClientPolicy(host=self.config.host, port=self.config.port)

    def _get_client(self):
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def get_optim_params(self) -> dict:
        raise NotImplementedError("RemoteClientPolicy does not support training.")

    def reset(self):
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("RemoteClientPolicy does not support training.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        del kwargs
        self.eval()

        if self.config.payload_format != "infer_pi0":
            raise ValueError(f"Unsupported payload_format: {self.config.payload_format}")

        observation = batch_to_client_observation(batch, self.config)
        result = self._get_client().infer(observation)
        # print('result:',result.keys(), result, batch['observation.state'])
        # import pdb;pdb.set_trace()

        expected_action_dim = None
        if ACTION in self.config.output_features:
            expected_action_dim = self.config.output_features[ACTION].shape[0]

        action_chunk = normalize_remote_action_chunk(result, expected_action_dim=expected_action_dim, policy_name=self.config.policy_name)
        action_chunk = smooth_openvla_oft_action_chunk(action_chunk, batch[OBS_STATE], self.config)
        return torch.from_numpy(action_chunk).unsqueeze(0)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        del kwargs
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()
