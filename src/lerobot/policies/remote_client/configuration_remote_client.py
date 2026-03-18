#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("remote_client")
@dataclass
class RemoteClientConfig(PreTrainedConfig):
    # Local-side execution only needs CPU tensors before sending them over the wire.
    device: str | None = "cpu"
    use_amp: bool = False

    host: str = "127.0.0.1"
    port: int = 8000
    uri: str | None = None
    api_key: str | None = None

    protocol: str = "openpi_websocket"
    payload_format: str = "infer_pi0"
    policy_name: str = "pi0"

    chunk_size: int = 50
    n_action_steps: int = 50

    # Maps local LeRobot image keys to remote keys.
    # Both full keys (`observation.images.*`) and short keys are accepted.
    # Default is aligned with the current bi_piper_follower fold-clothes setup.
    image_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "right_front": "cam_high",
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        }
    )

    # Required when the remote server expects joints / grippers split out of observation.state.
    state_joints_indices: list[int] = field(default_factory=list)
    state_gripper_indices: list[int] = field(default_factory=list)

    convert_images_to_uint8: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.protocol != "openpi_websocket":
            raise ValueError(f"Unsupported protocol: {self.protocol}")

        if self.payload_format != "infer_pi0":
            raise ValueError(f"Unsupported payload_format: {self.payload_format}")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        return

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("RemoteClientPolicy does not support training.")

    def get_scheduler_preset(self):
        raise NotImplementedError("RemoteClientPolicy does not support training.")
