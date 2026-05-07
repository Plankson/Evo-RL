from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@dataclass
class RemoteMonitorEndpointConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    uri: str | None = None
    api_key: str | None = None

    protocol: str = "openpi_websocket"
    payload_format: str = "infer_pi0"
    policy_name: str = "pi0"

    chunk_size: int = 50
    n_action_steps: int = 50

    image_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "right_front": "cam_high",
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
        }
    )
    state_joints_indices: list[int] = field(default_factory=list)
    state_gripper_indices: list[int] = field(default_factory=list)
    convert_images_to_uint8: bool = True

    def __post_init__(self) -> None:
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


@PreTrainedConfig.register_subclass("remote_monitor")
@dataclass
class RemoteMonitorConfig(PreTrainedConfig):
    detector_remote: RemoteMonitorEndpointConfig = field(
        default_factory=lambda: RemoteMonitorEndpointConfig(port=8089)
    )
    predictor_remote: RemoteMonitorEndpointConfig = field(
        default_factory=lambda: RemoteMonitorEndpointConfig(port=8088)
    )

    host: str = "127.0.0.1"

    device: str | None = "cpu"
    use_amp: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.detector_remote.host == "127.0.0.1" and self.host != "127.0.0.1":
            self.detector_remote.host = self.host
        if self.predictor_remote.host == "127.0.0.1" and self.host != "127.0.0.1":
            self.predictor_remote.host = self.host

    def validate_features(self) -> None:
        return

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.predictor_remote.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("RemoteMonitor does not support training.")

    def get_scheduler_preset(self):
        raise NotImplementedError("RemoteMonitor does not support training.")
