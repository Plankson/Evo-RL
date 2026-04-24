from dataclasses import dataclass, field
from typing import List, Optional, Any
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.remote_client.configuration_remote_client import RemoteClientConfig

@PreTrainedConfig.register_subclass("remote_monitor")
@dataclass
class RemoteMonitorConfig(PreTrainedConfig):
    # Remote configurations
    detector_remote: RemoteClientConfig = field(default_factory=lambda: RemoteClientConfig(port=8089))
    predictor_remote: RemoteClientConfig = field(default_factory=lambda: RemoteClientConfig(port=8088))
    
    # Common host
    host: str = "127.0.0.1"
    
    # Policy execution settings
    device: str | None = "cpu"
    use_amp: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Sync host
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
