import logging
from collections import deque
from typing import Dict, Any, Optional
import torch

from lerobot.policies.pretrained import PreTrainedPolicy, ActionSelectKwargs
from lerobot.policies.remote_client.modeling_remote_client import (
    batch_to_client_observation,
    normalize_remote_action_chunk,
)
from lerobot.utils.constants import ACTION
from .configuration_remote_monitor import RemoteMonitorConfig

logger = logging.getLogger(__name__)

class RemoteMonitorPolicy(PreTrainedPolicy):
    """
    Interface for both Predictor and Detector.
    Concurrency is managed by the recording script.
    """
    config_class = RemoteMonitorConfig
    name = "remote_monitor"

    def __init__(self, config: RemoteMonitorConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._action_queue: deque[torch.Tensor] = deque()

        self.predictor_client = None
        self.detector_client = None
        self.last_predictor_safety: Optional[Dict[str, Any]] = None

    def _make_client(self, endpoint_config):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        return WebsocketClientPolicy(
            host=endpoint_config.host,
            port=endpoint_config.port,
            api_key=endpoint_config.api_key,
        )

    def _get_predictor_client(self):
        if self.predictor_client is None:
            self.predictor_client = self._make_client(self.config.predictor_remote)
        return self.predictor_client

    def _get_detector_client(self):
        if self.detector_client is None:
            self.detector_client = self._make_client(self.config.detector_remote)
        return self.detector_client

    def reset(self):
        self._action_queue.clear()
        self.last_predictor_safety = None

    @torch.no_grad()
    def infer_detector(self, batch: dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Support function for the background detector process.
        """
        client_obs = batch_to_client_observation(batch, self.config.detector_remote)
        result = self._get_detector_client().infer(client_obs)
        return {
            "source": "detector",
            "is_dangerous": result.get("is_dangerous", False),
            "score": result.get("score", 0.0),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        client_obs = batch_to_client_observation(batch, self.config.predictor_remote)
        result = self._get_predictor_client().infer(client_obs)
        
        # Store safety info in the mailbox
        self.last_predictor_safety = {
            "source": "predictor",
            "is_dangerous": result.get("is_dangerous", False),
            "score": result.get("safety_score", 0.0),
        }
        
        # Extract actions
        expected_dim = self.config.output_features.get(ACTION, None)
        if expected_dim: expected_dim = expected_dim.shape[0]
        
        action_chunk = normalize_remote_action_chunk(
            result, 
            expected_action_dim=expected_dim, 
            policy_name=self.config.predictor_remote.policy_name
        )
        return torch.from_numpy(action_chunk).unsqueeze(0)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: ActionSelectKwargs) -> torch.Tensor:
        # standard caching logic
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.predictor_remote.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        else:
            self.last_predictor_safety = None # Only new info when queue was empty

        return self._action_queue.popleft()

    def get_optim_params(self) -> dict:
        raise NotImplementedError()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        raise NotImplementedError()
