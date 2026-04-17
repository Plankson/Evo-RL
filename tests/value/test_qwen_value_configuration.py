#!/usr/bin/env python

from lerobot.policies.factory import make_policy_config
from lerobot.values.qwen_value.configuration_qwen_value import QwenValueConfig


def test_qwen_value_config_from_dict():
    payload = {
        "type": "qwen_value",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "siglip_repo_id": "google/siglip-so400m-patch14-384",
        "dinov2_repo_id": "facebook/dinov2-base",
    }
    cfg = make_policy_config(payload.pop("type"), **payload)
    assert isinstance(cfg, QwenValueConfig)
    assert cfg.type == "qwen_value"
