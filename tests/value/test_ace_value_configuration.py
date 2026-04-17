#!/usr/bin/env python

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import make_policy_config
from lerobot.values.ace_value.configuration_ace_value import AceValueConfig


def test_ace_value_config_from_dict():
    payload = {
        "type": "ace_value",
        "num_bins": 101,
        "bin_min": -1.0,
        "bin_max": 0.0,
        "task_index_feature": "task_index",
        "task_field": "task",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "siglip_repo_id": "google/siglip-so400m-patch14-384",
        "dinov2_repo_id": "facebook/dinov2-base",
    }
    cfg = make_policy_config(payload.pop("type"), **payload)
    assert isinstance(cfg, AceValueConfig)
    assert cfg.type == "ace_value"


def test_ace_value_preset_uses_cosine_decay_with_warmup():
    cfg = AceValueConfig()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    assert scheduler_cfg.peak_lr == cfg.optimizer_lr
