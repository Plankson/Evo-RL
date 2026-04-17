#!/usr/bin/env python

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import make_policy_config
from lerobot.values.qwen_siglip_dinov2.configuration_qwen_siglip_dinov2 import QwenSiglipDinov2Config


def test_qwen_siglip_dinov2_config_from_dict():
    payload = {
        "type": "qwen_siglip_dinov2",
        "num_bins": 101,
        "bin_min": -1.0,
        "bin_max": 0.0,
        "task_index_feature": "task_index",
        "task_field": "task",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "siglip_repo_id": "google/siglip-so400m-patch14-384",
        "dinov2_repo_id": "facebook/dinov2-base",
        "dropout": 0.2,
    }
    cfg = make_policy_config(payload.pop("type"), **payload)
    assert isinstance(cfg, QwenSiglipDinov2Config)
    assert cfg.type == "qwen_siglip_dinov2"
    assert cfg.num_bins == 101
    assert cfg.camera_features == ["observation.images.front"]
    assert cfg.loss_weight_key == "observation.value_loss_weight"


def test_qwen_siglip_dinov2_preset_uses_cosine_decay_with_warmup():
    cfg = QwenSiglipDinov2Config()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    assert scheduler_cfg.peak_lr == cfg.optimizer_lr
    assert scheduler_cfg.decay_lr == cfg.scheduler_decay_lr
    assert scheduler_cfg.num_warmup_steps == cfg.scheduler_warmup_steps
    assert scheduler_cfg.num_decay_steps == cfg.scheduler_decay_steps
