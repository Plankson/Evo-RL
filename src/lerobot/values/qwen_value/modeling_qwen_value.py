#!/usr/bin/env python

import logging
from typing import Any

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.values.pistar06.modeling_pistar06 import build_bin_centers
from lerobot.values.qwen_value.configuration_qwen_value import QwenValueConfig
from lerobot.values.qwen_siglip_dinov2.modeling_qwen_siglip_dinov2 import (
    QwenSiglipDinov2Model,
    QwenSiglipDinov2Policy,
)


class QwenValueModel(QwenSiglipDinov2Model):
    pass


class QwenValuePolicy(QwenSiglipDinov2Policy):
    config_class = QwenValueConfig
    name = "qwen_value"

    def __init__(
        self,
        config: QwenValueConfig,
        dataset_meta=None,
        **kwargs: Any,
    ):
        del dataset_meta
        hf_token = kwargs.pop("hf_token", None)
        hf_cache_dir = kwargs.pop("hf_cache_dir", None)
        hf_local_files_only = bool(kwargs.pop("hf_local_files_only", False))
        if kwargs:
            logging.debug("Ignoring unsupported QwenValuePolicy init kwargs: %s", sorted(kwargs))
        PreTrainedPolicy.__init__(self, config)
        self.config = config
        self.model = QwenValueModel(
            config,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            hf_local_files_only=hf_local_files_only,
        )
        self.register_buffer(
            "bin_centers",
            build_bin_centers(config.num_bins, config.bin_min, config.bin_max),
            persistent=False,
        )
