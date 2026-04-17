#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.values.qwen_siglip_dinov2.configuration_qwen_siglip_dinov2 import QwenSiglipDinov2Config


@PreTrainedConfig.register_subclass("ace_value")
@dataclass
class AceValueConfig(QwenSiglipDinov2Config):
    """ACE value model entrypoint for the Qwen text + SigLIP + DINOv2 stack."""
