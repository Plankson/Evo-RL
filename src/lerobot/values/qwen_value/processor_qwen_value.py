#!/usr/bin/env python

from lerobot.values.qwen_siglip_dinov2.configuration_qwen_siglip_dinov2 import QwenSiglipDinov2Config
from lerobot.values.qwen_siglip_dinov2.processor_qwen_siglip_dinov2 import (
    QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY,
    QWEN_SIGLIP_DINOV2_IMAGES_KEY,
    make_qwen_siglip_dinov2_pre_post_processors,
)

QWEN_VALUE_IMAGES_KEY = QWEN_SIGLIP_DINOV2_IMAGES_KEY
QWEN_VALUE_IMAGE_MASK_KEY = QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY


def make_qwen_value_pre_post_processors(config: QwenSiglipDinov2Config, dataset_stats=None):
    return make_qwen_siglip_dinov2_pre_post_processors(config=config, dataset_stats=dataset_stats)
