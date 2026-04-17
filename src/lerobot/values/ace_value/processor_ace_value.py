#!/usr/bin/env python

from lerobot.values.ace_value.configuration_ace_value import AceValueConfig
from lerobot.values.qwen_siglip_dinov2.processor_qwen_siglip_dinov2 import (
    QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY,
    QWEN_SIGLIP_DINOV2_IMAGES_KEY,
    make_qwen_siglip_dinov2_pre_post_processors,
)

ACE_VALUE_IMAGES_KEY = QWEN_SIGLIP_DINOV2_IMAGES_KEY
ACE_VALUE_IMAGE_MASK_KEY = QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY


def make_ace_value_pre_post_processors(config: AceValueConfig, dataset_stats=None):
    return make_qwen_siglip_dinov2_pre_post_processors(config=config, dataset_stats=dataset_stats)
