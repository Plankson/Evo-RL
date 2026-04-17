#!/usr/bin/env python

from __future__ import annotations

from typing import Any

from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.values.pistar06.processor_pistar06 import (
    Pistar06PrepareImagesProcessorStep,
    Pistar06PrepareTaskPromptProcessorStep,
)
from lerobot.values.qwen_siglip_dinov2.configuration_qwen_siglip_dinov2 import QwenSiglipDinov2Config

QWEN_SIGLIP_DINOV2_IMAGES_KEY = "observation.qwen_siglip_dinov2.images"
QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY = "observation.qwen_siglip_dinov2.image_attention_mask"


class QwenSiglipDinov2PrepareImagesProcessorStep(Pistar06PrepareImagesProcessorStep):
    def __call__(self, transition):
        transition = super().__call__(transition)
        observation = dict(transition["observation"])
        observation[QWEN_SIGLIP_DINOV2_IMAGES_KEY] = observation.pop("observation.pistar06.images")
        observation[QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY] = observation.pop("observation.pistar06.image_attention_mask")
        transition["observation"] = observation
        return transition


def make_qwen_siglip_dinov2_pre_post_processors(
    config: QwenSiglipDinov2Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [k for k in (config.input_features or {}) if k.startswith(OBS_IMAGES)]

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        Pistar06PrepareTaskPromptProcessorStep(
            task_key=config.task_field,
            include_state_in_prompt=config.include_state_in_prompt,
            state_feature=config.state_feature,
            max_state_dim=config.max_state_dim,
            state_discretization_bins=config.state_discretization_bins,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.language_repo_id,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        QwenSiglipDinov2PrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [DeviceProcessorStep(device="cpu")]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
