#!/usr/bin/env python

from typing import Any

from lerobot.processor import (
    IdentityProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_remote_client import RemoteClientConfig


def make_remote_client_pre_post_processors(
    config: RemoteClientConfig,
    dataset_stats=None,  # noqa: ARG001
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return (
        PolicyProcessorPipeline(
            steps=[IdentityProcessorStep()],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[IdentityProcessorStep()],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
