#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.datasets.data_constant import ace_fold_cloth_open_only
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    _build_hf_load_kwargs,
    _extract_hidden_size,
    _extract_vision_feature_size,
    _format_gated_repo_access_error,
    _freeze_module,
    _is_gated_repo_access_error,
    _load_language_model,
    _maybe_enable_gradient_checkpointing,
    _resolve_image_size,
    _resolve_load_dtype,
    _resolve_norm_stats,
    build_bin_centers,
    compute_normalized_value_targets,
    expected_value_from_logits,
    project_values_to_bins,
)
from lerobot.values.qwen_siglip_dinov2.configuration_qwen_siglip_dinov2 import QwenSiglipDinov2Config
from lerobot.values.qwen_siglip_dinov2.processor_qwen_siglip_dinov2 import (
    QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY,
    QWEN_SIGLIP_DINOV2_IMAGES_KEY,
)

if _transformers_available:
    from transformers import AutoImageProcessor, AutoModel
else:
    AutoImageProcessor = None
    AutoModel = None


QWEN_SIGLIP_DINOV2_SAVE_INFO = "qwen_siglip_dinov2_save_info.json"


class QwenSiglipDinov2Model(nn.Module):
    def __init__(
        self,
        cfg: QwenSiglipDinov2Config,
        *,
        hf_token: str | bool | None = None,
        hf_cache_dir: str | Path | None = None,
        hf_local_files_only: bool = False,
    ):
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'`."
            )

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        self.siglip_encoder = self._load_vision_encoder(
            repo_id=cfg.siglip_repo_id,
            revision=cfg.siglip_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            hf_local_files_only=hf_local_files_only,
        )
        self.dinov2_encoder = self._load_vision_encoder(
            repo_id=cfg.dinov2_repo_id,
            revision=cfg.dinov2_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            hf_local_files_only=hf_local_files_only,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
            token=hf_token,
            cache_dir=hf_cache_dir,
            local_files_only=hf_local_files_only,
        )

        (
            self.siglip_image_resolution,
            siglip_image_mean,
            siglip_image_std,
        ) = self._load_image_stats(
            repo_id=cfg.siglip_repo_id,
            revision=cfg.siglip_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            hf_local_files_only=hf_local_files_only,
        )
        (
            self.dinov2_image_resolution,
            dinov2_image_mean,
            dinov2_image_std,
        ) = self._load_image_stats(
            repo_id=cfg.dinov2_repo_id,
            revision=cfg.dinov2_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            hf_local_files_only=hf_local_files_only,
        )

        self.register_buffer(
            "siglip_image_mean",
            torch.tensor(siglip_image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_image_std",
            torch.tensor(siglip_image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dinov2_image_mean",
            torch.tensor(dinov2_image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dinov2_image_std",
            torch.tensor(dinov2_image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        siglip_feature_size = _extract_vision_feature_size(self.siglip_encoder)
        dinov2_feature_size = _extract_vision_feature_size(self.dinov2_encoder)
        language_hidden_size = _extract_hidden_size(self.language_model)

        self.siglip_projector = nn.Sequential(
            nn.Linear(siglip_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.dinov2_projector = nn.Sequential(
            nn.Linear(dinov2_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_hidden_dim * 3)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim * 3, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_bins),
        )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.siglip_encoder)
            _maybe_enable_gradient_checkpointing(self.dinov2_encoder)

        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_siglip_encoder:
            _freeze_module(self.siglip_encoder)
        if cfg.freeze_dinov2_encoder:
            _freeze_module(self.dinov2_encoder)

    def _load_vision_encoder(
        self,
        repo_id: str,
        revision: str | None,
        hf_token: str | bool | None,
        hf_cache_dir: str | Path | None,
        hf_local_files_only: bool,
    ) -> nn.Module:
        load_kwargs = _build_hf_load_kwargs(
            revision=revision,
            token=hf_token,
            cache_dir=hf_cache_dir,
            local_files_only=hf_local_files_only,
            dtype=self.model_dtype,
        )
        try:
            return AutoModel.from_pretrained(repo_id, **load_kwargs)
        except (HfHubHTTPError, OSError) as exc:
            if _is_gated_repo_access_error(exc):
                raise _format_gated_repo_access_error(repo_id, exc) from exc
            raise

    def _load_image_stats(
        self,
        repo_id: str,
        revision: str | None,
        hf_token: str | bool | None,
        hf_cache_dir: str | Path | None,
        hf_local_files_only: bool,
    ) -> tuple[tuple[int, int], tuple[float, float, float], tuple[float, float, float]]:
        load_kwargs = _build_hf_load_kwargs(
            revision=revision,
            token=hf_token,
            cache_dir=hf_cache_dir,
            local_files_only=hf_local_files_only,
        )
        try:
            image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=True, **load_kwargs)
        except (HfHubHTTPError, OSError) as exc:
            if _is_gated_repo_access_error(exc):
                raise _format_gated_repo_access_error(repo_id, exc) from exc
            raise
        resolution = _resolve_image_size(image_processor)
        mean, std = _resolve_norm_stats(image_processor)
        return resolution, mean, std

    def _encode_images(self, encoder: nn.Module, flat_images: Tensor) -> Tensor:
        if hasattr(encoder, "get_image_features"):
            return encoder.get_image_features(pixel_values=flat_images)

        vision_outputs = encoder(pixel_values=flat_images, return_dict=True)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported vision encoder output. Expected pooler_output or last_hidden_state.")

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Language model output does not contain `last_hidden_state`.")

        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def _preprocess_images(
        self,
        images: Tensor,
        image_attention_mask: Tensor,
        *,
        resolution: tuple[int, int],
        mean_buffer: Tensor,
        std_buffer: Tensor,
    ) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B,N,C,H,W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(
                f"'image_attention_mask' must have shape [B,N], got {tuple(image_attention_mask.shape)}."
            )

        bsize, num_cameras = images.shape[:2]
        if image_attention_mask.shape != (bsize, num_cameras):
            raise ValueError("Batch shape mismatch between images and image_attention_mask.")

        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32) / 255.0
        else:
            images = images.to(dtype=torch.float32)
            if bool(torch.max(images) > 1.0) or bool(torch.min(images) < 0.0):
                images = (images / 255.0).clamp(0.0, 1.0)

        flat_images = images.view(bsize * num_cameras, *images.shape[2:])
        if flat_images.shape[-2:] != resolution:
            flat_images = functional.interpolate(
                flat_images,
                size=resolution,
                mode="bilinear",
                align_corners=False,
            )

        mean = mean_buffer.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        std = std_buffer.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        flat_images = (flat_images - mean) / std
        flat_images = flat_images.view(
            bsize,
            num_cameras,
            flat_images.shape[1],
            flat_images.shape[2],
            flat_images.shape[3],
        )

        camera_mask = image_attention_mask.to(device=flat_images.device, dtype=flat_images.dtype).view(
            bsize, num_cameras, 1, 1, 1
        )
        return flat_images * camera_mask

    def _pool_projected_camera_features(self, features: Tensor, image_attention_mask: Tensor) -> Tensor:
        bsize, num_cameras = image_attention_mask.shape
        projected = features.view(bsize, num_cameras, -1)
        mask = image_attention_mask.unsqueeze(-1).to(dtype=projected.dtype)
        projected = projected * mask
        denom = image_attention_mask.sum(dim=1, keepdim=True).to(dtype=projected.dtype).clamp_min(1.0)
        return projected.sum(dim=1) / denom

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("Language inputs must have shape [B, T].")
        if images.ndim != 5 or image_attention_mask.ndim != 2:
            raise ValueError("Image inputs must be [B, N, C, H, W] with a [B, N] mask.")

        bsize = input_ids.shape[0]
        if attention_mask.shape[0] != bsize or images.shape[0] != bsize or image_attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch across language and image inputs.")
        if images.shape[1] == 0:
            raise ValueError("At least one camera is required for QwenSiglipDinov2Model.")

        image_attention_mask = image_attention_mask.to(dtype=torch.bool, device=images.device)
        if not torch.all(image_attention_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera input.")
        language_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if not torch.all(language_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token.")

        siglip_images = self._preprocess_images(
            images,
            image_attention_mask,
            resolution=self.siglip_image_resolution,
            mean_buffer=self.siglip_image_mean,
            std_buffer=self.siglip_image_std,
        )
        dinov2_images = self._preprocess_images(
            images,
            image_attention_mask,
            resolution=self.dinov2_image_resolution,
            mean_buffer=self.dinov2_image_mean,
            std_buffer=self.dinov2_image_std,
        )

        num_cameras = images.shape[1]
        siglip_flat = siglip_images.reshape(bsize * num_cameras, *siglip_images.shape[2:]).to(dtype=self.model_dtype)
        dinov2_flat = dinov2_images.reshape(bsize * num_cameras, *dinov2_images.shape[2:]).to(dtype=self.model_dtype)

        siglip_context = torch.no_grad() if self.cfg.freeze_siglip_encoder else nullcontext()
        with siglip_context:
            siglip_features = self._encode_images(self.siglip_encoder, siglip_flat)

        dinov2_context = torch.no_grad() if self.cfg.freeze_dinov2_encoder else nullcontext()
        with dinov2_context:
            dinov2_features = self._encode_images(self.dinov2_encoder, dinov2_flat)

        language_context = torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        with language_context:
            language_features = self._encode_language(input_ids=input_ids, attention_mask=language_mask.long())

        siglip_features = self.siglip_projector(siglip_features.to(dtype=torch.float32))
        dinov2_features = self.dinov2_projector(dinov2_features.to(dtype=torch.float32))
        language_features = self.language_projector(language_features.to(dtype=torch.float32))

        siglip_pooled = self._pool_projected_camera_features(siglip_features, image_attention_mask)
        dinov2_pooled = self._pool_projected_camera_features(dinov2_features, image_attention_mask)
        joint_features = torch.cat([siglip_pooled, dinov2_pooled, language_features], dim=-1)
        return self.value_head(self.final_norm(joint_features))


class QwenSiglipDinov2Policy(PreTrainedPolicy):
    config_class = QwenSiglipDinov2Config
    name = "qwen_siglip_dinov2"

    def __init__(
        self,
        config: QwenSiglipDinov2Config,
        dataset_meta=None,
        **kwargs: Any,
    ):
        del dataset_meta
        hf_token = kwargs.pop("hf_token", None)
        hf_cache_dir = kwargs.pop("hf_cache_dir", None)
        hf_local_files_only = bool(kwargs.pop("hf_local_files_only", False))
        if kwargs:
            logging.debug("Ignoring unsupported QwenSiglipDinov2Policy init kwargs: %s", sorted(kwargs))
        super().__init__(config)
        self.config = config
        self.model = QwenSiglipDinov2Model(
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

    def _frozen_checkpoint_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        if self.config.freeze_siglip_encoder:
            prefixes.append("model.siglip_encoder.")
        if self.config.freeze_dinov2_encoder:
            prefixes.append("model.dinov2_encoder.")
        if self.config.freeze_language_model:
            prefixes.append("model.language_model.")
        return prefixes

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)

        model_to_save = self.module if hasattr(self, "module") else self
        state_dict = model_to_save.state_dict()
        excluded_prefixes = self._frozen_checkpoint_prefixes()
        if excluded_prefixes:
            state_dict = {
                key: tensor
                for key, tensor in state_dict.items()
                if not any(key.startswith(prefix) for prefix in excluded_prefixes)
            }

        save_file(state_dict, str(save_directory / SAFETENSORS_SINGLE_FILE))
        save_info = {
            "format_version": 1,
            "weights_mode": "partial" if excluded_prefixes else "full",
            "freeze_siglip_encoder": bool(self.config.freeze_siglip_encoder),
            "freeze_dinov2_encoder": bool(self.config.freeze_dinov2_encoder),
            "freeze_language_model": bool(self.config.freeze_language_model),
            "excluded_prefixes": excluded_prefixes,
            "saved_tensor_count": len(state_dict),
        }
        with open(save_directory / QWEN_SIGLIP_DINOV2_SAVE_INFO, "w", encoding="utf-8") as f:
            json.dump(save_info, f, indent=2, sort_keys=True)

    @classmethod
    def _load_save_info(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> dict[str, Any] | None:
        del cls
        model_id = str(pretrained_name_or_path)
        save_info_path: Path | None = None
        if os.path.isdir(model_id):
            candidate = Path(model_id) / QWEN_SIGLIP_DINOV2_SAVE_INFO
            if candidate.is_file():
                save_info_path = candidate
        else:
            try:
                resolved = hf_hub_download(
                    repo_id=model_id,
                    filename=QWEN_SIGLIP_DINOV2_SAVE_INFO,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except (HfHubHTTPError, FileNotFoundError):
                resolved = None
            if resolved is not None:
                save_info_path = Path(resolved)

        if save_info_path is None:
            return None
        try:
            with open(save_info_path, encoding="utf-8") as f:
                parsed = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: QwenSiglipDinov2Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> "QwenSiglipDinov2Policy":
        model_init_kwargs = dict(kwargs)
        model_init_kwargs.setdefault("hf_token", token)
        model_init_kwargs.setdefault("hf_cache_dir", cache_dir)
        model_init_kwargs.setdefault("hf_local_files_only", local_files_only)

        save_info = cls._load_save_info(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        effective_strict = strict
        if save_info is not None and save_info.get("weights_mode") == "partial":
            effective_strict = False

        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=effective_strict,
            **model_init_kwargs,
        )

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        return

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("QwenSiglipDinov2Policy is a value model and does not support action prediction.")

    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("QwenSiglipDinov2Policy is a value model and does not support action selection.")

    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        logits = self.model(
            input_ids=batch[OBS_LANGUAGE_TOKENS],
            attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
            images=batch[QWEN_SIGLIP_DINOV2_IMAGES_KEY],
            image_attention_mask=batch[QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY],
        )
        bin_centers = self.bin_centers.to(device=logits.device)
        return expected_value_from_logits(logits, bin_centers)

    def build_training_raw_batch_hook(self, dataset, targets_cfg):
        raw_frames = dataset.hf_dataset.with_format(None)
        if len(raw_frames) == 0:
            raise ValueError("Dataset has no frames.")

        episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes = episodes_ds[:]
        has_success = targets_cfg.success_field in episodes_ds.column_names
        open_only_repo_ids = set(ace_fold_cloth_open_only)

        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}
        for i in range(len(episodes_ds)):
            ep_idx = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved_success = resolve_episode_success_label(
                explicit_success,
                default_label=targets_cfg.default_success,
                require_label=True,
            )
            ep_success = resolved_success == EPISODE_SUCCESS
            repo_id = str(episodes["repo_id"][i]) if "repo_id" in episodes_ds.column_names else str(dataset.repo_id)
            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx,
                task_index=task_index,
                length=ep_length,
                success=ep_success,
                open_only=repo_id in open_only_repo_ids,
            )
            task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=self.config.bin_min,
            clip_max=self.config.bin_max,
        )

        max_index = int(np.max(absolute_indices))
        value_target_lookup = np.zeros(max_index + 1, dtype=np.float32)
        value_target_lookup[absolute_indices] = value_targets.astype(np.float32, copy=False)
        target_key = targets_cfg.target_field

        def value_target_hook(batch: dict[str, Any], step: int) -> dict[str, Any]:
            del step
            batch_indices = batch.get("index")
            if batch_indices is None:
                raise KeyError("Missing 'index' in batch while building value targets.")
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            batch_indices_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            batch[target_key] = torch.from_numpy(value_target_lookup[batch_indices_np]).to(dtype=torch.float32)
            return batch

        return value_target_hook

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.target_key not in batch:
            raise KeyError(
                f"Missing target key '{self.config.target_key}' in batch. "
                "Make sure lerobot-value-train target hook is enabled."
            )

        device = next(self.model.parameters()).device
        value_target = batch[self.config.target_key]
        if not isinstance(value_target, Tensor):
            value_target = torch.as_tensor(value_target)
        value_target = value_target.to(device=device, dtype=torch.float32, non_blocking=True)
        if value_target.ndim == 2 and value_target.shape[-1] == 1:
            value_target = value_target.squeeze(-1)
        if value_target.ndim != 1:
            raise ValueError(
                f"Value target must be rank-1 or [B,1], got shape={tuple(value_target.shape)} "
                f"for key '{self.config.target_key}'."
            )

        logits = self.model(
            input_ids=batch[OBS_LANGUAGE_TOKENS],
            attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
            images=batch[QWEN_SIGLIP_DINOV2_IMAGES_KEY],
            image_attention_mask=batch[QWEN_SIGLIP_DINOV2_IMAGE_MASK_KEY],
        )
        bin_centers = self.bin_centers.to(device=device)
        soft_target = project_values_to_bins(value_target, bin_centers)
        log_probs = functional.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

        sample_weight = None
        if self.config.loss_weight_key in batch:
            sample_weight = batch[self.config.loss_weight_key]
            if not isinstance(sample_weight, Tensor):
                sample_weight = torch.as_tensor(sample_weight)
            sample_weight = sample_weight.to(device=device, dtype=torch.float32, non_blocking=True)
            if sample_weight.ndim == 2 and sample_weight.shape[-1] == 1:
                sample_weight = sample_weight.squeeze(-1)
            if sample_weight.ndim != 1:
                raise ValueError(
                    f"Loss weight must be rank-1 or [B,1], got shape={tuple(sample_weight.shape)} "
                    f"for key '{self.config.loss_weight_key}'."
                )
            per_sample_loss = per_sample_loss * sample_weight

        pred_value = expected_value_from_logits(logits, bin_centers)
        value_mae = (pred_value - value_target).abs().mean()
        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()

        loss_dict = {
            "loss": float(loss.mean().detach().item()) if reduction == "none" else float(loss.detach().item()),
            "value_mae": float(value_mae.detach().item()),
        }
        if sample_weight is not None:
            loss_dict["loss_weight_mean"] = float(sample_weight.mean().detach().item())
        return loss, loss_dict
