"""Explicit CUDA-only two-stage training for the one selected four-class model."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .checkpoint import (
    load_checkpoint_payload, load_trained_state_excluding_board_encoder,
    load_trained_state_with_wld_migration,
    load_transferred_model, load_transferred_profile_model, sha256_file,
)
from .data_contract import validate_model_ready_npz
from .board_perspective import SUPPORTED_BOARD_PERSPECTIVES, require_board_perspective
from .feature_policy import INPUT_POLICY
from .model import (
    ProfileConditionedLossModel, SEVERITY_CLASS_NAMES, WLD_CLASS_NAMES,
    TimeConditionedLossModel, multitask_loss,
)
from .board_cnn import (
    BoardCNNAuxiliaryHeads,
    legal_move_targets_from_board_tokens,
    sample_auxiliary_node_indices,
)
from .oq_profile_features import (
    OQ_PROFILE_FEATURE_NAMES,
    profile_ablation_hash,
    profile_ablation_indices,
)
from .progress import atomic_write_json, write_progress

MODEL_NAME = "board-cnn-causal-tcn-time-conditioned-severe-loss"
BASELINE_MODEL_SCHEMA = "time-plus-four-class-severity-plus-three-class-wld-v1"
PROFILE_MODEL_SCHEMA = "time-plus-four-class-severity-plus-three-class-wld-oq-profile-v1"
LEGACY_PROFILE_MODEL_SCHEMA = "time-plus-four-class-severity-oq-profile-v1"
MODEL_SCHEMA = BASELINE_MODEL_SCHEMA


@dataclass(frozen=True)
class TrainingConfig:
    head_epochs: int = 8
    fine_tune_epochs: int = 52
    batch_size: int = 32
    head_learning_rate: float = 1.0e-3
    fine_tune_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    time_task_weight: float = 0.25
    severity_classification_weight: float = 1.0
    wld_classification_weight: float = 1.0
    severity_class_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    num_workers: int = 0
    seed: int = 42
    training_mode: str = "joint"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    board_perspective: str = ""
    epochs: int = 0
    last_unfrozen_residual_blocks: int = 0
    cnn_learning_rate: float = 0.0
    non_cnn_learning_rate: float = 0.0
    auxiliary_nodes_per_game: int = 0
    legal_aux_weight: float = 0.0
    value_aux_weight: float = 0.0
    initial_stage_checkpoint_sha256: str = ""
    auxiliary_checkpoint_sha256: str = ""
    early_stopping_patience: int = 0
    initial_profile_checkpoint_sha256: str = ""

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("input_policy") != INPUT_POLICY:
            raise ValueError(f"config input_policy must be {INPUT_POLICY!r}")
        payload = document["training"]
        payload["board_perspective"] = document.get("board_perspective", "")
        payload["severity_class_weights"] = tuple(payload["severity_class_weights"])
        cfg = cls(**payload)
        if len(cfg.severity_class_weights) != 4 or any(not np.isfinite(x) or x <= 0 for x in cfg.severity_class_weights):
            raise ValueError("severity_class_weights must have four finite positive entries")
        if not np.isfinite(cfg.wld_classification_weight) or cfg.wld_classification_weight < 0:
            raise ValueError("wld_classification_weight must be finite and nonnegative")
        if cfg.training_mode not in {
            "joint", "wld-head-only", "cnn-stage-a", "cnn-stage-b", "cnn-stage-c", "cnn-direct-full"
        }:
            raise ValueError("unsupported training_mode")
        if cfg.training_mode == "wld-head-only" and cfg.head_epochs != 0:
            raise ValueError("wld-head-only training requires head_epochs=0")
        if cfg.training_mode == "cnn-stage-a" and (cfg.head_epochs <= 0 or cfg.fine_tune_epochs != 0):
            raise ValueError("cnn-stage-a requires head_epochs > 0 and fine_tune_epochs=0")
        if cfg.training_mode == "cnn-stage-b":
            if cfg.epochs <= 0 or cfg.head_epochs != 0 or cfg.fine_tune_epochs != 0:
                raise ValueError("cnn-stage-b requires epochs > 0 and head_epochs=fine_tune_epochs=0")
            if cfg.last_unfrozen_residual_blocks != 2:
                raise ValueError("cnn-stage-b requires exactly the last two residual blocks")
            if cfg.cnn_learning_rate <= 0 or cfg.non_cnn_learning_rate <= 0:
                raise ValueError("cnn-stage-b learning rates must be positive")
            if not np.isclose(cfg.cnn_learning_rate / cfg.non_cnn_learning_rate, 0.1, rtol=0, atol=1e-12):
                raise ValueError("cnn-stage-b CNN learning rate must be exactly 0.1 times non-CNN learning rate")
            if cfg.auxiliary_nodes_per_game != 2:
                raise ValueError("cnn-stage-b requires auxiliary_nodes_per_game=2")
            if cfg.legal_aux_weight <= 0 or cfg.value_aux_weight <= 0:
                raise ValueError("cnn-stage-b auxiliary loss weights must be positive")
            for name, value in (
                ("initial_stage_checkpoint_sha256", cfg.initial_stage_checkpoint_sha256),
                ("auxiliary_checkpoint_sha256", cfg.auxiliary_checkpoint_sha256),
            ):
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                    raise ValueError(f"cnn-stage-b requires a lowercase/uppercase 64-character {name}")
        if cfg.training_mode == "cnn-stage-c":
            if cfg.epochs <= 0 or cfg.head_epochs != 0 or cfg.fine_tune_epochs != 0:
                raise ValueError("cnn-stage-c requires epochs > 0 and head_epochs=fine_tune_epochs=0")
            if cfg.cnn_learning_rate <= 0 or cfg.non_cnn_learning_rate <= 0:
                raise ValueError("cnn-stage-c learning rates must be positive")
            if not np.isclose(cfg.cnn_learning_rate, 1.0e-5, rtol=0, atol=1e-12):
                raise ValueError("cnn-stage-c CNN learning rate must be 1e-5")
            if not np.isclose(cfg.non_cnn_learning_rate, 5.0e-5, rtol=0, atol=1e-12):
                raise ValueError("cnn-stage-c non-CNN learning rate must be 5e-5")
            if cfg.auxiliary_nodes_per_game != 2 or cfg.legal_aux_weight <= 0 or cfg.value_aux_weight <= 0:
                raise ValueError("cnn-stage-c requires the audited two-node positive-weight auxiliary contract")
            if cfg.early_stopping_patience <= 0:
                raise ValueError("cnn-stage-c requires a positive early_stopping_patience")
            value = cfg.initial_stage_checkpoint_sha256
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError("cnn-stage-c requires a 64-character initial_stage_checkpoint_sha256")
        if cfg.training_mode == "cnn-direct-full":
            if cfg.epochs != 38 or cfg.head_epochs != 0 or cfg.fine_tune_epochs != 0:
                raise ValueError("cnn-direct-full requires epochs=38 and head_epochs=fine_tune_epochs=0")
            if cfg.last_unfrozen_residual_blocks != 6:
                raise ValueError("cnn-direct-full requires all six residual blocks")
            if not np.isclose(cfg.cnn_learning_rate, 1.0e-5, rtol=0, atol=1e-12):
                raise ValueError("cnn-direct-full CNN learning rate must be 1e-5")
            if not np.isclose(cfg.non_cnn_learning_rate, 1.0e-4, rtol=0, atol=1e-12):
                raise ValueError("cnn-direct-full non-CNN learning rate must be 1e-4")
            if cfg.auxiliary_nodes_per_game != 2 or cfg.legal_aux_weight <= 0 or cfg.value_aux_weight <= 0:
                raise ValueError("cnn-direct-full requires the audited auxiliary contract")
            if cfg.early_stopping_patience != 6:
                raise ValueError("cnn-direct-full requires early_stopping_patience=6")
            for name, value in (
                ("initial_profile_checkpoint_sha256", cfg.initial_profile_checkpoint_sha256),
                ("auxiliary_checkpoint_sha256", cfg.auxiliary_checkpoint_sha256),
            ):
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                    raise ValueError(f"cnn-direct-full requires a 64-character {name}")
        if cfg.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if cfg.board_perspective not in SUPPORTED_BOARD_PERSPECTIVES:
            raise ValueError(
                "config board_perspective must explicitly be "
                "'snapshot_side_to_move_v1' or 'legacy_fixed_color'"
            )
        return cfg


class SequenceDataset(Dataset):
    TENSOR_NAMES = (
        "X", "board_tokens", "board_move_tokens", "current_hint_tokens",
        "current_hint_values", "prev_own_hint_values", "actual_thinking_time_ms",
        "severity_class", "mask", "wld_class", "wld_label_available",
        "global_placement_ply",
    )

    def __init__(self, path: Path, split: str, require_oq_profile: bool = False) -> None:
        with np.load(path, allow_pickle=False) as source:
            selected = np.flatnonzero(source["split"].astype(str) == split)
            if not len(selected):
                raise ValueError(f"model-ready data has no {split} games")
            names = list(self.TENSOR_NAMES)
            if require_oq_profile:
                names.extend(("oq_profile_features", "oq_profile_missing"))
            if split == "train":
                names.append("current_score")
            self.arrays = {name: torch.from_numpy(source[name][selected].copy()) for name in names}
        if split == "train":
            current = self.arrays["board_tokens"][:, :, 0, :]
            self.arrays["legal_move_target"] = legal_move_targets_from_board_tokens(current)

    def __len__(self) -> int:
        return self.arrays["X"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.arrays.items()}


def configure_cnn_stage_b(
    model: TimeConditionedLossModel,
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    cfg: TrainingConfig,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Freeze the CNN prefix and construct explicit Stage-B optimizer groups."""
    encoder = model.backbone.board_encoder
    if not hasattr(encoder, "shared"):
        raise ValueError("cnn-stage-b requires the residual-v2 shared board CNN")
    shared = encoder.shared
    if len(shared.blocks) != 6 or cfg.last_unfrozen_residual_blocks != 2:
        raise ValueError("cnn-stage-b requires six residual blocks and unfreezes blocks 4 and 5")
    for parameter in model.parameters():
        parameter.requires_grad = True
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for block in shared.blocks[-cfg.last_unfrozen_residual_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for module in (shared.projection, shared.embedding_norm):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for parameter in auxiliary_heads.parameters():
        parameter.requires_grad = True

    cnn_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    non_cnn_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("backbone.board_encoder.") and parameter.requires_grad
    ]
    auxiliary_parameters = list(auxiliary_heads.parameters())
    frozen_cnn = [parameter for parameter in encoder.parameters() if not parameter.requires_grad]
    if not cnn_parameters or not non_cnn_parameters or not auxiliary_parameters:
        raise AssertionError("cnn-stage-b optimizer groups must all be non-empty")
    optimizer = torch.optim.AdamW(
        [
            {"params": cnn_parameters, "lr": cfg.cnn_learning_rate, "name": "cnn_partial"},
            {"params": non_cnn_parameters, "lr": cfg.non_cnn_learning_rate, "name": "formal_non_cnn"},
            {"params": auxiliary_parameters, "lr": cfg.non_cnn_learning_rate, "name": "auxiliary_heads"},
        ],
        weight_decay=cfg.weight_decay,
    )
    groups = [
        {
            "name": group["name"],
            "learningRate": float(group["lr"]),
            "parameterCount": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    report = {
        "lastUnfrozenResidualBlocks": cfg.last_unfrozen_residual_blocks,
        "parameterGroups": groups,
        "frozenCnnParameterCount": sum(parameter.numel() for parameter in frozen_cnn),
        "unfrozenCnnParameterCount": sum(parameter.numel() for parameter in cnn_parameters),
        "residualBlockRequiresGrad": [
            bool(any(parameter.requires_grad for parameter in block.parameters()))
            for block in shared.blocks
        ],
        "stemRequiresGrad": bool(any(parameter.requires_grad for parameter in shared.stem.parameters())),
        "stemNormRequiresGrad": bool(any(parameter.requires_grad for parameter in shared.stem_norm.parameters())),
        "projectionRequiresGrad": bool(any(parameter.requires_grad for parameter in shared.projection.parameters())),
        "embeddingNormRequiresGrad": bool(any(parameter.requires_grad for parameter in shared.embedding_norm.parameters())),
    }
    expected_blocks = [False, False, False, False, True, True]
    if report["residualBlockRequiresGrad"] != expected_blocks:
        raise AssertionError("cnn-stage-b residual block freeze contract failed")
    if report["stemRequiresGrad"] or report["stemNormRequiresGrad"]:
        raise AssertionError("cnn-stage-b stem must be frozen")
    if not report["projectionRequiresGrad"] or not report["embeddingNormRequiresGrad"]:
        raise AssertionError("cnn-stage-b projection and final LayerNorm must be trainable")
    return optimizer, report


def configure_cnn_stage_c(
    model: TimeConditionedLossModel,
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    cfg: TrainingConfig,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Fully unfreeze the residual CNN with the confirmed reduced non-CNN LR."""
    encoder = model.backbone.board_encoder
    if not hasattr(encoder, "shared") or len(encoder.shared.blocks) != 6:
        raise ValueError("cnn-stage-c requires the six-block residual-v2 shared board CNN")
    for parameter in model.parameters():
        parameter.requires_grad = True
    for parameter in auxiliary_heads.parameters():
        parameter.requires_grad = True
    cnn_parameters = list(encoder.parameters())
    non_cnn_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("backbone.board_encoder.")
    ]
    auxiliary_parameters = list(auxiliary_heads.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": cnn_parameters, "lr": cfg.cnn_learning_rate, "name": "cnn_full"},
            {"params": non_cnn_parameters, "lr": cfg.non_cnn_learning_rate, "name": "formal_non_cnn"},
            {"params": auxiliary_parameters, "lr": cfg.non_cnn_learning_rate, "name": "auxiliary_heads"},
        ],
        weight_decay=cfg.weight_decay,
    )
    groups = [
        {
            "name": group["name"], "learningRate": float(group["lr"]),
            "parameterCount": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    shared = encoder.shared
    report = {
        "allCnnTrainable": bool(all(parameter.requires_grad for parameter in encoder.parameters())),
        "parameterGroups": groups,
        "frozenCnnParameterCount": sum(
            parameter.numel() for parameter in encoder.parameters() if not parameter.requires_grad
        ),
        "unfrozenCnnParameterCount": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "residualBlockRequiresGrad": [
            bool(all(parameter.requires_grad for parameter in block.parameters())) for block in shared.blocks
        ],
        "stemRequiresGrad": bool(all(parameter.requires_grad for parameter in shared.stem.parameters())),
        "stemNormRequiresGrad": bool(all(parameter.requires_grad for parameter in shared.stem_norm.parameters())),
        "projectionRequiresGrad": bool(all(parameter.requires_grad for parameter in shared.projection.parameters())),
        "embeddingNormRequiresGrad": bool(all(parameter.requires_grad for parameter in shared.embedding_norm.parameters())),
        "earlyStoppingPatience": cfg.early_stopping_patience,
    }
    if not report["allCnnTrainable"] or report["frozenCnnParameterCount"] != 0:
        raise AssertionError("cnn-stage-c requires every CNN parameter to be trainable")
    if report["residualBlockRequiresGrad"] != [True] * 6:
        raise AssertionError("cnn-stage-c residual block unfreeze contract failed")
    return optimizer, report


def configure_cnn_direct_full(
    model: TimeConditionedLossModel,
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    cfg: TrainingConfig,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Train the complete joint model from epoch one with differential LRs."""
    optimizer, report = configure_cnn_stage_c(model, auxiliary_heads, cfg)
    report = dict(report)
    report["strategy"] = "full-joint-from-epoch-one"
    report["maximumEpochs"] = cfg.epochs
    return optimizer, report


def _strict_load_stage_a_checkpoint(
    model: TimeConditionedLossModel,
    path: Path,
    expected_sha256: str,
    identity: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("Stage A checkpoint SHA-256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "tcn-loss-profile-wld-checkpoint-v2":
        raise ValueError("initial-stage checkpoint schema is not the Stage A profile/WLD schema")
    if saved.get("stage") != "cnn-stage-a" or saved.get("manifest", {}).get("trainingMode") != "cnn-stage-a":
        raise ValueError("initial-stage checkpoint is not cnn-stage-a")
    if saved.get("board_perspective") != "snapshot_side_to_move_v1":
        raise ValueError("initial-stage checkpoint has a legacy or missing board perspective")
    source_manifest = saved.get("manifest", {})
    mismatches = [key for key, value in identity.items() if source_manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Stage A checkpoint identity mismatch: {mismatches}")
    model.load_state_dict(saved["modelStateDict"], strict=True)
    return saved, actual_sha256


def _strict_load_stage_b_checkpoint(
    model: TimeConditionedLossModel,
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    path: Path,
    expected_sha256: str,
    identity: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("Stage B checkpoint SHA-256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "tcn-loss-profile-wld-stage-b-checkpoint-v1":
        raise ValueError("initial-stage checkpoint schema is not the Stage B schema")
    if saved.get("stage") != "cnn-stage-b" or saved.get("manifest", {}).get("trainingMode") != "cnn-stage-b":
        raise ValueError("initial-stage checkpoint is not cnn-stage-b")
    if saved.get("board_perspective") != "snapshot_side_to_move_v1":
        raise ValueError("initial-stage checkpoint has a legacy or missing board perspective")
    source_manifest = saved.get("manifest", {})
    mismatches = [key for key, value in identity.items() if source_manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Stage B checkpoint identity mismatch: {mismatches}")
    model.load_state_dict(saved["modelStateDict"], strict=True)
    auxiliary_heads.load_state_dict(saved["auxiliaryHeadsStateDict"], strict=True)
    return saved, actual_sha256


def _strict_load_auxiliary_heads(
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    path: Path,
    expected_sha256: str,
    base_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("auxiliary checkpoint SHA-256 mismatch")
    expected_transfer_hash = base_payload.get("board_cnn_pretrain_transfer", {}).get("pretrainedSha256")
    if expected_transfer_hash != actual_sha256:
        raise ValueError("auxiliary checkpoint is not the CNN checkpoint used for board-CNN transfer")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "board-cnn-pretrain-checkpoint-v2":
        raise ValueError("auxiliary checkpoint format mismatch")
    auxiliary_heads.legal_head.load_state_dict(payload["legal_move_head_state_dict"], strict=True)
    auxiliary_heads.value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
    return payload, actual_sha256


def stage_b_auxiliary_loss(
    model: TimeConditionedLossModel,
    auxiliary_heads: BoardCNNAuxiliaryHeads,
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
) -> dict[str, torch.Tensor]:
    valid = batch["mask"].bool() & torch.isfinite(batch["current_score"])
    selected = sample_auxiliary_node_indices(valid, cfg.auxiliary_nodes_per_game)
    if selected.numel() == 0:
        raise ValueError("Stage B batch has no valid auxiliary nodes")
    game, node = selected[:, 0], selected[:, 1]
    encoder = model.backbone.board_encoder
    board_planes, _, _ = encoder.build_input_planes(
        batch["board_tokens"][game, node].unsqueeze(1),
        batch["board_move_tokens"][game, node].unsqueeze(1),
        batch["current_hint_tokens"][game, node].unsqueeze(1),
        batch["current_hint_values"][game, node].unsqueeze(1),
        batch["prev_own_hint_values"][game, node].unsqueeze(1),
    )
    state_only = torch.zeros_like(board_planes)
    state_only[:, :3] = board_planes[:, :3]
    if int(torch.count_nonzero(state_only[:, 3:]).item()) != 0:
        raise AssertionError("state-only auxiliary channels 3..22 are not zero")
    spatial = encoder.shared.spatial_features(state_only)
    embedding = encoder.shared.embedding_from_spatial(spatial)
    legal_logits, values = auxiliary_heads(spatial, embedding)
    legal_target = batch["legal_move_target"][game, node].to(legal_logits.dtype)
    value_target = batch["current_score"][game, node].float() / 64.0
    legal_loss = torch.nn.functional.binary_cross_entropy_with_logits(legal_logits, legal_target)
    value_loss = torch.nn.functional.smooth_l1_loss(values, value_target)
    return {
        "legal_auxiliary": legal_loss,
        "value_auxiliary": value_loss,
        "weighted_total": cfg.legal_aux_weight * legal_loss + cfg.value_aux_weight * value_loss,
        "sampled_nodes": torch.as_tensor(selected.shape[0], device=legal_loss.device),
        "state_only_nonzero_forbidden_channels": torch.count_nonzero(state_only[:, 3:]),
    }


def _json_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _is_completed_epoch_extension(
    saved_config: dict[str, Any], current_config: dict[str, Any], saved_epoch: int
) -> bool:
    saved = json.loads(json.dumps(saved_config))
    current = json.loads(json.dumps(current_config))
    saved_training = saved.get("training", {})
    current_training = current.get("training", {})
    old_fine_tune_epochs = int(saved_training.get("fine_tune_epochs", -1))
    new_fine_tune_epochs = int(current_training.get("fine_tune_epochs", -1))
    old_head_epochs = int(saved_training.get("head_epochs", -1))
    if new_fine_tune_epochs <= old_fine_tune_epochs:
        return False
    if saved_epoch != old_head_epochs + old_fine_tune_epochs:
        return False
    current_training["fine_tune_epochs"] = old_fine_tune_epochs
    return saved == current


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _device_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA; torch.cuda.is_available() is false")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_count": torch.cuda.device_count(), "cuda_version": torch.version.cuda or "",
        "torch_version": torch.__version__, "gpu_total_memory_bytes": int(properties.total_memory),
    }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _model_output(model: TimeConditionedLossModel, batch: dict[str, torch.Tensor], use_oq_profile: bool = False):
    base_args = (
        batch["X"].float(), batch["board_tokens"], batch["board_move_tokens"],
        batch["current_hint_tokens"], batch["current_hint_values"].float(),
        batch["prev_own_hint_values"].float(), batch["actual_thinking_time_ms"].float(),
    )
    if use_oq_profile:
        if not isinstance(model, ProfileConditionedLossModel):
            raise TypeError("profile batch requires ProfileConditionedLossModel")
        return model(*base_args, batch["oq_profile_features"].float(), batch["oq_profile_missing"])
    return model(*base_args)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    probabilities = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    labels = labels.astype(int)
    calibration = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        in_bin = (probabilities >= edges[index]) & (probabilities < edges[index + 1] if index < bins - 1 else probabilities <= edges[index + 1])
        calibration.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(in_bin.sum()),
            "actual_rate": float(labels[in_bin].mean()) if in_bin.any() else None,
            "mean_probability": float(probabilities[in_bin].mean()) if in_bin.any() else None,
        })
    both_classes = len(np.unique(labels)) == 2
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.any() else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "actual_positive_rate": float(labels.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
        "calibration": calibration,
    }


@torch.no_grad()
def evaluate(model: TimeConditionedLossModel, loader: DataLoader, device: torch.device,
             cfg: TrainingConfig, use_oq_profile: bool = False) -> dict[str, Any]:
    model.eval()
    loss_sums = {key: 0.0 for key in ("total", "thinking_time", "severity_classification", "wld_classification")}
    targets, probabilities, wld_targets, wld_probabilities = [], [], [], []
    for batch in loader:
        batch = _move(batch, device)
        output = _model_output(model, batch, use_oq_profile)
        losses = multitask_loss(
            output, batch["actual_thinking_time_ms"].float(), batch["severity_class"].float(), batch["mask"],
            cfg.time_task_weight, cfg.severity_classification_weight, cfg.severity_class_weights,
            batch["wld_class"].float(), batch["wld_label_available"],
            batch["global_placement_ply"], cfg.wld_classification_weight,
        )
        for key in loss_sums:
            loss_sums[key] += float(losses[key].item())
        valid = batch["mask"].bool() & torch.isfinite(batch["severity_class"])
        targets.append(batch["severity_class"][valid].detach().cpu().numpy().astype(int))
        probabilities.append(output.severity_class_probabilities[valid].detach().cpu().numpy())
        wld_valid = (
            batch["mask"].bool() & batch["wld_label_available"].bool()
            & (batch["global_placement_ply"] >= 39)
        )
        if bool(wld_valid.any()):
            wld_targets.append(batch["wld_class"][wld_valid].detach().cpu().numpy().astype(int))
            wld_probabilities.append(output.wld_probabilities[wld_valid].detach().cpu().numpy())
    losses = {key: value / len(loader) for key, value in loss_sums.items()}
    classes = np.concatenate(targets)
    class_probabilities = np.concatenate(probabilities)
    zero = classes == 0
    ge4 = classes >= 2
    ge10 = classes == 3
    zero_probability = class_probabilities[:, 0]
    ge4_probability = class_probabilities[:, 2] + class_probabilities[:, 3]
    ge10_probability = class_probabilities[:, 3]
    if np.any(ge10_probability > ge4_probability + 1e-6) or np.any(ge4_probability > 1 - zero_probability + 1e-6):
        raise AssertionError("derived severity probabilities violate monotonicity")
    if not wld_targets:
        raise ValueError("evaluation split has no valid WLD nodes at global placement ply >= 39")
    wld_classes = np.concatenate(wld_targets)
    wld_class_probabilities = np.concatenate(wld_probabilities)
    expected_wld = 0.5 * wld_class_probabilities[:, 1] + wld_class_probabilities[:, 2]
    actual_wld = wld_classes / 2.0
    wld_metrics = {
        "nodes": int(len(wld_classes)),
        "cross_entropy": float(log_loss(wld_classes, wld_class_probabilities, labels=[0, 1, 2])),
        "accuracy": float((wld_class_probabilities.argmax(axis=1) == wld_classes).mean()),
        "expected_wld_loss_mae": float(np.abs(expected_wld - actual_wld).mean()),
        "expected_wld_loss_mean": float(expected_wld.mean()),
        "actual_wld_loss_mean": float(actual_wld.mean()),
        "class_actual_rates": {name: float((wld_classes == i).mean()) for i, name in enumerate(WLD_CLASS_NAMES)},
        "class_mean_probabilities": {name: float(wld_class_probabilities[:, i].mean()) for i, name in enumerate(WLD_CLASS_NAMES)},
    }
    return {
        **losses,
        "zero": _binary_metrics(zero, zero_probability),
        "ge4": _binary_metrics(ge4, ge4_probability),
        "ge10": _binary_metrics(ge10, ge10_probability),
        "class_actual_rates": {name: float((classes == index).mean()) for index, name in enumerate(SEVERITY_CLASS_NAMES)},
        "class_mean_probabilities": {name: float(class_probabilities[:, index].mean()) for index, name in enumerate(SEVERITY_CLASS_NAMES)},
        "wld": wld_metrics,
    }


def train_cuda(data_path: Path, base_checkpoint: Path, config_path: Path, output_dir: Path,
               run_name: str, context_metadata: Path | None = None, resume: Path | None = None,
               evaluate_test: bool = True, use_oq_profile: bool = False,
               initial_profile_checkpoint: Path | None = None,
               initial_stage_checkpoint: Path | None = None,
               auxiliary_checkpoint: Path | None = None) -> None:
    device_info = _device_info()
    cfg = TrainingConfig.load(config_path)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    model_variant = str(config_document.get("model_variant") or "baseline")
    profile_ablation = str(config_document.get("oq_profile_ablation") or "")
    if use_oq_profile:
        if model_variant != "oq-profile":
            raise ValueError("profile training requires config model_variant='oq-profile'")
        profile_ablation_indices(profile_ablation)
        model_schema = PROFILE_MODEL_SCHEMA
    else:
        if model_variant not in {"", "baseline"}:
            raise ValueError("baseline training refuses an OQ profile model config")
        profile_ablation = ""
        model_schema = BASELINE_MODEL_SCHEMA
    config_payload = {
        "training": asdict(cfg), "model_schema": model_schema,
        "input_policy": INPUT_POLICY,
        "model_variant": model_variant,
        "oq_profile_ablation": profile_ablation,
        "board_perspective": cfg.board_perspective,
    }
    config_hash = _json_hash(config_payload)
    data_hash, base_hash = sha256_file(data_path), sha256_file(base_checkpoint)
    if initial_profile_checkpoint is not None and not use_oq_profile:
        raise ValueError("initial profile checkpoint requires profile training")
    warm_start_hash = sha256_file(initial_profile_checkpoint) if initial_profile_checkpoint else ""
    if cfg.training_mode in {"cnn-stage-b", "cnn-stage-c"}:
        if not use_oq_profile:
            raise ValueError(f"{cfg.training_mode} requires OQ profile training")
        if evaluate_test:
            raise ValueError(f"{cfg.training_mode} forbids test evaluation")
        if initial_stage_checkpoint is None:
            raise ValueError(f"{cfg.training_mode} requires --initial-stage-checkpoint")
        if cfg.training_mode == "cnn-stage-b" and auxiliary_checkpoint is None:
            raise ValueError("cnn-stage-b requires --initial-stage-checkpoint and --auxiliary-checkpoint")
        if cfg.training_mode == "cnn-stage-c" and auxiliary_checkpoint is not None:
            raise ValueError("cnn-stage-c restores auxiliary heads from Stage B and forbids --auxiliary-checkpoint")
        if initial_profile_checkpoint is not None:
            raise ValueError(f"{cfg.training_mode} cannot use --initial-profile-checkpoint")
    elif initial_stage_checkpoint is not None or auxiliary_checkpoint is not None:
        if cfg.training_mode != "cnn-direct-full":
            raise ValueError("stage/auxiliary checkpoints are not valid for this training mode")
    if cfg.training_mode == "cnn-direct-full":
        if not use_oq_profile or evaluate_test:
            raise ValueError("cnn-direct-full requires OQ profile validation-only training")
        if initial_stage_checkpoint is not None:
            raise ValueError("cnn-direct-full starts before Stage A and forbids --initial-stage-checkpoint")
        if initial_profile_checkpoint is None or auxiliary_checkpoint is None:
            raise ValueError("cnn-direct-full requires --initial-profile-checkpoint and --auxiliary-checkpoint")
        if sha256_file(initial_profile_checkpoint).lower() != cfg.initial_profile_checkpoint_sha256.lower():
            raise ValueError("cnn-direct-full initial profile checkpoint SHA-256 mismatch")
    existing = [item for item in output_dir.iterdir()] if output_dir.exists() else []
    if [item for item in existing if item.name not in {"stdout.log", "stderr.log"}] and resume is None:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = (
        cfg.epochs
        if cfg.training_mode in {"cnn-stage-b", "cnn-stage-c", "cnn-direct-full"}
        else cfg.head_epochs + cfg.fine_tune_epochs
    )
    write_progress(output_dir, status="validating-data", stage="validating-data", model_name=MODEL_NAME,
                   run_id=run_name, base_checkpoint=str(base_checkpoint.resolve()), max_epochs=max_epochs,
                   training_config=config_payload, **device_info)
    base_payload = load_checkpoint_payload(base_checkpoint)
    require_board_perspective(
        base_payload.get("board_perspective"), cfg.board_perspective
    )
    preprocessing_hash = _json_hash(base_payload["preprocessing"])
    validation = validate_model_ready_npz(
        data_path, expected_input_features=base_payload["input_features"],
        expected_board_channels=base_payload["board_encoding"]["cnn_channels"],
        expected_preprocessing_sha256=preprocessing_hash,
        expected_board_perspective=cfg.board_perspective,
        require_oq_profile=use_oq_profile,
        expected_oq_profile_feature_names=OQ_PROFILE_FEATURE_NAMES if use_oq_profile else None,
    )
    write_progress(output_dir, status="preparing-features", stage="verifying-model-ready-features", data_manifest=validation)
    atomic_write_json(output_dir / "config.json", config_payload)
    manifest = {
        "schema": "tcn-loss-run-manifest-v1", "modelSchema": model_schema, "runName": run_name,
        "modelVariant": model_variant,
        "boardPerspective": cfg.board_perspective,
        "inputPolicy": INPUT_POLICY,
        "dataPath": str(data_path.resolve()), "dataSha256": data_hash,
        "contextMetadata": str(context_metadata.resolve()) if context_metadata else "",
        "baseCheckpoint": str(base_checkpoint.resolve()), "baseCheckpointSha256": base_hash,
        "testEvaluationPlanned": bool(evaluate_test),
        "warmStartCheckpoint": str(initial_profile_checkpoint.resolve()) if initial_profile_checkpoint else "",
        "warmStartCheckpointSha256": warm_start_hash,
        "initialStageCheckpoint": str(initial_stage_checkpoint.resolve()) if initial_stage_checkpoint else "",
        "initialStageCheckpointSha256": sha256_file(initial_stage_checkpoint) if initial_stage_checkpoint else "",
        "auxiliaryCheckpoint": str(auxiliary_checkpoint.resolve()) if auxiliary_checkpoint else "",
        "auxiliaryCheckpointSha256": sha256_file(auxiliary_checkpoint) if auxiliary_checkpoint else "",
        "configSha256": config_hash,
        "featureOrderSha256": hashlib.sha256("\n".join(base_payload["input_features"]).encode("utf-8")).hexdigest(),
        "boardChannelOrderSha256": hashlib.sha256("\n".join(base_payload["board_encoding"]["cnn_channels"]).encode("utf-8")).hexdigest(),
        "preprocessingSha256": preprocessing_hash, "dataset": validation, **device_info,
    }
    manifest.update({
        "oqProfileAblation": profile_ablation,
        "oqProfileAblationSha256": profile_ablation_hash(profile_ablation) if use_oq_profile else "",
        "oqProfileFeatureOrderSha256": (
            hashlib.sha256("\n".join(OQ_PROFILE_FEATURE_NAMES).encode("utf-8")).hexdigest()
            if use_oq_profile else ""
        ),
        "oqProfilePreprocessingSha256": validation["oqProfile"]["preprocessingSha256"] if use_oq_profile else "",
        "oqProfilePolicy": validation["oqProfile"]["policy"] if use_oq_profile else "",
        "oqProfileTemporalLeakageAuthorized": validation["oqProfile"]["temporalLeakageAuthorized"] if use_oq_profile else False,
    })
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    write_progress(output_dir, status="loading-checkpoint", stage="loading-checkpoint", data_manifest=manifest)
    model, _ = (
        load_transferred_profile_model(base_checkpoint, profile_ablation)
        if use_oq_profile else load_transferred_model(base_checkpoint)
    )
    auxiliary_heads: BoardCNNAuxiliaryHeads | None = None
    if initial_profile_checkpoint is not None:
        warm_start = torch.load(initial_profile_checkpoint, map_location="cpu", weights_only=False)
        if warm_start.get("schema") not in {
            "tcn-loss-profile-checkpoint-v1", "tcn-loss-profile-wld-checkpoint-v2"
        }:
            raise ValueError("warm-start checkpoint is not a profile checkpoint")
        warm_manifest = warm_start["manifest"]
        expected_warm_base_hash = base_hash
        if cfg.training_mode in {"cnn-stage-a", "cnn-direct-full"}:
            transfer_identity = base_payload.get("board_cnn_pretrain_transfer")
            if not isinstance(transfer_identity, dict) or not transfer_identity.get("strictNonCnnLoad"):
                raise ValueError("cnn-stage-a requires a strictly transferred board-CNN base checkpoint")
            expected_warm_base_hash = str(transfer_identity.get("targetSha256") or "")
            if not expected_warm_base_hash:
                raise ValueError("transferred board-CNN checkpoint lacks its original target SHA-256")
        required_warm_identity = {
            "modelVariant": "oq-profile",
            "baseCheckpointSha256": expected_warm_base_hash,
            "oqProfileAblation": profile_ablation,
            "oqProfileAblationSha256": profile_ablation_hash(profile_ablation),
        }
        warm_mismatches = [key for key, value in required_warm_identity.items() if warm_manifest.get(key) != value]
        if warm_manifest.get("modelSchema") not in {PROFILE_MODEL_SCHEMA, LEGACY_PROFILE_MODEL_SCHEMA}:
            warm_mismatches.append("modelSchema")
        if warm_mismatches:
            raise ValueError(f"warm-start profile identity mismatch: {warm_mismatches}")
        migration = (
            load_trained_state_excluding_board_encoder(model, warm_start["modelStateDict"])
            if cfg.training_mode in {"cnn-stage-a", "cnn-direct-full"}
            else load_trained_state_with_wld_migration(model, warm_start["modelStateDict"])
        )
        manifest["warmStartStateMigration"] = migration
        manifest["warmStartEpoch"] = int(warm_start.get("epoch", 0))
        manifest["warmStartStage"] = str(warm_start.get("stage", ""))
        manifest["warmStartProfilePreprocessingSha256"] = warm_manifest.get("oqProfilePreprocessingSha256", "")
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    if cfg.training_mode == "cnn-direct-full":
        auxiliary_heads = BoardCNNAuxiliaryHeads(model.backbone.cfg.board_channels)
        _, auxiliary_hash = _strict_load_auxiliary_heads(
            auxiliary_heads, auxiliary_checkpoint, cfg.auxiliary_checkpoint_sha256, base_payload
        )
        manifest["directFullInitialization"] = {
            "strategy": "full-joint-from-epoch-one",
            "transferredBoardCnnCheckpoint": str(base_checkpoint.resolve()),
            "transferredBoardCnnCheckpointSha256": base_hash,
            "profileCheckpoint": str(initial_profile_checkpoint.resolve()),
            "profileCheckpointSha256": warm_start_hash,
            "strictNonCnnLoadExcludingLegacyBoardEncoder": True,
            "auxiliaryCheckpoint": str(auxiliary_checkpoint.resolve()),
            "auxiliaryCheckpointSha256": auxiliary_hash,
            "strictAuxiliaryHeadLoad": True,
            "randomizedFormalModules": False,
        }
        manifest["auxiliaryContract"] = {
            "nodesPerGame": cfg.auxiliary_nodes_per_game,
            "legalLossWeight": cfg.legal_aux_weight,
            "valueLossWeight": cfg.value_aux_weight,
            "stateOnlyChannelsRetained": [0, 1, 2],
            "stateOnlyChannelsZeroed": list(range(3, 23)),
            "legalTargetSource": "derived-by-othello-rules-from-current-board-tokens-only",
            "valueTargetSource": "current_score_divided_by_64",
            "auxiliaryHeadsAreFormalInferenceOutputs": False,
        }
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    if cfg.training_mode == "cnn-stage-b":
        stage_identity = {key: manifest[key] for key in (
            "modelSchema", "modelVariant", "boardPerspective", "inputPolicy",
            "dataSha256", "baseCheckpointSha256", "featureOrderSha256",
            "boardChannelOrderSha256", "preprocessingSha256", "oqProfileAblation",
            "oqProfileAblationSha256", "oqProfileFeatureOrderSha256",
            "oqProfilePreprocessingSha256", "oqProfilePolicy",
            "oqProfileTemporalLeakageAuthorized", "testEvaluationPlanned",
        )}
        stage_a_saved, stage_a_hash = _strict_load_stage_a_checkpoint(
            model, initial_stage_checkpoint, cfg.initial_stage_checkpoint_sha256, stage_identity
        )
        auxiliary_heads = BoardCNNAuxiliaryHeads(model.backbone.cfg.board_channels)
        _, auxiliary_hash = _strict_load_auxiliary_heads(
            auxiliary_heads, auxiliary_checkpoint, cfg.auxiliary_checkpoint_sha256, base_payload
        )
        manifest["stageTransition"] = {
            "sourceCheckpoint": str(initial_stage_checkpoint.resolve()),
            "sourceCheckpointSha256": stage_a_hash,
            "sourceSchema": stage_a_saved["schema"],
            "sourceStage": stage_a_saved["stage"],
            "sourceEpoch": int(stage_a_saved["epoch"]),
            "strictFullModelLoad": True,
            "auxiliaryCheckpoint": str(auxiliary_checkpoint.resolve()),
            "auxiliaryCheckpointSha256": auxiliary_hash,
            "strictAuxiliaryHeadLoad": True,
        }
        manifest["auxiliaryContract"] = {
            "nodesPerGame": cfg.auxiliary_nodes_per_game,
            "legalLossWeight": cfg.legal_aux_weight,
            "valueLossWeight": cfg.value_aux_weight,
            "stateOnlyChannelsRetained": [0, 1, 2],
            "stateOnlyChannelsZeroed": list(range(3, 23)),
            "legalTargetSource": "derived-by-othello-rules-from-current-board-tokens-only",
            "valueTargetSource": "current_score_divided_by_64",
            "forbiddenAuxiliaryInputs": [
                "numeric-362", "hint-planes", "hint-values", "history-boards",
                "history-actual-moves", "oq-profile",
            ],
            "auxiliaryHeadsAreFormalInferenceOutputs": False,
        }
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    if cfg.training_mode == "cnn-stage-c":
        stage_identity = {key: manifest[key] for key in (
            "modelSchema", "modelVariant", "boardPerspective", "inputPolicy",
            "dataSha256", "baseCheckpointSha256", "featureOrderSha256",
            "boardChannelOrderSha256", "preprocessingSha256", "oqProfileAblation",
            "oqProfileAblationSha256", "oqProfileFeatureOrderSha256",
            "oqProfilePreprocessingSha256", "oqProfilePolicy",
            "oqProfileTemporalLeakageAuthorized", "testEvaluationPlanned",
        )}
        auxiliary_heads = BoardCNNAuxiliaryHeads(model.backbone.cfg.board_channels)
        stage_b_saved, stage_b_hash = _strict_load_stage_b_checkpoint(
            model, auxiliary_heads, initial_stage_checkpoint,
            cfg.initial_stage_checkpoint_sha256, stage_identity,
        )
        manifest["stageTransition"] = {
            "sourceCheckpoint": str(initial_stage_checkpoint.resolve()),
            "sourceCheckpointSha256": stage_b_hash,
            "sourceSchema": stage_b_saved["schema"],
            "sourceStage": stage_b_saved["stage"],
            "sourceEpoch": int(stage_b_saved["epoch"]),
            "strictFullModelLoad": True,
            "strictAuxiliaryHeadLoad": True,
        }
        manifest["auxiliaryContract"] = dict(stage_b_saved["manifest"]["auxiliaryContract"])
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    device = torch.device(device_info["device"])
    model.to(device)
    if auxiliary_heads is not None:
        auxiliary_heads.to(device)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)
    train_loader = DataLoader(SequenceDataset(data_path, "train", use_oq_profile), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, generator=loader_generator)
    validation_loader = DataLoader(SequenceDataset(data_path, "validation", use_oq_profile), batch_size=cfg.batch_size,
                                   num_workers=cfg.num_workers, pin_memory=True)
    write_progress(output_dir, status="ready-to-train", stage="checkpoint-loaded", total_batches=len(train_loader))
    selection_metric = "validation_wld_classification_loss" if cfg.training_mode == "wld-head-only" else "validation_total_loss"
    manifest["trainingMode"] = cfg.training_mode
    manifest["selectionMetric"] = selection_metric
    manifest["datasetSplitsConstructed"] = ["train", "validation"]
    manifest["testSplitDatasetConstructed"] = False
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    start_epoch, best_value, best_epoch = 1, float("inf"), None
    epochs_without_improvement = 0
    if cfg.training_mode == "wld-head-only":
        stage = "wld-head-only"
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.wld_head.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.wld_head.parameters(), lr=cfg.fine_tune_learning_rate,
                                      weight_decay=cfg.weight_decay)
    elif cfg.training_mode == "cnn-stage-a":
        stage = "cnn-stage-a"
        for parameter in model.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.board_encoder.parameters():
            parameter.requires_grad = False
        if any(parameter.requires_grad for parameter in model.backbone.board_encoder.parameters()):
            raise AssertionError("stage A board CNN must be completely frozen")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=cfg.head_learning_rate, weight_decay=cfg.weight_decay,
        )
        manifest["stageA"] = {
            "frozenModule": "backbone.board_encoder",
            "boardCnnAllFrozen": True,
            "trainableParameterCount": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozenBoardCnnParameterCount": sum(
                parameter.numel() for parameter in model.backbone.board_encoder.parameters()
            ),
        }
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    elif cfg.training_mode == "cnn-stage-b":
        stage = "cnn-stage-b"
        if auxiliary_heads is None:
            raise AssertionError("cnn-stage-b auxiliary heads were not initialized")
        optimizer, stage_b_report = configure_cnn_stage_b(model, auxiliary_heads, cfg)
        manifest["stageB"] = stage_b_report
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    elif cfg.training_mode == "cnn-stage-c":
        stage = "cnn-stage-c"
        if auxiliary_heads is None:
            raise AssertionError("cnn-stage-c auxiliary heads were not restored")
        optimizer, stage_c_report = configure_cnn_stage_c(model, auxiliary_heads, cfg)
        manifest["stageC"] = stage_c_report
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    elif cfg.training_mode == "cnn-direct-full":
        stage = "cnn-direct-full"
        if auxiliary_heads is None:
            raise AssertionError("cnn-direct-full auxiliary heads were not restored")
        optimizer, direct_full_report = configure_cnn_direct_full(model, auxiliary_heads, cfg)
        manifest["directFull"] = direct_full_report
        atomic_write_json(output_dir / "run_manifest.json", manifest)
    else:
        stage = "training-heads"
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.head_learning_rate,
                                      weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        expected = {key: manifest[key] for key in (
            "modelSchema", "dataSha256", "baseCheckpointSha256", "configSha256",
            "featureOrderSha256", "boardChannelOrderSha256", "preprocessingSha256", "inputPolicy",
            "testEvaluationPlanned",
            "modelVariant", "oqProfileAblation", "oqProfileAblationSha256",
            "oqProfileFeatureOrderSha256", "oqProfilePreprocessingSha256", "oqProfilePolicy",
            "oqProfileTemporalLeakageAuthorized",
            "warmStartCheckpoint", "warmStartCheckpointSha256",
            "initialStageCheckpoint", "initialStageCheckpointSha256",
            "auxiliaryCheckpoint", "auxiliaryCheckpointSha256",
        )}
        mismatches = [key for key, value in expected.items() if saved["manifest"].get(key) != value]
        epoch_extension = set(mismatches) == {"configSha256"} and _is_completed_epoch_extension(
            saved["config"], config_payload, int(saved["epoch"])
        )
        if epoch_extension:
            mismatches = []
        if mismatches:
            raise ValueError(f"resume manifest mismatch: {mismatches}")
        if epoch_extension:
            manifest["epochExtension"] = {
                "resumedFromEpoch": int(saved["epoch"]),
                "previousConfigSha256": saved["manifest"]["configSha256"],
                "targetMaxEpochs": max_epochs,
            }
            atomic_write_json(output_dir / "run_manifest.json", manifest)
        if cfg.training_mode in {"cnn-stage-b", "cnn-stage-c", "cnn-direct-full"}:
            expected_schema = (
                "tcn-loss-profile-wld-stage-b-checkpoint-v1"
                if cfg.training_mode == "cnn-stage-b"
                else (
                    "tcn-loss-profile-wld-stage-c-checkpoint-v1"
                    if cfg.training_mode == "cnn-stage-c"
                    else "tcn-loss-profile-wld-direct-full-checkpoint-v1"
                )
            )
            if saved.get("schema") != expected_schema:
                raise ValueError(f"{cfg.training_mode} resume checkpoint schema mismatch")
            if saved.get("stage") != cfg.training_mode or saved.get("board_perspective") != cfg.board_perspective:
                raise ValueError(f"{cfg.training_mode} resume stage or perspective mismatch")
            model.load_state_dict(saved["modelStateDict"], strict=True)
            if auxiliary_heads is None:
                raise AssertionError(f"{cfg.training_mode} auxiliary heads are missing")
            auxiliary_heads.load_state_dict(saved["auxiliaryHeadsStateDict"], strict=True)
            stage_report = (
                manifest["stageB"] if cfg.training_mode == "cnn-stage-b"
                else (manifest["stageC"] if cfg.training_mode == "cnn-stage-c" else manifest["directFull"])
            )
            if saved.get("optimizerParameterGroups") != stage_report["parameterGroups"]:
                raise ValueError(f"{cfg.training_mode} optimizer parameter-group contract mismatch")
        else:
            load_trained_state_with_wld_migration(model, saved["modelStateDict"])
        stage = saved["stage"]
        if stage == "fine-tuning":
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fine_tune_learning_rate, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(saved["optimizerStateDict"])
        if "gradScalerStateDict" in saved:
            scaler.load_state_dict(saved["gradScalerStateDict"])
        random.setstate(saved["pythonRandomState"])
        np.random.set_state(saved["numpyRandomState"])
        torch.set_rng_state(saved["torchRandomState"])
        torch.cuda.set_rng_state_all(saved["cudaRandomStates"])
        loader_generator.set_state(saved["loaderGeneratorState"])
        start_epoch = int(saved["epoch"]) + 1
        best_value, best_epoch = float(saved["bestValidationLoss"]), saved["bestEpoch"]
        epochs_without_improvement = int(saved.get("epochsWithoutImprovement", 0))
    elif cfg.training_mode in {"cnn-stage-b", "cnn-stage-c", "cnn-direct-full"}:
        initial_validation = evaluate(model, validation_loader, device, cfg, use_oq_profile)
        best_value = float(initial_validation["total"])
        best_epoch = 0
        atomic_write_json(output_dir / "initial_validation_metrics.json", {
            "schema": f"tcn-loss-{cfg.training_mode}-initial-validation-metrics-v1",
            "sourceCheckpoint": str(
                (initial_profile_checkpoint if cfg.training_mode == "cnn-direct-full" else initial_stage_checkpoint).resolve()
            ),
            "sourceCheckpointSha256": (
                manifest["warmStartCheckpointSha256"]
                if cfg.training_mode == "cnn-direct-full" else manifest["initialStageCheckpointSha256"]
            ),
            "sourceEpoch": int(
                manifest.get("warmStartEpoch", 0)
                if cfg.training_mode == "cnn-direct-full" else manifest["stageTransition"]["sourceEpoch"]
            ),
            "selectionMetric": selection_metric,
            **initial_validation,
        })
        initial_checkpoint = {
            "schema": (
                "tcn-loss-profile-wld-stage-b-checkpoint-v1"
                if cfg.training_mode == "cnn-stage-b"
                else (
                    "tcn-loss-profile-wld-stage-c-checkpoint-v1"
                    if cfg.training_mode == "cnn-stage-c"
                    else "tcn-loss-profile-wld-direct-full-checkpoint-v1"
                )
            ),
            "modelStateDict": model.state_dict(),
            "auxiliaryHeadsStateDict": auxiliary_heads.state_dict(),
            "optimizerStateDict": optimizer.state_dict(), "epoch": 0, "stage": stage,
            "optimizerParameterGroups": (
                manifest["stageB"]["parameterGroups"] if cfg.training_mode == "cnn-stage-b"
                else (
                    manifest["stageC"]["parameterGroups"]
                    if cfg.training_mode == "cnn-stage-c" else manifest["directFull"]["parameterGroups"]
                )
            ),
            "bestValidationLoss": best_value, "bestEpoch": best_epoch,
            "epochsWithoutImprovement": 0,
            "selectionMetric": selection_metric, "manifest": manifest, "config": config_payload,
            "board_perspective": cfg.board_perspective,
            "pythonRandomState": random.getstate(), "numpyRandomState": np.random.get_state(),
            "torchRandomState": torch.get_rng_state(), "cudaRandomStates": torch.cuda.get_rng_state_all(),
            "loaderGeneratorState": loader_generator.get_state(),
        }
        _atomic_torch_save(output_dir / "best.pt", initial_checkpoint)
        write_progress(
            output_dir, status=cfg.training_mode, stage=stage, epoch=0,
            validation_total_loss=initial_validation["total"],
            thinking_time_loss=initial_validation["thinking_time"],
            severity_classification_loss=initial_validation["severity_classification"],
            wld_classification_loss=initial_validation["wld_classification"],
            wld_validation_metrics=initial_validation["wld"], best_metric=best_value,
            best_epoch=best_epoch, selection_metric=selection_metric,
            learning_rate=cfg.non_cnn_learning_rate,
        )
    elif cfg.training_mode == "wld-head-only" and initial_profile_checkpoint is not None:
        initial_validation = evaluate(model, validation_loader, device, cfg, use_oq_profile)
        best_value = float(initial_validation["wld_classification"])
        best_epoch = 0
        atomic_write_json(output_dir / "initial_validation_metrics.json", {
            "schema": "tcn-loss-initial-validation-metrics-v1",
            "sourceCheckpoint": str(initial_profile_checkpoint.resolve()),
            "sourceEpoch": int(manifest.get("warmStartEpoch", 0)),
            "selectionMetric": selection_metric,
            **initial_validation,
        })
        _atomic_torch_save(output_dir / "best.pt", {
            "schema": "tcn-loss-profile-wld-checkpoint-v2" if use_oq_profile else "tcn-loss-wld-checkpoint-v2",
            "modelStateDict": model.state_dict(),
            "optimizerStateDict": optimizer.state_dict(), "epoch": 0, "stage": stage,
            "bestValidationLoss": best_value, "bestEpoch": best_epoch,
            "selectionMetric": selection_metric,
            "manifest": manifest, "config": config_payload,
            "board_perspective": cfg.board_perspective,
            "pythonRandomState": random.getstate(),
            "numpyRandomState": np.random.get_state(),
            "torchRandomState": torch.get_rng_state(),
            "cudaRandomStates": torch.cuda.get_rng_state_all(),
            "loaderGeneratorState": loader_generator.get_state(),
        })
        write_progress(
            output_dir, status="wld-head-only", stage=stage, epoch=0,
            validation_total_loss=initial_validation["total"],
            wld_classification_loss=initial_validation["wld_classification"],
            wld_validation_metrics=initial_validation["wld"],
            best_metric=best_value, best_epoch=best_epoch,
            selection_metric=selection_metric, learning_rate=optimizer.param_groups[0]["lr"],
        )
    history_path, started = output_dir / "training_history.csv", time.time()
    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, max_epochs + 1):
        if cfg.training_mode == "joint" and epoch == cfg.head_epochs + 1 and stage != "fine-tuning":
            stage = "fine-tuning"
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fine_tune_learning_rate, weight_decay=cfg.weight_decay)
            write_progress(output_dir, status="fine-tuning", stage=stage, epoch=epoch - 1)
        model.train()
        if auxiliary_heads is not None:
            auxiliary_heads.train()
        totals = {key: 0.0 for key in (
            "total", "formal_total", "thinking_time", "severity_classification",
            "wld_classification", "legal_auxiliary", "value_auxiliary",
        )}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
                output = _model_output(model, batch, use_oq_profile)
                losses = multitask_loss(
                    output, batch["actual_thinking_time_ms"].float(), batch["severity_class"].float(), batch["mask"],
                    cfg.time_task_weight, cfg.severity_classification_weight, cfg.severity_class_weights,
                    batch["wld_class"].float(), batch["wld_label_available"],
                    batch["global_placement_ply"], cfg.wld_classification_weight,
                )
                losses["formal_total"] = losses["total"]
                if cfg.training_mode in {"cnn-stage-b", "cnn-stage-c", "cnn-direct-full"}:
                    if auxiliary_heads is None:
                        raise AssertionError(f"{cfg.training_mode} auxiliary heads are missing")
                    auxiliary_losses = stage_b_auxiliary_loss(model, auxiliary_heads, batch, cfg)
                    losses["legal_auxiliary"] = auxiliary_losses["legal_auxiliary"]
                    losses["value_auxiliary"] = auxiliary_losses["value_auxiliary"]
                    losses["total"] = losses["formal_total"] + auxiliary_losses["weighted_total"]
                else:
                    zero = losses["total"] * 0.0
                    losses["legal_auxiliary"] = zero
                    losses["value_auxiliary"] = zero
                scaled_loss = losses["total"] / cfg.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if batch_index % cfg.gradient_accumulation_steps == 0 or batch_index == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key in totals:
                totals[key] += float(losses[key].item())
        train_metrics = {key: value / len(train_loader) for key, value in totals.items()}
        validation_metrics = evaluate(model, validation_loader, device, cfg, use_oq_profile)
        selection_value = (
            validation_metrics["wld_classification"]
            if cfg.training_mode == "wld-head-only"
            else validation_metrics["total"]
        )
        improved = selection_value < best_value
        if improved:
            best_value, best_epoch = selection_value, epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "schema": (
                "tcn-loss-profile-wld-stage-b-checkpoint-v1"
                if cfg.training_mode == "cnn-stage-b"
                else (
                    "tcn-loss-profile-wld-stage-c-checkpoint-v1"
                    if cfg.training_mode == "cnn-stage-c"
                    else (
                        "tcn-loss-profile-wld-direct-full-checkpoint-v1"
                        if cfg.training_mode == "cnn-direct-full"
                        else ("tcn-loss-profile-wld-checkpoint-v2" if use_oq_profile else "tcn-loss-wld-checkpoint-v2")
                    )
                )
            ),
            "modelStateDict": model.state_dict(),
            "optimizerStateDict": optimizer.state_dict(), "epoch": epoch, "stage": stage,
            "gradScalerStateDict": scaler.state_dict(),
            "bestValidationLoss": best_value, "bestEpoch": best_epoch,
            "epochsWithoutImprovement": epochs_without_improvement,
            "selectionMetric": selection_metric,
            "manifest": manifest, "config": config_payload,
            "board_perspective": cfg.board_perspective,
            "pythonRandomState": random.getstate(),
            "numpyRandomState": np.random.get_state(),
            "torchRandomState": torch.get_rng_state(),
            "cudaRandomStates": torch.cuda.get_rng_state_all(),
            "loaderGeneratorState": loader_generator.get_state(),
        }
        if cfg.training_mode in {"cnn-stage-b", "cnn-stage-c", "cnn-direct-full"}:
            checkpoint["auxiliaryHeadsStateDict"] = auxiliary_heads.state_dict()
            checkpoint["optimizerParameterGroups"] = (
                manifest["stageB"]["parameterGroups"] if cfg.training_mode == "cnn-stage-b"
                else (
                    manifest["stageC"]["parameterGroups"]
                    if cfg.training_mode == "cnn-stage-c" else manifest["directFull"]["parameterGroups"]
                )
            )
        _atomic_torch_save(output_dir / "latest.pt", checkpoint)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", checkpoint)
        row = {
            "epoch": epoch, "stage": stage, "train_total_loss": train_metrics["total"],
            "train_formal_total_loss": train_metrics["formal_total"],
            "train_thinking_time_loss": train_metrics["thinking_time"],
            "train_severity_classification_loss": train_metrics["severity_classification"],
            "train_wld_classification_loss": train_metrics["wld_classification"],
            "train_legal_auxiliary_loss": train_metrics["legal_auxiliary"],
            "train_value_auxiliary_loss": train_metrics["value_auxiliary"],
            "validation_total_loss": validation_metrics["total"],
            "validation_thinking_time_loss": validation_metrics["thinking_time"],
            "validation_severity_classification_loss": validation_metrics["severity_classification"],
            "validation_wld_classification_loss": validation_metrics["wld_classification"],
            "validation_wld_accuracy": validation_metrics["wld"]["accuracy"],
            "validation_expected_wld_loss_mae": validation_metrics["wld"]["expected_wld_loss_mae"],
            "zero_log_loss": validation_metrics["zero"]["log_loss"],
            "ge4_log_loss": validation_metrics["ge4"]["log_loss"],
            "ge10_log_loss": validation_metrics["ge10"]["log_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not history_path.exists() or history_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        elapsed = time.time() - started
        eta = elapsed / max(epoch - start_epoch + 1, 1) * (max_epochs - epoch)
        write_progress(
            output_dir, status=stage, stage=stage, epoch=epoch, batch=len(train_loader),
            total_batches=len(train_loader), train_total_loss=train_metrics["total"],
            train_formal_total_loss=train_metrics["formal_total"],
            train_legal_auxiliary_loss=train_metrics["legal_auxiliary"],
            train_value_auxiliary_loss=train_metrics["value_auxiliary"],
            validation_total_loss=validation_metrics["total"],
            thinking_time_loss=validation_metrics["thinking_time"],
            severity_classification_loss=validation_metrics["severity_classification"],
            wld_classification_loss=validation_metrics["wld_classification"],
            wld_validation_metrics=validation_metrics["wld"],
            zero_loss_log_loss=validation_metrics["zero"]["log_loss"],
            ge4_log_loss=validation_metrics["ge4"]["log_loss"], ge10_log_loss=validation_metrics["ge10"]["log_loss"],
            zero_loss_brier=validation_metrics["zero"]["brier_score"],
            ge4_brier=validation_metrics["ge4"]["brier_score"], ge10_brier=validation_metrics["ge10"]["brier_score"],
            zero_validation_metrics=validation_metrics["zero"], ge4_validation_metrics=validation_metrics["ge4"],
            ge10_validation_metrics=validation_metrics["ge10"],
            severity_class_actual_rates=validation_metrics["class_actual_rates"],
            severity_class_mean_probabilities=validation_metrics["class_mean_probabilities"],
            best_metric=best_value, best_epoch=best_epoch, learning_rate=optimizer.param_groups[0]["lr"],
            selection_metric=selection_metric,
            elapsed_seconds=elapsed, eta_seconds=eta,
        )
        completed_epoch = epoch
        if (
            cfg.training_mode in {"cnn-stage-c", "cnn-direct-full"}
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            stopped_early = True
            break
    if not evaluate_test:
        atomic_write_json(output_dir / "validation_only_completion.json", {
            "schema": "tcn-loss-validation-only-completion-v1",
            "ok": True,
            "testEvaluated": False,
            "selectionData": "validation-only",
            "bestValidationTotalLoss": best_value,
            "bestEpoch": best_epoch,
            "maxEpochs": max_epochs,
            "completedEpochs": completed_epoch,
            "stoppedEarly": stopped_early,
            "earlyStoppingPatience": (
                cfg.early_stopping_patience
                if cfg.training_mode in {"cnn-stage-c", "cnn-direct-full"} else 0
            ),
        })
        write_progress(
            output_dir, status="completed-validation-only", stage="completed-validation-only",
            epoch=completed_epoch, validation_total_loss=validation_metrics["total"], best_metric=best_value,
            best_epoch=best_epoch, elapsed_seconds=time.time() - started, eta_seconds=0,
        )
        return
    write_progress(output_dir, status="evaluating", stage="evaluating", epoch=max_epochs)
    test_loader = DataLoader(SequenceDataset(data_path, "test", use_oq_profile), batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, pin_memory=True)
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    load_trained_state_with_wld_migration(model, best["modelStateDict"])
    test_metrics = evaluate(model, test_loader, device, cfg, use_oq_profile)
    atomic_write_json(output_dir / "test_metrics.json", {"schema": "tcn-loss-test-metrics-v1", **test_metrics})
    write_progress(output_dir, status="completed", stage="completed", epoch=max_epochs,
                   validation_total_loss=validation_metrics["total"], best_metric=best_value, best_epoch=best_epoch,
                   elapsed_seconds=time.time() - started, eta_seconds=0)
