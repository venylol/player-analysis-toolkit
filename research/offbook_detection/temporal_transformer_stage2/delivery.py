"""Shared validation and loading helpers for the frozen stage-2 delivery."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from research.offbook_detection.temporal_transformer_stage2.data import sha256_file
from research.offbook_detection.temporal_transformer_stage2.model import (
    TemporalTransformer,
    TemporalTransformerConfig,
)


def load_time_stats(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage2-train-only-time-stats-v1":
        raise ValueError("unsupported time statistics")
    stats = {
        name: payload[name]
        for name in ("thinking_time", "target_remaining_time", "opponent_remaining_time")
    }
    return stats, payload


def load_frozen_temporal_model(
    checkpoint_path: Path,
    board_manifest_path: Path,
    time_stats_path: Path,
    device: torch.device,
) -> tuple[TemporalTransformer, dict[str, object]]:
    checkpoint_path = checkpoint_path.resolve(strict=True)
    board_manifest_path = board_manifest_path.resolve(strict=True)
    time_stats_path = time_stats_path.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "stage2-temporal-transformer-checkpoint-v1":
        raise ValueError("unsupported temporal checkpoint")
    config = TemporalTransformerConfig(**checkpoint["model_config"])
    run_config = checkpoint.get("run_config", {})
    expected_board_hash = run_config.get("board_cache_manifest_sha256")
    expected_stats_hash = run_config.get("time_stats_sha256")
    if sha256_file(board_manifest_path) != expected_board_hash:
        raise ValueError("board cache manifest does not match the checkpoint training input")
    if sha256_file(time_stats_path) != expected_stats_hash:
        raise ValueError("time statistics do not match the checkpoint training input")
    model = TemporalTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint
