"""Strictly replace an old formal board encoder with an independently pretrained CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .backbone import BoardConditionedBackbone, ModelConfig
from .board_cnn import BOARD_EMBEDDING_DIM, DEFAULT_BOARD_CHANNELS, DEFAULT_RESIDUAL_BLOCKS
from .board_cnn_pretrain import sha256_file, write_json
from .board_perspective import BOARD_PERSPECTIVE

BOARD_PREFIX = "board_encoder."
SHARED_PREFIX = "board_encoder.shared."


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return value


def _formal_config(target: dict[str, Any]) -> ModelConfig:
    if all(key in target for key in ("config", "input_dim", "board_encoding")):
        source = ModelConfig.from_checkpoint(target)
        return ModelConfig(
            input_dim=source.input_dim, channels=source.channels, levels=source.levels,
            kernel_size=source.kernel_size, dropout=source.dropout,
            board_embedding_dim=BOARD_EMBEDDING_DIM, board_channels=DEFAULT_BOARD_CHANNELS,
            board_dropout=source.board_dropout, current_hint_planes=source.current_hint_planes,
            residual_blocks=DEFAULT_RESIDUAL_BLOCKS, embedding_projection_kernel=1,
            board_cnn_architecture="residual-v2",
        )
    return ModelConfig()


def transfer_board_cnn(pretrained_path: Path, target_path: Path, output_path: Path) -> dict[str, Any]:
    pretrained_path = pretrained_path.resolve(strict=True)
    target_path = target_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite transfer output: {output_path}")
    pretrained = _load(pretrained_path)
    if pretrained.get("format") != "board-cnn-pretrain-checkpoint-v2":
        raise ValueError("pretrained checkpoint has the wrong format")
    target = _load(target_path)
    if "model_state_dict" not in target:
        raise ValueError("target checkpoint lacks model_state_dict")

    old_state = target["model_state_dict"]
    cfg = _formal_config(target)
    new_state = BoardConditionedBackbone(cfg).state_dict()
    old_non_cnn = {key: value for key, value in old_state.items() if not key.startswith(BOARD_PREFIX)}
    expected_non_cnn = {key for key in new_state if not key.startswith(BOARD_PREFIX)}
    missing_non_cnn = sorted(expected_non_cnn - set(old_non_cnn))
    unexpected_non_cnn = sorted(set(old_non_cnn) - expected_non_cnn)
    if missing_non_cnn or unexpected_non_cnn:
        raise RuntimeError(
            "non-CNN checkpoint state mismatch: "
            f"missing={missing_non_cnn}, unexpected={unexpected_non_cnn}"
        )
    for key, source_tensor in old_non_cnn.items():
        if source_tensor.shape != new_state[key].shape:
            raise ValueError(
                f"non-CNN shape mismatch: {key} {tuple(source_tensor.shape)} != {tuple(new_state[key].shape)}"
            )
        new_state[key] = source_tensor.clone()

    source_shared = pretrained["shared_cnn_state_dict"]
    expected_shared = {key[len(SHARED_PREFIX):] for key in new_state if key.startswith(SHARED_PREFIX)}
    if set(source_shared) != expected_shared:
        raise RuntimeError(
            "pretrained shared CNN state mismatch: "
            f"missing={sorted(expected_shared - set(source_shared))}, "
            f"unexpected={sorted(set(source_shared) - expected_shared)}"
        )
    copied: list[dict[str, Any]] = []
    for source_key, source_tensor in source_shared.items():
        target_key = SHARED_PREFIX + source_key
        target_tensor = new_state[target_key]
        if source_key == "stem.weight":
            if tuple(source_tensor.shape) != (64, 3, 3, 3) or tuple(target_tensor.shape) != (64, 23, 3, 3):
                raise ValueError("stem convolution must transfer [64,3,3,3] -> [64,23,3,3]")
            replacement = torch.zeros_like(target_tensor)
            replacement[:, :3] = source_tensor
        else:
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"shared CNN shape mismatch: {source_key} {tuple(source_tensor.shape)} "
                    f"!= {target_key} {tuple(target_tensor.shape)}"
                )
            replacement = source_tensor.clone()
        new_state[target_key] = replacement
        copied.append({"source": source_key, "target": target_key, "shape": list(replacement.shape)})

    stem = new_state[SHARED_PREFIX + "stem.weight"]
    checks: dict[str, bool] = {
        "firstThreeChannelsExact": torch.equal(stem[:, :3], source_shared["stem.weight"]),
        "remainingTwentyChannelsStrictZero": bool(torch.count_nonzero(stem[:, 3:]).item() == 0),
        "allNonCnnKeysStrictlyLoaded": not missing_non_cnn and not unexpected_non_cnn,
    }
    for source_key, source_tensor in source_shared.items():
        if source_key == "stem.weight":
            continue
        checks[f"{SHARED_PREFIX}{source_key}Exact"] = torch.equal(
            new_state[SHARED_PREFIX + source_key], source_tensor
        )
    if not all(checks.values()):
        raise AssertionError(f"post-transfer verification failed: {checks}")

    excluded_old_cnn_keys = sorted(key for key in old_state if key.startswith(BOARD_PREFIX))
    target["model_state_dict"] = new_state
    config = target.setdefault("config", {})
    config.update({
        "board_channels": 64, "board_embedding_dim": 96, "residual_blocks": 6,
        "embedding_projection_kernel": 1, "board_cnn_architecture": "residual-v2",
    })
    target["board_cnn_pretrain_transfer"] = {
        "pretrainedCheckpoint": str(pretrained_path),
        "pretrainedSha256": sha256_file(pretrained_path),
        "targetCheckpoint": str(target_path),
        "targetSha256": sha256_file(target_path),
        "excludedOldBoardEncoderKeys": excluded_old_cnn_keys,
        "nonCnnMissingKeys": missing_non_cnn,
        "nonCnnUnexpectedKeys": unexpected_non_cnn,
        "strictNonCnnLoad": True,
        "rules": {"target[:,0:3]": "pretrained", "target[:,3:23]": "zero"},
        "auxiliaryHeadsTransferred": False,
    }
    target["board_perspective"] = BOARD_PERSPECTIVE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, output_path)
    manifest = {
        "format": "board-cnn-transfer-manifest-v2",
        "pretrainedCheckpoint": str(pretrained_path), "pretrainedSha256": sha256_file(pretrained_path),
        "targetCheckpoint": str(target_path), "targetSha256": sha256_file(target_path),
        "outputCheckpoint": str(output_path), "outputSha256": sha256_file(output_path),
        "outputBytes": output_path.stat().st_size, "copiedTensors": copied,
        "excludedOldBoardEncoderKeys": excluded_old_cnn_keys,
        "nonCnnMissingKeys": missing_non_cnn, "nonCnnUnexpectedKeys": unexpected_non_cnn,
        "checks": checks, "auxiliaryHeadsTransferred": False, "encoding": "UTF-8",
        "boardPerspective": BOARD_PERSPECTIVE,
    }
    write_json(output_path.with_suffix(output_path.suffix + ".transfer-manifest.json"), manifest)
    return manifest
