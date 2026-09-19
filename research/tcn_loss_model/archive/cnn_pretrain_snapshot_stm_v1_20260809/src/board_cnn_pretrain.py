"""Independent current-board-only CNN pretraining pipeline.

The binary record is deliberately small and fixed-width:
16 bytes of 2-bit board cells, 8 bytes of legal-move bitboard, and one int8 score.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import random
import struct
import time
import uuid
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .board_cnn import (
    BOARD_EMBEDDING_DIM,
    DEFAULT_BOARD_CHANNELS,
    DEFAULT_PROJECTION_KERNEL,
    DEFAULT_RESIDUAL_BLOCKS,
    BoardCNNAuxiliaryHeads,
    SharedBoardCNN,
    parameter_count,
)

SHARD_MAGIC = b"BCNNDS01"
SHARD_VERSION = 1
HEADER = struct.Struct("<8sHHQ12x")
RECORD_SIZE = 25
BOARD_BYTES = 16
PLANE_ORDER = ("current_empty", "current_X", "current_O")
CELL_TO_CODE = {"-": 0, "X": 1, "O": 2}
DATA_CONTRACT = {
    "name": "egaroucid-board-cnn-pretrain-v1",
    "boardOrder": "a1,b1,...,h8",
    "planeOrder": list(PLANE_ORDER),
    "cellEncoding": {"-": 0, "X": 1, "O": 2},
    "recordLayout": {"board2BitBytes": 16, "legalMoveUint64Bytes": 8, "scoreInt8Bytes": 1},
    "legalMoveTarget": "64-cell multi-label mask generated from board; all-zero pass supported",
    "valueTarget": "current-player score divided by 64.0",
    "forbiddenInputs": [
        "legal moves", "score", "hint1/hint6", "numeric features", "history boards",
        "history moves", "OQ player data", "TCN", "formal-model checkpoint",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def legal_moves_bitboard(board: str) -> int:
    """Return legal X moves with square bit i matching board character i."""
    if len(board) != 64 or any(cell not in CELL_TO_CODE for cell in board):
        raise ValueError("board must contain exactly 64 characters from '-', 'X', 'O'")
    result = 0
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for square, cell in enumerate(board):
        if cell != "-":
            continue
        row, col = divmod(square, 8)
        for dr, dc in directions:
            r, c = row + dr, col + dc
            seen_opponent = False
            while 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == "O":
                seen_opponent = True
                r += dr
                c += dc
            if seen_opponent and 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == "X":
                result |= 1 << square
                break
    return result


def pack_board(board: str) -> bytes:
    if len(board) != 64 or any(cell not in CELL_TO_CODE for cell in board):
        raise ValueError("board must contain exactly 64 characters from '-', 'X', 'O'")
    output = bytearray(BOARD_BYTES)
    for index, cell in enumerate(board):
        output[index // 4] |= CELL_TO_CODE[cell] << ((index % 4) * 2)
    return bytes(output)


def unpack_board(packed: bytes) -> np.ndarray:
    if len(packed) != BOARD_BYTES:
        raise ValueError(f"packed board must be {BOARD_BYTES} bytes")
    raw = np.frombuffer(packed, dtype=np.uint8)
    shifts = np.array((0, 2, 4, 6), dtype=np.uint8)
    return ((raw[:, None] >> shifts[None, :]) & 3).reshape(64)


def parse_source_line(line: str) -> tuple[str, int]:
    parts = line.rstrip("\r\n").split()
    if len(parts) != 2:
        raise ValueError("expected '<64-character-board> <integer-score>'")
    board, score_text = parts
    score = int(score_text)
    if not -64 <= score <= 64:
        raise ValueError(f"score outside [-64, 64]: {score}")
    pack_board(board)
    return board, score


def _completed_manifest(shard_path: Path) -> dict[str, Any] | None:
    manifest_path = shard_path.with_suffix(shard_path.suffix + ".manifest.json")
    if not shard_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return None
    expected_size = HEADER.size + int(manifest["records"]) * RECORD_SIZE
    if shard_path.stat().st_size != expected_size:
        raise ValueError(f"completed shard size mismatch: {shard_path}")
    if sha256_file(shard_path) != manifest["sha256"]:
        raise ValueError(f"completed shard checksum mismatch: {shard_path}")
    return manifest


def materialize_source_file(source: Path, output_dir: Path, max_samples: int | None = None) -> dict[str, Any]:
    """Create one resumable shard. Existing complete output is verified and reused."""
    source = source.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{source.stem}.bcnn"
    completed = _completed_manifest(shard_path)
    if completed is not None:
        return {**completed, "reused": True}
    if shard_path.exists():
        raise FileExistsError(
            f"incomplete final shard is preserved: {shard_path}; move it aside manually before retrying"
        )

    attempt_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    attempt_path = output_dir / f"{source.stem}.attempt-{attempt_id}.part"
    attempt_manifest = attempt_path.with_suffix(attempt_path.suffix + ".manifest.json")
    records = 0
    started = time.time()
    try:
        with source.open("r", encoding="utf-8", newline="") as input_handle, attempt_path.open("w+b") as output_handle:
            output_handle.write(HEADER.pack(SHARD_MAGIC, SHARD_VERSION, RECORD_SIZE, 0))
            for line_number, line in enumerate(input_handle, start=1):
                if max_samples is not None and records >= max_samples:
                    break
                try:
                    board, score = parse_source_line(line)
                except Exception as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                output_handle.write(pack_board(board))
                output_handle.write(struct.pack("<Qb", legal_moves_bitboard(board), score))
                records += 1
            output_handle.seek(0)
            output_handle.write(HEADER.pack(SHARD_MAGIC, SHARD_VERSION, RECORD_SIZE, records))
            output_handle.flush()
            os.fsync(output_handle.fileno())
        attempt_payload = {
            "status": "attempt-complete",
            "attemptId": attempt_id,
            "source": str(source),
            "sourceBytes": source.stat().st_size,
            "records": records,
            "bytes": attempt_path.stat().st_size,
            "sha256": sha256_file(attempt_path),
            "maxSamples": max_samples,
            "elapsedSeconds": time.time() - started,
            "dataContract": DATA_CONTRACT,
        }
        write_json(attempt_manifest, attempt_payload)
        if shard_path.exists():
            raise FileExistsError(f"final shard appeared during attempt and was not overwritten: {shard_path}")
        attempt_path.rename(shard_path)
        final_manifest_path = shard_path.with_suffix(shard_path.suffix + ".manifest.json")
        final_payload = {**attempt_payload, "status": "complete", "shard": str(shard_path.resolve()), "reused": False}
        write_json(final_manifest_path, final_payload)
        return final_payload
    except BaseException as exc:
        failure = {
            "status": "failed",
            "attemptId": attempt_id,
            "source": str(source),
            "attemptPath": str(attempt_path.resolve()),
            "recordsWritten": records,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(attempt_manifest, failure)
        raise


def materialize_files(
    source_dir: Path,
    output_dir: Path,
    file_names: Iterable[str],
    max_samples: int | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.resolve(strict=True)
    output_dir = output_dir.resolve() if output_dir.exists() else output_dir.absolute()
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    shards = [materialize_source_file(source_dir / name, output_dir, max_samples) for name in file_names]
    manifest = {
        "format": "board-cnn-shards-v1",
        "sourceDirectory": str(source_dir),
        "outputDirectory": str(output_dir),
        "maxSamplesPerFile": max_samples,
        "shards": shards,
        "dataContract": DATA_CONTRACT,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


class BoardShardDataset(Dataset[bytes]):
    def __init__(self, shard_paths: Sequence[Path]) -> None:
        if not shard_paths:
            raise ValueError("at least one shard is required")
        self.paths = [Path(path).resolve(strict=True) for path in shard_paths]
        self.counts: list[int] = []
        self.cumulative: list[int] = []
        total = 0
        for path in self.paths:
            with path.open("rb") as handle:
                magic, version, record_size, count = HEADER.unpack(handle.read(HEADER.size))
            if (magic, version, record_size) != (SHARD_MAGIC, SHARD_VERSION, RECORD_SIZE):
                raise ValueError(f"unsupported shard header: {path}")
            if path.stat().st_size != HEADER.size + count * RECORD_SIZE:
                raise ValueError(f"shard size mismatch: {path}")
            self.counts.append(count)
            total += count
            self.cumulative.append(total)
        self._handles: dict[int, Any] = {}
        self._maps: dict[int, mmap.mmap] = {}

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_maps"] = {}
        return state

    def close(self) -> None:
        for mapping in self._maps.values():
            mapping.close()
        for handle in self._handles.values():
            handle.close()
        self._maps.clear()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()

    def _mapping(self, shard_index: int) -> mmap.mmap:
        if shard_index not in self._maps:
            handle = self.paths[shard_index].open("rb")
            self._handles[shard_index] = handle
            self._maps[shard_index] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[shard_index]

    def __getitem__(self, index: int) -> bytes:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        local_index = index - (self.cumulative[shard_index - 1] if shard_index else 0)
        start = HEADER.size + local_index * RECORD_SIZE
        return self._mapping(shard_index)[start:start + RECORD_SIZE]


def collate_records(records: Sequence[bytes]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    raw = np.frombuffer(b"".join(records), dtype=np.uint8).reshape(batch, RECORD_SIZE)
    packed = raw[:, :BOARD_BYTES]
    shifts = np.array((0, 2, 4, 6), dtype=np.uint8)
    codes = ((packed[:, :, None] >> shifts[None, None, :]) & 3).reshape(batch, 64)
    planes = np.stack((codes == 0, codes == 1, codes == 2), axis=1).astype(np.float32)
    legal_bytes = np.ascontiguousarray(raw[:, BOARD_BYTES:BOARD_BYTES + 8])
    legal_words = legal_bytes.view("<u8").reshape(batch)
    bit_shifts = np.arange(64, dtype=np.uint64)
    legal = ((legal_words[:, None] >> bit_shifts[None, :]) & 1).astype(np.float32)
    scores = raw[:, -1].view(np.int8).astype(np.float32) / 64.0
    return torch.from_numpy(planes.reshape(batch, 3, 8, 8)), torch.from_numpy(legal), torch.from_numpy(scores)


class BoardCNNPretrainModel(nn.Module):
    """Spatial trunk aligned with the transferable portion of BoardCNNEncoder."""

    def __init__(
        self,
        input_channels: int = 3,
        board_channels: int = DEFAULT_BOARD_CHANNELS,
        residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
        board_embedding_dim: int = BOARD_EMBEDDING_DIM,
        embedding_projection_kernel: int = DEFAULT_PROJECTION_KERNEL,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("independent pretraining input_channels must be 3")
        self.shared = SharedBoardCNN(
            input_channels, board_channels, residual_blocks,
            board_embedding_dim, embedding_projection_kernel,
        )
        self.auxiliary_heads = BoardCNNAuxiliaryHeads(board_channels)

    @property
    def legal_head(self) -> nn.Conv2d:
        return self.auxiliary_heads.legal_head

    @property
    def value_head(self) -> nn.Sequential:
        return self.auxiliary_heads.value_head

    def forward(self, current_board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if current_board_planes.ndim != 4 or tuple(current_board_planes.shape[1:]) != (3, 8, 8):
            raise ValueError(f"expected Bx3x8x8 input, got {tuple(current_board_planes.shape)}")
        spatial = self.shared.spatial_features(current_board_planes)
        embedding = self.shared.embedding_from_spatial(spatial)
        return self.auxiliary_heads(spatial, embedding)


@dataclass(frozen=True)
class BoardCNNTrainingConfig:
    data_dir: Path
    shard_dir: Path
    splits: dict[str, list[str]]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    legal_loss_weight: float
    value_loss_weight: float
    num_workers: int
    seed: int
    checkpoint_interval_epochs: int
    validation_interval_epochs: int
    mixed_precision: bool
    resume_strategy: str
    smoke_max_samples: int
    board_channels: int
    residual_blocks: int
    board_embedding_dim: int
    embedding_projection_kernel: int
    input_channels: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.input_channels != 3:
            raise ValueError("independent pretraining input_channels must be 3")
        if self.board_embedding_dim != BOARD_EMBEDDING_DIM:
            raise ValueError(f"board_embedding_dim must be {BOARD_EMBEDDING_DIM}")

    @classmethod
    def load(cls, path: Path) -> "BoardCNNTrainingConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent.parent
        def resolved(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        return cls(
            data_dir=resolved(raw["data_dir"]), shard_dir=resolved(raw["shard_dir"]),
            splits={key: list(value) for key, value in raw["splits"].items()},
            batch_size=int(raw["batch_size"]), epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]), weight_decay=float(raw["weight_decay"]),
            legal_loss_weight=float(raw["legal_loss_weight"]), value_loss_weight=float(raw["value_loss_weight"]),
            num_workers=int(raw["num_workers"]), seed=int(raw["seed"]),
            checkpoint_interval_epochs=int(raw["checkpoint_interval_epochs"]),
            validation_interval_epochs=int(raw["validation_interval_epochs"]),
            mixed_precision=bool(raw["mixed_precision"]), resume_strategy=str(raw["resume_strategy"]),
            smoke_max_samples=int(raw["smoke_max_samples"]),
            board_channels=int(raw["board_channels"]),
            residual_blocks=int(raw["residual_blocks"]),
            board_embedding_dim=int(raw["board_embedding_dim"]),
            embedding_projection_kernel=int(raw["embedding_projection_kernel"]),
            input_channels=int(raw["input_channels"]),
            gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        )

    def as_manifest(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["data_dir"] = str(self.data_dir)
        result["shard_dir"] = str(self.shard_dir)
        return result


def split_shards(cfg: BoardCNNTrainingConfig, split: str, shard_dir: Path | None = None) -> list[Path]:
    root = shard_dir or cfg.shard_dir
    return [root / f"{Path(name).stem}.bcnn" for name in cfg.splits[split]]


def checkpoint_payload(
    model: BoardCNNPretrainModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    cfg: BoardCNNTrainingConfig,
    data_manifest: dict[str, Any],
    best_validation: dict[str, float],
) -> dict[str, Any]:
    file_inventory = [
        {
            "sourcePath": shard["source"], "sourceBytes": shard["sourceBytes"],
            "shardPath": shard["shard"], "shardBytes": shard["bytes"],
            "records": shard["records"], "shardSha256": shard["sha256"],
        }
        for shard in data_manifest["shards"]
    ]
    return {
        "format": "board-cnn-pretrain-checkpoint-v2",
        "shared_cnn_state_dict": model.shared.state_dict(),
        "legal_move_head_state_dict": model.legal_head.state_dict(),
        "value_head_state_dict": model.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": cfg.as_manifest(),
        "data_contract": DATA_CONTRACT,
        "splits": cfg.splits,
        "seed": cfg.seed,
        "data_files": file_inventory,
        "best_validation": best_validation,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        "format": payload["format"], "checkpoint": str(path.resolve()),
        "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "epoch": payload["epoch"], "step": payload["step"],
        "bestValidation": payload["best_validation"], "encoding": "UTF-8",
    }
    write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def load_pretrain_checkpoint(
    path: Path, model: BoardCNNPretrainModel, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "board-cnn-pretrain-checkpoint-v2":
        raise ValueError("not a board-CNN pretraining checkpoint")
    model.shared.load_state_dict(payload["shared_cnn_state_dict"], strict=True)
    model.legal_head.load_state_dict(payload["legal_move_head_state_dict"], strict=True)
    model.value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _metrics(legal_logits: torch.Tensor, legal_targets: torch.Tensor, values: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    prediction = legal_logits >= 0
    truth = legal_targets.bool()
    tp = (prediction & truth).sum().item()
    fp = (prediction & ~truth).sum().item()
    fn = (~prediction & truth).sum().item()
    tn = (~prediction & ~truth).sum().item()
    errors = (values - targets) * 64.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "exact": (prediction == truth).all(dim=1).sum().item(), "samples": targets.numel(),
        "absoluteErrorSum": errors.abs().sum().item(), "squaredErrorSum": errors.square().sum().item(),
        "signCorrect": (torch.sign(values) == torch.sign(targets)).sum().item(),
    }


def _finalize_metrics(total: dict[str, float], legal_bce_sum: float) -> dict[str, float]:
    tp, fp, fn, tn, samples = (total[key] for key in ("tp", "fp", "fn", "tn", "samples"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "legalBcePerCell": legal_bce_sum / max(samples * 64, 1),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exactLegalSetAccuracy": total["exact"] / max(samples, 1),
        "illegalCellFalsePositiveRate": fp / max(fp + tn, 1),
        "valueMaeDiscs": total["absoluteErrorSum"] / max(samples, 1),
        "valueRmseDiscs": math.sqrt(total["squaredErrorSum"] / max(samples, 1)),
        "valueSignAccuracy": total["signCorrect"] / max(samples, 1),
    }


@torch.no_grad()
def evaluate(model: BoardCNNPretrainModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("tp", "fp", "fn", "tn", "exact", "samples", "absoluteErrorSum", "squaredErrorSum", "signCorrect")}
    legal_bce_sum = 0.0
    for boards, legal, values in loader:
        boards, legal, values = boards.to(device), legal.to(device), values.to(device)
        logits, prediction = model(boards)
        legal_bce_sum += F.binary_cross_entropy_with_logits(logits, legal, reduction="sum").item()
        for key, value in _metrics(logits, legal, prediction, values).items():
            totals[key] += value
    return _finalize_metrics(totals, legal_bce_sum)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_board_cnn(
    cfg: BoardCNNTrainingConfig,
    output_dir: Path,
    device: torch.device,
    resume: Path | None = None,
    shard_dir: Path | None = None,
    max_steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    _seed_everything(cfg.seed)
    root = shard_dir or cfg.shard_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_data = BoardShardDataset(split_shards(cfg, "train", root))
    validation_data = BoardShardDataset(split_shards(cfg, "validation", root))
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_records)
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    model = BoardCNNPretrainModel(
        cfg.input_channels, cfg.board_channels, cfg.residual_blocks,
        cfg.board_embedding_dim, cfg.embedding_projection_kernel,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_epoch, global_step, best = 0, 0, {}
    if resume is not None:
        restored = load_pretrain_checkpoint(resume, model, optimizer)
        if restored["data_contract"] != DATA_CONTRACT:
            raise ValueError("resume checkpoint data contract differs from the current contract")
        if restored["splits"] != cfg.splits or int(restored["seed"]) != cfg.seed:
            raise ValueError("resume checkpoint split or seed differs from the current configuration")
        architecture_fields = (
            "input_channels", "board_channels", "residual_blocks",
            "board_embedding_dim", "embedding_projection_kernel",
        )
        architecture_mismatches = [
            field for field in architecture_fields
            if int(restored["config"][field]) != int(getattr(cfg, field))
        ]
        if architecture_mismatches:
            raise ValueError(f"resume checkpoint CNN architecture differs: {architecture_mismatches}")
        start_epoch, global_step = int(restored["epoch"]) + 1, int(restored["step"])
        best = dict(restored["best_validation"])
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    saved: list[dict[str, Any]] = []
    last_losses: dict[str, float] = {}
    training_started = time.time()
    samples_processed = 0
    write_json(output_dir / "progress.json", {
        "status": "training",
        "stage": "board-cnn-pretraining",
        "device": str(device),
        "epoch": start_epoch + 1,
        "max_epochs": cfg.epochs,
        "batch": 0,
        "batches_per_epoch": len(train_loader),
        "optimizer_step": global_step,
        "samples_processed_this_run": 0,
        "batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "mixed_precision_active": use_amp,
        "formal_training_started": device.type == "cuda",
        "updated_at_unix": training_started,
    })
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0
        for batch_index, (boards, legal, values) in enumerate(train_loader):
            boards, legal, values = boards.to(device), legal.to(device), values.to(device)
            samples_processed += boards.shape[0]
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, prediction = model(boards)
                legal_loss = F.binary_cross_entropy_with_logits(logits, legal)
                value_loss = F.smooth_l1_loss(prediction, values)
                loss = cfg.legal_loss_weight * legal_loss + cfg.value_loss_weight * value_loss
            scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
            accumulated_batches += 1
            should_stop = max_steps_per_epoch is not None and batch_index + 1 >= max_steps_per_epoch
            should_step = accumulated_batches == cfg.gradient_accumulation_steps or should_stop
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulated_batches = 0
            last_losses = {"total": loss.item(), "legal": legal_loss.item(), "value": value_loss.item()}
            if batch_index == 0 or (should_step and global_step % 100 == 0):
                elapsed = max(time.time() - training_started, 1e-9)
                legal_exact_set_accuracy = (
                    ((logits >= 0) == legal.bool()).all(dim=1).float().mean().item()
                )
                value_mae_discs = ((prediction - values).abs().mean() * 64.0).item()
                write_json(output_dir / "progress.json", {
                    "status": "training",
                    "stage": "board-cnn-pretraining",
                    "device": str(device),
                    "epoch": epoch + 1,
                    "max_epochs": cfg.epochs,
                    "batch": batch_index + 1,
                    "batches_per_epoch": len(train_loader),
                    "optimizer_step": global_step,
                    "samples_processed_this_run": samples_processed,
                    "samples_per_second": samples_processed / elapsed,
                    "last_losses": last_losses,
                    "last_batch_legal_exact_set_accuracy": legal_exact_set_accuracy,
                    "last_batch_value_mae_discs": value_mae_discs,
                    "batch_size": cfg.batch_size,
                    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                    "mixed_precision_active": use_amp,
                    "formal_training_started": device.type == "cuda",
                    "elapsed_seconds": elapsed,
                    "updated_at_unix": time.time(),
                })
            if should_stop:
                break
        if accumulated_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        validation = evaluate(model, validation_loader, device) if (epoch + 1) % cfg.validation_interval_epochs == 0 else {}
        if validation:
            selection = validation["legalBcePerCell"] + validation["valueMaeDiscs"] / 64.0
            if selection < best.get("selectionMetric", float("inf")):
                best = {**validation, "selectionMetric": selection, "epoch": epoch}
        if (epoch + 1) % cfg.checkpoint_interval_epochs == 0:
            payload = checkpoint_payload(model, optimizer, epoch, global_step, cfg, manifest, best)
            saved.append(save_checkpoint(output_dir / "checkpoints" / f"epoch-{epoch:04d}-step-{global_step:08d}.pt", payload))
    train_data.close()
    validation_data.close()
    write_json(output_dir / "progress.json", {
        "status": "completed",
        "stage": "board-cnn-pretraining",
        "device": str(device),
        "max_epochs": cfg.epochs,
        "optimizer_step": global_step,
        "samples_processed_this_run": samples_processed,
        "last_losses": last_losses,
        "best_validation": best,
        "formal_training_started": device.type == "cuda",
        "elapsed_seconds": time.time() - training_started,
        "updated_at_unix": time.time(),
    })
    return {
        "device": str(device), "epochsCompleted": max(0, cfg.epochs - start_epoch),
        "globalStep": global_step, "lastLosses": last_losses, "bestValidation": best,
        "checkpoints": saved,
    }


def smoke_run(cfg: BoardCNNTrainingConfig, output_dir: Path) -> dict[str, Any]:
    smoke_shards = cfg.shard_dir.parent / "board_cnn_pretrain_smoke_shards"
    smoke_files = [cfg.splits[split][0] for split in ("train", "validation", "test")]
    data_manifest = materialize_files(cfg.data_dir, smoke_shards, smoke_files, cfg.smoke_max_samples)
    smoke_splits = {split: [cfg.splits[split][0]] for split in ("train", "validation", "test")}
    first_cfg = replace(cfg, shard_dir=smoke_shards, splits=smoke_splits, epochs=1, num_workers=0, mixed_precision=False)
    first = train_board_cnn(first_cfg, output_dir, torch.device("cpu"), max_steps_per_epoch=2)
    first_checkpoint = Path(first["checkpoints"][-1]["checkpoint"])
    second_cfg = replace(first_cfg, epochs=2)
    second = train_board_cnn(second_cfg, output_dir, torch.device("cpu"), resume=first_checkpoint, max_steps_per_epoch=1)
    test_loader = DataLoader(
        BoardShardDataset(split_shards(second_cfg, "test", smoke_shards)),
        batch_size=second_cfg.batch_size, collate_fn=collate_records,
    )
    restored_model = BoardCNNPretrainModel(
        second_cfg.input_channels, second_cfg.board_channels, second_cfg.residual_blocks,
        second_cfg.board_embedding_dim, second_cfg.embedding_projection_kernel,
    )
    load_pretrain_checkpoint(Path(second["checkpoints"][-1]["checkpoint"]), restored_model)
    test_metrics = evaluate(restored_model, test_loader, torch.device("cpu"))
    report = {
        "ok": True, "formalTrainingStarted": False, "optimizationDevice": "cpu",
        "materializedSamplesPerSplitFile": cfg.smoke_max_samples,
        "forwardBackward": True, "checkpointSaved": True, "checkpointResumed": True,
        "firstRun": first, "resumedRun": second, "testMetrics": test_metrics,
        "dataManifest": data_manifest,
    }
    write_json(output_dir / "smoke-report.json", report)
    return report


def benchmark_board_cnn(
    cfg: BoardCNNTrainingConfig,
    device: torch.device,
    steps: int = 5,
    warmup_steps: int = 2,
) -> dict[str, Any]:
    """Run a bounded synthetic compute benchmark; it never reads or preprocesses the full dataset."""
    if steps <= 0 or warmup_steps < 0:
        raise ValueError("benchmark steps must be positive and warmup_steps non-negative")
    model = BoardCNNPretrainModel(
        cfg.input_channels, cfg.board_channels, cfg.residual_blocks,
        cfg.board_embedding_dim, cfg.embedding_projection_kernel,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    boards = torch.randn(cfg.batch_size, 3, 8, 8, device=device)
    legal = torch.zeros(cfg.batch_size, 64, device=device)
    values = torch.zeros(cfg.batch_size, device=device)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward_once() -> None:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            model(boards)

    def train_once() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, prediction = model(boards)
            loss = F.binary_cross_entropy_with_logits(logits, legal) + F.smooth_l1_loss(prediction, values)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup_steps):
        forward_once()
        train_once()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        forward_once()
    synchronize()
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(steps):
        train_once()
    synchronize()
    train_seconds = time.perf_counter() - started
    return {
        "device": str(device), "syntheticInput": True, "batchSize": cfg.batch_size,
        "measuredSteps": steps, "forwardStepMilliseconds": forward_seconds * 1000.0 / steps,
        "trainingStepMilliseconds": train_seconds * 1000.0 / steps,
        "forwardSamplesPerSecond": cfg.batch_size * steps / forward_seconds,
        "trainingSamplesPerSecond": cfg.batch_size * steps / train_seconds,
        "peakGpuMemoryBytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "sharedCnnParameters": parameter_count(model.shared),
        "legalHeadParameters": parameter_count(model.legal_head),
        "valueHeadParameters": parameter_count(model.value_head),
    }
