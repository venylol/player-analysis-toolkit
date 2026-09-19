"""Unified reader for the retained LV17 data and the 11,200-game pass supplement."""

from __future__ import annotations

import json
import mmap
import struct
from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SHARD_MAGIC = b"BCNNDS01"
SHARD_VERSION = 1
HEADER = struct.Struct("<8sHHQ12x")
BASE_RECORD_SIZE = 25
BOARD_BYTES = 16


class BcnnShardDataset(Dataset[tuple[bytes, int, bool]]):
    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths:
            raise ValueError("at least one base shard is required")
        self.paths = [Path(path).resolve(strict=True) for path in paths]
        self.counts: list[int] = []
        self.cumulative: list[int] = []
        total = 0
        for path in self.paths:
            with path.open("rb") as handle:
                magic, version, record_size, count = HEADER.unpack(handle.read(HEADER.size))
            if (magic, version, record_size) != (SHARD_MAGIC, SHARD_VERSION, BASE_RECORD_SIZE):
                raise ValueError(f"unsupported base shard header: {path}")
            if path.stat().st_size != HEADER.size + count * BASE_RECORD_SIZE:
                raise ValueError(f"base shard size mismatch: {path}")
            self.counts.append(count)
            total += count
            self.cumulative.append(total)
        self._handles: dict[int, object] = {}
        self._maps: dict[int, mmap.mmap] = {}

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_maps"] = {}
        return state

    def close(self) -> None:
        for mapping in self._maps.values():
            mapping.close()
        for handle in self._handles.values():
            handle.close()  # type: ignore[attr-defined]
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

    def __getitem__(self, index: int) -> tuple[bytes, int, bool]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        prior = self.cumulative[shard_index - 1] if shard_index else 0
        local_index = index - prior
        start = HEADER.size + local_index * BASE_RECORD_SIZE
        record = self._mapping(shard_index)[start:start + BASE_RECORD_SIZE]
        legal_mask = struct.unpack("<Q", record[BOARD_BYTES:BOARD_BYTES + 8])[0]
        return record[:BOARD_BYTES], legal_mask, legal_mask == 0


class PassSupplementDataset(Dataset[tuple[bytes, int, bool]]):
    def __init__(self, path: Path, split: str) -> None:
        with np.load(path.resolve(strict=True), allow_pickle=False) as data:
            selected = data["split"] == split
            self.packed_boards = np.ascontiguousarray(data["packed_boards"][selected])
            self.legal_masks = np.ascontiguousarray(data["legal_masks"][selected])
        if self.packed_boards.ndim != 2 or self.packed_boards.shape[1] != BOARD_BYTES:
            raise ValueError("pass supplement packed_boards must have shape Nx16")
        if np.any(self.legal_masks != 0):
            raise ValueError("pass supplement contains a nonzero legal mask")

    def __len__(self) -> int:
        return self.packed_boards.shape[0]

    def __getitem__(self, index: int) -> tuple[bytes, int, bool]:
        return self.packed_boards[index].tobytes(), int(self.legal_masks[index]), True


class CombinedLegalDataset(Dataset[tuple[bytes, int, bool]]):
    def __init__(self, base: BcnnShardDataset, supplement: PassSupplementDataset) -> None:
        self.base = base
        self.supplement = supplement

    def __len__(self) -> int:
        return len(self.base) + len(self.supplement)

    def __getitem__(self, index: int) -> tuple[bytes, int, bool]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.base[index] if index < len(self.base) else self.supplement[index - len(self.base)]


class PassAwareBatchSampler:
    """Cover every base record once while sampling a fixed pass fraction per batch."""

    def __init__(
        self,
        base_size: int,
        supplement_size: int,
        batch_size: int,
        pass_fraction: float,
        seed: int,
        epoch: int = 0,
    ) -> None:
        if base_size <= 0 or supplement_size <= 0 or batch_size <= 1:
            raise ValueError("base, supplement, and batch sizes must be positive")
        if not 0 < pass_fraction < 1:
            raise ValueError("pass_fraction must be between zero and one")
        self.base_size = base_size
        self.supplement_size = supplement_size
        self.batch_size = batch_size
        self.pass_per_batch = max(1, round(batch_size * pass_fraction))
        self.base_per_batch = batch_size - self.pass_per_batch
        if self.base_per_batch <= 0:
            raise ValueError("pass fraction leaves no base records in a batch")
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return (self.base_size + self.base_per_batch - 1) // self.base_per_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        base_order = torch.randperm(self.base_size, generator=generator)
        supplement_offset = self.base_size
        for start in range(0, self.base_size, self.base_per_batch):
            base_indices = base_order[start:start + self.base_per_batch].tolist()
            pass_indices = (
                torch.randint(self.supplement_size, (self.pass_per_batch,), generator=generator)
                + supplement_offset
            ).tolist()
            batch = base_indices + pass_indices
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


def load_combined_dataset(config_path: Path, split: str) -> CombinedLegalDataset:
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "transformer-legal-pretrain-data-v1":
        raise ValueError("unsupported legal pretraining data config")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unsupported split: {split}")
    root = config_path.parent
    base_dir = (root / config["base"]["directory"]).resolve(strict=True)
    base = BcnnShardDataset([base_dir / name for name in config["base"]["splits"][split]])
    supplement_path = root / config["passSupplement"]["path"]
    supplement = PassSupplementDataset(supplement_path, split)
    combined = CombinedLegalDataset(base, supplement)
    expected = int(config["combinedCounts"][split])
    if len(combined) != expected:
        raise ValueError(f"combined {split} count differs: expected {expected}, got {len(combined)}")
    return combined


def collate_legal_records(
    records: Sequence[tuple[bytes, int, bool]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    packed = np.frombuffer(b"".join(record[0] for record in records), dtype=np.uint8).reshape(batch, BOARD_BYTES)
    shifts = np.array((0, 2, 4, 6), dtype=np.uint8)
    codes = ((packed[:, :, None] >> shifts[None, None, :]) & 3).reshape(batch, 64)
    legal_words = np.asarray([record[1] for record in records], dtype=np.uint64)
    bit_shifts = np.arange(64, dtype=np.uint64)
    legal = ((legal_words[:, None] >> bit_shifts[None, :]) & 1).astype(np.float32)
    is_pass = np.asarray([record[2] for record in records], dtype=np.bool_)
    return torch.from_numpy(codes.copy()), torch.from_numpy(legal), torch.from_numpy(is_pass)
