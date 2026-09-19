"""Readers and feature construction for the sharded stage-2 board cache."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from research.offbook_detection.temporal_transformer_stage2.data import sha256_file


@dataclass(frozen=True)
class SequenceRef:
    shard_index: int
    game_index: int
    target_view: int
    split: str
    game_id: str
    node_count: int


class BoardCache:
    def __init__(self, directory: Path, verify_hashes: bool = True, max_open_shards: int = 2) -> None:
        self.directory = directory.resolve(strict=True)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "stage2-canonical-board-cache-v1":
            raise ValueError("unsupported board cache manifest")
        if self.manifest.get("status") not in ("complete", "smoke-test-subset"):
            raise ValueError("board cache is not complete")
        if self.manifest.get("dtype") != "float32" or int(self.manifest.get("embedding_dim", 0)) != 96:
            raise ValueError("board cache violates the frozen embedding contract")
        self.shards = list(self.manifest["shards"])
        if verify_hashes:
            for shard in self.shards:
                path = self.directory / shard["file"]
                if sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"board cache shard hash mismatch: {path}")
        self.max_open_shards = max_open_shards
        self._loaded: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.sequence_refs: list[SequenceRef] = []
        for shard_index in range(len(self.shards)):
            with np.load(self.directory / self.shards[shard_index]["file"], allow_pickle=False) as shard:
                for game_index, (game_id, split, count) in enumerate(
                    zip(shard["game_id"].tolist(), shard["split"].tolist(), shard["game_node_count"].tolist())
                ):
                    for target_view in (0, 1):
                        self.sequence_refs.append(
                            SequenceRef(shard_index, game_index, target_view, str(split), str(game_id), int(count))
                        )

    def close(self) -> None:
        self._loaded.clear()

    def __del__(self) -> None:
        self.close()

    def load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index in self._loaded:
            self._loaded.move_to_end(shard_index)
            return self._loaded[shard_index]
        with np.load(self.directory / self.shards[shard_index]["file"], allow_pickle=False) as archive:
            shard = {name: archive[name] for name in archive.files}
        self._loaded[shard_index] = shard
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return shard

    def sequence(self, ref: SequenceRef) -> dict[str, np.ndarray | str | int]:
        shard = self.load_shard(ref.shard_index)
        offset = int(shard["game_node_offset"][ref.game_index])
        stop = offset + ref.node_count
        target_is_black = ref.target_view == 0
        actor_is_black = shard["actor_is_black"][offset:stop].astype(np.bool_, copy=True)
        black_remaining = shard["black_remaining_time_ms_after"][offset:stop].astype(np.float32)
        white_remaining = shard["white_remaining_time_ms_after"][offset:stop].astype(np.float32)
        return {
            "game_id": ref.game_id,
            "split": ref.split,
            "target_view": ref.target_view,
            "board_embedding": shard["embeddings"][ref.target_view, offset:stop].astype(np.float32, copy=True),
            "thinking_time_ms": shard["thinking_time_ms"][offset:stop].astype(np.float32),
            "target_remaining_time_ms_after": black_remaining if target_is_black else white_remaining,
            "opponent_remaining_time_ms_after": white_remaining if target_is_black else black_remaining,
            "strict_ply": shard["strict_ply"][offset:stop].astype(np.float32),
            "node_index": shard["node_index"][offset:stop].astype(np.int64),
            "actor_is_target": actor_is_black == target_is_black,
            "actor_is_black": actor_is_black,
            "is_pass": shard["is_pass"][offset:stop].astype(np.bool_, copy=True),
            "time_control_index": 0,
        }


class TemporalSequenceDataset(Dataset[dict[str, np.ndarray | str | int]]):
    def __init__(self, cache: BoardCache, split: str) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unsupported split: {split}")
        self.cache = cache
        self.refs = [ref for ref in cache.sequence_refs if ref.split == split]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str | int]:
        return self.cache.sequence(self.refs[index])


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle sequences while keeping each batch inside one cache shard."""

    def __init__(self, dataset: TemporalSequenceDataset, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.by_shard: dict[int, list[int]] = {}
        for index, ref in enumerate(dataset.refs):
            self.by_shard.setdefault(ref.shard_index, []).append(index)

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.by_shard.values())

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(len(self.by_shard), generator=generator).tolist()
        shard_ids = list(self.by_shard)
        for position in shard_order:
            indices = self.by_shard[shard_ids[position]]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[index] for index in order]
            for start in range(0, len(shuffled), self.batch_size):
                yield shuffled[start:start + self.batch_size]


def robust_standardize(values_ms: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (np.log1p(values_ms.astype(np.float32)) - center) / scale


def collate_sequences(
    records: list[dict[str, np.ndarray | str | int]], stats: dict[str, dict[str, float]]
) -> dict[str, torch.Tensor | list[str]]:
    batch = len(records)
    lengths = torch.tensor([len(record["node_index"]) for record in records], dtype=torch.long)  # type: ignore[arg-type]
    max_length = int(lengths.max())
    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)

    def padded(name: str, dtype: np.dtype, trailing: tuple[int, ...] = ()) -> np.ndarray:
        output = np.zeros((batch, max_length, *trailing), dtype=dtype)
        for row, record in enumerate(records):
            values = np.asarray(record[name])
            output[row, : len(values)] = values
        return output

    board = padded("board_embedding", np.float32, (96,))
    raw_thinking = padded("thinking_time_ms", np.float32)
    raw_target = padded("target_remaining_time_ms_after", np.float32)
    raw_opponent = padded("opponent_remaining_time_ms_after", np.float32)
    thinking = robust_standardize(
        raw_thinking, stats["thinking_time"]["center"], stats["thinking_time"]["scale"]
    )
    target_remaining = robust_standardize(
        raw_target, stats["target_remaining_time"]["center"], stats["target_remaining_time"]["scale"]
    )
    opponent_remaining = robust_standardize(
        raw_opponent, stats["opponent_remaining_time"]["center"], stats["opponent_remaining_time"]["scale"]
    )
    # Padding is explicitly masked; keeping its numerical features at zero avoids artificial extremes.
    for values in (thinking, target_remaining, opponent_remaining):
        values[padding_mask.numpy()] = 0.0
    return {
        "game_id": [str(record["game_id"]) for record in records],
        "target_view": torch.tensor([int(record["target_view"]) for record in records]),
        "lengths": lengths,
        "padding_mask": padding_mask,
        "board_embedding": torch.from_numpy(board),
        "thinking_time": torch.from_numpy(thinking),
        "target_remaining_time": torch.from_numpy(target_remaining),
        "opponent_remaining_time": torch.from_numpy(opponent_remaining),
        "strict_ply": torch.from_numpy(padded("strict_ply", np.float32)),
        "node_index": torch.from_numpy(padded("node_index", np.int64)),
        "actor_is_target": torch.from_numpy(padded("actor_is_target", np.bool_)),
        "actor_is_black": torch.from_numpy(padded("actor_is_black", np.bool_)),
        "is_pass": torch.from_numpy(padded("is_pass", np.bool_)),
        "time_control_index": torch.tensor([int(record["time_control_index"]) for record in records]),
    }
