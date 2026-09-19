"""Read-only adapter from the frozen PCA cache to HMM observation sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitData:
    values: np.ndarray
    lengths: np.ndarray
    game_id: np.ndarray
    target_id: np.ndarray
    target_view: np.ndarray
    target_decision: np.ndarray
    original_node_index: np.ndarray
    strict_ply: np.ndarray

    @property
    def sequence_count(self) -> int:
        return int(len(self.lengths))

    @property
    def decision_count(self) -> int:
        return int(self.lengths.sum())


def load_manifest(cache_dir: Path) -> tuple[Path, dict[str, object]]:
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "stage3-projected-sequence-cache-v1" or manifest.get("status") != "complete":
        raise ValueError("complete stage3 projected cache required")
    return path, manifest


def selected_positions(count: int, strict_ply: np.ndarray, config: dict[str, object]) -> np.ndarray:
    decisions = np.arange(1, count + 1, dtype=np.int16)
    return np.flatnonzero(
        (decisions >= int(config["target_decision_min_inclusive"]))
        & (decisions <= int(config["target_decision_max_inclusive"]))
        & (strict_ply <= int(config["strict_ply_max_inclusive"]))
    )


def load_split(cache_dir: Path, manifest: dict[str, object], config: dict[str, object], split: str) -> SplitData:
    sequences: list[np.ndarray] = []
    node_sequences: list[np.ndarray] = []
    ply_sequences: list[np.ndarray] = []
    decision_sequences: list[np.ndarray] = []
    game_ids: list[str] = []
    target_ids: list[str] = []
    target_views: list[int] = []
    for item in manifest["shards"]:  # type: ignore[index]
        with np.load(cache_dir / item["file"], allow_pickle=False) as shard:  # type: ignore[index]
            for row, split_value in enumerate(shard["split"].tolist()):
                if str(split_value) != split:
                    continue
                offset = int(shard["sequence_offset"][row])
                count = int(shard["sequence_count"][row])
                plies = shard["strict_ply"][offset:offset + count]
                positions = selected_positions(count, plies, config)
                if not len(positions):
                    raise ValueError(f"empty HMM sequence: {shard['game_id'][row]}/{int(shard['target_view'][row])}")
                sequences.append(shard["z"][offset:offset + count][positions].astype(np.float32))
                node_sequences.append(shard["original_node_index"][offset:offset + count][positions].astype(np.int16))
                ply_sequences.append(plies[positions].astype(np.int16))
                decision_sequences.append((positions + 1).astype(np.int16))
                game_ids.append(str(shard["game_id"][row]))
                target_ids.append(str(shard["target_id"][row]))
                target_views.append(int(shard["target_view"][row]))
    if not sequences:
        raise ValueError(f"no sequences in split {split}")
    max_length = max(map(len, sequences))
    dimension = int(config["projection_dim"])
    values = np.zeros((len(sequences), max_length, dimension), dtype=np.float32)
    target_decision = np.full((len(sequences), max_length), -1, dtype=np.int16)
    original_node_index = np.full((len(sequences), max_length), -1, dtype=np.int16)
    strict_ply = np.full((len(sequences), max_length), -1, dtype=np.int16)
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        values[row, :length] = sequence
        target_decision[row, :length] = decision_sequences[row]
        original_node_index[row, :length] = node_sequences[row]
        strict_ply[row, :length] = ply_sequences[row]
    return SplitData(
        values=values,
        lengths=lengths,
        game_id=np.asarray(game_ids),
        target_id=np.asarray(target_ids),
        target_view=np.asarray(target_views, dtype=np.uint8),
        target_decision=target_decision,
        original_node_index=original_node_index,
        strict_ply=strict_ply,
    )


def flattened_valid_values(data: SplitData) -> np.ndarray:
    mask = np.arange(data.values.shape[1])[None, :] < data.lengths[:, None]
    return data.values[mask]
