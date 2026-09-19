"""Fit log1p robust time statistics using only train-split nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.cache import BoardCache  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.data import write_json  # noqa: E402


def summary(values: np.ndarray) -> dict[str, float]:
    logged = np.log1p(values.astype(np.float64))
    q25, median, q75 = np.quantile(logged, (0.25, 0.5, 0.75))
    scale = float(q75 - q25)
    if scale <= 0:
        raise ValueError("training time feature has zero interquartile range")
    return {"center": float(median), "scale": scale, "q25": float(q25), "q75": float(q75)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cache = BoardCache(args.board_cache, verify_hashes=True)
    thinking: list[np.ndarray] = []
    black_remaining: list[np.ndarray] = []
    white_remaining: list[np.ndarray] = []
    train_games = 0
    train_nodes = 0
    for shard_index in range(len(cache.shards)):
        shard = cache.load_shard(shard_index)
        for game_index, split in enumerate(shard["split"].tolist()):
            if split != "train":
                continue
            offset = int(shard["game_node_offset"][game_index])
            count = int(shard["game_node_count"][game_index])
            stop = offset + count
            thinking.append(shard["thinking_time_ms"][offset:stop].astype(np.float32))
            black_remaining.append(shard["black_remaining_time_ms_after"][offset:stop].astype(np.float32))
            white_remaining.append(shard["white_remaining_time_ms_after"][offset:stop].astype(np.float32))
            train_games += 1
            train_nodes += count
    both_remaining = np.concatenate((*black_remaining, *white_remaining))
    payload = {
        "schema": "stage2-train-only-time-stats-v1",
        "source_board_cache_manifest": str((cache.directory / "manifest.json").resolve()),
        "fit_split": "train",
        "method": "log1p_then_median_center_and_IQR_scale",
        "train_games": train_games,
        "train_nodes": train_nodes,
        "thinking_time": summary(np.concatenate(thinking)),
        # Across the two target views, target and opponent remaining-time samples have the same multiset.
        "target_remaining_time": summary(both_remaining),
        "opponent_remaining_time": summary(both_remaining),
    }
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()

