"""Verify every shard and mapping in a completed stage-2 sequence cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.build_sequence_cache import encode_shard  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.cache import BoardCache  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.delivery import (  # noqa: E402
    load_frozen_temporal_model,
    load_time_stats,
)


SHARED_ARRAYS = (
    "game_id",
    "split",
    "black_id",
    "white_id",
    "transform_id",
    "time_control_id",
    "game_node_offset",
    "game_node_count",
    "node_index",
    "strict_ply",
    "actor_is_black",
    "is_pass",
    "thinking_time_ms",
    "black_remaining_time_ms_after",
    "white_remaining_time_ms_after",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--time-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sequence_dir = args.sequence_cache.resolve(strict=True)
    board_dir = args.board_cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification report: {output}")
    sequence_manifest_path = sequence_dir / "manifest.json"
    board_manifest_path = board_dir / "manifest.json"
    sequence_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
    board_manifest = json.loads(board_manifest_path.read_text(encoding="utf-8"))
    if sequence_manifest.get("schema") != "stage2-temporal-sequence-cache-v1":
        raise ValueError("unsupported sequence cache manifest")
    if sequence_manifest.get("status") != "complete":
        raise ValueError("sequence cache is not complete")
    if sequence_manifest["board_cache_manifest_sha256"] != sha256_file(board_manifest_path):
        raise ValueError("sequence cache points to a different board cache manifest")
    if len(sequence_manifest["shards"]) != len(board_manifest["shards"]):
        raise ValueError("sequence and board shard counts differ")
    totals = Counter()
    split_games = Counter()
    hidden_min = float("inf")
    hidden_max = float("-inf")
    for sequence_item, board_item in zip(sequence_manifest["shards"], board_manifest["shards"]):
        index = int(sequence_item["index"])
        if index != int(board_item["index"]):
            raise ValueError(f"shard index mismatch at {index}")
        sequence_path = sequence_dir / sequence_item["file"]
        board_path = board_dir / board_item["file"]
        if sha256_file(sequence_path) != sequence_item["sha256"]:
            raise ValueError(f"sequence shard hash mismatch: {sequence_path}")
        if sha256_file(board_path) != sequence_item["input_board_shard_sha256"]:
            raise ValueError(f"board shard hash mismatch: {board_path}")
        with np.load(sequence_path, allow_pickle=False) as sequence, np.load(board_path, allow_pickle=False) as board:
            for name in SHARED_ARRAYS:
                if not np.array_equal(sequence[name], board[name]):
                    raise ValueError(f"mapping array {name} differs in shard {index}")
            nodes = int(board_item["nodes"])
            games = int(board_item["games"])
            hidden = sequence["hidden_states"]
            if hidden.shape != (2, nodes, 128) or hidden.dtype != np.float32:
                raise ValueError(f"invalid hidden state contract in shard {index}: {hidden.shape}, {hidden.dtype}")
            if not np.isfinite(hidden).all():
                raise ValueError(f"non-finite hidden state in shard {index}")
            actor_is_black = board["actor_is_black"].astype(np.bool_, copy=False)
            expected_actor_is_target = np.stack((actor_is_black, ~actor_is_black))
            if not np.array_equal(sequence["actor_is_target"], expected_actor_is_target):
                raise ValueError(f"actor_is_target mapping differs in shard {index}")
            if not np.array_equal(sequence["target_view"], np.asarray([0, 1], dtype=np.uint8)):
                raise ValueError(f"target view ids differ in shard {index}")
            if sequence["target_view_name"].tolist() != ["black", "white"]:
                raise ValueError(f"target view names differ in shard {index}")
            if sequence_item["hidden_states_shape"] != [2, nodes, 128]:
                raise ValueError(f"manifest shape differs in shard {index}")
            hidden_min = min(hidden_min, float(hidden.min()))
            hidden_max = max(hidden_max, float(hidden.max()))
            split_games.update(str(value) for value in sequence["split"].tolist())
        totals["shards"] += 1
        totals["games"] += games
        totals["nodes"] += nodes
        totals["target_view_sequences"] += games * 2
        totals["target_view_nodes"] += nodes * 2
        totals["bytes"] += int(sequence_item["bytes"])
    for name in ("games", "nodes", "target_view_sequences", "target_view_nodes"):
        if totals[name] != int(sequence_manifest[name]):
            raise ValueError(f"manifest total mismatch for {name}")
    expected_splits = {"train": 48922, "validation": 6118, "test": 6105}
    if dict(split_games) != expected_splits:
        raise ValueError(f"split counts differ: {dict(split_games)}")
    device = torch.device(args.device)
    cache = BoardCache(board_dir, verify_hashes=True)
    stats, _ = load_time_stats(args.time_stats)
    model, _ = load_frozen_temporal_model(
        args.checkpoint, board_manifest_path, args.time_stats, device
    )
    recomputed_shards: dict[str, float] = {}
    for shard_index in sorted({0, len(sequence_manifest["shards"]) // 2, len(sequence_manifest["shards"]) - 1}):
        recomputed = encode_shard(
            cache, shard_index, model, stats, args.batch_size, device, device.type == "cuda"
        )["hidden_states"]
        with np.load(sequence_dir / sequence_manifest["shards"][shard_index]["file"], allow_pickle=False) as saved:
            difference = float(np.max(np.abs(recomputed - saved["hidden_states"])))
        if difference > 1e-6:
            raise ValueError(f"recomputed hidden states differ in shard {shard_index}: {difference}")
        recomputed_shards[str(shard_index)] = difference
    report = {
        "schema": "stage2-temporal-sequence-cache-verification-v1",
        "status": "passed",
        "sequence_cache_manifest": str(sequence_manifest_path.resolve()),
        "sequence_cache_manifest_sha256": sha256_file(sequence_manifest_path),
        "board_cache_manifest": str(board_manifest_path.resolve()),
        "board_cache_manifest_sha256": sha256_file(board_manifest_path),
        "verified": dict(totals),
        "split_games": dict(split_games),
        "split_target_view_sequences": {name: count * 2 for name, count in split_games.items()},
        "hidden_states": {
            "dtype": "float32",
            "hidden_size": 128,
            "all_finite": True,
            "minimum": hidden_min,
            "maximum": hidden_max,
        },
        "recomputed_hidden_state_shards": recomputed_shards,
        "checks": [
            "all sequence shard SHA-256 hashes",
            "all linked board shard SHA-256 hashes",
            "all shared game and node mapping arrays",
            "target view and actor_is_target mappings",
            "hidden state shape, dtype, and finite values",
            "global and split counts",
            "full hidden-state recomputation for first, middle, and last shards",
        ],
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
