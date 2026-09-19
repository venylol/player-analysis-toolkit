"""Export the frozen temporal Transformer's last-layer hidden states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.cache import (  # noqa: E402
    BoardCache,
    SequenceRef,
    collate_sequences,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.delivery import (  # noqa: E402
    load_frozen_temporal_model,
    load_time_stats,
)
from research.offbook_detection.temporal_transformer_stage2.train import to_device  # noqa: E402


SCHEMA = "stage2-temporal-sequence-cache-v1"


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


@torch.inference_mode()
def encode_shard(
    cache: BoardCache,
    shard_index: int,
    model: torch.nn.Module,
    stats: dict[str, dict[str, float]],
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    source = cache.load_shard(shard_index)
    counts = source["game_node_count"].astype(np.int64)
    offsets = source["game_node_offset"].astype(np.int64)
    total_nodes = int(counts.sum())
    hidden_size = int(model.config.hidden_size)
    hidden_states = np.empty((2, total_nodes, hidden_size), dtype=np.float32)
    actor_is_black = source["actor_is_black"].astype(np.bool_, copy=False)
    actor_is_target = np.stack((actor_is_black, ~actor_is_black))
    for target_view in (0, 1):
        refs = [
            SequenceRef(
                shard_index=shard_index,
                game_index=game_index,
                target_view=target_view,
                split=str(source["split"][game_index]),
                game_id=str(source["game_id"][game_index]),
                node_count=int(count),
            )
            for game_index, count in enumerate(counts)
        ]
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start:start + batch_size]
            records = [cache.sequence(ref) for ref in batch_refs]
            host_batch = collate_sequences(records, stats)
            batch = to_device(host_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                encoded = model.encode(batch)
            encoded_cpu = encoded.float().cpu().numpy()
            for row, ref in enumerate(batch_refs):
                offset = int(offsets[ref.game_index])
                hidden_states[target_view, offset:offset + ref.node_count] = encoded_cpu[row, :ref.node_count]
    return {
        "hidden_states": hidden_states,
        "game_id": source["game_id"],
        "split": source["split"],
        "black_id": source["black_id"],
        "white_id": source["white_id"],
        "transform_id": source["transform_id"],
        "time_control_id": source["time_control_id"],
        "target_view": np.asarray([0, 1], dtype=np.uint8),
        "target_view_name": np.asarray(["black", "white"]),
        "game_node_offset": source["game_node_offset"],
        "game_node_count": source["game_node_count"],
        "node_index": source["node_index"],
        "original_node_index": source["node_index"],
        "strict_ply": source["strict_ply"],
        "actor_is_target": actor_is_target,
        "actor_is_black": source["actor_is_black"],
        "is_pass": source["is_pass"],
        "thinking_time_ms": source["thinking_time_ms"],
        "black_remaining_time_ms_after": source["black_remaining_time_ms_after"],
        "white_remaining_time_ms_after": source["white_remaining_time_ms_after"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--time-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed sequence cache already exists: {manifest_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cache = BoardCache(args.board_cache, verify_hashes=True)
    stats, stats_payload = load_time_stats(args.time_stats)
    model, checkpoint = load_frozen_temporal_model(
        args.checkpoint, cache.directory / "manifest.json", args.time_stats, device
    )
    amp = device.type == "cuda" and not args.no_amp
    checkpoint_path = args.checkpoint.resolve(strict=True)
    stats_path = args.time_stats.resolve(strict=True)
    board_manifest_path = (cache.directory / "manifest.json").resolve()
    contract = {
        "schema": SCHEMA,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "board_cache_manifest_sha256": sha256_file(board_manifest_path),
        "time_stats_sha256": sha256_file(stats_path),
        "model_config": checkpoint["model_config"],
        "time_standardization": stats_payload,
        "target_views": ["black", "white"],
        "orientation": cache.manifest["orientation"],
        "dtype": "float32",
        "hidden_size": int(model.config.hidden_size),
        "encoder_output": "direct final-layer hidden state h_i",
        "recovery_heads_used": False,
        "inference_batch_size": args.batch_size,
        "inference_amp_fp16": amp,
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract:
            raise ValueError("existing partial sequence cache contract does not match this run")
    else:
        progress = {"schema": "stage2-temporal-sequence-cache-progress-v1", "contract": contract, "shards": []}
        write_json(progress_path, progress)
    completed = {int(item["index"]): item for item in progress["shards"]}
    for shard_index, source_item in enumerate(cache.shards):
        output_path = output_dir / f"shard-{shard_index:05d}.npz"
        prior = completed.get(shard_index)
        if prior is not None:
            if not output_path.exists() or sha256_file(output_path) != prior["sha256"]:
                raise ValueError(f"completed sequence shard {shard_index} is missing or has changed")
            if prior["input_board_shard_sha256"] != source_item["sha256"]:
                raise ValueError(f"source board shard {shard_index} changed")
            continue
        if output_path.exists():
            raise FileExistsError(f"untracked sequence shard already exists: {output_path}")
        payload = encode_shard(cache, shard_index, model, stats, args.batch_size, device, amp)
        save_npz_atomic(output_path, payload)
        item = {
            "index": shard_index,
            "file": output_path.name,
            "games": int(source_item["games"]),
            "nodes": int(source_item["nodes"]),
            "target_view_sequences": int(source_item["games"]) * 2,
            "target_view_nodes": int(source_item["target_view_nodes"]),
            "hidden_states_shape": [2, int(source_item["nodes"]), int(model.config.hidden_size)],
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "input_board_shard_file": source_item["file"],
            "input_board_shard_sha256": source_item["sha256"],
            "first_game_id": source_item["first_game_id"],
            "last_game_id": source_item["last_game_id"],
        }
        progress["shards"].append(item)
        write_json(progress_path, progress)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    manifest = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "board_cache_manifest": str(board_manifest_path),
        "source_data_hashes": cache.manifest["source_hashes"],
        "time_stats": str(stats_path),
        "games": int(cache.manifest["games"]),
        "nodes": int(cache.manifest["nodes"]),
        "target_view_sequences": int(cache.manifest["target_view_sequences"]),
        "target_view_nodes": int(cache.manifest["target_view_nodes"]),
        "shards": progress["shards"],
    }
    write_json(manifest_path, manifest)
    write_json(
        progress_path,
        {
            **progress,
            "status": "complete",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps({"manifest": str(manifest_path), "shards": len(progress["shards"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
