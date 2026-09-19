"""Fit the frozen train-only PCA and build the projected stage-3 sequence cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_single_change_point.core import (  # noqa: E402
    candidate_decisions, raw_stage3_features,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


SOURCE_ARRAYS = (
    "hidden_states", "game_id", "split", "black_id", "white_id", "game_node_offset",
    "game_node_count", "node_index", "strict_ply", "actor_is_target", "is_pass",
    "thinking_time_ms", "black_remaining_time_ms_after", "white_remaining_time_ms_after",
)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in SOURCE_ARRAYS}


def shard_train_features(source: dict[str, np.ndarray], stats: dict[str, dict[str, float]]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for game_index, split in enumerate(source["split"].tolist()):
        if str(split) != "train":
            continue
        offset = int(source["game_node_offset"][game_index])
        stop = offset + int(source["game_node_count"][game_index])
        for target_view in (0, 1):
            values, _ = raw_stage3_features(source, target_view, offset, stop, stats)  # type: ignore[arg-type]
            if len(values):
                blocks.append(values)
    if not blocks:
        return np.empty((0, 131), dtype=np.float32)
    return np.concatenate(blocks)


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def projected_shard(
    source: dict[str, np.ndarray], pca: IncrementalPCA, stats: dict[str, dict[str, float]]
) -> dict[str, np.ndarray]:
    z_blocks: list[np.ndarray] = []
    node_blocks: list[np.ndarray] = []
    ply_blocks: list[np.ndarray] = []
    offsets: list[int] = []
    counts: list[int] = []
    game_ids: list[str] = []
    splits: list[str] = []
    target_views: list[int] = []
    target_ids: list[str] = []
    finite_candidate_counts: list[int] = []
    cursor = 0
    for game_index, game_id in enumerate(source["game_id"].tolist()):
        offset = int(source["game_node_offset"][game_index])
        stop = offset + int(source["game_node_count"][game_index])
        for target_view in (0, 1):
            values, selected = raw_stage3_features(source, target_view, offset, stop, stats)  # type: ignore[arg-type]
            z = pca.transform(values).astype(np.float32) if len(values) else np.empty((0, pca.n_components_), np.float32)
            nodes = source["node_index"][offset:stop][selected].astype(np.int16)
            plies = source["strict_ply"][offset:stop][selected].astype(np.int16)
            offsets.append(cursor)
            counts.append(len(z))
            cursor += len(z)
            z_blocks.append(z)
            node_blocks.append(nodes)
            ply_blocks.append(plies)
            game_ids.append(str(game_id))
            splits.append(str(source["split"][game_index]))
            target_views.append(target_view)
            target_ids.append(str(source["black_id"][game_index] if target_view == 0 else source["white_id"][game_index]))
            finite_candidate_counts.append(len(candidate_decisions(plies)))
    return {
        "z": np.concatenate(z_blocks),
        "original_node_index": np.concatenate(node_blocks),
        "strict_ply": np.concatenate(ply_blocks),
        "sequence_offset": np.asarray(offsets, dtype=np.int64),
        "sequence_count": np.asarray(counts, dtype=np.int16),
        "game_id": np.asarray(game_ids),
        "split": np.asarray(splits),
        "target_view": np.asarray(target_views, dtype=np.uint8),
        "target_id": np.asarray(target_ids),
        "finite_candidate_count": np.asarray(finite_candidate_counts, dtype=np.int16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.sequence_cache.resolve(strict=True)
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != "stage2-temporal-sequence-cache-v1" or source_manifest.get("status") != "complete":
        raise ValueError("a complete stage-2 sequence cache is required")
    if config.get("schema") != "stage3-single-change-point-config-v1" or config.get("status") != "frozen":
        raise ValueError("a frozen stage-3 configuration is required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("completed projected cache already exists")
    stats_payload = source_manifest["time_standardization"]
    stats = {name: stats_payload[name] for name in ("thinking_time", "target_remaining_time", "opponent_remaining_time")}
    pca = IncrementalPCA(n_components=int(config["projection_dim"]), batch_size=None)
    fit_nodes = 0
    for item in source_manifest["shards"]:
        source = load_archive(source_dir / item["file"])
        values = shard_train_features(source, stats)
        if len(values):
            pca.partial_fit(values)
            fit_nodes += len(values)
        print(json.dumps({"phase": "pca_fit", "shard": item["index"], "train_nodes": fit_nodes}), flush=True)
    pca_path = output_dir / "pca.joblib"
    joblib.dump(pca, pca_path)
    shard_items: list[dict[str, object]] = []
    totals = {"sequences": 0, "decisions": 0, "sequences_with_candidates": 0}
    for item in source_manifest["shards"]:
        index = int(item["index"])
        source = load_archive(source_dir / item["file"])
        payload = projected_shard(source, pca, stats)
        path = output_dir / f"shard-{index:05d}.npz"
        save_npz_atomic(path, payload)
        sequences = len(payload["sequence_count"])
        decisions = len(payload["z"])
        eligible = int((payload["finite_candidate_count"] > 0).sum())
        totals["sequences"] += sequences
        totals["decisions"] += decisions
        totals["sequences_with_candidates"] += eligible
        shard_item = {
            "index": index, "file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "sequences": sequences, "decisions": decisions, "sequences_with_candidates": eligible,
            "source_shard_sha256": item["sha256"],
        }
        shard_items.append(shard_item)
        print(json.dumps({"phase": "projection", **shard_item}), flush=True)
    manifest = {
        "schema": "stage3-projected-sequence-cache-v1",
        "status": "complete",
        "source_sequence_manifest": str(source_manifest_path.resolve()),
        "source_sequence_manifest_sha256": sha256_file(source_manifest_path),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "pca": str(pca_path), "pca_sha256": sha256_file(pca_path),
        "pca_fit_split": "train", "pca_fit_decisions": fit_nodes,
        "input_dimension": 131, "projection_dimension": int(config["projection_dim"]),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        **totals, "shards": shard_items,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
