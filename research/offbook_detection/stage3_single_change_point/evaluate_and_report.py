"""Evaluate a frozen stage-3 model and emit complete per-sequence posteriors."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_single_change_point.core import (  # noqa: E402
    DiagonalSegmentModel, candidate_decisions, posterior_summary, state_log_scores,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "stage3-model-selection-v1" or selection.get("status") != "frozen_before_test":
        raise ValueError("frozen pre-test model selection is required")
    cache_dir = args.projected_cache.resolve(strict=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != selection["projected_cache_manifest_sha256"]:
        raise ValueError("projected cache differs from frozen selection")
    for item in manifest["shards"]:
        if sha256_file(cache_dir / item["file"]) != item["sha256"]:
            raise ValueError(f"projected shard hash mismatch: {item['file']}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "posteriors.jsonl"
    summary_path = output_dir / "summary.json"
    if records_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to repeat or overwrite {args.split} evaluation")
    checkpoint_path = Path(selection["selected"]["checkpoint"])
    if sha256_file(checkpoint_path) != selection["selected"]["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiagonalSegmentModel(
        int(checkpoint["dimension"]), str(checkpoint["family"]), float(checkpoint["degrees_of_freedom"]),
        float(checkpoint["scale_floor"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.eval().to(device)
    model_version = selection["selected"]["checkpoint_sha256"]
    temporary = records_path.with_suffix(".jsonl.tmp")
    sequences = decisions = eligible_sequences = no_offbook = 0
    total_nll = 0.0
    map_decisions: Counter[int] = Counter()
    p_no_change_sum = entropy_sum = 0.0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for item in manifest["shards"]:
            with np.load(cache_dir / item["file"], allow_pickle=False) as shard:
                shard_splits = shard["split"]
                offsets = shard["sequence_offset"]
                counts = shard["sequence_count"]
                z = shard["z"]
                plies = shard["strict_ply"]
                nodes = shard["original_node_index"]
                for row, split_value in enumerate(shard_splits.tolist()):
                    if str(split_value) != args.split:
                        continue
                    offset = int(offsets[row])
                    count = int(counts[row])
                    sequence_plies = plies[offset:offset + count]
                    sequence_nodes = nodes[offset:offset + count]
                    candidates = candidate_decisions(sequence_plies)
                    if len(candidates):
                        values = torch.from_numpy(z[offset:offset + count][None].astype(np.float32)).to(device)
                        lengths = torch.tensor([count], device=device)
                        mask = torch.zeros((1, count), dtype=torch.bool, device=device)
                        mask[0, torch.from_numpy(candidates.astype(np.int64) - 1).to(device)] = True
                        with torch.inference_mode():
                            padded_scores, _ = state_log_scores(model, values, lengths, mask)
                        selected_scores = padded_scores[0, torch.from_numpy(np.concatenate(([0], candidates)).astype(np.int64)).to(device)]
                        score_values = selected_scores.float().cpu().numpy()
                        total_nll += float(-torch.logsumexp(selected_scores, dim=0))
                        decisions += count
                        eligible_sequences += 1
                        posterior = posterior_summary(score_values, candidates, sequence_plies)
                    else:
                        posterior = {
                            "p_no_change": 1.0, "map_state": "no_change", "map_probability": 1.0,
                            "map_decision": None, "posterior_entropy": 0.0,
                            "finite_strict_ply_mean": None, "finite_strict_ply_std": None,
                            "candidate_probabilities": [],
                        }
                    map_decision = posterior["map_decision"]
                    if map_decision is None:
                        no_offbook += 1
                        map_node = map_ply = None
                    else:
                        decision_index = int(map_decision) - 1
                        map_node = int(sequence_nodes[decision_index])
                        map_ply = int(sequence_plies[decision_index])
                        map_decisions[int(map_decision)] += 1
                    candidate_records = [
                        {
                            "target_decision": int(k), "original_node_index": int(sequence_nodes[int(k) - 1]),
                            "strict_ply": int(sequence_plies[int(k) - 1]), "probability": probability,
                        }
                        for k, probability in zip(candidates, posterior["candidate_probabilities"])
                    ]
                    record = {
                        "game_id": str(shard["game_id"][row]), "target_id": str(shard["target_id"][row]),
                        "target_view": int(shard["target_view"][row]),
                        "target_color": "black" if int(shard["target_view"][row]) == 0 else "white",
                        "effective_decisions": count, "result": "no_offbook" if map_decision is None else "change_point",
                        "map_target_decision": map_decision, "map_original_node_index": map_node,
                        "map_strict_ply": map_ply, **{k: v for k, v in posterior.items() if k != "candidate_probabilities"},
                        "candidates": candidate_records, "model_version_sha256": model_version,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    sequences += 1
                    p_no_change_sum += float(posterior["p_no_change"])
                    entropy_sum += float(posterior["posterior_entropy"])
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, records_path)
    summary = {
        "schema": "stage3-single-change-point-evaluation-v1", "status": "complete", "split": args.split,
        "test_used_for_selection": False, "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path), "selected_family": checkpoint["family"],
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": model_version,
        "projected_cache_manifest": str(manifest_path), "projected_cache_manifest_sha256": sha256_file(manifest_path),
        "records": str(records_path), "records_sha256": sha256_file(records_path),
        "sequences": sequences, "eligible_sequences": eligible_sequences,
        "eligible_decisions": decisions, "nll_per_decision": total_nll / decisions,
        "no_offbook_sequences": no_offbook, "change_point_sequences": sequences - no_offbook,
        "mean_p_no_change": p_no_change_sum / sequences, "mean_posterior_entropy": entropy_sum / sequences,
        "map_target_decision_counts": {str(key): value for key, value in sorted(map_decisions.items())},
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
