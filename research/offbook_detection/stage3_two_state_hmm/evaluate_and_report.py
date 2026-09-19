"""Evaluate a frozen HMM family and emit complete semantic posteriors and diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_two_state_hmm.core import (  # noqa: E402
    DiagonalHMM,
    infer_batch,
    semantic_indices,
    thirds,
)
from research.offbook_detection.stage3_two_state_hmm.data import load_manifest, load_split  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def load_model(candidate: dict[str, object], device: torch.device) -> tuple[DiagonalHMM, dict[str, object]]:
    checkpoint_path = Path(str(candidate["checkpoint"]))
    if sha256_file(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiagonalHMM(
        int(checkpoint["dimension"]),
        str(checkpoint["family"]),
        float(checkpoint["degrees_of_freedom"]),
        float(checkpoint["scale_floor"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().to(device), checkpoint


def transition_record(data, row: int, time_index: int, from_state: str, to_state: str, probability: float) -> dict[str, object]:
    return {
        "arrival_target_decision": int(data.target_decision[row, time_index]),
        "arrival_original_node_index": int(data.original_node_index[row, time_index]),
        "arrival_strict_ply": int(data.strict_ply[row, time_index]),
        "from_state": from_state,
        "to_state": to_state,
        "transition_posterior": probability,
    }


def off_book_intervals(data, row: int, semantic_path: np.ndarray) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    start: int | None = None
    for index, state in enumerate(np.append(semantic_path, -1)):
        if state == 1 and start is None:
            start = index
        elif state != 1 and start is not None:
            stop = index - 1
            result.append({
                "start_target_decision": int(data.target_decision[row, start]),
                "start_original_node_index": int(data.original_node_index[row, start]),
                "start_strict_ply": int(data.strict_ply[row, start]),
                "end_target_decision": int(data.target_decision[row, stop]),
                "end_original_node_index": int(data.original_node_index[row, stop]),
                "end_strict_ply": int(data.strict_ply[row, stop]),
            })
            start = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=("student_t", "gaussian"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--test-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "stage3-two-state-hmm-selection-v1" or selection.get("status") != "frozen_before_test":
        raise ValueError("frozen pre-test HMM selection required")
    if args.split == "test":
        if args.test_authorization is None:
            raise ValueError("test evaluation requires an explicit authorization manifest")
        authorization_path = args.test_authorization.resolve(strict=True)
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if authorization.get("schema") != "stage3-two-state-hmm-test-authorization-v1":
            raise ValueError("invalid test authorization schema")
        if authorization.get("selection_sha256") != sha256_file(selection_path):
            raise ValueError("test authorization does not match frozen selection")
    elif args.test_authorization is not None:
        raise ValueError("test authorization is only valid for the test split")

    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != selection["config_sha256"]:
        raise ValueError("config differs from frozen selection")
    cache_dir = args.projected_cache.resolve(strict=True)
    cache_manifest_path, cache_manifest = load_manifest(cache_dir)
    if sha256_file(cache_manifest_path) != selection["projected_cache_manifest_sha256"]:
        raise ValueError("projected cache differs from frozen selection")
    candidate = selection["family_best"][args.family]
    mapping = selection["state_mappings"][args.family]
    raw_to_semantic = [int(value) for value in mapping["raw_to_semantic"]]
    in_raw, off_raw = semantic_indices(raw_to_semantic)
    device = torch.device(args.device)
    model, checkpoint = load_model(candidate, device)
    data = load_split(cache_dir, cache_manifest, config, args.split)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "posteriors.jsonl"
    summary_path = output_dir / "summary.json"
    if records_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite HMM evaluation")
    temporary = records_path.with_suffix(".jsonl.tmp")
    total_log_likelihood = 0.0
    soft_counts = np.zeros(2, dtype=np.float64)
    hard_counts = np.zeros(2, dtype=np.int64)
    expected_transitions = np.zeros((2, 2), dtype=np.float64)
    total_entropy = 0.0
    sequence_entropy_sum = 0.0
    switch_sum = single_state_sequences = multi_round_trip_sequences = 0
    early_sum = np.zeros(2, dtype=np.float64)
    late_sum = np.zeros(2, dtype=np.float64)
    early_nodes = late_nodes = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        batch_size = int(config["batch_size"])
        for batch_start in range(0, data.sequence_count, batch_size):
            batch_stop = min(batch_start + batch_size, data.sequence_count)
            result = infer_batch(
                model,
                torch.from_numpy(data.values[batch_start:batch_stop]).to(device),
                torch.from_numpy(data.lengths[batch_start:batch_stop]).to(device),
            )
            for local_row, global_row in enumerate(range(batch_start, batch_stop)):
                length = int(data.lengths[global_row])
                posterior_raw = result.posterior[local_row, :length]
                posterior_semantic = posterior_raw[:, [in_raw, off_raw]]
                path_raw = result.viterbi[local_row, :length]
                path_semantic = np.asarray([raw_to_semantic[int(state)] for state in path_raw], dtype=np.int8)
                xi_raw = result.transition_posterior[local_row, :length]
                xi_semantic = xi_raw[:, [in_raw, off_raw]][:, :, [in_raw, off_raw]]
                entropy = -(posterior_semantic * np.log(np.clip(posterior_semantic, 1e-300, None))).sum(axis=1)
                switches = int(np.count_nonzero(path_semantic[1:] != path_semantic[:-1]))
                viterbi_transitions: list[dict[str, object]] = []
                in_to_off: list[dict[str, object]] = []
                off_to_in: list[dict[str, object]] = []
                for time_index in range(1, length):
                    if path_semantic[time_index] == path_semantic[time_index - 1]:
                        continue
                    from_code = int(path_semantic[time_index - 1])
                    to_code = int(path_semantic[time_index])
                    from_name = "in_book" if from_code == 0 else "off_book"
                    to_name = "in_book" if to_code == 0 else "off_book"
                    record = transition_record(
                        data,
                        global_row,
                        time_index,
                        from_name,
                        to_name,
                        float(xi_semantic[time_index, from_code, to_code]),
                    )
                    viterbi_transitions.append(record)
                    (in_to_off if (from_code, to_code) == (0, 1) else off_to_in).append(record)
                if in_to_off:
                    primary_index = int(np.nanargmax(xi_semantic[1:, 0, 1])) + 1
                    primary = transition_record(
                        data,
                        global_row,
                        primary_index,
                        "in_book",
                        "off_book",
                        float(xi_semantic[primary_index, 0, 1]),
                    )
                    primary["is_viterbi_in_to_off_transition"] = any(
                        int(item["arrival_target_decision"]) == int(primary["arrival_target_decision"])
                        for item in in_to_off
                    )
                else:
                    primary = None
                steps = []
                for time_index in range(length):
                    steps.append({
                        "original_node_index": int(data.original_node_index[global_row, time_index]),
                        "strict_ply": int(data.strict_ply[global_row, time_index]),
                        "target_decision": int(data.target_decision[global_row, time_index]),
                        "p_in_book_state": float(posterior_semantic[time_index, 0]),
                        "p_off_book_state": float(posterior_semantic[time_index, 1]),
                        "viterbi_state": "in_book" if path_semantic[time_index] == 0 else "off_book",
                        "p_in_to_off": None if time_index == 0 else float(xi_semantic[time_index, 0, 1]),
                        "p_off_to_in": None if time_index == 0 else float(xi_semantic[time_index, 1, 0]),
                    })
                record = {
                    "game_id": str(data.game_id[global_row]),
                    "target_id": str(data.target_id[global_row]),
                    "target_view": int(data.target_view[global_row]),
                    "target_color": "black" if int(data.target_view[global_row]) == 0 else "white",
                    "analyzed_decisions": length,
                    "steps": steps,
                    "viterbi_path": ["in_book" if value == 0 else "off_book" for value in path_semantic],
                    "in_to_off_transitions": in_to_off,
                    "off_to_in_transitions": off_to_in,
                    "all_viterbi_transitions": viterbi_transitions,
                    "off_book_intervals": off_book_intervals(data, global_row, path_semantic),
                    "state_switch_count": switches,
                    "primary_offbook_entry": primary,
                    "mean_state_posterior_entropy_nats": float(entropy.mean()),
                    "model": {
                        "family": args.family,
                        "is_selected_family": args.family == selection["selected_family"],
                        "checkpoint_sha256": candidate["checkpoint_sha256"],
                        "config_sha256": selection["config_sha256"],
                        "projected_cache_manifest_sha256": selection["projected_cache_manifest_sha256"],
                        "pca_sha256": selection["pca_sha256"],
                        "seed": candidate["seed"],
                        "state_mapping": raw_to_semantic,
                    },
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_log_likelihood += float(result.log_likelihood[local_row])
                soft_counts += posterior_semantic.sum(axis=0)
                hard_counts += np.bincount(path_semantic, minlength=2)
                if length > 1:
                    expected_transitions += np.nansum(xi_semantic[1:], axis=0)
                total_entropy += float(entropy.sum())
                sequence_entropy_sum += float(entropy.mean())
                switch_sum += switches
                single_state_sequences += int(switches == 0)
                multi_round_trip_sequences += int(switches >= 3)
                split = thirds(length)
                if split is not None:
                    early, late = split
                    early_sum += posterior_semantic[early].sum(axis=0)
                    late_sum += posterior_semantic[late].sum(axis=0)
                    early_nodes += len(early)
                    late_nodes += len(late)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, records_path)
    semantic_transition = model.log_transition().exp().detach().cpu().double().numpy()[[in_raw, off_raw]][:, [in_raw, off_raw]]
    semantic_initial = model.log_initial().exp().detach().cpu().double().numpy()[[in_raw, off_raw]]
    summary = {
        "schema": "stage3-two-state-hmm-evaluation-v1",
        "status": "complete",
        "split": args.split,
        "family": args.family,
        "is_selected_family": args.family == selection["selected_family"],
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "test_used_for_selection": False,
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "config": str(config_path),
        "config_sha256": selection["config_sha256"],
        "projected_cache_manifest": str(cache_manifest_path),
        "projected_cache_manifest_sha256": selection["projected_cache_manifest_sha256"],
        "pca_sha256": selection["pca_sha256"],
        "records": str(records_path),
        "records_sha256": sha256_file(records_path),
        "sequences": data.sequence_count,
        "decisions": data.decision_count,
        "total_nll": -total_log_likelihood,
        "nll_per_decision": -total_log_likelihood / data.decision_count,
        "semantic_initial_probabilities": semantic_initial.tolist(),
        "semantic_transition_matrix": semantic_transition.tolist(),
        "posterior_soft_occupancy": (soft_counts / data.decision_count).tolist(),
        "viterbi_hard_occupancy": (hard_counts / data.decision_count).tolist(),
        "posterior_expected_transition_counts": expected_transitions.tolist(),
        "mean_viterbi_switches_per_sequence": switch_sum / data.sequence_count,
        "single_state_sequences": single_state_sequences,
        "single_state_sequence_ratio": single_state_sequences / data.sequence_count,
        "multi_round_trip_sequences": multi_round_trip_sequences,
        "multi_round_trip_sequence_ratio": multi_round_trip_sequences / data.sequence_count,
        "early_mean_state_posterior": (early_sum / early_nodes).tolist(),
        "late_mean_state_posterior": (late_sum / late_nodes).tolist(),
        "node_weighted_mean_state_posterior_entropy_nats": total_entropy / data.decision_count,
        "sequence_weighted_mean_state_posterior_entropy_nats": sequence_entropy_sum / data.sequence_count,
        "state_mapping": mapping,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
