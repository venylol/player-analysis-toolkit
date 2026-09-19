"""Select validation-best HMMs and freeze train-only semantic state mappings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_two_state_hmm.core import DiagonalHMM, infer_batch, thirds  # noqa: E402
from research.offbook_detection.stage3_two_state_hmm.data import load_manifest, load_split  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def load_candidate(directory: Path) -> dict[str, object]:
    complete_path = directory / "training_complete.json"
    checkpoint_path = directory / "best.pt"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != "stage3-two-state-hmm-training-complete-v1" or complete.get("status") != "complete":
        raise ValueError(f"incomplete HMM run: {directory}")
    if complete["best_checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    return {
        "family": str(complete["family"]),
        "initialization_index": int(complete["initialization_index"]),
        "seed": int(complete["seed"]),
        "best_epoch": int(complete["best_epoch"]),
        "epochs_run": int(complete["epochs_run"]),
        "validation_nll_per_decision": float(complete["best_validation_nll_per_decision"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": complete["best_checkpoint_sha256"],
        "training_complete": str(complete_path.resolve()),
        "training_complete_sha256": sha256_file(complete_path),
    }


def load_model(candidate: dict[str, object], device: torch.device) -> DiagonalHMM:
    checkpoint = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=False)
    model = DiagonalHMM(
        int(checkpoint["dimension"]),
        str(checkpoint["family"]),
        float(checkpoint["degrees_of_freedom"]),
        float(checkpoint["scale_floor"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().to(device)


def name_states(
    model: DiagonalHMM,
    values: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    early_sum = np.zeros(2, dtype=np.float64)
    late_sum = np.zeros(2, dtype=np.float64)
    early_nodes = late_nodes = excluded_sequences = 0
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        result = infer_batch(
            model,
            torch.from_numpy(values[start:stop]).to(device),
            torch.from_numpy(lengths[start:stop]).to(device),
        )
        for row, length_value in enumerate(lengths[start:stop]):
            length = int(length_value)
            split = thirds(length)
            if split is None:
                excluded_sequences += 1
                continue
            early, late = split
            early_sum += result.posterior[row, early].sum(axis=0)
            late_sum += result.posterior[row, late].sum(axis=0)
            early_nodes += len(early)
            late_nodes += len(late)
    early_mean = early_sum / early_nodes
    late_mean = late_sum / late_nodes
    score_state0_in = float(early_mean[0] + late_mean[1])
    score_state1_in = float(early_mean[1] + late_mean[0])
    raw_to_semantic = [0, 1] if score_state0_in >= score_state1_in else [1, 0]
    return {
        "semantic_codes": {"in_book": 0, "off_book": 1},
        "raw_to_semantic": raw_to_semantic,
        "in_book_raw_state": raw_to_semantic.index(0),
        "off_book_raw_state": raw_to_semantic.index(1),
        "mapping_scores": {
            "raw0_in_raw1_off": score_state0_in,
            "raw1_in_raw0_off": score_state1_in,
        },
        "raw_state_early_mean_posterior": early_mean.tolist(),
        "raw_state_late_mean_posterior": late_mean.tolist(),
        "early_nodes": early_nodes,
        "late_nodes": late_nodes,
        "excluded_short_sequences": excluded_sequences,
        "tie_rule": "raw state 0 is in_book when scores are exactly equal",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-t-dir", type=Path, action="append", required=True)
    parser.add_argument("--gaussian-dir", type=Path, action="append", required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initializations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {output}")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_runs = len(config["initialization_seeds"])
    if len(args.student_t_dir) != expected_runs or len(args.gaussian_dir) != expected_runs:
        raise ValueError(f"exactly {expected_runs} runs per family are required")
    all_candidates = {
        "student_t": [load_candidate(path.resolve(strict=True)) for path in args.student_t_dir],
        "gaussian": [load_candidate(path.resolve(strict=True)) for path in args.gaussian_dir],
    }
    for family, candidates in all_candidates.items():
        if {candidate["family"] for candidate in candidates} != {family}:
            raise ValueError(f"wrong family in {family} run list")
        if sorted(int(candidate["initialization_index"]) for candidate in candidates) != list(range(expected_runs)):
            raise ValueError(f"initialization indices incomplete for {family}")
        if sorted(int(candidate["seed"]) for candidate in candidates) != sorted(int(x) for x in config["initialization_seeds"]):
            raise ValueError(f"initialization seeds differ for {family}")
        candidates.sort(key=lambda item: (float(item["validation_nll_per_decision"]), int(item["initialization_index"])))
    family_best = {family: candidates[0] for family, candidates in all_candidates.items()}
    selected_family = min(
        family_best,
        key=lambda family: (
            float(family_best[family]["validation_nll_per_decision"]),
            0 if family == "student_t" else 1,
        ),
    )
    cache_dir = args.projected_cache.resolve(strict=True)
    cache_manifest_path, cache_manifest = load_manifest(cache_dir)
    train = load_split(cache_dir, cache_manifest, config, "train")
    device = torch.device(args.device)
    mappings: dict[str, object] = {}
    for family in ("student_t", "gaussian"):
        model = load_model(family_best[family], device)
        mappings[family] = name_states(model, train.values, train.lengths, int(config["batch_size"]), device)
    initialization_manifest_path = args.initializations.resolve(strict=True) / "manifest.json"
    payload = {
        "schema": "stage3-two-state-hmm-selection-v1",
        "status": "frozen_before_test",
        "selection_split": "validation",
        "selection_metric": "marginal NLL per analyzed decision",
        "test_used": False,
        "selected_family": selected_family,
        "selected": family_best[selected_family],
        "family_best": family_best,
        "all_runs": all_candidates,
        "state_mapping_split": "train",
        "state_mappings": mappings,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "projected_cache_manifest": str(cache_manifest_path),
        "projected_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "pca_sha256": cache_manifest["pca_sha256"],
        "initialization_manifest": str(initialization_manifest_path),
        "initialization_manifest_sha256": sha256_file(initialization_manifest_path),
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
