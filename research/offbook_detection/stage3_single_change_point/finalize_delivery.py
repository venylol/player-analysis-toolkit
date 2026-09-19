"""Verify stage-3 posterior files and write the final delivery manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def verify_report(directory: Path) -> dict[str, object]:
    summary_path = directory / "summary.json"
    records_path = directory / "posteriors.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary["records_sha256"] != sha256_file(records_path):
        raise ValueError(f"report manifest mismatch: {directory}")
    lines = changes = no_offbook = 0
    max_probability_error = 0.0
    seen: set[tuple[str, int]] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            key = (record["game_id"], int(record["target_view"]))
            if key in seen:
                raise ValueError(f"duplicate game/target view at line {line_number}")
            seen.add(key)
            candidates = record["candidates"]
            probability_sum = float(record["p_no_change"]) + sum(float(item["probability"]) for item in candidates)
            max_probability_error = max(max_probability_error, abs(probability_sum - 1.0))
            if not math.isfinite(float(record["posterior_entropy"])):
                raise ValueError(f"non-finite posterior entropy at line {line_number}")
            for item in candidates:
                decision = int(item["target_decision"])
                if not 3 <= decision <= 19 or int(item["strict_ply"]) > 38:
                    raise ValueError(f"illegal candidate at line {line_number}")
                if int(record["effective_decisions"]) - decision + 1 < 3:
                    raise ValueError(f"post segment too short at line {line_number}")
            if record["result"] == "no_offbook":
                no_offbook += 1
                if record["map_target_decision"] is not None:
                    raise ValueError(f"no_offbook has a finite MAP at line {line_number}")
            else:
                changes += 1
                matches = [item for item in candidates if item["target_decision"] == record["map_target_decision"]]
                if len(matches) != 1 or matches[0]["original_node_index"] != record["map_original_node_index"]:
                    raise ValueError(f"MAP mapping mismatch at line {line_number}")
            lines += 1
    if max_probability_error > 1e-6:
        raise ValueError(f"posterior normalization error: {max_probability_error}")
    if lines != int(summary["sequences"]) or changes != int(summary["change_point_sequences"]) or no_offbook != int(summary["no_offbook_sequences"]):
        raise ValueError(f"record totals differ from summary: {directory}")
    return {
        "split": summary["split"], "summary": str(summary_path.resolve()), "summary_sha256": sha256_file(summary_path),
        "records": str(records_path.resolve()), "records_sha256": sha256_file(records_path),
        "sequences": lines, "change_point_sequences": changes, "no_offbook_sequences": no_offbook,
        "maximum_posterior_sum_error": max_probability_error, "all_entropies_finite": True,
        "candidate_and_mapping_checks": "passed", "nll_per_decision": summary["nll_per_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite delivery: {output}")
    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cache_manifest = args.projected_cache.resolve(strict=True) / "manifest.json"
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    config_path = args.config.resolve(strict=True)
    spec_path = args.spec.resolve(strict=True)
    validation = verify_report(args.validation_report.resolve(strict=True))
    test = verify_report(args.test_report.resolve(strict=True))
    payload = {
        "schema": "stage3-final-delivery-v1", "status": "complete",
        "protocol": str(spec_path), "protocol_sha256": sha256_file(spec_path),
        "config": str(config_path), "config_sha256": sha256_file(config_path), "seed": 42,
        "projected_cache": {
            "manifest": str(cache_manifest), "manifest_sha256": sha256_file(cache_manifest),
            "pca_sha256": cache["pca_sha256"], "projection_dimension": cache["projection_dimension"],
            "pca_fit_train_decisions": cache["pca_fit_decisions"], "sequences": cache["sequences"],
            "effective_decisions": cache["decisions"], "shards": len(cache["shards"]),
        },
        "model_selection": {
            "manifest": str(selection_path), "manifest_sha256": sha256_file(selection_path),
            "selection_split": "validation", "test_used": False,
            "selected_family": selection["selected"]["family"],
            "checkpoint": selection["selected"]["checkpoint"],
            "checkpoint_sha256": selection["selected"]["checkpoint_sha256"],
            "student_t_validation_nll_per_decision": selection["candidates"][0]["validation_nll_per_decision"],
            "gaussian_validation_nll_per_decision": selection["candidates"][1]["validation_nll_per_decision"],
        },
        "validation": validation, "fixed_test": test,
        "superseded_artifacts": [
            {
                "path": "outputs/validation_student_t_pca16_seed42",
                "reason": "FP32 posterior entropy could be NaN; replaced by validation_student_t_pca16_seed42_v2; NLL and MAP unaffected"
            }
        ],
        "verification": {
            "posterior_normalization": "passed", "candidate_boundaries": "passed",
            "MAP_node_mapping": "passed", "record_counts_and_hashes": "passed",
        },
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
