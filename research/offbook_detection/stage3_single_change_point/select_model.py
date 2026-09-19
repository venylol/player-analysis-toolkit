"""Freeze the stage-3 family selection using validation metrics only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def load_candidate(directory: Path) -> dict[str, object]:
    complete_path = directory / "training_complete.json"
    checkpoint_path = directory / "best.pt"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise ValueError(f"incomplete training run: {directory}")
    if complete["best_checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    return {
        "family": complete["family"],
        "validation_nll_per_decision": complete["best_validation_nll_per_decision"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": complete["best_checkpoint_sha256"],
        "training_complete": str(complete_path.resolve()),
        "training_complete_sha256": sha256_file(complete_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-t-dir", type=Path, required=True)
    parser.add_argument("--gaussian-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {output}")
    candidates = [load_candidate(args.student_t_dir.resolve(strict=True)), load_candidate(args.gaussian_dir.resolve(strict=True))]
    candidates.sort(key=lambda item: float(item["validation_nll_per_decision"]))
    config_path = args.config.resolve(strict=True)
    cache_manifest = args.projected_cache.resolve(strict=True) / "manifest.json"
    payload = {
        "schema": "stage3-model-selection-v1", "status": "frozen_before_test",
        "selection_split": "validation", "selection_metric": "marginal NLL per effective decision",
        "test_used": False, "candidates": candidates, "selected": candidates[0],
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "projected_cache_manifest": str(cache_manifest),
        "projected_cache_manifest_sha256": sha256_file(cache_manifest),
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
