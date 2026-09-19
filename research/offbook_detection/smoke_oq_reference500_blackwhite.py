#!/usr/bin/env python3
"""Run the minimal dynamic-band Sentinel smoke test for the BW Reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from player_analysis_toolkit import sentinel  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def resolve_root(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = sentinel.read_json(config_path)
    sentinel.configure_from_reference_config(config_path)
    derived = resolve_root(str(config["derivedDirectory"]))
    records_path = resolve_root(str(config["directedTargetRecords"]))
    manifest_path = resolve_root(str(config["referenceManifest"]))
    manifest = sentinel.read_json(manifest_path)
    manifest_ok = True
    for row in manifest.get("files") or []:
        path = derived / str(row["path"])
        if not path.is_file() or sentinel.sha256_file(path) != row.get("sha256"):
            manifest_ok = False
            break
    records = read_jsonl(records_path)
    formal = [row for row in records if row.get("formalReferenceEligible")]
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in formal:
        by_game.setdefault(str(row["gameId"]), []).append(row)
    target_game_id, target = next(
        (game_id, rows)
        for game_id, rows in sorted(by_game.items())
        if {str(row.get("targetColor")) for row in rows} == {"black", "white"}
    )
    score = sentinel.score_target_records(target, records)
    checks = {
        "manifestVerified": manifest_ok,
        "directedRecordsLoaded": len(records) > 0 and len(records) % 2 == 0,
        "dynamicMaximumApplied": sentinel.MAX_ELO == int(config["formalEloMaximum"]),
        "nineFormalBands": len(sentinel.BANDS) == 9,
        "twoTargetSidesScored": score.get("targetRecordCount") == 2,
        "minimalScoreCalibratable": score.get("calibratableGameCount") == 2,
        "leaveOneGameExcluded": score.get("excludedReferenceGameIds") == [target_game_id],
    }
    audit = {
        "schema": "oq-reference500-blackwhite-sentinel-smoke-audit-v1",
        "ok": all(checks.values()),
        "config": str(config_path),
        "directedTargetRecords": str(records_path),
        "directedRecordCount": len(records),
        "formalEloMaximum": sentinel.MAX_ELO,
        "formalBandCount": len(sentinel.BANDS),
        "targetGameId": target_game_id,
        "scoreSummary": {
            "targetRecordCount": score.get("targetRecordCount"),
            "calibratableGameCount": score.get("calibratableGameCount"),
            "excludedReferenceGameIds": score.get("excludedReferenceGameIds"),
        },
        "checks": checks,
    }
    atomic_json(args.output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
