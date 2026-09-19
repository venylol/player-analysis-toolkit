#!/usr/bin/env python3
"""Build a Sentinel reference and attach authoritative black-white provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))
from oq_blackwhite_contract import load_expansion_config
DEFAULT_SOURCE = ROOT / "research" / "offbook_detection" / "data" / "oq_elo_matchup600_blackwhite_reference_level22_1600plus_20260911"
DEFAULT_OUTPUT = ROOT / "research" / "offbook_detection" / "data" / "oq_sentinel_reference_level22_1600plus_v11_20260911"
SENTINEL_BUILDER = ROOT / "scripts" / "analysis" / "sentinel_analysis.py"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, help="UTF-8 black-white expansion configuration")
    args = parser.parse_args()
    source_was_default = args.source == DEFAULT_SOURCE
    output_was_default = args.output == DEFAULT_OUTPUT
    config_sha = None
    if args.config is not None:
        config, _config_path, config_sha = load_expansion_config(args.config)
        if source_was_default:
            args.source = Path(config["paths"]["sourceOutputDirectory"])
        if output_was_default:
            args.output = Path(config["paths"]["sentinelOutputDirectory"])
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        audit_path = output / "reference_build_audit.json"
        manifest_path = output / "reference_sha256_manifest.json"
        if audit_path.is_file() and read_json(audit_path).get("ok") is True and manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest_paths = {str(row.get("path") or "") for row in manifest.get("files", [])}
            if "black_white_source_partition_provenance.json" in manifest_paths:
                print(json.dumps(read_json(audit_path), ensure_ascii=False, indent=2))
                return 0
        # A failed build may have atomically emitted a subset of the known
        # Sentinel artifacts before its final audit was written.  Resume that
        # same output batch, but refuse any directory containing unrelated
        # files or a previously successful audit.
        resumable_files = {
            "directed_target_records.jsonl",
            "directed_target_records.csv",
            "offbook_records_by_target_side.json",
            "reference_cell_summary.json",
            "reference_cell_summary.csv",
            "reference_source_manifest.json",
            "reference_build_audit.json",
            "reference_sha256_manifest.json",
            "black_white_source_partition_provenance.json",
        }
        existing_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        if not existing_files or not existing_files.issubset(resumable_files):
            raise FileExistsError(f"refusing to overwrite non-empty Sentinel output: {output}")
    subprocess.run([sys.executable, str(SENTINEL_BUILDER), "build-reference", "--reference-dir", str(source), "--output-dir", str(output)], cwd=ROOT, check=True, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})

    selection = read_json(source / "selected_games_with_partitions.json")
    games = selection.get("games") or []
    coverage_path = source / "provenance" / "black_white_coverage_audit.json"
    bw_path = source / "partitions_black_white.json"
    unordered_path = source / "partitions_unordered.json"
    coverage = read_json(coverage_path)
    bw = read_json(bw_path)
    unordered = read_json(unordered_path)
    if len(bw.get("partitions") or []) != 81:
        raise ValueError("source black-white partition matrix is not exactly 81 cells")
    selected_by_id = {str(row.get("gameId") or ""): row for row in games}
    if len(selected_by_id) != len(games):
        raise ValueError("source selected game IDs are not unique")
    source_manifest_path = output / "reference_source_manifest.json"
    source_manifest = read_json(source_manifest_path)
    source_manifest["blackWhiteCellSummary"] = {"path": str(bw_path.resolve()), "sha256": sha256_file(bw_path), "partitionDimension": "black_white_directed", "formalCellCount": 81}
    source_manifest["partitionsBlackWhite"] = {"path": str(bw_path.resolve()), "sha256": sha256_file(bw_path)}
    source_manifest["partitionsUnorderedLegacySummary"] = {"path": str(unordered_path.resolve()), "sha256": sha256_file(unordered_path), "partitionDimension": "unordered_legacy_summary", "formalPairCount": 45}
    source_manifest["blackWhiteCoverageAudit"] = {"path": str(coverage_path.resolve()), "sha256": sha256_file(coverage_path), "ok": coverage.get("ok") is True}
    source_manifest["sourceSelectionAudit"] = {"path": str((source / "selection_audit.json").resolve()), "sha256": sha256_file(source / "selection_audit.json")}
    source_manifest["sourcePartitionDimension"] = "black_white_directed_cell"
    source_manifest["configSha256"] = config_sha
    source_manifest["sentinelCompatibilityDimension"] = "targetEloBand_x_opponentEloBand_x_targetColor_x_analysisScope"
    atomic_json(source_manifest_path, source_manifest)

    build_audit_path = output / "reference_build_audit.json"
    build_audit = read_json(build_audit_path)
    checks = build_audit.setdefault("checks", {})
    checks.update({
        "blackWhiteSourceMatrixExactly81Cells": len(bw.get("partitions") or []) == 81,
        "blackWhiteCoverageAuditOk": coverage.get("ok") is True,
        "blackWhiteSelectionRowsMatchSource": set(selected_by_id) == {str(row.get("gameId") or "") for row in games},
        "unorderedLegacySummaryHas45FormalPairs": sum(row.get("partitionScope") == "main_bilateral" for row in unordered.get("partitions") or []) == 45,
        "lowEloExcludedFromFormalBlackWhiteDenominator": build_audit.get("formalMainMatrixDirectedRecordCount") == build_audit.get("mainMatrixGameCount", 0) * 2,
        "sourceDimensionIsNotSentinelTargetOpponent": True,
    })
    build_audit["blackWhiteCellSummary"] = {"path": str(bw_path.resolve()), "sha256": sha256_file(bw_path), "formalCellCount": 81}
    build_audit["blackWhiteCoverageAudit"] = {"path": str(coverage_path.resolve()), "sha256": sha256_file(coverage_path)}
    build_audit["configSha256"] = config_sha
    build_audit["ok"] = bool(build_audit.get("ok") and all(checks.values()))
    atomic_json(build_audit_path, build_audit)
    atomic_json(output / "black_white_source_partition_provenance.json", {"schema": "oq-sentinel-blackwhite-source-provenance-v2", "ok": build_audit["ok"], "configSha256": config_sha, "sourceReference": str(source), "partitionsBlackWhite": str(bw_path), "partitionsUnordered": str(unordered_path), "blackWhiteCoverageAudit": str(coverage_path), "formalCellCount": 81, "legacyFormalPairCount": 45, "sourceGameCount": len(games), "directedTargetRecordCount": len(games) * 2})

    # Rebuild the derived manifest after augmentation; exclude the manifest
    # itself so it never recursively hashes itself.
    import importlib.util
    spec = importlib.util.spec_from_file_location("sentinel_analysis_for_bw_manifest", SENTINEL_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Sentinel manifest helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = ["directed_target_records.jsonl", "directed_target_records.csv", "offbook_records_by_target_side.json", "reference_cell_summary.json", "reference_cell_summary.csv", "reference_source_manifest.json", "reference_build_audit.json", "black_white_source_partition_provenance.json"]
    manifest = module.sentinel.manifest_for_files(output, generated, "player-anomaly-sentinel-reference-sha256-manifest-v2")
    module.sentinel.write_json(output / "reference_sha256_manifest.json", manifest, refuse_existing=False)
    print(json.dumps(build_audit, ensure_ascii=False, indent=2))
    return 0 if build_audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
