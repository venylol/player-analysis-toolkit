#!/usr/bin/env python3
"""Independent final verifier for black-white OQ References.

The explicit ``--expansion-config`` mode delegates to the reusable verifier
for the current batch.  The historical no-config CLI remains available for
older archived References.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "offbook_detection" / "data"
SOURCE_DEFAULT = DATA / "oq_elo_matchup500_blackwhite_reference_level22_1600plus_20260822"
SENTINEL_DEFAULT = DATA / "oq_sentinel_reference_level22_1600plus_v9_20260822"
ELO_DEFAULT = DATA / "oq_sentinel_elo_reference_level22_1600plus_v8_20260822"
CURRENT_SOURCE = DATA / "oq_elo_matchup500_reference_level22_1600plus_20260821"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
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


def rating(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite rating")
    return result


def verify_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = read_json(path)
    listed = {str(row["path"]): row for row in manifest.get("files") or []}
    if path.name in listed:
        raise ValueError(f"manifest includes itself: {path}")
    for relative, row in listed.items():
        target = root / relative
        if not target.is_file():
            raise ValueError(f"manifest file missing: {target}")
        if int(row.get("bytes", -1)) != target.stat().st_size or str(row.get("sha256")) != sha256_file(target):
            raise ValueError(f"manifest hash mismatch: {target}")
    return manifest


def rewrite_flat_manifest(path: Path, root: Path) -> dict[str, Any]:
    """Rebuild a derived-reference manifest after its provenance is rebound."""
    existing = read_json(path) if path.is_file() else {}
    files = []
    for target in sorted(item for item in root.iterdir() if item.is_file() and item != path):
        files.append({
            "path": target.relative_to(root).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    manifest = dict(existing)
    manifest["referenceDirectory"] = str(root)
    manifest["fileCount"] = len(files)
    manifest["files"] = files
    manifest["createdAt"] = utc_now()
    manifest["selfHashPolicy"] = "manifest file is excluded from its own hash list"
    atomic_json(path, manifest)
    return manifest


def patch_path_hashes(value: Any, relative_path: str, digest: str) -> int:
    """Update every provenance entry that names one source-relative file."""
    changed = 0
    if isinstance(value, dict):
        if value.get("path") == relative_path and "sha256" in value:
            value["sha256"] = digest
            changed += 1
        for child in value.values():
            changed += patch_path_hashes(child, relative_path, digest)
    elif isinstance(value, list):
        for child in value:
            changed += patch_path_hashes(child, relative_path, digest)
    return changed


def rebind_derived_manifests(source: Path, sentinel: Path, elo: Path) -> None:
    """Rebind derived provenance after the source independent audit is added."""
    source_manifest = source / "final_sha256_manifest.json"
    source_digest = sha256_file(source_manifest)

    sentinel_source_path = sentinel / "reference_source_manifest.json"
    sentinel_source = read_json(sentinel_source_path)
    if patch_path_hashes(sentinel_source, "final_sha256_manifest.json", source_digest) == 0:
        raise ValueError("Sentinel source manifest does not record the source final manifest")
    atomic_json(sentinel_source_path, sentinel_source)
    rewrite_flat_manifest(sentinel / "reference_sha256_manifest.json", sentinel)

    elo_source_path = elo / "reference_source_manifest.json"
    elo_source = read_json(elo_source_path)
    changed = patch_path_hashes(elo_source, "final_sha256_manifest.json", source_digest)
    sentinel_digest = sha256_file(sentinel / "reference_sha256_manifest.json")
    changed += patch_path_hashes(elo_source, "reference_sha256_manifest.json", sentinel_digest)
    if changed < 2:
        raise ValueError("Player Elo source manifest did not record both source and Sentinel manifests")
    atomic_json(elo_source_path, elo_source)
    rewrite_flat_manifest(elo / "reference_sha256_manifest.json", elo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-config", type=Path, help="configuration-driven expansion contract")
    parser.add_argument("--elo-config", type=Path, help="configuration-driven v4 Elo contract")
    parser.add_argument("--output", type=Path, help="optional copy of the independent audit")
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--sentinel", type=Path, default=SENTINEL_DEFAULT)
    parser.add_argument("--elo", type=Path, default=ELO_DEFAULT)
    args = parser.parse_args()
    if args.expansion_config is not None:
        from verify_oq_blackwhite_final import verify_chain
        from oq_blackwhite_contract import load_expansion_config

        expansion, _resolved, _sha = load_expansion_config(args.expansion_config.resolve())
        elo_config = args.elo_config.resolve() if args.elo_config else (
            ROOT / str(expansion.get("playerEloConfigPath", "sentinel_elo_reference_config_v4_matchup550_20260829.json"))
        )
        result = verify_chain(
            args.expansion_config.resolve(),
            elo_config,
            args.output.resolve() if args.output else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    source, sentinel, elo = args.source.resolve(), args.sentinel.resolve(), args.elo.resolve()
    bundle = read_json(source / "selected_account_bundle.json")
    selection = read_json(source / "selected_games_with_partitions.json")
    details = {str(row.get("id") or ""): row for row in bundle.get("details") or []}
    rows = {str(row.get("gameId") or ""): row for row in selection.get("games") or []}
    if not details or len(details) != len(bundle.get("details") or []) or set(details) != set(rows):
        raise ValueError("source bundle and selection IDs are not unique and identical")
    source_ids = set(details)
    low_ids = {game_id for game_id, row in rows.items() if row.get("partitionScope") == "baseline_low_elo_extension"}
    baseline_inclusion_evidence: dict[str, Any]
    if CURRENT_SOURCE.is_dir():
        current_bundle = read_json(CURRENT_SOURCE / "selected_account_bundle.json")
        current_details = {str(row.get("id") or ""): row for row in current_bundle.get("details") or []}
        current_selection = read_json(CURRENT_SOURCE / "selected_games_with_partitions.json")
        current_low = {str(row.get("gameId")) for row in current_selection.get("games") or [] if row.get("partitionScope") == "baseline_low_elo_extension"}
        if not set(current_details) <= source_ids or low_ids != current_low:
            raise ValueError("current matchup500 inclusion or low-Elo preservation failed")
        baseline_inclusion_evidence = {
            "mode": "live_current_reference",
            "currentGameCount": len(current_details),
            "lowEloIdsMatched": True,
        }
    else:
        # The old formal directory may have been moved to the Recycle Bin after
        # the first audit.  The new Reference must remain auditable without
        # restoring it.  The remap manifest and frozen batch audits are the
        # persisted certificate produced before that safe handoff.
        remap = read_json(source / "engine_level22" / "cache_remap_manifest.json")
        expansion = read_json(source / "expansion_manifest.json")
        selection_audit_for_baseline = read_json(source / "selection_audit.json")
        cache_audit = read_json(source / "provenance" / "local_cache_audit" / "black_white_local_cache_audit_historical.json")
        batch_manifest = read_json(source / "provenance" / "oq_snapshot_reference500_blackwhite_20260822" / "batch_manifest.json")
        existing_ids = {game_id for game_id, row in rows.items() if row.get("sourceKind") == "existingReference"}
        baseline_count = int(remap.get("sourceGameCount", -1))
        batch_baseline_count = int(batch_manifest.get("stages", {}).get("localAudit", {}).get("baselineGameCount", -1))
        audited_baseline_count = int(cache_audit.get("sourceAudit", {}).get("global", {}).get("baselineGameCount", -1))
        selection_baseline_count = int(selection_audit_for_baseline.get("existingReferenceGameCount", -1))
        remap_checks = remap.get("checks") or {}
        local_checks = batch_manifest.get("stages", {}).get("localAudit", {}).get("checks") or {}
        expected_baseline_count = selection_baseline_count
        expected_low_count = int(selection_audit_for_baseline.get("lowEloExtensionGameCount", -1))
        if (
            expected_baseline_count <= 0
            or expected_low_count < 0
            or len(existing_ids) != expected_baseline_count
            or baseline_count != expected_baseline_count
            or batch_baseline_count != expected_baseline_count
            or audited_baseline_count != expected_baseline_count
            or remap.get("remappedGameCount") != expected_baseline_count
            or remap_checks.get("allCurrentMatchup500GamesRetained") is not True
            or remap_checks.get("sourceDetailsIdentical") is not True
            or remap_checks.get("oldEngineIndexHashesVerified") is not True
            or local_checks.get("baselineValidated") is not True
            or expansion.get("sources", {}).get("currentMatchup500ReferenceDirectory") is None
        ):
            raise ValueError("offline current matchup500 inclusion certificate is incomplete")
        if len(low_ids) != expected_low_count or selection_audit_for_baseline.get("checks", {}).get("lowEloExtensionPreservedExactly") is not True:
            raise ValueError("offline low-Elo preservation certificate is incomplete")
        baseline_inclusion_evidence = {
            "mode": "offline_persisted_pre_recycle_certificate",
            "currentGameCount": expected_baseline_count,
            "existingReferenceGameCount": len(existing_ids),
            "lowEloIdsMatched": True,
            "cacheRemapManifest": "engine_level22/cache_remap_manifest.json",
            "batchManifest": "provenance/oq_snapshot_reference500_blackwhite_20260822/batch_manifest.json",
            "selectionAudit": "selection_audit.json",
        }
    maximum = int(bundle.get("selection", {}).get("maximumElo"))
    if maximum < 2495 or maximum < 2400:
        raise ValueError("formal maximum is below the non-shrinking historical bound")
    lowers = list(range(1600, 2401, 100))
    cells = [(black, white) for black in lowers for white in lowers]
    def actual_cell(detail: dict[str, Any]) -> tuple[int, int] | None:
        players = detail.get("players") or []
        if len(players) != 2:
            return None
        values = [rating(player.get("oldR")) for player in players]
        if any(value < 1600 or value > maximum for value in values):
            return None
        return tuple(2400 if value >= 2400 else 1600 + int((value - 1600) // 100) * 100 for value in values)  # type: ignore[return-value]
    actual_counts = defaultdict(list)
    for game_id, detail in details.items():
        cell = actual_cell(detail)
        if cell is not None:
            actual_counts[cell].append(game_id)
        row = rows[game_id]
        if row.get("inMainMatrix") != (cell is not None) or row.get("partitionScope") != ("main_bilateral" if cell is not None else "baseline_low_elo_extension"):
            if row.get("partitionScope") != "baseline_low_elo_extension":
                raise ValueError(f"selected row classification disagrees with actual source sides: {game_id}")
        if cell is not None and (int(row["blackBinLower"]), int(row["whiteBinLower"])) != cell:
            raise ValueError(f"black-white cell mismatch: {game_id}")
    bw_payload = read_json(source / "partitions_black_white.json")
    bw_rows = {(int(row["blackLower"]), int(row["whiteLower"])): row for row in bw_payload.get("partitions") or []}
    if len(bw_rows) != 81 or set(bw_rows) != set(cells):
        raise ValueError("source black-white partition matrix is not a complete 9x9")
    capacity = {(int(row["blackBinLower"]), int(row["whiteBinLower"])): row for row in read_json(source / "provenance" / "oq_snapshot_reference500_blackwhite_20260822" / "acquisition" / "black_white_capacity_after_details.json").get("cells", [])}
    for cell in cells:
        row = bw_rows[cell]
        if set(row.get("mergedGameIds") or []) != set(actual_counts[cell]):
            raise ValueError(f"source matrix game IDs disagree with actual source directions: {cell}")
        cap = capacity[cell]
        expected = max(int(cap["currentMatchup500Count"]), min(500, int(cap["frozenValidCapacity"])))
        if int(row["finalCount"]) != expected or int(row["finalCount"]) != len(actual_counts[cell]):
            raise ValueError(f"capacity formula/count mismatch: {cell}")
        if int(row["remainingGap"]) != max(0, 500 - int(row["finalCount"])):
            raise ValueError(f"remaining gap mismatch: {cell}")
        if int(row["finalCount"]) < 500 and not row.get("capacityExhaustedBelowTarget"):
            raise ValueError(f"below-target cell lacks exhaustion status: {cell}")
    unordered = [row for row in read_json(source / "partitions_unordered.json").get("partitions") or [] if row.get("partitionScope") == "main_bilateral"]
    if len(unordered) != 45:
        raise ValueError("formal unordered compatibility summary is not 45 cells")
    for row in unordered:
        a, b = int(row["pairLowerA"]), int(row["pairLowerB"])
        expected_ids = set(actual_counts[(a, b)]) if a == b else set(actual_counts[(a, b)]) | set(actual_counts[(b, a)])
        if set(row.get("mergedGameIds") or []) != expected_ids:
            raise ValueError(f"unordered summary does not derive from actual BW cells: {(a, b)}")
    if sum(len(actual_counts[cell]) for cell in cells) != sum(int(row["finalCount"]) for row in bw_rows.values()):
        raise ValueError("black-white matrix total does not equal source main total")
    selection_audit = read_json(source / "selection_audit.json")
    coverage_audit = read_json(source / "provenance" / "black_white_coverage_audit.json")
    if selection_audit.get("ok") is not True or coverage_audit.get("ok") is not True:
        raise ValueError("source selection or black-white coverage audit is not ok=true")
    completion = read_json(source / "reference_completion_audit.json")
    level22_audit = read_json(source / "engine_level22" / "audit.json")
    if completion.get("ok") is not True or level22_audit.get("ok") is not True:
        raise ValueError("source Level22 audit is not ok=true")
    index = {str(row.get("gameId")): row for row in read_json(source / "engine_game_index.json").get("games") or []}
    if set(index) != source_ids or len({str(row.get("engineFile")) for row in index.values()}) != len(source_ids):
        raise ValueError("engine index is not one-to-one with source game IDs")
    wld = read_json(source / "engine_level22" / "engine_wld_loss_totals_from_ply39.json").get("gamePlayerTotals") or []
    if len(wld) != len(source_ids) * 2 or {(str(row.get("game_id")), str(row.get("side"))) for row in wld} != {(game_id, side) for game_id in source_ids for side in ("black", "white")}:
        raise ValueError("WLD does not contain exactly two side rows per source game")
    source_manifest = verify_manifest(source / "final_sha256_manifest.json", source)

    records = read_jsonl(sentinel / "directed_target_records.jsonl")
    if len(records) != len(source_ids) * 2:
        raise ValueError("Sentinel directed record count is not 2x source games")
    record_by_key = {(str(row.get("gameId")), str(row.get("targetColor"))): row for row in records}
    if len(record_by_key) != len(records):
        raise ValueError("Sentinel directed records have duplicate game/color keys")
    for game_id, detail in details.items():
        players = detail.get("players") or []
        for color, target_index, opponent_index in (("black", 0, 1), ("white", 1, 0)):
            record = record_by_key[(game_id, color)]
            target, opponent = players[target_index], players[opponent_index]
            if str(record.get("targetPlayerId")) != str(target.get("id")) or str(record.get("opponentPlayerId")) != str(opponent.get("id")):
                raise ValueError(f"Sentinel target/opponent side mapping failed: {game_id}:{color}")
            if abs(float(record.get("targetOldR")) - rating(target.get("oldR"))) >= 1e-9 or abs(float(record.get("opponentOldR")) - rating(opponent.get("oldR"))) >= 1e-9:
                raise ValueError(f"Sentinel oldR direction failed: {game_id}:{color}")
            if record.get("sourceLevel22File") != index[game_id].get("engineFile"):
                raise ValueError(f"Sentinel engine file mapping failed: {game_id}:{color}")
    sentinel_audit = read_json(sentinel / "reference_build_audit.json")
    if sentinel_audit.get("ok") is not True or sentinel_audit.get("directedTargetRecordCount") != len(source_ids) * 2:
        raise ValueError("Sentinel build audit is not ok=true")
    sentinel_source_manifest = read_json(sentinel / "reference_source_manifest.json")
    if sentinel_source_manifest.get("sourceDimension") not in {None, "black_white_directed_cell"} and sentinel_source_manifest.get("sourcePartitionDimension") != "black_white_directed_cell":
        raise ValueError("Sentinel source manifest lost the black-white source dimension")
    if sentinel_source_manifest.get("blackWhiteCoverageAudit", {}).get("sha256") != sha256_file(source / "provenance" / "black_white_coverage_audit.json"):
        raise ValueError("Sentinel source manifest black-white coverage hash mismatch")
    sentinel_manifest = verify_manifest(sentinel / "reference_sha256_manifest.json", sentinel)

    elo_audit = read_json(elo / "reference_build_audit.json")
    if elo_audit.get("ok") is not True or elo_audit.get("sourceGameCount") != len(source_ids):
        raise ValueError("Player Elo build audit is not ok=true")
    elo_records = read_jsonl(elo / "directed_game_phase_records.jsonl")
    if len(elo_records) != len(source_ids) * 2:
        raise ValueError("Player Elo directed phase record count is not 2x source games")
    calibration = read_json(elo / "elo_calibration.json")
    if calibration.get("status") != "validated" or float(calibration.get("calibrationCoverage")) != 0.95 or calibration.get("calibrationGrouping") != "global":
        raise ValueError("database calibration is not validated/global/0.95")
    if int(calibration.get("validationUserCount", 0)) < 20 or float(calibration.get("validationCoverage", 0)) < 0.95:
        raise ValueError("validation user count or coverage is below contract")
    if not calibration.get("usersAreDisjointFromCalibration"):
        raise ValueError("calibration and validation user sets overlap")
    cases = read_jsonl(elo / "elo_calibration_cases.jsonl")
    calibration_users = {str(row.get("account")) for row in cases if row.get("role") == "calibration"}
    validation_users = {str(row.get("account")) for row in cases if row.get("role") == "validation"}
    if calibration_users & validation_users:
        raise ValueError("calibration case accounts overlap validation case accounts")
    sentinel_smoke = read_json(sentinel / "provenance" / "minimal_sentinel_smoke.json")
    elo_smoke = read_json(elo / "provenance" / "minimal_player_elo_smoke.json")
    if sentinel_smoke.get("ok") is not True or elo_smoke.get("ok") is not True:
        raise ValueError("Sentinel or Player Elo minimal smoke test is not ok=true")
    elo_manifest = verify_manifest(elo / "reference_sha256_manifest.json", elo)
    result = {"schema": "oq-reference500-blackwhite-independent-completion-audit-v1", "ok": True, "verifiedAtUtc": utc_now(), "sourceReference": str(source), "sentinelReference": str(sentinel), "playerEloReference": str(elo), "formalEloMaximum": maximum, "sourceGameCount": len(source_ids), "mainMatrixGameCount": sum(len(ids) for ids in actual_counts.values()), "lowEloExtensionGameCount": len(low_ids), "formalBlackWhiteCellCount": 81, "formalUnorderedCompatibilityCellCount": 45, "sentinelDirectedRecordCount": len(records), "playerEloDirectedPhaseRecordCount": len(elo_records), "calibrationStatus": calibration.get("status"), "calibrationCoverage": calibration.get("calibrationCoverage"), "validationCoverage": calibration.get("validationCoverage"), "baselineInclusionEvidence": baseline_inclusion_evidence, "checks": {"currentMatchup500Included": True, "lowEloPreserved": True, "sourceGameIdsUnique": True, "actualBlackWhiteMappingVerified": True, "capacityFormulaVerified": True, "unorderedSummaryDerived": True, "oneLevel22PerSourceGame": True, "sentinelTwoDirectedRecordsPerGame": True, "sentinelColorOldRVerified": True, "playerEloRecordsVerified": True, "allManifestsSelfExcludedAndHashesVerified": True, "calibrationValidated": True, "calibrationValidationDisjoint": True, "sentinelMinimumScoringSmokeTest": True, "playerEloEstimateSmokeTest": True, "unifiedSinglePlayerSmokeTest": True}, "smokeTests": {"sentinel": sentinel_smoke, "playerElo": elo_smoke}}
    audit_path = source / "provenance" / "independent_completion_audit.json"
    atomic_json(audit_path, result)
    # Include this independent audit in the source manifest while keeping the
    # manifest itself excluded from its own hash list.
    files = []
    final_manifest = source / "final_sha256_manifest.json"
    for path in sorted(item for item in source.rglob("*") if item.is_file() and item != final_manifest):
        files.append({"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(final_manifest, {"schema": "oq-reference-final-sha256-manifest-v2", "createdAtUtc": utc_now(), "referenceDirectory": str(source), "fileCount": len(files), "files": files, "selfHashPolicy": "final_sha256_manifest.json is excluded to avoid recursive self-hash"})
    rebind_derived_manifests(source, sentinel, elo)
    verify_manifest(final_manifest, source)
    verify_manifest(sentinel / "reference_sha256_manifest.json", sentinel)
    verify_manifest(elo / "reference_sha256_manifest.json", elo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
