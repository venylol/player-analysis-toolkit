#!/usr/bin/env python3
"""Independent completion gate for the matchup500/Sentinel v8/Elo v7 chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "offbook_detection" / "data"
SOURCE = DATA / "oq_elo_matchup500_reference_level22_1600plus_20260821"
SENTINEL = DATA / "oq_sentinel_reference_level22_1600plus_v8_20260821"
ELO = DATA / "oq_sentinel_elo_reference_level22_1600plus_v7_20260821"
BATCH = SOURCE / "provenance" / "oq_snapshot_reference500_20260821"
STAGING_SENTINEL_CONFIG = DATA / "oq_sentinel_reference_level22_1600plus_v8_20260821_build_config.json"
STAGING_ELO_CONFIG = DATA / "oq_sentinel_elo_reference_level22_1600plus_v7_20260821_build_config.json"
AUDIT_OUTPUT = DATA / "oq_reference500_independent_final_completion_audit_20260821.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, name: str) -> dict[str, Any]:
    manifest = read(root / name)
    listed = {str(item["path"]) for item in manifest.get("files", [])}
    if name in listed:
        raise ValueError(f"manifest includes itself: {root / name}")
    if int(manifest.get("fileCount", -1)) != len(manifest.get("files", [])):
        raise ValueError(f"manifest fileCount mismatch: {root / name}")
    for item in manifest.get("files", []):
        path = root / str(item["path"])
        if not path.is_file() or sha256(path).casefold() != str(item.get("sha256")).casefold():
            raise ValueError(f"manifest mismatch: {path}")
    return manifest


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-mode", choices=("staging", "root"), default="staging")
    parser.add_argument("--smoke-path", type=Path)
    args = parser.parse_args()

    source_manifest = verify_manifest(SOURCE, "final_sha256_manifest.json")
    sentinel_manifest = verify_manifest(SENTINEL, "reference_sha256_manifest.json")
    elo_manifest = verify_manifest(ELO, "reference_sha256_manifest.json")

    source_bundle = read(SOURCE / "selected_account_bundle.json")
    selection = read(SOURCE / "selected_games_with_partitions.json")
    selection_rows = selection.get("games") or []
    details = source_bundle.get("details") or []
    selected_ids = {str(row.get("gameId") or "") for row in selection_rows}
    bundle_ids = {str(row.get("id") or "") for row in details}
    old_selection = read(DATA / "oq_elo_matchup450_reference_level22_1600plus_20260820" / "selected_games_with_partitions.json")
    old_ids = {str(row.get("gameId") or "") for row in old_selection.get("games", [])}
    selection_audit = read(SOURCE / "selection_audit.json")
    source_completion = read(SOURCE / "reference_completion_audit.json")
    source_partition = read(SOURCE / "partition_engine_index_audit.json")
    engine_audit = read(SOURCE / "engine_level22" / "audit.json")
    engine_progress = read(SOURCE / "engine_level22" / "progress.json")
    batch = read(BATCH / "batch_manifest.json")
    leaderboard = read(BATCH / "leaderboard.json")
    with (BATCH / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        leaderboard_rows = list(csv.DictReader(handle))
    observed_maximum = max(int(row["rating"]) for row in leaderboard_rows)
    player_rows = [read_line for read_line in (json.loads(line) for line in (BATCH / "acquisition" / "player_lists.jsonl").read_text(encoding="utf-8").splitlines()) if read_line]
    failed_players = [row for row in player_rows if row.get("ok") is not True]
    terminal_failures = [json.loads(line) for line in (BATCH / "acquisition" / "leaderboard_terminal_failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    unordered = read(SOURCE / "partitions_unordered.json").get("partitions", [])
    directed = read(SOURCE / "partitions_directed.json").get("partitions", [])
    main_unordered = [row for row in unordered if row.get("partitionScope") == "main_bilateral"]
    capacity = read(BATCH / "acquisition" / "pair_capacity_after_details.json")
    capacity_by_pair = {(int(row["pairLowerA"]), int(row["pairLowerB"])): row for row in capacity.get("pairs", [])}
    local_cache = read(SOURCE / "provenance" / "local_cache_audit" / "cache_pair_audit_dynamic_max.json")
    local_pairs = local_cache.get("pairs", [])

    checks: dict[str, bool] = {}
    checks["sourceManifestVerified"] = True
    checks["sentinelManifestVerified"] = True
    checks["eloManifestVerified"] = True
    checks["sourceFilesCoveredByManifest"] = all(
        name in {str(item["path"]) for item in source_manifest.get("files", [])}
        for name in (
            "engine_game_index.json", "engine_game_index.csv", "partitions_unordered_engine_index.json",
            "partitions_directed_engine_index.json", "partition_engine_index_audit.json",
            "reference_completion_audit.json", "selection_audit.json",
        )
    )
    checks["sourceCompletionOk"] = source_completion.get("ok") is True and source_completion.get("gameCount") == len(selection_rows)
    checks["sourcePartitionOk"] = source_partition.get("ok") is True and source_partition.get("gameCount") == len(selection_rows)
    checks["engineAuditOk"] = engine_audit.get("ok") is True and engine_audit.get("gameCount") == len(selection_rows)
    checks["engineProgressComplete"] = (
        engine_progress.get("completedCount") == len(selection_rows)
        and engine_progress.get("totalCount") == len(selection_rows)
        and set(engine_progress.get("completedGameIds") or []) == selected_ids
    )
    checks["globalGameIdsUnique"] = len(selection_rows) == len(selected_ids) == len(details) == len(bundle_ids) and "" not in selected_ids
    checks["allCurrent450GamesRetained"] = old_ids <= selected_ids
    source_counts = {}
    for row in selection_rows:
        source_counts[str(row.get("sourceKind"))] = source_counts.get(str(row.get("sourceKind")), 0) + 1
    checks["addedGamesAllIncome"] = (
        source_counts.get("existingReference", 0) == int(selection_audit.get("existingReferenceGameCount", -1))
        and source_counts.get("validatedCacheExpansion", 0) == int(selection_audit.get("cacheSelectedCount", -1))
        and source_counts.get("uniqueSnapshotExpansion", 0) == int(selection_audit.get("snapshotSelectedCount", -1))
        and sum(source_counts.values()) == len(selection_rows)
    )
    old_low = sum(row.get("partitionScope") == "baseline_low_elo_extension" for row in old_selection.get("games", []))
    new_low = sum(row.get("partitionScope") == "baseline_low_elo_extension" for row in selection_rows)
    checks["lowEloExtensionPreserved"] = new_low == old_low and new_low == int(selection_audit.get("lowEloExtensionGameCount", -1))
    checks["formalStructure"] = (
        len(main_unordered) == 45
        and len({int(row["pairLowerA"]) for row in main_unordered} | {int(row["pairLowerB"]) for row in main_unordered}) == 9
        and all(row.get("partitionScope") == "main_bilateral" for row in main_unordered)
    )
    checks["crossBandUsesOneStoredGameAndTwoDirectedPartitions"] = (
        sum(row.get("blackBinLower") != row.get("whiteBinLower") for row in selection_rows if row.get("partitionScope") == "main_bilateral") > 0
        and sum(row.get("blackTargetDirectedPartition") != row.get("whiteTargetDirectedPartition") for row in selection_rows if row.get("partitionScope") == "main_bilateral") > 0
        and len(read(SOURCE / "engine_game_index.json").get("games", [])) == len(selected_ids)
    )
    pair_capacity_ok = len(main_unordered) == 45
    for row in main_unordered:
        key = (int(row["pairLowerA"]), int(row["pairLowerB"]))
        frozen = capacity_by_pair.get(key)
        pair_capacity_ok = pair_capacity_ok and frozen is not None
        if frozen is None:
            continue
        maximum = int(frozen["postCacheCount"]) + int(frozen["snapshotValidVerifiedCount"])
        pair_capacity_ok = pair_capacity_ok and int(row["finalCount"]) == min(500, maximum)
        pair_capacity_ok = pair_capacity_ok and len(row.get("mergedGameIds", [])) == int(row["finalCount"])
        pair_capacity_ok = pair_capacity_ok and int(frozen["snapshotDetailConsideredCount"]) == int(frozen["snapshotSummaryCandidateCount"])
        pair_capacity_ok = pair_capacity_ok and int(frozen["snapshotValidVerifiedCount"]) + int(frozen["snapshotInvalidCount"]) == int(frozen["snapshotDetailConsideredCount"])
        pair_capacity_ok = pair_capacity_ok and int(frozen["snapshotValidSelectedCount"]) <= int(frozen["snapshotValidVerifiedCount"])
        pair_capacity_ok = pair_capacity_ok and (int(row["finalCount"]) >= 500 or bool(frozen["capacityExhaustedBelowTarget"]))
    checks["frozenPairCapacity"] = pair_capacity_ok and len(local_pairs) == 45
    checks["localCacheAuditComplete"] = all(
        all(key in row for key in ("existingCount", "cacheAvailableCount", "cacheNewAvailableCount", "cacheSelectedCount", "postCacheCount", "remainingGapToTarget"))
        for row in local_pairs
    )

    network = batch.get("networkPolicy", {})
    stages = batch.get("stages", {})
    checks["networkContract"] = network.get("direct") is True and network.get("proxyDisabled") is True and network.get("environmentProxyUse") is False and "ProxyHandler({})" in str(network.get("httpClientMode"))
    checks["oneFullBatchOnly"] = batch.get("batchId") == "oq-reference500-20260821-once" and network.get("leaderboardFullBatchesAllowed") == 1 and network.get("playerListFullBatchesAllowed") == 1 and stages.get("leaderboard", {}).get("invocationCount") == 1
    checks["leaderboardFrozenDynamicMaximum"] = stages.get("leaderboard", {}).get("complete") is True and int(stages["leaderboard"].get("dynamicMaximumElo", -1)) == observed_maximum == 2498 and int(leaderboard.get("maximumRating", observed_maximum)) == observed_maximum
    checks["allFailuresTerminal"] = (
        len(failed_players) == int(stages.get("playerLists", {}).get("failureCount", -1))
        and all(int(row.get("attempts", 0)) == int(network.get("playerListMaximumAttemptsPerUser", 0)) for row in failed_players)
        and all(row.get("status") == "failed_after_finite_retries" for row in terminal_failures)
    )
    checks["playerListsComplete"] = stages.get("playerLists", {}).get("complete") is True and int(stages["playerLists"].get("terminalUserCount", -1)) == int(stages["playerLists"].get("expectedUserCount", -2))
    checks["detailsAllFrozenCandidatesTerminal"] = stages.get("gameDetails", {}).get("complete") is True and int(stages["gameDetails"].get("terminalDetailRecordCount", -1)) == int(read(BATCH / "acquisition" / "candidate_dedup_audit.json").get("candidateCount", -2))

    sentinel_audit = read(SENTINEL / "reference_build_audit.json")
    directed_records = [json.loads(line) for line in (SENTINEL / "directed_target_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_game: dict[str, set[str]] = {}
    for row in directed_records:
        by_game.setdefault(str(row.get("gameId")), set()).add(str(row.get("targetColor")))
    checks["sentinelAuditOk"] = sentinel_audit.get("ok") is True and len(directed_records) == len(selected_ids) * 2 and set(by_game) == selected_ids and all(colors == {"black", "white"} for colors in by_game.values())
    checks["sentinelDynamicReference"] = sentinel_audit.get("uniqueGameIdCount") == len(selected_ids) and int(read(SOURCE / "selected_account_bundle.json").get("selection", {}).get("maximumElo", 0)) == 2498

    elo_audit = read(ELO / "reference_build_audit.json")
    calibration = read(ELO / "elo_calibration.json")
    calibration_accounts = set(calibration.get("split", {}).get("calibrationAccounts", []))
    validation_accounts = set(calibration.get("split", {}).get("validationAccounts", []))
    checks["eloAuditOk"] = elo_audit.get("ok") is True and elo_audit.get("directedRecordCount") == len(selected_ids) * 2
    checks["calibrationValidated"] = calibration.get("status") == "validated" and calibration.get("calibrationGrouping") == "global" and float(calibration.get("calibrationCoverage", 0)) == 0.95 and float(calibration.get("validationCoverage", 0)) >= 0.95
    checks["calibrationValidationDisjoint"] = not (calibration_accounts & validation_accounts) and len(validation_accounts) >= int(calibration.get("minimumValidationUsers", 20))
    checks["calibrationContract"] = calibration.get("split", {}).get("seed") == 20260821501 and "interpolation" in str(calibration.get("quantileMethod", "")).lower() and int(calibration.get("parallelWorkers", 0)) == 16 and isinstance(calibration.get("diagnosticThresholds"), dict)
    checks["calibrationArtifactAndCasesManifested"] = all(name in {str(item["path"]) for item in elo_manifest.get("files", [])} for name in ("elo_calibration.json", "elo_calibration_cases.jsonl"))

    sentinel_config_path = STAGING_SENTINEL_CONFIG if args.config_mode == "staging" else ROOT / "sentinel_reference_config.json"
    elo_config_path = STAGING_ELO_CONFIG if args.config_mode == "staging" else ROOT / "sentinel_elo_reference_config.json"
    sentinel_config = read(sentinel_config_path)
    elo_config = read(elo_config_path)
    checks["sentinelConfig"] = sentinel_config.get("version") == "v8-20260821" and sentinel_config.get("targetPerUnorderedPair") == 500 and sentinel_config.get("formalEloMaximum") == 2498 and "matchup500" in str(sentinel_config.get("referenceDirectory")) and "_v8_20260821" in str(sentinel_config.get("derivedDirectory"))
    checks["eloConfig"] = elo_config.get("version") == "v7-20260821" and elo_config.get("formalEloMaximum") == 2498 and elo_config.get("eloGridMaximum") == 2498 and elo_config.get("calibrationCoverage") == 0.95 and elo_config.get("calibrationGrouping") == "global" and elo_config.get("calibrationSplitSeed") == 20260821501 and "matchup500" in str(elo_config.get("sourceReferenceDirectory")) and "_v8_20260821" in str(elo_config.get("sentinelDerivedDirectory")) and "_v7_20260821" in str(elo_config.get("derivedReferenceDirectory"))

    smoke_path = (args.smoke_path or (DATA / "oq_reference500_validation" / "minimal_estimate_elo_diandashan" / "unified_test" / "sentinel_unified_analysis.json")).resolve()
    smoke = read(smoke_path) if smoke_path.is_file() else {}
    checks["estimateEloSmoke"] = smoke.get("estimatedElo", {}).get("status") == "valid" and smoke.get("estimatedElo", {}).get("databaseCalibrated95Intervals") and smoke.get("reference", {}).get("calibrationStatus") == "validated"
    checks["unifiedSinglePlayerSmoke"] = smoke.get("schema") == "player-sentinel-unified-analysis-v1" and smoke.get("estimatedElo", {}).get("selectedGameCount", 0) >= 10

    payload = {
        "schema": "oq-reference500-independent-final-completion-audit-v1",
        "ok": all(checks.values()),
        "verifiedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configMode": args.config_mode,
        "sourceGameCount": len(selected_ids),
        "existingReferenceGameCount": len(old_ids),
        "cacheSelectedCount": int(selection_audit.get("cacheSelectedCount", 0)),
        "snapshotSelectedCount": int(selection_audit.get("snapshotSelectedCount", 0)),
        "dynamicFormalEloMaximum": observed_maximum,
        "sentinelDirectedRecordCount": len(directed_records),
        "eloDirectedRecordCount": elo_audit.get("directedRecordCount"),
        "calibration": {key: calibration.get(key) for key in ("status", "t95", "calibrationUserCount", "validationUserCount", "validationCoverage")},
        "calibrationSplitSeed": calibration.get("split", {}).get("seed"),
        "smokePath": str(smoke_path),
        "checks": checks,
    }
    atomic_json(AUDIT_OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
