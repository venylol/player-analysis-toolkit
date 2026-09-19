"""Independent final gate for the 2026-08-20 matchup450 and derived References."""

from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "offbook_detection" / "data"
SOURCE = DATA / "oq_elo_matchup450_reference_level22_1600plus_20260820"
SENTINEL = DATA / "oq_sentinel_reference_level22_1600plus_v7_20260820"
ELO = DATA / "oq_sentinel_elo_reference_level22_1600plus_v6_20260820"
BATCH = SOURCE / "provenance" / "oq_snapshot_reference450_20260820"
AUDIT_OUTPUT = DATA / "oq_reference450_final_completion_audit_20260820.json"


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
    mismatches = []
    for item in manifest.get("files", []):
        path = root / str(item["path"])
        if not path.is_file() or sha256(path).casefold() != str(item.get("sha256")).casefold():
            mismatches.append(str(path))
    if mismatches:
        raise ValueError(f"manifest mismatches: {mismatches[:5]}")
    return manifest


def main() -> int:
    source_manifest = verify_manifest(SOURCE, "final_sha256_manifest.json")
    sentinel_manifest = verify_manifest(SENTINEL, "reference_sha256_manifest.json")
    elo_manifest = verify_manifest(ELO, "reference_sha256_manifest.json")

    source_bundle = read(SOURCE / "selected_account_bundle.json")
    source_selection = read(SOURCE / "selected_games_with_partitions.json")
    old_bundle = read(DATA / "oq_elo_matchup400_reference_level22_1600plus_20260815" / "selected_account_bundle.json")
    source_details = {str(row.get("id")): row for row in source_bundle.get("details", [])}
    old_ids = {str(row.get("id")) for row in old_bundle.get("details", [])}
    selection_rows = source_selection.get("games", [])
    selected_ids = {str(row.get("gameId")) for row in selection_rows}
    added_rows = [row for row in selection_rows if row.get("sourceKind") != "existingReference"]
    source_completion = read(SOURCE / "reference_completion_audit.json")
    source_partition = read(SOURCE / "partition_engine_index_audit.json")
    engine_audit = read(SOURCE / "engine_level22" / "audit.json")
    selection_audit = read(SOURCE / "selection_audit.json")
    capacity = read(BATCH / "acquisition" / "pair_capacity_after_details.json")
    capacity_by_pair = {
        (int(row["pairLowerA"]), int(row["pairLowerB"])): row
        for row in capacity.get("pairs", [])
    }
    unordered = read(SOURCE / "partitions_unordered.json").get("partitions", [])
    main_unordered = [row for row in unordered if row.get("partitionScope") == "main_bilateral"]
    partition_capacity_ok = True
    for row in main_unordered:
        key = (int(row["pairLowerA"]), int(row["pairLowerB"]))
        frozen = capacity_by_pair[key]
        partition_capacity_ok = partition_capacity_ok and (
            int(row["finalCount"]) == int(frozen["finalCount"])
            and len(row.get("mergedGameIds", [])) == int(row["finalCount"])
            and (int(row["finalCount"]) >= 450 or bool(frozen["capacityExhaustedBelowTarget"]))
        )

    directed_lines = sum(1 for line in (SENTINEL / "directed_target_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    sentinel_audit = read(SENTINEL / "reference_build_audit.json")
    elo_audit = read(ELO / "reference_build_audit.json")
    calibration = read(ELO / "elo_calibration.json")
    calibration_accounts = set(calibration.get("split", {}).get("calibrationAccounts", []))
    validation_accounts = set(calibration.get("split", {}).get("validationAccounts", []))
    batch = read(BATCH / "batch_manifest.json")
    leaderboard = read(BATCH / "leaderboard.json")
    with (BATCH / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        leaderboard_rows = list(csv.DictReader(handle))
    observed_leaderboard_maximum = max(int(row["rating"]) for row in leaderboard_rows)
    player_stage = batch.get("stages", {}).get("playerLists", {})
    smoke = DATA / "oq_player_elo_smoke_reynard_20260820" / "sentinel_unified_analysis.json"
    sentinel_config = read(ROOT / "sentinel_reference_config.json")
    elo_config = read(ROOT / "sentinel_elo_reference_config.json")

    checks = {
        "sourceManifestVerified": True,
        "sentinelManifestVerified": True,
        "eloManifestVerified": True,
        "sourceCompletionOk": source_completion.get("ok") is True and source_completion.get("gameCount") == len(selected_ids),
        "sourcePartitionOk": source_partition.get("ok") is True and source_partition.get("gameCount") == len(selected_ids),
        "engineAuditOk": engine_audit.get("ok") is True and engine_audit.get("gameCount") == len(selected_ids),
        "allCurrentReferenceRetained": old_ids <= selected_ids,
        "globalGameIdsUnique": len(source_details) == len(selected_ids) == len(selection_rows),
        "addedCounts": len(added_rows) == int(selection_audit.get("cacheSelectedCount", -1)) + int(selection_audit.get("snapshotSelectedCount", -1)) and sum(row.get("sourceKind") == "validatedCacheExpansion" for row in added_rows) == int(selection_audit.get("cacheSelectedCount", -2)) and sum(row.get("sourceKind") == "uniqueSnapshotExpansion" for row in added_rows) == int(selection_audit.get("snapshotSelectedCount", -2)),
        "lowEloExtensionPreserved": sum(row.get("partitionScope") == "baseline_low_elo_extension" for row in selection_rows) == 116,
        "formalStructure": len(main_unordered) == 45 and len({int(row["pairLowerA"]) for row in main_unordered} | {int(row["pairLowerB"]) for row in main_unordered}) == 9,
        "frozenPairCapacity": partition_capacity_ok,
        "sentinelAuditOk": sentinel_audit.get("ok") is True and directed_lines == len(selected_ids) * 2,
        "eloAuditOk": elo_audit.get("ok") is True and elo_audit.get("directedRecordCount") == len(selected_ids) * 2,
        "calibrationValidated": calibration.get("status") == "validated" and calibration.get("calibrationGrouping") == "global" and float(calibration.get("validationCoverage", 0)) >= 0.95,
        "calibrationValidationDisjoint": not (calibration_accounts & validation_accounts) and len(validation_accounts) >= int(calibration.get("minimumValidationUsers", 20)),
        "networkContract": batch.get("networkPolicy", {}).get("direct") is True and batch.get("networkPolicy", {}).get("proxyDisabled") is True and batch.get("networkPolicy", {}).get("environmentProxyUse") is False,
        "oneBatch": batch.get("batchId") == "oq-reference450-20260820-once" and batch.get("networkPolicy", {}).get("leaderboardFullBatchesAllowed") == 1 and batch.get("networkPolicy", {}).get("playerListFullBatchesAllowed") == 1,
        "leaderboardDynamicMaximum": int(batch.get("stages", {}).get("leaderboard", {}).get("dynamicMaximumElo", 0)) == observed_leaderboard_maximum,
        "playerListsTerminal": player_stage.get("complete") is True and int(player_stage.get("terminalUserCount", 0)) == int(player_stage.get("expectedUserCount", 0)) and int(player_stage.get("failureCount", 0)) == 17,
        "unifiedSmokePassed": smoke.is_file() and read(smoke).get("estimatedElo", {}).get("status") == "valid",
        "configSwitch": sentinel_config.get("version") == "v7-20260820" and sentinel_config.get("targetPerUnorderedPair") == 450 and "matchup450" in sentinel_config.get("referenceDirectory", "") and elo_config.get("version") == "v6-20260820" and "matchup450" in elo_config.get("sourceReferenceDirectory", ""),
    }
    # The leaderboard JSON stores page summaries, while the stage records the observed maximum.
    checks["leaderboardDynamicMaximum"] = int(batch["stages"]["leaderboard"]["dynamicMaximumElo"]) == observed_leaderboard_maximum and int(sentinel_config["formalEloMaximum"]) == int(batch["stages"]["leaderboard"]["dynamicMaximumElo"])
    payload = {
        "schema": "oq-reference450-independent-final-completion-audit-v1",
        "ok": all(checks.values()),
        "sourceGameCount": len(selected_ids),
        "sourceManifestFileCount": source_manifest.get("fileCount"),
        "sentinelManifestFileCount": sentinel_manifest.get("fileCount"),
        "eloManifestFileCount": elo_manifest.get("fileCount"),
        "dynamicFormalEloMaximum": batch["stages"]["leaderboard"]["dynamicMaximumElo"],
        "checks": checks,
        "calibration": {
            "status": calibration.get("status"),
            "t95": calibration.get("t95"),
            "calibrationUserCount": calibration.get("calibrationUserCount"),
            "validationUserCount": calibration.get("validationUserCount"),
            "validationCoverage": calibration.get("validationCoverage"),
        },
    }
    AUDIT_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
