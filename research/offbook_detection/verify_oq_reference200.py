#!/usr/bin/env python3
"""Verify the completed OQ Reference400 source, derivative, and active config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "research/offbook_detection/data/oq_elo_matchup400_reference_level22_1600plus_20260815"
DEFAULT_DERIVED = ROOT / "research/offbook_detection/data/oq_sentinel_reference_level22_1600plus_v6_20260819"
DEFAULT_CONFIG = ROOT / "sentinel_reference_config.json"
DEFAULT_OUTPUT = ROOT / "investigations/oq_reference400_20260815_completion_audit.json"
OLD_SOURCE = ROOT / "research/offbook_detection/data/oq_elo_matchup200_reference_level22_1600plus_20260815"
TARGET_PER_PAIR = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def verify_manifest(root: Path, path: Path) -> tuple[bool, int]:
    manifest = read_json(path)
    files = manifest.get("files") or []
    ok = True
    for row in files:
        target = root / str(row["path"])
        if not target.is_file() or sha256_file(target) != row.get("sha256"):
            ok = False
    return ok, len(files)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    derived = args.derived.resolve()
    config_path = args.config.resolve()

    selection = read_json(source / "selected_games_with_partitions.json")["games"]
    game_ids = [str(row["gameId"]) for row in selection]
    selected = set(game_ids)
    bundle = read_json(source / "selected_account_bundle.json")
    bundle_ids = {str(row.get("id") or "") for row in bundle.get("details") or []}
    engine_index_rows = read_json(source / "engine_game_index.json")["games"]
    engine_index = {str(row["gameId"]): row for row in engine_index_rows}
    engine_files = list((source / "engine_level22").glob("game_*.json"))
    engine_ids = [str(read_json(path).get("gameId") or "") for path in engine_files]
    source_completion = read_json(source / "reference_completion_audit.json")
    source_partition = read_json(source / "partition_engine_index_audit.json")
    runner_audit = read_json(source / "engine_level22/audit.json")
    selection_audit = read_json(source / "selection_audit.json")

    partitions = read_json(source / "partitions_unordered.json")["partitions"]
    formal_pairs = [row for row in partitions if row.get("partitionScope") == "main_bilateral"]
    pair_capacity_ok = all(
        int(row["finalCount"]) >= TARGET_PER_PAIR
        or (row.get("capacityExhaustedBelowTarget") is True and int(row["remainingGap"]) > 0)
        for row in formal_pairs
    )
    no_old_bucket_trim = all(int(row["finalCount"]) >= int(row["existingCount"]) for row in formal_pairs)
    low_ids = {str(row["gameId"]) for row in selection if row.get("partitionScope") == "baseline_low_elo_extension"}

    directed_path = derived / "directed_target_records.jsonl"
    directed_count = 0
    directed_by_game: Counter[str] = Counter()
    colors_by_game: dict[str, set[str]] = {}
    with directed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            game_id = str(row.get("gameId") or "")
            directed_count += 1
            directed_by_game[game_id] += 1
            colors_by_game.setdefault(game_id, set()).add(str(row.get("targetColor") or ""))
    derived_audit = read_json(derived / "reference_build_audit.json")
    config = read_json(config_path)
    old_selection = read_json(OLD_SOURCE / "selected_games_with_partitions.json")["games"]
    old_ids = {str(row["gameId"]) for row in old_selection}
    old_low_ids = {
        str(row["gameId"]) for row in old_selection
        if row.get("partitionScope") == "baseline_low_elo_extension"
    }
    batch = read_json(
        source / "provenance/oq_snapshot_reference400_20260815/batch_manifest.json"
    )
    stages = batch.get("stages") or {}
    source_manifest_ok, source_manifest_count = verify_manifest(source, source / "final_sha256_manifest.json")
    derived_manifest_ok, derived_manifest_count = verify_manifest(derived, derived / "reference_sha256_manifest.json")

    checks = {
        "sourceSelectionIdsUnique": len(game_ids) == len(selected),
        "sourceBundleContainsEverySelectedGame": bundle_ids == selected,
        "engineIndexContainsEverySelectedGame": set(engine_index) == selected,
        "exactlyOneEngineFilePerSelectedGame": len(engine_files) == len(selected) and len(engine_ids) == len(set(engine_ids)) and set(engine_ids) == selected,
        "engineIndexHashesMatch": all(
            (source / str(row["engineFile"])).is_file()
            and sha256_file(source / str(row["engineFile"])) == row.get("engineFileSha256")
            for row in engine_index_rows
        ),
        "sourceAuditsOk": all(value.get("ok") is True for value in (source_completion, source_partition, runner_audit, selection_audit)),
        "formalPairCountIs45": len(formal_pairs) == 45,
        "everyPairMeetsTargetOrFrozenCapacity": pair_capacity_ok,
        "oldBucketsWereNotTrimmed": no_old_bucket_trim,
        "allOldReferenceGamesRetained": old_ids <= selected and len(old_ids) == 6020,
        "lowEloExtensionPreserved": low_ids == old_low_ids and len(low_ids) == 116,
        "twoDirectedRecordsPerGame": directed_count == len(selected) * 2 and set(directed_by_game) == selected and all(count == 2 for count in directed_by_game.values()),
        "bothColorsPresentPerGame": all(colors == {"black", "white"} for colors in colors_by_game.values()),
        "derivedAuditOk": derived_audit.get("ok") is True,
        "sourceManifestVerified": source_manifest_ok,
        "derivedManifestVerified": derived_manifest_ok,
        "configPointsToNewSource": resolve_config_path(config_path, config["referenceDirectory"]) == source,
        "configPointsToNewDerived": resolve_config_path(config_path, config["derivedDirectory"]) == derived,
        "configTargetIs400": int(config.get("targetPerUnorderedPair", -1)) == TARGET_PER_PAIR,
        "configMaximumMatchesBundle": int(config.get("formalEloMaximum", -1)) == int((bundle.get("selection") or {})["maximumElo"]),
        "configHistoricalMaximumIs2495": int(config.get("historicalFormalEloMaximum", -1)) == 2495,
        "fixedNineEloBands": len({int(row["pairLowerA"]) for row in formal_pairs} | {int(row["pairLowerB"]) for row in formal_pairs}) == 9,
        "singleFrozenNetworkBatch": batch.get("batchId") == "oq-reference400-20260815-once"
            and int((stages.get("leaderboard") or {}).get("batchOrdinal", -1)) == 1
            and int((stages.get("playerLists") or {}).get("batchOrdinal", -1)) == 1,
        "networkStagesHaveTerminalStatus": bool((stages.get("leaderboard") or {}).get("complete"))
            and bool((stages.get("playerLists") or {}).get("complete"))
            and int((stages.get("playerLists") or {}).get("terminalUserCount", -1))
                == int((stages.get("playerLists") or {}).get("expectedUserCount", -2)),
        "runtimeEngineFilesAreSelfContained": all((source / str(row["engineFile"])).resolve().is_relative_to(source) for row in engine_index_rows),
    }
    audit = {
        "schema": "oq-reference400-final-completion-audit-v1",
        "ok": all(checks.values()), "verifiedAtUtc": utc_now(),
        "sourceDirectory": str(source), "derivedDirectory": str(derived),
        "gameCount": len(selected), "mainMatrixGameCount": len(selected) - len(low_ids),
        "lowEloExtensionGameCount": len(low_ids), "directedRecordCount": directed_count,
        "formalPairCount": len(formal_pairs),
        "pairsAt400": sum(int(row["finalCount"]) >= TARGET_PER_PAIR for row in formal_pairs),
        "pairsCapacityExhaustedBelow400": sum(bool(row.get("capacityExhaustedBelowTarget")) for row in formal_pairs),
        "sourceManifestFileCount": source_manifest_count,
        "derivedManifestFileCount": derived_manifest_count,
        "checks": checks,
    }
    atomic_json(args.output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
