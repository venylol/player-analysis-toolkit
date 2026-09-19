#!/usr/bin/env python3
"""Configuration-driven OQ black/white expansion CLI.

This runner is intentionally versioned separately from the historical unordered
pair runners.  The source corpus is partitioned by the actual players[0] and
players[1] sides; Sentinel's target/opponent compatibility partitions are only
derived views and never drive source selection.

Every network stage is resumable from the same batch manifest.  Successful
leaderboard pages, player lists, and game details are terminal checkpoints and
are not requested again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from build_elo_reference import replay_is_legal
from pull_oq_transformer_dataset import (
    GAME_BASE_URL,
    GTYPE,
    HttpClient,
    LeaderboardUser,
    normal_score,
    valid_detail,
)
from oq_blackwhite_contract import (
    load_expansion_config,
    resolve_repo_path,
    source_priority,
    stable_summary_sha,
)


DATA = ROOT / "research" / "offbook_detection" / "data"
CURRENT_REFERENCE = DATA / "oq_elo_matchup550_blackwhite_reference_level22_1600plus_20260829"
CURRENT_SENTINEL = DATA / "oq_sentinel_reference_level22_1600plus_v10_20260829"
CURRENT_ELO = DATA / "oq_sentinel_elo_reference_level22_1600plus_v4_matchup550_20260829"
RAW_CACHE = DATA / "oq_transformer_100000_20260813" / "acquisition" / "game_details.jsonl"
MATERIALIZED_GAMES = DATA / "oq_transformer_61145_20260813" / "games.csv"
CURRENT_SNAPSHOT = CURRENT_REFERENCE / "provenance" / "oq_snapshot_reference550_blackwhite_20260829"

MINIMUM_ELO = 1600
HISTORICAL_FORMAL_MAXIMUM = 2495
TOP_BIN_LOWER = 2400
BIN_WIDTH = 100
CELL_COUNT = 9
TARGET_PER_CELL = 600
SELECTION_SEED = 20260911501
REQUEST_TIMEOUT = 20.0
WORKERS = 16
PLAYER_LIST_MAX_ATTEMPTS = 3
DETAIL_MAX_ATTEMPTS = 3
RUN_DATE = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
BATCH_ID = f"oq-reference600-blackwhite-{RUN_DATE}-once"
DEFAULT_OUTPUT = DATA / f"oq_elo_matchup600_blackwhite_reference_level22_1600plus_{RUN_DATE}"
SNAPSHOT_NAME = f"oq_snapshot_reference600_blackwhite_{RUN_DATE}"
DEFAULT_CONFIG_PATH = ROOT / "oq_reference_blackwhite_expansion_config_v3_matchup600_20260911.json"

EXPANSION_CONFIG: dict[str, Any] | None = None
EXPANSION_CONFIG_PATH: Path | None = None
EXPANSION_CONFIG_SHA256: str | None = None
BASELINE_SNAPSHOT = CURRENT_SNAPSHOT
SNAPSHOT_OUTPUT_DIRECTORY: Path | None = None
LEADERBOARD_MAX_ATTEMPTS = 3


def apply_expansion_config(config_path: Path | None) -> dict[str, Any] | None:
    """Inject one validated expansion contract into this compatibility CLI."""

    global EXPANSION_CONFIG, EXPANSION_CONFIG_PATH, EXPANSION_CONFIG_SHA256
    global CURRENT_REFERENCE, CURRENT_SENTINEL, CURRENT_ELO, RAW_CACHE
    global MATERIALIZED_GAMES, CURRENT_SNAPSHOT, BASELINE_SNAPSHOT
    global SNAPSHOT_OUTPUT_DIRECTORY, MINIMUM_ELO, HISTORICAL_FORMAL_MAXIMUM
    global TOP_BIN_LOWER, BIN_WIDTH, CELL_COUNT, TARGET_PER_CELL, SELECTION_SEED
    global REQUEST_TIMEOUT, WORKERS, PLAYER_LIST_MAX_ATTEMPTS
    global DETAIL_MAX_ATTEMPTS, LEADERBOARD_MAX_ATTEMPTS, RUN_DATE, BATCH_ID
    global DEFAULT_OUTPUT, SNAPSHOT_NAME

    if config_path is None:
        return None
    config, resolved_path, config_sha = load_expansion_config(config_path)
    EXPANSION_CONFIG = config
    EXPANSION_CONFIG_PATH = resolved_path
    EXPANSION_CONFIG_SHA256 = config_sha
    CURRENT_REFERENCE = Path(config["paths"]["baselineSourceReference"])
    CURRENT_SENTINEL = Path(config["paths"]["baselineSentinelReference"])
    CURRENT_ELO = resolve_repo_path(config.get("baselinePlayerEloReference", CURRENT_ELO))
    RAW_CACHE = resolve_repo_path(config["localSources"]["rawCache"])
    materialized = resolve_repo_path(config["localSources"]["materializedDataset"])
    MATERIALIZED_GAMES = materialized / "games.csv" if materialized.is_dir() else materialized
    BASELINE_SNAPSHOT = Path(config["paths"]["baselineSnapshot"])
    CURRENT_SNAPSHOT = BASELINE_SNAPSHOT
    SNAPSHOT_OUTPUT_DIRECTORY = Path(config["paths"]["snapshotOutputDirectory"])
    MINIMUM_ELO = int(config["minimumElo"])
    HISTORICAL_FORMAL_MAXIMUM = int(config["historicalFormalMaximum"])
    BIN_WIDTH = int(config["binWidth"])
    CELL_COUNT = int(config["formalBinCount"])
    TOP_BIN_LOWER = MINIMUM_ELO + BIN_WIDTH * (CELL_COUNT - 1)
    TARGET_PER_CELL = int(config["targetPerBlackWhiteCell"])
    SELECTION_SEED = int(config["selectionSeed"])
    network = config["network"]
    REQUEST_TIMEOUT = float(network["requestTimeoutSeconds"])
    WORKERS = int(network["workers"])
    PLAYER_LIST_MAX_ATTEMPTS = int(network["playerListMaximumAttempts"])
    DETAIL_MAX_ATTEMPTS = int(network["detailMaximumAttempts"])
    LEADERBOARD_MAX_ATTEMPTS = int(network["leaderboardMaximumAttempts"])
    RUN_DATE = str(config["runDateAsiaShanghai"])
    BATCH_ID = str(config["batchId"])
    DEFAULT_OUTPUT = Path(config["paths"]["sourceOutputDirectory"])
    SNAPSHOT_NAME = SNAPSHOT_OUTPUT_DIRECTORY.name
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short: the Windows workspace is close to
    # MAX_PATH for the dated snapshot paths.
    temporary = path.parent / f".{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV without a schema: {path}")
    fields = list(rows[0])
    output = ["\ufeff" + ",".join(fields)]
    # csv.writer over a temporary file is clearer and preserves UTF-8 exactly.
    temporary = path.parent / f".{os.getpid()}.csv.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot_dir(output: Path) -> Path:
    if SNAPSHOT_OUTPUT_DIRECTORY is not None:
        return SNAPSHOT_OUTPUT_DIRECTORY
    return output / "provenance" / SNAPSHOT_NAME


def batch_path(output: Path) -> Path:
    return snapshot_dir(output) / "batch_manifest.json"


def baseline_reference_label() -> str:
    return CURRENT_REFERENCE.name


def load_batch(output: Path) -> dict[str, Any]:
    path = batch_path(output)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("batchId") != BATCH_ID:
            raise RuntimeError(f"existing output belongs to another batch: {value.get('batchId')!r}")
        if EXPANSION_CONFIG_SHA256 is not None and value.get("configSha256") != EXPANSION_CONFIG_SHA256:
            raise RuntimeError("existing output belongs to a different expansion configuration contract")
        return value
    if output.exists() and any(output.iterdir()):
        # The first implementation attempt can fail after writing only the
        # read-only local-audit artifacts and before its first checkpoint.  It
        # is safe to adopt precisely that known partial state; every other
        # non-empty directory is a conflicting batch and must not be touched.
        allowed = {
            "provenance/local_cache_audit/black_white_local_cache_audit_historical.json",
            "provenance/local_cache_audit/black_white_local_cache_audit_historical.csv",
            "provenance/local_cache_audit/black_white_local_cache_audit_dynamic.json",
            "provenance/local_cache_audit/black_white_local_cache_audit_dynamic.csv",
            "provenance/local_cache_validated_records.jsonl",
            "provenance/expansion_config.json",
            "provenance/lifecycle_progress.json",
            "provenance/local-audit.log",
            ".lifecycle/progress.json",
        }
        present = {str(path.relative_to(output)).replace("\\", "/") for path in output.rglob("*") if path.is_file()}
        if not present <= allowed:
            raise RuntimeError(f"refusing to use non-empty output without the expected batch manifest: {output}")
    return {
        "schema": "oq-reference-blackwhite-one-shot-batch-v2",
        "batchId": BATCH_ID,
        "configSha256": EXPANSION_CONFIG_SHA256,
        "runDateAsiaShanghai": RUN_DATE,
        "createdAtUtc": utc_now(),
        "networkPolicy": {
            "direct": True,
            "proxyDisabled": True,
            "httpClientMode": "urllib.ProxyHandler({})",
            "leaderboardFullBatchesAllowed": 1,
            "playerListFullBatchesAllowed": 1,
            "leaderboardMaximumAttempts": LEADERBOARD_MAX_ATTEMPTS,
            "playerListMaximumAttemptsPerUser": PLAYER_LIST_MAX_ATTEMPTS,
            "detailMaximumAttemptsPerGame": DETAIL_MAX_ATTEMPTS,
            "requestTimeoutSeconds": REQUEST_TIMEOUT,
            "workers": WORKERS,
            "resumePolicy": "same batch; successful pages/users/details are terminal and are never requested again",
        },
        "selection": {
            "seed": SELECTION_SEED,
            "sorting": ["canonicalBlackWhiteCellKey", "gameId", "sourcePriority", "stableSha256"],
            "canonicalization": "UTF-8 JSON with sorted keys and compact separators for summary hash",
            "sourcePriority": {
                "existingReference": 0,
                "validatedCacheExpansion": 1,
                "uniqueSnapshotExpansion": 2,
            },
        },
        "stages": {},
    }


def save_batch(output: Path, batch: dict[str, Any]) -> None:
    batch["updatedAtUtc"] = utc_now()
    atomic_json(batch_path(output), batch)


def baseline_config_check() -> dict[str, Any]:
    if EXPANSION_CONFIG is not None:
        sentinel_config_path = resolve_repo_path(EXPANSION_CONFIG.get("baselineSentinelConfigPath", "")) if EXPANSION_CONFIG.get("baselineSentinelConfigPath") else None
        elo_config_path = resolve_repo_path(EXPANSION_CONFIG.get("baselinePlayerEloConfigPath", "")) if EXPANSION_CONFIG.get("baselinePlayerEloConfigPath") else None
    else:
        sentinel_config_path = ROOT / "sentinel_reference_config.json"
        elo_config_path = ROOT / "sentinel_elo_reference_config.json"
    if sentinel_config_path is None or not sentinel_config_path.is_file():
        raise FileNotFoundError(f"baseline Sentinel configuration is missing: {sentinel_config_path}")
    if elo_config_path is None or not elo_config_path.is_file():
        raise FileNotFoundError(f"baseline Player Elo configuration is missing: {elo_config_path}")
    sentinel_config = json.loads(sentinel_config_path.read_text(encoding="utf-8")) if sentinel_config_path.is_file() else {}
    elo_config = json.loads(elo_config_path.read_text(encoding="utf-8")) if elo_config_path.is_file() else {}
    expected = {
        "sentinelReferenceDirectory": str(CURRENT_REFERENCE.resolve()),
        "sentinelDerivedDirectory": str(CURRENT_SENTINEL.resolve()),
        "eloSourceReferenceDirectory": str(CURRENT_REFERENCE.resolve()),
        "eloSentinelDerivedDirectory": str(CURRENT_SENTINEL.resolve()),
        "eloDerivedDirectory": str(CURRENT_ELO.resolve()),
    }
    actual = {
        "sentinelReferenceDirectory": str(resolve_repo_path(sentinel_config.get("referenceDirectory"))) if sentinel_config.get("referenceDirectory") else "",
        "sentinelDerivedDirectory": str(resolve_repo_path(sentinel_config.get("derivedDirectory"))) if sentinel_config.get("derivedDirectory") else "",
        "eloSourceReferenceDirectory": str(resolve_repo_path(elo_config.get("sourceReferenceDirectory"))) if elo_config.get("sourceReferenceDirectory") else "",
        "eloSentinelDerivedDirectory": str(resolve_repo_path(elo_config.get("sentinelDerivedDirectory"))) if elo_config.get("sentinelDerivedDirectory") else "",
        "eloDerivedDirectory": str(resolve_repo_path(elo_config.get("derivedReferenceDirectory"))) if elo_config.get("derivedReferenceDirectory") else "",
    }
    if actual != expected:
        raise RuntimeError(f"baseline configurations do not match the configured baseline: {actual} != {expected}")
    audit_paths = (
        CURRENT_REFERENCE / "reference_completion_audit.json",
        CURRENT_SENTINEL / "reference_build_audit.json",
        CURRENT_ELO / "reference_build_audit.json",
    )
    for path in (CURRENT_REFERENCE, CURRENT_SENTINEL, CURRENT_ELO, RAW_CACHE, MATERIALIZED_GAMES, CURRENT_SNAPSHOT, *audit_paths):
        if not path.exists():
            raise FileNotFoundError(path)
    for path, name in ((audit_paths[0], "source"), (audit_paths[1], "Sentinel"), (audit_paths[2], "Player Elo")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("ok") is not True:
            raise RuntimeError(f"baseline {name} audit is not ok=true: {path}")
    return {"sentinel": sentinel_config, "elo": elo_config, "paths": actual, "referenceResolution": (EXPANSION_CONFIG or {}).get("baselineResolutionAudit")}


def parse_rating(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def archive_bin(value: Any) -> int | None:
    rating = parse_rating(value)
    if rating is None:
        return None
    return int(math.floor(rating / BIN_WIDTH) * BIN_WIDTH)


def formal_bin(value: Any, maximum: int) -> int | None:
    rating = parse_rating(value)
    if rating is None or rating < MINIMUM_ELO or rating > maximum:
        return None
    return TOP_BIN_LOWER if rating >= TOP_BIN_LOWER else MINIMUM_ELO + int((rating - MINIMUM_ELO) // BIN_WIDTH) * BIN_WIDTH


def label(lower: int, maximum: int) -> str:
    return f"[{lower},{maximum}]" if lower == TOP_BIN_LOWER else f"[{lower},{lower + BIN_WIDTH})"


def formal_cells() -> list[tuple[int, int]]:
    lowers = list(range(MINIMUM_ELO, TOP_BIN_LOWER + 1, BIN_WIDTH))
    return [(black, white) for black in lowers for white in lowers]


def cell_for_players(players: Any, maximum: int) -> tuple[int, int] | None:
    if not isinstance(players, list) or len(players) != 2:
        return None
    black = formal_bin((players[0] or {}).get("oldR"), maximum)
    white = formal_bin((players[1] or {}).get("oldR"), maximum)
    return (black, white) if black is not None and white is not None else None


def actual_bin_pair(detail: dict[str, Any]) -> tuple[int | None, int | None]:
    players = detail.get("players") or []
    if not isinstance(players, list) or len(players) != 2:
        return None, None
    return archive_bin((players[0] or {}).get("oldR")), archive_bin((players[1] or {}).get("oldR"))


def row_for_detail(detail: dict[str, Any], source_kind: str, maximum: int, table: int | None = None) -> dict[str, Any]:
    players = detail.get("players") or []
    black = players[0] if len(players) == 2 else {}
    white = players[1] if len(players) == 2 else {}
    black_rating = parse_rating(black.get("oldR"))
    white_rating = parse_rating(white.get("oldR"))
    black_lower, white_lower = actual_bin_pair(detail)
    black_formal = formal_bin(black_rating, maximum)
    white_formal = formal_bin(white_rating, maximum)
    in_main = black_formal is not None and white_formal is not None
    low = not in_main and black_lower is not None and white_lower is not None and (black_lower < MINIMUM_ELO or white_lower < MINIMUM_ELO)
    scope = "main_bilateral" if in_main else "baseline_low_elo_extension" if low else "outside_unpartitioned"
    black_label = label(black_lower, maximum) if black_lower is not None and black_lower >= MINIMUM_ELO else f"[{black_lower}," if black_lower is not None else "outside_partition_range"
    white_label = label(white_lower, maximum) if white_lower is not None and white_lower >= MINIMUM_ELO else f"[{white_lower}," if white_lower is not None else "outside_partition_range"
    cell_key = f"{black_formal}__{white_formal}" if in_main else "outside_unpartitioned"
    result: dict[str, Any] = {
        "gameId": str(detail.get("id") or ""),
        "created": str(detail.get("created") or ""),
        "sourceKind": source_kind,
        "blackPlayerId": str(black.get("id") or ""),
        "blackOldR": black_rating,
        "blackBinLower": black_lower,
        "blackBinUpper": (maximum if black_lower == TOP_BIN_LOWER else black_lower + BIN_WIDTH) if black_lower is not None else None,
        "blackBinLabel": black_label,
        "whitePlayerId": str(white.get("id") or ""),
        "whiteOldR": white_rating,
        "whiteBinLower": white_lower,
        "whiteBinUpper": (maximum if white_lower == TOP_BIN_LOWER else white_lower + BIN_WIDTH) if white_lower is not None else None,
        "whiteBinLabel": white_label,
        "blackWhiteCellKey": cell_key,
        "blackWhitePartitionKey": cell_key,
        "partitionScope": scope,
        "inMainMatrix": in_main,
        "unorderedPartitionKey": (
            "__".join(sorted((label(black_formal, maximum), label(white_formal, maximum))))
            if in_main else "outside_unpartitioned"
        ),
        "blackTargetDirectedPartition": f"{label(black_formal, maximum)}__vs__{label(white_formal, maximum)}" if in_main else "outside_unpartitioned",
        "whiteTargetDirectedPartition": f"{label(white_formal, maximum)}__vs__{label(black_formal, maximum)}" if in_main else "outside_unpartitioned",
    }
    if table is not None:
        result["bundleTable"] = table
        result["expectedEngineFile"] = f"engine_level22/game_0_{table}_{result['gameId']}.json"
    return result


def canonical_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_sha(cell: tuple[int, int], game_id: str, source_kind: str, summary: dict[str, Any]) -> str:
    return stable_summary_sha(cell, game_id, source_kind, summary, SELECTION_SEED)


def validate_record(summary: dict[str, Any], detail: dict[str, Any]) -> tuple[bool, str]:
    if str(detail.get("id") or "") != str(summary.get("id") or ""):
        return False, "game_id_mismatch"
    ok, reason = valid_detail(summary, detail)
    if not ok:
        return False, reason
    if str(detail.get("gtype") or "") != GTYPE:
        return False, "wrong_game_type"
    if detail.get("finished") is not True:
        return False, "not_finished"
    summary_players = summary.get("players") or []
    detail_players = detail.get("players") or []
    if len(summary_players) != 2 or len(detail_players) != 2:
        return False, "invalid_players"
    summary_ids = [str(item.get("id") or "").strip().casefold() for item in summary_players]
    detail_ids = [str(item.get("id") or "").strip().casefold() for item in detail_players]
    if not all(detail_ids) or len(set(detail_ids)) != 2 or summary_ids != detail_ids:
        return False, "player_id_mismatch"
    for index in range(2):
        summary_rating = parse_rating(summary_players[index].get("oldR"))
        detail_rating = parse_rating(detail_players[index].get("oldR"))
        # OQ list summaries commonly expose integer oldR while the detail
        # endpoint preserves the decimal value.  The detail value is the
        # authoritative provenance; require the summary to be its rounded
        # representation rather than demanding byte-for-byte equality.
        if summary_rating is None or detail_rating is None or abs(summary_rating - detail_rating) >= 1.0:
            return False, "oldR_provenance_mismatch"
    if not replay_is_legal(detail):
        return False, "illegal_replay"
    return True, "ok"


def load_baseline() -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    config = baseline_config_check()
    bundle = json.loads((CURRENT_REFERENCE / "selected_account_bundle.json").read_text(encoding="utf-8"))
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    indexes = bundle.get("index") if isinstance(bundle.get("index"), list) else []
    if len(details) != len(indexes) or not details:
        raise RuntimeError(f"{baseline_reference_label()} bundle details/index lengths disagree")
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for summary, detail in zip(indexes, details, strict=True):
        game_id = str(detail.get("id") or "")
        if not game_id or game_id in records:
            raise RuntimeError(f"{baseline_reference_label()} contains an empty or duplicate gameId")
        ok, reason = validate_record(summary, detail)
        if not ok:
            raise RuntimeError(f"{baseline_reference_label()} game {game_id} failed contract validation: {reason}")
        records[game_id] = (summary, detail)
    selection_rows = json.loads((CURRENT_REFERENCE / "selected_games_with_partitions.json").read_text(encoding="utf-8")).get("games") or []
    row_ids = [str(row.get("gameId") or "") for row in selection_rows]
    if set(row_ids) != set(records) or len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"{baseline_reference_label()} selected-games partition map does not match bundle IDs")
    low_ids = {str(row["gameId"]) for row in selection_rows if row.get("partitionScope") == "baseline_low_elo_extension"}
    expected_total = int((EXPANSION_CONFIG or {}).get("baselineGameCount", len(records)))
    expected_low = int((EXPANSION_CONFIG or {}).get("baselineLowEloExtensionCount", 116))
    if len(records) != expected_total:
        raise RuntimeError(f"configured baseline game count changed: {len(records)} != {expected_total}")
    if len(low_ids) != expected_low:
        raise RuntimeError(f"configured baseline low-Elo extension count changed: {len(low_ids)} != {expected_low}")
    expected_main = int((EXPANSION_CONFIG or {}).get("baselineMainMatrixGameCount", len(records) - len(low_ids)))
    if sum(bool(row.get("inMainMatrix")) for row in selection_rows) != expected_main:
        raise RuntimeError("configured baseline formal directed matrix count changed")
    for path, key in ((CURRENT_SENTINEL / "reference_build_audit.json", "Sentinel"), (CURRENT_ELO / "reference_build_audit.json", "Player Elo")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("ok") is not True:
            raise RuntimeError(f"current {key} audit is not ok=true")
    return records, {"bundle": bundle, "rows": selection_rows, "lowIds": low_ids, "config": config}, config


def scan_local_jsonl(path: Path, source_name: str) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], set[str], dict[str, Any]]:
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    invalid_ids: set[str] = set()
    counts: Counter[str] = Counter()
    if not path.is_file():
        return records, invalid_ids, {"path": str(path.resolve()), "exists": False, "counts": {}}
    for row in read_jsonl(path):
        counts["jsonlLines"] += 1
        game_id = str(row.get("game_id") or row.get("gameId") or "").strip()
        if not game_id:
            counts["missingGameId"] += 1
            continue
        if row.get("valid") is False:
            counts[f"knownInvalid:{row.get('reason') or 'unknown'}"] += 1
            invalid_ids.add(game_id)
            continue
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else row.get("game") if isinstance(row.get("game"), dict) else {}
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        if not summary or not detail:
            counts["missingSummaryOrDetail"] += 1
            invalid_ids.add(game_id)
            continue
        ok, reason = validate_record(summary, detail)
        if not ok:
            counts[f"contractFailure:{reason}"] += 1
            invalid_ids.add(game_id)
            continue
        if game_id in records:
            counts["duplicateGameIdWithinSource"] += 1
            continue
        records[game_id] = (summary, detail)
        counts["validUnique"] += 1
    return records, invalid_ids, {"path": str(path.resolve()), "exists": True, "sha256": sha256_file(path), "counts": dict(counts), "validUniqueCount": len(records), "knownInvalidCount": len(invalid_ids)}


def local_source_paths() -> list[tuple[str, Path]]:
    paths = [("rawCache", RAW_CACHE), ("priorFrozenSnapshot", BASELINE_SNAPSHOT / "acquisition" / "game_details.jsonl")]
    for path in sorted(CURRENT_REFERENCE.rglob("game_details.jsonl")):
        if path != paths[1][1]:
            paths.append((f"currentReferenceProvenance:{path.relative_to(CURRENT_REFERENCE)}", path))
    # The materialized directory is audited separately as an inventory below;
    # its authoritative detail JSONL files, when present, are still eligible
    # local candidates through the same global gameId de-duplication path.
    materialized_root = MATERIALIZED_GAMES.parent
    for path in (sorted(materialized_root.rglob("*.jsonl")) if materialized_root.is_dir() else []):
        paths.append((f"materializedDataset:{path.relative_to(materialized_root)}", path))
    return paths


def collect_local_union(baseline: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any], str]], set[str], dict[str, Any]]:
    union: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    invalid_ids: set[str] = set()
    source_audit: list[dict[str, Any]] = []
    for source_name, path in local_source_paths():
        records, invalid, audit = scan_local_jsonl(path, source_name)
        invalid_ids.update(invalid)
        duplicate = 0
        for game_id, (summary, detail) in records.items():
            if game_id in union:
                duplicate += 1
                continue
            union[game_id] = (summary, detail, source_name)
        audit["sourceName"] = source_name
        audit["newAfterGlobalDedupCount"] = len(records) - duplicate
        audit["duplicateAgainstEarlierSources"] = duplicate
        source_audit.append(audit)
    conflicts = invalid_ids & set(baseline)
    if conflicts:
        raise RuntimeError(f"baseline gameIds are marked invalid by a local provenance source: {sorted(conflicts)[:5]}")
    for game_id in invalid_ids:
        union.pop(game_id, None)
    materialized_ids: set[str] = set()
    if MATERIALIZED_GAMES.is_file():
        with MATERIALIZED_GAMES.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                game_id = str(row.get("game_id") or row.get("gameId") or row.get("id") or "").strip()
                if game_id:
                    materialized_ids.add(game_id)
    audit = {
        "sources": source_audit,
        "materializedDataset": {
            "path": str(MATERIALIZED_GAMES.resolve()),
            "sha256": sha256_file(MATERIALIZED_GAMES),
            "uniqueGameIdCount": len(materialized_ids),
            "withValidatedDetailInLocalUnion": len(materialized_ids & set(union)),
        },
        "global": {
            "validatedUniqueLocalGameCount": len(union),
            "knownInvalidGameIdCount": len(invalid_ids),
            "baselineGameCount": len(baseline),
            "baselineOverlapCount": len(set(baseline) & set(union)),
        },
    }
    return union, invalid_ids, audit


def cell_audit_template(black: int, white: int, maximum: int) -> dict[str, Any]:
    return {
        "blackLower": black,
        "whiteLower": white,
        "blackBinLower": black,
        "whiteBinLower": white,
        "blackBinLabel": label(black, maximum),
        "whiteBinLabel": label(white, maximum),
        "blackLabel": label(black, maximum),
        "whiteLabel": label(white, maximum),
        "partitionDimension": "black_white_directed",
        "partitionScope": "main_bilateral",
        "targetPerBlackWhiteCell": TARGET_PER_CELL,
        "existingCount": 0,
        "baselineReferenceCount": 0,
        "localCacheAvailableCount": 0,
        "localCacheDedupBeforeCount": 0,
        "localCacheDedupAfterCount": 0,
        "localCacheQualifiedCapacity": 0,
        "localCacheSelectedCount": 0,
        "localCacheNewSelectionCount": 0,
        "usedCacheCount": 0,
        "priorSnapshotAvailableCount": 0,
        "priorSnapshotSelectedCount": 0,
        "snapshotSummaryCandidateCount": 0,
        "newSnapshotSummaryCandidateCount": 0,
        "snapshotDetailConsideredCount": 0,
        "newSnapshotDetailConsideredCount": 0,
        "snapshotValidVerifiedCount": 0,
        "newSnapshotValidVerifiedCount": 0,
        "newSnapshotSelectedCount": 0,
        "snapshotNewSelectionCount": 0,
        "finalCount": 0,
        "remainingGap": TARGET_PER_CELL,
        "knownInvalidCount": 0,
        "duplicateCount": 0,
        "contractFailureCount": 0,
        "capacityExhaustedBelowTarget": False,
        "mergedGameIds": [],
    }


def write_local_audit(output: Path, maximum: int, baseline: dict[str, tuple[dict[str, Any], dict[str, Any]]], union: dict[str, tuple[dict[str, Any], dict[str, Any], str]], invalid_ids: set[str], audit: dict[str, Any], name: str) -> None:
    cells = {(black, white): cell_audit_template(black, white, maximum) for black, white in formal_cells()}
    baseline_by = defaultdict(set)
    local_by = defaultdict(set)
    prior_by = defaultdict(set)
    for game_id, (_, detail) in baseline.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            baseline_by[cell].add(game_id)
    for game_id, (_, detail, source_name) in union.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            local_by[cell].add(game_id)
            if source_name == "priorFrozenSnapshot":
                prior_by[cell].add(game_id)
    for cell, row in cells.items():
        row["baselineReferenceCount"] = len(baseline_by[cell])
        row["existingCount"] = len(baseline_by[cell])
        row["localCacheDedupBeforeCount"] = len(local_by[cell]) + len(baseline_by[cell])
        row["localCacheDedupAfterCount"] = len(local_by[cell] | baseline_by[cell])
        row["localCacheQualifiedCapacity"] = row["localCacheDedupAfterCount"]
        row["localCacheAvailableCount"] = max(0, row["localCacheQualifiedCapacity"] - row["existingCount"])
        row["priorSnapshotAvailableCount"] = len(prior_by[cell] - baseline_by[cell])
        row["localCacheSelectedCount"] = max(
            0,
            min(TARGET_PER_CELL, row["localCacheQualifiedCapacity"]) - row["existingCount"],
        )
        row["localCacheNewSelectionCount"] = row["localCacheSelectedCount"]
        row["priorSnapshotSelectedCount"] = min(row["priorSnapshotAvailableCount"], row["localCacheSelectedCount"])
        row["usedCacheCount"] = row["localCacheDedupAfterCount"]
        row["finalCount"] = max(row["existingCount"], min(TARGET_PER_CELL, row["localCacheQualifiedCapacity"]))
        row["remainingGap"] = max(0, TARGET_PER_CELL - row["finalCount"])
        row["capacityExhaustedBelowTarget"] = row["finalCount"] < TARGET_PER_CELL and row["localCacheQualifiedCapacity"] < TARGET_PER_CELL
    payload = {
        "schema": "oq-reference-blackwhite-local-cache-audit-v3",
        "createdAtUtc": utc_now(),
        "maximumElo": maximum,
        "minimumElo": MINIMUM_ELO,
        "binWidth": BIN_WIDTH,
        "targetPerBlackWhiteCell": TARGET_PER_CELL,
        "seed": SELECTION_SEED,
        "network": {"direct": True, "proxyDisabled": True, "httpClientMode": "urllib.ProxyHandler({})"},
        "sourceAudit": audit,
        "knownInvalidGameIdCount": len(invalid_ids),
        "cells": list(cells.values()),
    }
    directory = output / "provenance" / "local_cache_audit"
    atomic_json(directory / f"{name}.json", payload)
    csv_rows = [{key: value for key, value in row.items() if key != "mergedGameIds"} for row in cells.values()]
    write_csv(directory / f"{name}.csv", csv_rows)


def command_local_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline, baseline_meta, config = load_baseline()
    if EXPANSION_CONFIG is not None:
        atomic_json(output / "provenance" / "expansion_config.json", EXPANSION_CONFIG)
    maximum = max(HISTORICAL_FORMAL_MAXIMUM, int(baseline_meta["bundle"].get("selection", {}).get("maximumElo") or HISTORICAL_FORMAL_MAXIMUM))
    union, invalid, audit = collect_local_union(baseline)
    write_local_audit(output, maximum, baseline, union, invalid, audit, "black_white_local_cache_audit_historical")
    batch = load_batch(output)
    batch["stages"]["localAudit"] = {
        "complete": True,
        "completedAtUtc": utc_now(),
        "maximumEloAtAudit": maximum,
        "baselineGameCount": len(baseline),
        "localValidatedUnionCount": len(union),
        "knownInvalidCount": len(invalid),
        "auditSha256": sha256_file(output / "provenance" / "local_cache_audit" / "black_white_local_cache_audit_historical.json"),
        "configSha256": EXPANSION_CONFIG_SHA256,
        "checks": {"baselineValidated": True, "allGameIdsUnique": True, "materializedInventoryRead": True},
    }
    save_batch(output, batch)
    print(json.dumps(batch["stages"]["localAudit"], ensure_ascii=False, indent=2))
    return 0


def load_leaderboard_users(path: Path) -> list[LeaderboardUser]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = [
            LeaderboardUser(
                rank=int(row["rank"]), user_id=row["id"], name=row["name"], rating=int(row["rating"]),
                page=int(row["page"]), index_on_page=int(row["index_on_page"]), stratum=int(row.get("stratum", 1)) - 1,
            ) for row in csv.DictReader(handle)
        ]
    by_id: dict[str, LeaderboardUser] = {}
    for user in sorted(parsed, key=lambda item: (item.rank, item.user_id)):
        by_id.setdefault(user.user_id, user)
    return list(by_id.values())


def command_leaderboard(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir(output)
    acquisition = snapshot / "acquisition"
    batch = load_batch(output)
    stage = batch["stages"].get("leaderboard") or {}
    if stage.get("complete"):
        # Repair only the derived leaderboard CSV if the frozen source pages
        # contained repeated account IDs.  The raw page evidence is retained.
        csv_path = snapshot / "leaderboard.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        unique_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in raw_rows:
            if row.get("id") in seen_ids:
                continue
            seen_ids.add(str(row.get("id") or ""))
            unique_rows.append(row)
        if len(unique_rows) != len(raw_rows):
            duplicate_ids = sorted({str(row.get("id") or "") for row in raw_rows if sum(1 for item in raw_rows if item.get("id") == row.get("id")) > 1})
            write_csv(csv_path, unique_rows)
            meta_path = snapshot / "leaderboard.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["rawRowCountBeforeUserIdDeduplication"] = len(raw_rows)
            meta["duplicateUserIdCount"] = len(duplicate_ids)
            meta["duplicateUserIds"] = duplicate_ids
            meta["count"] = len(unique_rows)
            atomic_json(meta_path, meta)
            stage["rawUserRowCount"] = len(raw_rows)
            stage["duplicateUserIdCount"] = len(duplicate_ids)
            stage["duplicateUserIds"] = duplicate_ids
            stage["userCount"] = len(unique_rows)
            stage["leaderboardSha256"] = sha256_file(meta_path)
            save_batch(output, batch)
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0
    stage.setdefault("startedAtUtc", utc_now())
    stage["fullBatchOrdinal"] = 1
    batch["stages"]["leaderboard"] = stage
    save_batch(output, batch)
    pages: dict[int, dict[str, Any]] = {}
    network_page_numbers: set[int] = set()
    terminal_failure_pages: set[int] = set()
    attempt_rows = list(read_jsonl(acquisition / "leaderboard_attempts.jsonl"))
    attempts_by_page: Counter[int] = Counter(int(row.get("page")) for row in attempt_rows if row.get("page") is not None)
    for row in read_jsonl(acquisition / "leaderboard_pages.jsonl"):
        if isinstance(row.get("data"), dict):
            pages[int(row["page"])] = row["data"]
    client = HttpClient(REQUEST_TIMEOUT, True)
    terminal_page: int | None = None
    page = 0
    while terminal_page is None:
        if page in pages:
            data = pages[page]
        else:
            data = None
            last_error = None
            standard_attempt_start = attempts_by_page[page]
            for attempt in range(standard_attempt_start + 1, LEADERBOARD_MAX_ATTEMPTS + 1):
                try:
                    data = client.leaderboard_page(page)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    append_jsonl(acquisition / "leaderboard_attempts.jsonl", {"atUtc": utc_now(), "page": page, "attempt": attempt, "ok": False, "error": last_error, "direct": True, "proxyDisabled": True})
                else:
                    append_jsonl(acquisition / "leaderboard_attempts.jsonl", {"atUtc": utc_now(), "page": page, "attempt": attempt, "ok": True, "direct": True, "proxyDisabled": True})
                    break
            # A missing page may receive one separately audited, longer direct
            # request.  This is a recovery of the same page, never a second
            # leaderboard scan.  It is deliberately one-shot.
            if data is None and not stage.get("finalExtendedTimeoutAttempted"):
                stage["finalExtendedTimeoutAttempted"] = True
                save_batch(output, batch)
                extended_client = HttpClient(60.0, True)
                try:
                    data = extended_client.leaderboard_page(page)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    append_jsonl(acquisition / "leaderboard_attempts.jsonl", {"atUtc": utc_now(), "page": page, "attempt": 6, "attemptKind": "final-extended-timeout", "ok": False, "error": last_error, "direct": True, "proxyDisabled": True, "timeoutSeconds": 60.0})
                else:
                    append_jsonl(acquisition / "leaderboard_attempts.jsonl", {"atUtc": utc_now(), "page": page, "attempt": 6, "attemptKind": "final-extended-timeout", "ok": True, "direct": True, "proxyDisabled": True, "timeoutSeconds": 60.0})
            if data is None:
                append_jsonl(acquisition / "leaderboard_pages.jsonl", {"page": page, "fetchedAtUtc": utc_now(), "terminal": True, "ok": False, "error": last_error, "direct": True, "proxyDisabled": True})
                stage.setdefault("terminalFailurePages", []).append(page)
        # Reuse the explicitly authorized baseline local
                # snapshot for pages beyond this objectively failed page.  The
                # failed page remains visible; no second full scan is started.
                prior_meta_path = CURRENT_SNAPSHOT / "leaderboard.json"
                prior_pages_path = CURRENT_SNAPSHOT / "acquisition" / "leaderboard_pages.jsonl"
                fallback_ok = False
                if prior_meta_path.is_file() and prior_pages_path.is_file():
                    prior_meta = json.loads(prior_meta_path.read_text(encoding="utf-8"))
                    prior_boundary = int(prior_meta.get("boundaryPage", -1))
                    prior_failures = {int(item) for item in prior_meta.get("terminalFailurePages") or []}
                    prior_pages = {int(row["page"]): row for row in read_jsonl(prior_pages_path) if isinstance(row.get("data"), dict)}
                    if (
                        prior_boundary >= page
                        and page in prior_pages
                        and all(
                            item in pages or item in prior_pages
                            for item in range(prior_boundary + 1)
                            if item != page
                        )
                    ):
                        for item in range(prior_boundary + 1):
                            if item == page or item in pages:
                                continue
                            pages[item] = prior_pages[item]["data"]
                            append_jsonl(acquisition / "leaderboard_pages.jsonl", {"page": item, "fetchedAtUtc": utc_now(), "terminal": True, "ok": True, "data": pages[item], "sourceKind": "baselineSnapshotReuse", "direct": True, "proxyDisabled": True})
                        terminal_failure_pages.add(page)
                        terminal_page = prior_boundary
                        stage["completeWithTerminalFailures"] = True
                        stage["reusedLocalSnapshotPages"] = [item for item in range(prior_boundary + 1) if item not in network_page_numbers and item != page]
                        stage["localFallbackSource"] = str(CURRENT_SNAPSHOT.resolve())
                        stage["localFallbackReason"] = "same-page terminal failure; previously frozen baseline pages reused without a second leaderboard batch"
                        fallback_ok = True
                if fallback_ok:
                    stage["complete"] = True
                    break
                stage["complete"] = False
                stage["terminalStatus"] = "failed_after_finite_retries"
                stage["lastFailureAtUtc"] = utc_now()
                save_batch(output, batch)
                raise RuntimeError(f"leaderboard page {page} failed after finite retries: {last_error}")
            pages[page] = data
            network_page_numbers.add(page)
            append_jsonl(acquisition / "leaderboard_pages.jsonl", {"page": page, "fetchedAtUtc": utc_now(), "terminal": True, "ok": True, "data": data, "direct": True, "proxyDisabled": True})
        ratings = [int(float(user.get("rating", 0))) for user in data.get("users") or []]
        if not ratings or ratings[-1] < MINIMUM_ELO:
            terminal_page = page
        page += 1
    rows: list[dict[str, Any]] = []
    page_audit: list[dict[str, Any]] = []
    for page_no in range(terminal_page + 1):
        if page_no in terminal_failure_pages:
            page_audit.append({"page": page_no, "terminalStatus": "failed_after_finite_retries", "sourceKind": "networkTerminalFailure", "count": None, "kept": None})
            continue
        data = pages[page_no]
        raw_users = data.get("users") or []
        ratings = [int(float(user.get("rating", 0))) for user in raw_users]
        kept = 0
        for index, raw in enumerate(raw_users):
            rating = int(float(raw.get("rating", 0)))
            if rating < MINIMUM_ELO:
                continue
            rows.append({"rank": int(data.get("start", page_no * len(raw_users))) + index + 1, "id": str(raw.get("id") or "").strip().lower(), "name": str(raw.get("name") or ""), "rating": rating, "page": page_no, "index_on_page": index})
            kept += 1
        page_audit.append({"page": page_no, "terminalStatus": "success", "sourceKind": "networkFrozen" if page_no in network_page_numbers else "baselineSnapshotReuse", "start": data.get("start"), "count": len(raw_users), "kept": kept, "first_rating": ratings[0] if ratings else None, "last_rating": ratings[-1] if ratings else None})
    rows.sort(key=lambda row: (row["rank"], row["id"]))
    for index, row in enumerate(rows):
        row["stratum"] = min((index * 10) // max(len(rows), 1), 9) + 1
    write_csv(snapshot / "leaderboard.csv", rows)
    atomic_json(snapshot / "leaderboard.json", {"schema": "oq-reference-blackwhite-leaderboard-snapshot-v2", "fetched_at": utc_now(), "cutoff_rating": MINIMUM_ELO, "count": len(rows), "boundaryPage": terminal_page, "terminalFailurePages": sorted(terminal_failure_pages), "pages": page_audit, "network": {"direct": True, "proxyDisabled": True, "httpClientMode": "urllib.ProxyHandler({})"}, "completenessPolicy": "one frozen batch; successful network pages plus explicitly recorded baseline local pages; failed page remains terminal and is not silently filled", "configSha256": EXPANSION_CONFIG_SHA256})
    maximum_observed = max(row["rating"] for row in rows)
    baseline_max = int(json.loads((CURRENT_REFERENCE / "selected_account_bundle.json").read_text(encoding="utf-8")).get("selection", {}).get("maximumElo") or HISTORICAL_FORMAL_MAXIMUM)
    dynamic_maximum = max(HISTORICAL_FORMAL_MAXIMUM, baseline_max, maximum_observed)
    stage.update({"complete": True, "completedAtUtc": utc_now(), "userCount": len(rows), "snapshotLeaderboardMaximum": maximum_observed, "dynamicMaximumElo": dynamic_maximum, "successfulPageCount": len(pages), "terminalFailurePages": sorted(terminal_failure_pages), "leaderboardSha256": sha256_file(snapshot / "leaderboard.json"), "rawPagesSha256": sha256_file(acquisition / "leaderboard_pages.jsonl"), "network": {"direct": True, "proxyDisabled": True, "httpClientMode": "urllib.ProxyHandler({})"}})
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def command_player_lists(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snapshot = snapshot_dir(output)
    batch = load_batch(output)
    if not (batch["stages"].get("leaderboard") or {}).get("complete"):
        raise RuntimeError("leaderboard must be frozen first")
    stage = batch["stages"].get("playerLists") or {}
    users = load_leaderboard_users(snapshot / "leaderboard.csv")
    if stage.get("complete"):
        final_by_user = {str(row.get("userId") or ""): row for row in read_jsonl(snapshot / "acquisition" / "player_lists.jsonl") if row.get("userId")}
        stage["expectedUserCount"] = len(users)
        stage["terminalUserCount"] = len(final_by_user)
        stage["successCount"] = sum(bool(row.get("ok")) for row in final_by_user.values())
        stage["failureCount"] = len(users) - stage["successCount"]
        stage["networkFetchedUserCount"] = max(0, len(users) - int(stage.get("localReusedSuccessCount") or 0))
        save_batch(output, batch)
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0
    stage.setdefault("startedAtUtc", utc_now())
    stage["fullBatchOrdinal"] = 1
    batch["stages"]["playerLists"] = stage
    save_batch(output, batch)
    acquisition = snapshot / "acquisition"
    final_path = acquisition / "player_lists.jsonl"
    attempts_path = acquisition / "player_list_attempts.jsonl"
    final_by_user = {str(row.get("userId") or ""): row for row in read_jsonl(final_path) if row.get("userId")}
    attempts_by_user: Counter[str] = Counter(str(row.get("userId") or "") for row in read_jsonl(attempts_path))
    # Reuse terminal successful records from the frozen baseline snapshot
    # before issuing any new request.  This is a local source merge, not a
    # second full player-list scan.
    prior_player_path = CURRENT_SNAPSHOT / "acquisition" / "player_lists.jsonl"
    prior_by_user = {str(row.get("userId") or ""): row for row in read_jsonl(prior_player_path) if row.get("userId") and row.get("ok")}
    local_reused = 0
    user_ids = {user.user_id for user in users}
    for user_id in sorted(user_ids & set(prior_by_user)):
        if user_id in final_by_user:
            continue
        prior = dict(prior_by_user[user_id])
        prior["sourceKind"] = "baselineSnapshotReuse"
        prior["sourceSnapshot"] = str(CURRENT_SNAPSHOT.resolve())
        prior["reusedAtUtc"] = utc_now()
        prior["directRequestThisBatch"] = False
        append_jsonl(final_path, prior)
        final_by_user[user_id] = prior
        local_reused += 1
    pending = [user for user in users if user.user_id not in final_by_user]
    lock = threading.Lock()
    client = HttpClient(REQUEST_TIMEOUT, True)

    def fetch(user: LeaderboardUser) -> tuple[LeaderboardUser, dict[str, Any] | None, str | None, int]:
        last_error = None
        attempt = attempts_by_user[user.user_id]
        for _ in range(attempt, PLAYER_LIST_MAX_ATTEMPTS):
            attempt += 1
            try:
                payload = client.json(f"{GAME_BASE_URL}/games/{GTYPE}/{user.user_id}.json")
                if not isinstance(payload, dict) or not isinstance(payload.get("games", []), list):
                    raise ValueError("player-list payload is not an object with a games list")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                with lock:
                    append_jsonl(attempts_path, {"atUtc": utc_now(), "userId": user.user_id, "attempt": attempt, "ok": False, "error": last_error, "direct": True, "proxyDisabled": True})
            else:
                with lock:
                    append_jsonl(attempts_path, {"atUtc": utc_now(), "userId": user.user_id, "attempt": attempt, "ok": True, "direct": True, "proxyDisabled": True})
                return user, payload, None, attempt
        return user, None, last_error or "failed after finite retries", attempt

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch, user): user for user in pending}
        for future in as_completed(futures):
            user, payload, error, attempts = future.result()
            row = {"ok": payload is not None, "fetchedAtUtc": utc_now(), "userId": user.user_id, "rank": user.rank, "rating": user.rating, "attempts": attempts, "direct": True, "proxyDisabled": True}
            if payload is not None:
                row["payload"] = payload
            else:
                row["error"] = error
            with lock:
                append_jsonl(final_path, row)
                final_by_user[user.user_id] = row
    if set(final_by_user) != {user.user_id for user in users}:
        raise RuntimeError("player-list snapshot does not have exactly one terminal row per frozen leaderboard user")
    success = sum(bool(row.get("ok")) for row in final_by_user.values())
    stage.update({"complete": True, "completedAtUtc": utc_now(), "expectedUserCount": len(users), "terminalUserCount": len(final_by_user), "successCount": success, "failureCount": len(users) - success, "localReusedSuccessCount": local_reused, "networkFetchedUserCount": len(users) - local_reused, "snapshotSha256": sha256_file(final_path), "attemptLogSha256": sha256_file(attempts_path), "network": {"direct": True, "proxyDisabled": True, "httpClientMode": "urllib.ProxyHandler({})"}})
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def load_snapshot_candidates(path: Path, baseline_ids: set[str], local_ids: set[str], invalid_ids: set[str], maximum: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: dict[str, dict[str, Any]] = {}
    sources: defaultdict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for player_row in read_jsonl(path):
        if not player_row.get("ok"):
            counts["failedPlayerLists"] += 1
            continue
        user_id = str(player_row.get("userId") or "")
        for summary in ((player_row.get("payload") or {}).get("games") or []):
            counts["listedSummaryRows"] += 1
            game_id = str(summary.get("id") or "").strip()
            if not game_id:
                counts["missingGameId"] += 1
                continue
            sources[game_id].add(user_id)
            if game_id in candidates:
                counts["duplicateSnapshotDiscovery"] += 1
                continue
            if game_id in baseline_ids:
                counts["deduplicatedBaselineReference"] += 1
                continue
            if game_id in local_ids:
                counts["deduplicatedValidatedLocalCache"] += 1
                continue
            if game_id in invalid_ids:
                counts["excludedKnownInvalid"] += 1
                continue
            if not normal_score(summary):
                counts["excludedNotScore"] += 1
                continue
            cell = cell_for_players(summary.get("players"), maximum)
            if cell is None:
                counts["excludedOutsideFormalRange"] += 1
                continue
            candidates[game_id] = {
                "gameId": game_id,
                "summary": summary,
                "blackBinLower": cell[0],
                "whiteBinLower": cell[1],
                "sourcePriority": source_priority("uniqueSnapshotExpansion"),
            }
    for game_id, row in candidates.items():
        row["discoveredFromUserIds"] = sorted(sources[game_id])
        row["stableSha256"] = stable_sha((int(row["blackBinLower"]), int(row["whiteBinLower"])), game_id, "uniqueSnapshotExpansion", row["summary"])
    ordered = sorted(candidates.values(), key=lambda row: (int(row["blackBinLower"]), int(row["whiteBinLower"]), row["gameId"], int(row["sourcePriority"]), row["stableSha256"]))
    return ordered, counts


def command_candidate_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snapshot = snapshot_dir(output)
    acquisition = snapshot / "acquisition"
    batch = load_batch(output)
    if not (batch["stages"].get("playerLists") or {}).get("complete"):
        raise RuntimeError("player lists must be frozen first")
    maximum = int(batch["stages"]["leaderboard"]["dynamicMaximumElo"])
    baseline, baseline_meta, _ = load_baseline()
    local, invalid, local_audit = collect_local_union(baseline)
    local_path = output / "provenance" / "local_cache_validated_records.jsonl"
    if not local_path.exists():
        temporary = local_path.with_name(f"{local_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for game_id in sorted(local):
                summary, detail, source_name = local[game_id]
                handle.write(json.dumps({"gameId": game_id, "sourceName": source_name, "summary": summary, "detail": detail}, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, local_path)
    baseline_by = defaultdict(set)
    local_by = defaultdict(set)
    for game_id, (_, detail) in baseline.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            baseline_by[cell].add(game_id)
    for game_id, (_, detail, _) in local.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            local_by[cell].add(game_id)
    candidates, counts = load_snapshot_candidates(acquisition / "player_lists.jsonl", set(baseline), set(local), invalid, maximum)
    by_cell: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cell[(int(row["blackBinLower"]), int(row["whiteBinLower"]))].append(row)
    cell_rows: list[dict[str, Any]] = []
    for black, white in formal_cells():
        cell = (black, white)
        row = cell_audit_template(black, white, maximum)
        row["baselineReferenceCount"] = len(baseline_by[cell])
        row["existingCount"] = row["baselineReferenceCount"]
        row["localCacheDedupBeforeCount"] = len(local_by[cell])
        row["localCacheDedupAfterCount"] = len(local_by[cell] | baseline_by[cell])
        row["localCacheQualifiedCapacity"] = row["localCacheDedupAfterCount"]
        row["localCacheAvailableCount"] = max(0, row["localCacheQualifiedCapacity"] - row["existingCount"])
        row["localCacheNewSelectionCount"] = max(0, min(TARGET_PER_CELL, row["localCacheQualifiedCapacity"]) - row["baselineReferenceCount"])
        row["localCacheSelectedCount"] = row["localCacheNewSelectionCount"]
        row["usedCacheCount"] = row["baselineReferenceCount"] + row["localCacheNewSelectionCount"]
        row["snapshotSummaryCandidateCount"] = len(by_cell[cell]) if row["usedCacheCount"] < TARGET_PER_CELL else 0
        row["newSnapshotSummaryCandidateCount"] = row["snapshotSummaryCandidateCount"]
        row["snapshotSummaryCandidateGameIds"] = [item["gameId"] for item in by_cell[cell]]
        row["maximumPossibleFinalCountBeforeDetails"] = max(row["baselineReferenceCount"], min(TARGET_PER_CELL, row["usedCacheCount"] + row["snapshotSummaryCandidateCount"]))
        row["remainingGapBeforeDetails"] = max(0, TARGET_PER_CELL - row["usedCacheCount"])
        cell_rows.append(row)
    atomic_json(acquisition / "candidate_dedup_audit.json", {"schema": "oq-reference-blackwhite-candidate-dedup-audit-v2", "createdAtUtc": utc_now(), "maximumElo": maximum, "targetPerBlackWhiteCell": TARGET_PER_CELL, "counts": dict(counts), "candidateCount": len(candidates), "candidates": candidates, "configSha256": EXPANSION_CONFIG_SHA256})
    atomic_json(acquisition / "black_white_capacity_before_details.json", {"schema": "oq-reference-blackwhite-capacity-before-details-v2", "createdAtUtc": utc_now(), "maximumElo": maximum, "targetPerBlackWhiteCell": TARGET_PER_CELL, "cells": cell_rows, "configSha256": EXPANSION_CONFIG_SHA256})
    write_csv(acquisition / "black_white_capacity_before_details.csv", [{key: value for key, value in row.items() if not isinstance(value, list)} for row in cell_rows])
    write_local_audit(output, maximum, baseline, local, invalid, local_audit, "black_white_local_cache_audit_dynamic")
    stage = {"complete": True, "completedAtUtc": utc_now(), "maximumElo": maximum, "candidateCount": len(candidates), "candidateAuditSha256": sha256_file(acquisition / "candidate_dedup_audit.json"), "network": {"direct": True, "proxyDisabled": True}}
    batch["stages"]["candidateAudit"] = stage
    save_batch(output, batch)
    print(json.dumps({**stage, "counts": dict(counts)}, ensure_ascii=False, indent=2))
    return 0


def command_details(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snapshot = snapshot_dir(output)
    acquisition = snapshot / "acquisition"
    batch = load_batch(output)
    if not (batch["stages"].get("candidateAudit") or {}).get("complete"):
        raise RuntimeError("candidate audit must be complete first")
    maximum = int(batch["stages"]["candidateAudit"]["maximumElo"])
    candidate_payload = json.loads((acquisition / "candidate_dedup_audit.json").read_text(encoding="utf-8"))
    candidates = candidate_payload.get("candidates") or []
    detail_path = acquisition / "game_details.jsonl"
    attempt_path = acquisition / "game_detail_attempts.jsonl"
    terminal_ids = {str(row.get("gameId") or "") for row in read_jsonl(detail_path) if row.get("gameId")}
    attempts_by_id: Counter[str] = Counter(str(row.get("gameId") or "") for row in read_jsonl(attempt_path))
    lock = threading.Lock()
    client = HttpClient(REQUEST_TIMEOUT, True)

    def fetch(candidate: dict[str, Any]) -> dict[str, Any]:
        game_id = str(candidate["gameId"])
        last_error = None
        detail = None
        attempt = attempts_by_id[game_id]
        while attempt < DETAIL_MAX_ATTEMPTS:
            attempt += 1
            try:
                value = client.json(f"{GAME_BASE_URL}/game/{game_id}.json")
                if not isinstance(value, dict):
                    raise ValueError("game detail payload is not an object")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                with lock:
                    append_jsonl(attempt_path, {"atUtc": utc_now(), "gameId": game_id, "attempt": attempt, "ok": False, "error": last_error, "direct": True, "proxyDisabled": True})
            else:
                detail = value
                with lock:
                    append_jsonl(attempt_path, {"atUtc": utc_now(), "gameId": game_id, "attempt": attempt, "ok": True, "direct": True, "proxyDisabled": True})
                break
        if detail is None:
            terminal = {"gameId": game_id, "valid": False, "reason": "request_failed_after_finite_retries", "error": last_error, "summary": candidate["summary"], "blackBinLower": candidate["blackBinLower"], "whiteBinLower": candidate["whiteBinLower"], "attempts": attempt}
        else:
            expected = (int(candidate["blackBinLower"]), int(candidate["whiteBinLower"]))
            ok, reason = validate_record(candidate["summary"], detail)
            actual = cell_for_players(detail.get("players"), maximum)
            if ok and actual != expected:
                ok, reason = False, "black_white_cell_mismatch"
            terminal = {"gameId": game_id, "valid": ok, "reason": reason, "summary": candidate["summary"], "detail": detail if ok else None, "blackBinLower": expected[0], "whiteBinLower": expected[1], "stableSha256": candidate.get("stableSha256"), "attempts": attempt, "fetchedAtUtc": utc_now()}
        with lock:
            append_jsonl(detail_path, terminal)
        return terminal

    stage = batch["stages"].get("details") or {}
    stage.setdefault("startedAtUtc", utc_now())
    batch["stages"]["details"] = stage
    save_batch(output, batch)
    pending_candidates = (candidate for candidate in candidates if str(candidate["gameId"]) not in terminal_ids)
    stage["alreadyTerminalCountAtStart"] = len(terminal_ids)
    stage["pendingCandidateCountAtStart"] = len(candidates) - len(terminal_ids)
    completed_count = len(terminal_ids)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures: dict[Any, dict[str, Any]] = {}
        for _ in range(min(WORKERS, stage["pendingCandidateCountAtStart"])):
            try:
                candidate = next(pending_candidates)
            except StopIteration:
                break
            futures[executor.submit(fetch, candidate)] = candidate
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
                futures.pop(future)
                completed_count += 1
                try:
                    candidate = next(pending_candidates)
                except StopIteration:
                    pass
                else:
                    futures[executor.submit(fetch, candidate)] = candidate
                if completed_count % 100 == 0:
                    stage["terminalDetailRecordCount"] = completed_count
                    stage["completedCandidateCount"] = completed_count - len(terminal_ids)
                    save_batch(output, batch)
    stage["terminalDetailRecordCount"] = completed_count
    stage["completedCandidateCount"] = completed_count - len(terminal_ids)
    save_batch(output, batch)
    baseline, _, _ = load_baseline()
    local, invalid, _ = collect_local_union(baseline)
    local_ids = set(local)
    cells: list[dict[str, Any]] = []
    terminal_rows = [row for row in read_jsonl(detail_path) if row.get("gameId")]
    terminal_ids = [str(row.get("gameId")) for row in terminal_rows]
    if len(terminal_ids) != len(set(terminal_ids)):
        raise RuntimeError("detail validation log contains duplicate terminal gameIds")
    candidate_ids = {str(row["gameId"]) for row in candidates}
    if set(terminal_ids) != candidate_ids:
        raise RuntimeError("detail validation did not reach a terminal result for every frozen candidate")
    terminal = {str(row.get("gameId")): row for row in terminal_rows}
    candidate_by_cell: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidate_by_cell[(int(row["blackBinLower"]), int(row["whiteBinLower"]))].append(row)
    baseline_by = defaultdict(set)
    local_by = defaultdict(set)
    for game_id, (_, detail) in baseline.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            baseline_by[cell].add(game_id)
    for game_id, (_, detail, _) in local.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            local_by[cell].add(game_id)
    for black, white in formal_cells():
        cell = (black, white)
        row = cell_audit_template(black, white, maximum)
        row["baselineReferenceCount"] = len(baseline_by[cell])
        row["existingCount"] = row["baselineReferenceCount"]
        row["localCacheDedupBeforeCount"] = len(local_by[cell])
        row["localCacheDedupAfterCount"] = len(local_by[cell] | baseline_by[cell])
        row["localCacheQualifiedCapacity"] = row["localCacheDedupAfterCount"]
        row["localCacheAvailableCount"] = max(0, row["localCacheQualifiedCapacity"] - row["existingCount"])
        row["localCacheNewSelectionCount"] = max(0, min(TARGET_PER_CELL, row["localCacheQualifiedCapacity"]) - row["baselineReferenceCount"])
        row["localCacheSelectedCount"] = row["localCacheNewSelectionCount"]
        row["usedCacheCount"] = row["baselineReferenceCount"] + row["localCacheNewSelectionCount"]
        cands = candidate_by_cell[cell]
        row["snapshotSummaryCandidateCount"] = len(cands)
        row["newSnapshotSummaryCandidateCount"] = len(cands)
        row["snapshotDetailConsideredCount"] = len(cands)
        row["newSnapshotDetailConsideredCount"] = len(cands)
        good = [item for item in cands if terminal.get(item["gameId"], {}).get("valid")]
        row["snapshotValidVerifiedCount"] = len(good)
        row["newSnapshotValidVerifiedCount"] = len(good)
        invalid_reasons = Counter(str(terminal.get(item["gameId"], {}).get("reason") or "unknown") for item in cands if not terminal.get(item["gameId"], {}).get("valid"))
        row["knownInvalidCount"] = sum(1 for item in cands if item["gameId"] in invalid)
        row["duplicateCount"] = int(candidate_payload.get("counts", {}).get("duplicateSnapshotDiscovery", 0)) if False else 0
        row["contractFailureCount"] = sum(value for reason, value in invalid_reasons.items() if reason != "request_failed_after_finite_retries")
        frozen_capacity = len((baseline_by[cell] | local_by[cell]) | {item["gameId"] for item in good})
        row["frozenValidCapacity"] = frozen_capacity
        row["targetFinalCount"] = max(row["baselineReferenceCount"], min(TARGET_PER_CELL, frozen_capacity))
        row["remainingGap"] = max(0, TARGET_PER_CELL - row["targetFinalCount"])
        row["capacityExhaustedBelowTarget"] = (
            row["targetFinalCount"] < TARGET_PER_CELL
            and set(terminal) >= {item["gameId"] for item in cands}
        )
        row["detailFailureReasons"] = dict(invalid_reasons)
        cells.append(row)
    atomic_json(acquisition / "black_white_capacity_after_details.json", {"schema": "oq-reference-blackwhite-capacity-after-details-v2", "createdAtUtc": utc_now(), "maximumElo": maximum, "targetPerBlackWhiteCell": TARGET_PER_CELL, "cells": cells, "network": {"direct": True, "proxyDisabled": True}, "configSha256": EXPANSION_CONFIG_SHA256})
    write_csv(acquisition / "black_white_capacity_after_details.csv", [{key: value for key, value in row.items() if not isinstance(value, (list, dict))} for row in cells])
    stage.update({"complete": True, "completedAtUtc": utc_now(), "maximumElo": maximum, "terminalDetailRecordCount": len(terminal), "snapshotCandidateCount": len(candidates), "snapshotValidVerifiedCount": sum(bool(row.get("valid")) for row in terminal.values()), "detailSnapshotSha256": sha256_file(detail_path), "detailAttemptLogSha256": sha256_file(attempt_path), "capacityAuditSha256": sha256_file(acquisition / "black_white_capacity_after_details.json"), "network": {"direct": True, "proxyDisabled": True}})
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def unordered_cells() -> list[tuple[int, int]]:
    lowers = list(range(MINIMUM_ELO, TOP_BIN_LOWER + 1, BIN_WIDTH))
    return [(a, b) for index, a in enumerate(lowers) for b in lowers[index:]]


def command_materialize(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snapshot = snapshot_dir(output)
    acquisition = snapshot / "acquisition"
    batch = load_batch(output)
    if not (batch["stages"].get("details") or {}).get("complete"):
        raise RuntimeError("details must be complete before materialization")
    maximum = int(batch["stages"]["details"]["maximumElo"])
    baseline, baseline_meta, _ = load_baseline()
    local, invalid, local_audit = collect_local_union(baseline)
    capacity_rows = { (int(row["blackBinLower"]), int(row["whiteBinLower"])): row for row in json.loads((acquisition / "black_white_capacity_after_details.json").read_text(encoding="utf-8")).get("cells", []) }
    terminal = {str(row.get("gameId")): row for row in read_jsonl(acquisition / "game_details.jsonl") if row.get("gameId")}
    baseline_by: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    local_by: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    prior_by: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    snapshot_by: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for game_id, (_, detail) in baseline.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell:
            baseline_by[cell].append(game_id)
    for game_id, (_, detail, source_name) in local.items():
        cell = cell_for_players(detail.get("players"), maximum)
        if cell and game_id not in baseline:
            local_by[cell].append(game_id)
            if source_name == "priorFrozenSnapshot":
                prior_by[cell].append(game_id)
    for game_id, row in terminal.items():
        if row.get("valid"):
            cell = (int(row["blackBinLower"]), int(row["whiteBinLower"]))
            snapshot_by[cell].append(game_id)
    source_by: dict[str, str] = {game_id: "existingReference" for game_id in baseline}
    detail_by: dict[str, dict[str, Any]] = {game_id: detail for game_id, (_, detail) in baseline.items()}
    summary_by: dict[str, dict[str, Any]] = {game_id: summary for game_id, (summary, _) in baseline.items()}
    selected_by_cell: dict[tuple[int, int], dict[str, list[str]]] = {}
    for cell in formal_cells():
        cap = capacity_rows[cell]
        existing = sorted(set(baseline_by[cell]))
        target = int(cap["targetFinalCount"])
        local_candidates = sorted(set(local_by[cell]), key=lambda game_id: (game_id, 1, stable_sha(cell, game_id, "validatedCacheExpansion", local[game_id][0])))
        local_need = max(0, target - len(existing))
        local_selected = local_candidates[:local_need]
        after_local = len(existing) + len(local_selected)
        snapshot_candidates = sorted(set(snapshot_by[cell]) - set(existing) - set(local_selected), key=lambda game_id: (game_id, 2, terminal[game_id].get("stableSha256") or stable_sha(cell, game_id, "uniqueSnapshotExpansion", terminal[game_id]["summary"])))
        snapshot_selected = snapshot_candidates[:max(0, target - after_local)]
        selected_by_cell[cell] = {"existing": existing, "local": local_selected, "snapshot": snapshot_selected}
        for game_id in local_selected:
            source_by[game_id] = "validatedCacheExpansion"
            detail_by[game_id] = local[game_id][1]
            summary_by[game_id] = local[game_id][0]
        for game_id in snapshot_selected:
            source_by[game_id] = "uniqueSnapshotExpansion"
            detail_by[game_id] = terminal[game_id]["detail"]
            summary_by[game_id] = terminal[game_id]["summary"]
    selected_ids = set(baseline)
    for item in selected_by_cell.values():
        selected_ids.update(item["local"])
        selected_ids.update(item["snapshot"])
    if selected_ids & invalid:
        raise RuntimeError("known-invalid gameId selected during materialization")
    if len(selected_ids) != len(baseline) + sum(len(item["local"]) + len(item["snapshot"]) for item in selected_by_cell.values()):
        raise RuntimeError("source gameId was selected in more than one black-white cell")
    ordered_ids = sorted(selected_ids, key=lambda game_id: (str(detail_by[game_id].get("created") or ""), game_id))
    table_by_id = {game_id: index for index, game_id in enumerate(ordered_ids, start=1)}
    rows: list[dict[str, Any]] = []
    for game_id in ordered_ids:
        row = row_for_detail(detail_by[game_id], source_by[game_id], maximum, table_by_id[game_id])
        row["sourceProvenance"] = {"sourceName": baseline_reference_label() if source_by[game_id] == "existingReference" else local.get(game_id, (None, None, ""))[2] if source_by[game_id] == "validatedCacheExpansion" else "newFrozenSnapshotDetail"}
        row["sourcePriority"] = source_priority(source_by[game_id])
        row["stableSha256"] = stable_sha(
            (int(row["blackBinLower"]), int(row["whiteBinLower"])),
            game_id,
            source_by[game_id],
            summary_by[game_id],
        )
        rows.append(row)
    row_by_id = {row["gameId"]: row for row in rows}
    if set(row_by_id) != selected_ids or len(rows) != len(selected_ids):
        raise RuntimeError("materialized selected rows are not globally unique")
    baseline_low_current = baseline_meta["lowIds"]
    new_low = {row["gameId"] for row in rows if row["partitionScope"] == "baseline_low_elo_extension"}
    if new_low != baseline_low_current:
        raise RuntimeError("low-Elo historical extension was not preserved exactly")
    # Formal black-white matrix.
    bw_rows: list[dict[str, Any]] = []
    for black, white in formal_cells():
        cell = (black, white)
        selected = selected_by_cell[cell]
        merged = sorted([*selected["existing"], *selected["local"], *selected["snapshot"]])
        cap = capacity_rows[cell]
        row = {
            "blackLower": black, "whiteLower": white,
            "blackBinLower": black, "whiteBinLower": white,
            "blackLabel": label(black, maximum), "whiteLabel": label(white, maximum),
            "partitionDimension": "black_white_directed",
            "partitionScope": "main_bilateral",
            "targetPerBlackWhiteCell": TARGET_PER_CELL,
            "baselineReferenceCount": len(selected["existing"]),
            "existingCount": len(selected["existing"]),
            "localCacheQualifiedCapacity": int(cap["localCacheQualifiedCapacity"]),
            "localCacheAvailableCount": max(0, int(cap["localCacheQualifiedCapacity"]) - len(selected["existing"])),
            "localCacheNewAvailableCount": max(0, int(cap["localCacheQualifiedCapacity"]) - len(selected["existing"])),
            "localCacheSelectedCount": len(selected["local"]),
            "usedCacheCount": len(selected["existing"]) + len(selected["local"]),
            "priorSnapshotAvailableCount": len(set(prior_by[cell]) - set(selected["existing"])),
            "priorSnapshotSelectedCount": sum(
                1 for game_id in selected["local"]
                if local.get(game_id, (None, None, ""))[2] == "priorFrozenSnapshot"
            ),
            "snapshotSummaryCandidateCount": int(cap["snapshotSummaryCandidateCount"]),
            "newSnapshotSummaryCandidateCount": int(cap["snapshotSummaryCandidateCount"]),
            "snapshotDetailConsideredCount": int(cap["snapshotDetailConsideredCount"]),
            "newSnapshotDetailConsideredCount": int(cap["snapshotDetailConsideredCount"]),
            "snapshotValidVerifiedCount": int(cap["snapshotValidVerifiedCount"]),
            "newSnapshotValidVerifiedCount": int(cap["snapshotValidVerifiedCount"]),
            "knownInvalidCount": int(cap.get("knownInvalidCount", 0)),
            "duplicateCount": int(cap.get("duplicateCount", 0)),
            "contractFailureCount": int(cap.get("contractFailureCount", 0)),
            "detailFailureCount": int(sum(int(value) for reason, value in (cap.get("detailFailureReasons") or {}).items() if reason != "request_failed_after_finite_retries")),
            "detailRequestFailureCount": int((cap.get("detailFailureReasons") or {}).get("request_failed_after_finite_retries", 0)),
            "snapshotSelectedCount": len(selected["snapshot"]),
            "newSnapshotSelectedCount": len(selected["snapshot"]),
            "finalSelectionCount": len(selected["local"]) + len(selected["snapshot"]),
            "finalCount": len(merged),
            "remainingGap": max(0, TARGET_PER_CELL - len(merged)),
            "capacityExhaustedBelowTarget": bool(cap["capacityExhaustedBelowTarget"]),
            "capacityFormula": "max(existingCount, min(targetPerBlackWhiteCell, frozenValidCapacity))",
            "mergedGameIds": merged,
        }
        if len(merged) != int(cap["targetFinalCount"]):
            raise RuntimeError(f"cell {cell} disagrees with frozen target formula")
        bw_rows.append(row)
    # Legacy unordered summary is derived from the real directed source cells.
    by_cell_ids = {(int(row["blackLower"]), int(row["whiteLower"])): set(row["mergedGameIds"]) for row in bw_rows}
    unordered: list[dict[str, Any]] = []
    for a, b in unordered_cells():
        ids = sorted(by_cell_ids[(a, b)] if a == b else by_cell_ids[(a, b)] | by_cell_ids[(b, a)])
        unordered.append({"pairLowerA": a, "pairLowerB": b, "pairLabelA": label(a, maximum), "pairLabelB": label(b, maximum), "partitionDimension": "unordered_legacy_summary", "partitionScope": "main_bilateral", "targetPerUnorderedPair": TARGET_PER_CELL, "finalCount": len(ids), "remainingGap": max(0, TARGET_PER_CELL - len(ids)), "mergedGameIds": ids, "blackWhiteCellKeys": [f"{a}__{b}"] if a == b else [f"{a}__{b}", f"{b}__{a}"]})
    low_by_pair: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for game_id in sorted(baseline_low_current):
        row = row_by_id[game_id]
        if row["blackBinLower"] is None or row["whiteBinLower"] is None:
            continue
        low_by_pair[tuple(sorted((int(row["blackBinLower"]), int(row["whiteBinLower"]))))].append(game_id)
    for (a, b), ids in sorted(low_by_pair.items()):
        unordered.append({"pairLowerA": a, "pairLowerB": b, "pairLabelA": f"[{a},", "pairLabelB": f"[{b},", "partitionDimension": "unordered_legacy_summary", "partitionScope": "baseline_low_elo_extension", "targetPerUnorderedPair": None, "finalCount": len(ids), "remainingGap": 0, "mergedGameIds": sorted(ids), "blackWhiteCellKeys": []})
    # target/opponent compatibility view; this is not a source partition.
    directed: list[dict[str, Any]] = []
    for target, opponent in [(a, b) for a in range(MINIMUM_ELO, TOP_BIN_LOWER + 1, BIN_WIDTH) for b in range(MINIMUM_ELO, TOP_BIN_LOWER + 1, BIN_WIDTH)]:
        ids = sorted(by_cell_ids[(target, opponent)] | by_cell_ids[(opponent, target)])
        directed.append({"targetBinLower": target, "opponentBinLower": opponent, "targetBinLabel": label(target, maximum), "opponentBinLabel": label(opponent, maximum), "partitionDimension": "target_opponent_compatibility", "sourcePartitionDimension": "black_white_directed", "partitionScope": "main_bilateral", "mergedGameCount": len(ids), "mergedGameIds": ids})
    for row in rows:
        if row["partitionScope"] != "baseline_low_elo_extension":
            continue
        for target, opponent in ((row["blackBinLower"], row["whiteBinLower"]), (row["whiteBinLower"], row["blackBinLower"])):
            if target is not None and opponent is not None:
                directed.append({"targetBinLower": target, "opponentBinLower": opponent, "targetBinLabel": f"[{target},", "opponentBinLabel": f"[{opponent},", "partitionDimension": "target_opponent_compatibility", "sourcePartitionDimension": "black_white_directed", "partitionScope": "baseline_low_elo_extension", "mergedGameCount": 1, "mergedGameIds": [row["gameId"]]})
    bundle = {
        "schema": "oq-account-bundle-elo-blackwhite-expansion-v2",
        "account": "elo_matchup_blackwhite_reference_multi_target",
        "fetchedAt": utc_now(),
        "selection": {"policy": "retain baseline; local validated cache and prior frozen snapshot first; unique frozen snapshot second; stable SHA-256", "seed": SELECTION_SEED, "targetDimension": "black_white_directed_cell", "targetPerBlackWhiteCell": TARGET_PER_CELL, "legacyUnorderedSummaryTargetPerPair": TARGET_PER_CELL, "minimumElo": MINIMUM_ELO, "maximumElo": maximum, "binWidth": BIN_WIDTH, "formalCellCount": CELL_COUNT * CELL_COUNT, "legacyUnorderedPairCount": CELL_COUNT * (CELL_COUNT + 1) // 2, "existingReferenceGameCount": len(baseline), "cacheSelectedCount": sum(len(item["local"]) for item in selected_by_cell.values()), "snapshotSelectedCount": sum(len(item["snapshot"]) for item in selected_by_cell.values()), "mergedGameCount": len(rows), "gameIds": ordered_ids, "sorting": ["canonicalBlackWhiteCellKey", "gameId", "sourcePriority", "stableSha256"], "canonicalization": "UTF-8 JSON with sorted keys and compact separators for summary hash"},
        "index": [summary_by[game_id] for game_id in ordered_ids],
        "details": [detail_by[game_id] for game_id in ordered_ids],
    }
    atomic_json(output / "selected_account_bundle.json", bundle)
    write_csv(output / "selected_games_with_partitions.csv", rows)
    atomic_json(output / "selected_games_with_partitions.json", {"schema": "oq-matchup-blackwhite-selected-games-partitions-v2", "games": rows, "configSha256": EXPANSION_CONFIG_SHA256})
    added_rows = [row for row in rows if row["sourceKind"] != "existingReference"]
    write_csv(output / "added_games.csv", added_rows)
    atomic_json(output / "added_games.json", {"schema": "oq-matchup-blackwhite-added-games-v2", "games": added_rows, "configSha256": EXPANSION_CONFIG_SHA256})
    write_csv(output / "partitions_black_white.csv", [{key: value for key, value in row.items() if key != "mergedGameIds"} for row in bw_rows])
    atomic_json(output / "partitions_black_white.json", {"schema": "oq-matchup-blackwhite-partitions-v2", "partitionDimension": "black_white_directed", "topLeftLabel": "黑棋\\白棋", "partitions": bw_rows, "configSha256": EXPANSION_CONFIG_SHA256})
    write_csv(output / "partitions_unordered.csv", [{key: value for key, value in row.items() if key != "mergedGameIds"} for row in unordered])
    atomic_json(output / "partitions_unordered.json", {"schema": "oq-matchup-blackwhite-unordered-legacy-summary-v2", "partitionDimension": "unordered_legacy_summary", "partitions": unordered, "configSha256": EXPANSION_CONFIG_SHA256})
    write_csv(output / "partitions_directed.csv", [{key: value for key, value in row.items() if key != "mergedGameIds"} for row in directed])
    atomic_json(output / "partitions_directed.json", {"schema": "oq-matchup-blackwhite-target-opponent-compatibility-v2", "partitionDimension": "target_opponent_compatibility", "notSourceBlackWhitePartition": True, "partitions": directed, "configSha256": EXPANSION_CONFIG_SHA256})
    atomic_json(output / "provenance" / "black_white_coverage_audit.json", {"schema": "oq-reference-blackwhite-coverage-audit-v2", "ok": True, "createdAtUtc": utc_now(), "maximumElo": maximum, "minimumElo": MINIMUM_ELO, "binWidth": BIN_WIDTH, "targetPerBlackWhiteCell": TARGET_PER_CELL, "formalCellCount": len(bw_rows), "formalMainMatrixGameCount": sum(row["finalCount"] for row in bw_rows), "lowEloExtensionGameCount": len(baseline_low_current), "cells": bw_rows, "checks": {"exactly81FormalCells": len(bw_rows) == CELL_COUNT * CELL_COUNT, "eachGameMapsToOneSourceCell": all(sum(game_id in ids for ids in by_cell_ids.values()) == 1 for game_id in [row["gameId"] for row in rows if row["inMainMatrix"]]), "capacityFormula": all(row["finalCount"] == max(row["existingCount"], min(TARGET_PER_CELL, int(capacity_rows[(row["blackLower"], row["whiteLower"])]["frozenValidCapacity"]))) for row in bw_rows)}, "selectionSeed": SELECTION_SEED, "calibrationSplitSeed": (EXPANSION_CONFIG or {}).get("calibrationSplitSeed")})
    write_csv(output / "provenance" / "black_white_coverage_audit.csv", [{key: value for key, value in row.items() if not isinstance(value, (list, dict))} for row in bw_rows])
    selection_audit = {"schema": "oq-reference-blackwhite-selection-audit-v2", "ok": True, "createdAtUtc": utc_now(), "gameCount": len(rows), "uniqueGameIdCount": len(selected_ids), "mainMatrixGameCount": sum(row["inMainMatrix"] for row in rows), "lowEloExtensionGameCount": len(baseline_low_current), "existingReferenceGameCount": len(baseline), "cacheSelectedCount": len(added_rows) - sum(1 for row in added_rows if row["sourceKind"] == "uniqueSnapshotExpansion"), "snapshotSelectedCount": sum(1 for row in added_rows if row["sourceKind"] == "uniqueSnapshotExpansion"), "formalBlackWhiteCellCount": CELL_COUNT * CELL_COUNT, "cellsAtTarget": sum(row["finalCount"] >= TARGET_PER_CELL for row in bw_rows), "cellsCapacityExhaustedBelowTarget": sum(bool(row["capacityExhaustedBelowTarget"]) for row in bw_rows), "selectionSeed": SELECTION_SEED, "calibrationSplitSeed": (EXPANSION_CONFIG or {}).get("calibrationSplitSeed"), "checks": {"allBaselineGamesRetained": set(baseline) <= selected_ids, "lowEloExtensionPreservedExactly": new_low == baseline_low_current, "allGameIdsUnique": len(rows) == len(selected_ids), "allFormalCellsMeetFrozenFormula": all(row["finalCount"] >= TARGET_PER_CELL or row["capacityExhaustedBelowTarget"] for row in bw_rows), "unorderedIsDerivedOnly": True, "noFormalOutsideCell": not any(row["partitionScope"] == "outside_unpartitioned" for row in rows)}}
    selection_audit["ok"] = all(selection_audit["checks"].values())
    atomic_json(output / "selection_audit.json", selection_audit)
    expansion_manifest = {"schema": "oq-reference-blackwhite-expansion-manifest-v2", "createdAtUtc": utc_now(), "batchId": BATCH_ID, "configSha256": EXPANSION_CONFIG_SHA256, "calibrationSplitSeed": (EXPANSION_CONFIG or {}).get("calibrationSplitSeed"), "baselineResolutionAudit": (EXPANSION_CONFIG or {}).get("baselineResolutionAudit"), "configuration": {"minimumElo": MINIMUM_ELO, "historicalFormalMaximumElo": HISTORICAL_FORMAL_MAXIMUM, "dynamicFormalMaximumElo": maximum, "topBinLower": TOP_BIN_LOWER, "binWidth": BIN_WIDTH, "formalBinCount": CELL_COUNT, "targetDimension": "black_white_directed_cell", "targetPerBlackWhiteCell": TARGET_PER_CELL, "legacyUnorderedSummaryTargetPerPair": TARGET_PER_CELL, "selectionSeed": SELECTION_SEED, "selectionSorting": ["canonicalBlackWhiteCellKey", "gameId", "sourcePriority", "stableSha256"], "canonicalization": "UTF-8 JSON with sorted keys and compact separators for summary hash"}, "sources": {"baselineSourceReferenceDirectory": str(CURRENT_REFERENCE.resolve()), "baselineBundleSha256": sha256_file(CURRENT_REFERENCE / "selected_account_bundle.json"), "rawCache": str(RAW_CACHE.resolve()), "rawCacheSha256": sha256_file(RAW_CACHE), "materializedDataset": str(MATERIALIZED_GAMES.resolve()), "materializedDatasetSha256": sha256_file(MATERIALIZED_GAMES), "baselineSnapshotDirectory": str(CURRENT_SNAPSHOT.resolve()), "newSnapshotDirectory": str(snapshot.resolve()), "batchManifest": str(batch_path(output).resolve())}, "selectionAudit": "selection_audit.json", "blackWhiteCoverageAudit": "provenance/black_white_coverage_audit.json", "manifestRule": "self excluded from final SHA-256 manifests"}
    atomic_json(output / "expansion_manifest.json", expansion_manifest)
    stage = batch["stages"].get("materialize") or {}
    stage.update({"complete": True, "completedAtUtc": utc_now(), "maximumElo": maximum, "gameCount": len(rows), "mainMatrixGameCount": sum(row["inMainMatrix"] for row in rows), "lowEloExtensionGameCount": len(baseline_low_current), "cacheSelectedCount": sum(1 for row in added_rows if row["sourceKind"] == "validatedCacheExpansion"), "snapshotSelectedCount": sum(1 for row in added_rows if row["sourceKind"] == "uniqueSnapshotExpansion"), "formalCellCount": CELL_COUNT * CELL_COUNT, "selectionAuditSha256": sha256_file(output / "selection_audit.json"), "coverageAuditSha256": sha256_file(output / "provenance" / "black_white_coverage_audit.json"), "configSha256": EXPANSION_CONFIG_SHA256})
    batch["stages"]["materialize"] = stage
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["local-audit", "leaderboard", "player-lists", "candidate-audit", "details", "materialize"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="UTF-8 configuration-driven black-white expansion contract")
    args = parser.parse_args()
    output_was_default = args.output == DEFAULT_OUTPUT
    apply_expansion_config(args.config)
    if output_was_default and EXPANSION_CONFIG is not None:
        args.output = Path(EXPANSION_CONFIG["paths"]["sourceOutputDirectory"])
    return {"local-audit": command_local_audit, "leaderboard": command_leaderboard, "player-lists": command_player_lists, "candidate-audit": command_candidate_audit, "details": command_details, "materialize": command_materialize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
