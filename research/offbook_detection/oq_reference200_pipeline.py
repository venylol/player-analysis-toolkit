#!/usr/bin/env python3
"""Build the one-shot OQ Elo-pair Reference expansion toward 400 games per cell."""

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from build_elo_reference import load_inputs, replay_is_legal
from build_matchup30_level22_expansion import (
    archive_label,
    archive_pair_for_detail,
    selected_game_row,
)
from pull_oq_transformer_dataset import (
    GAME_BASE_URL,
    GTYPE,
    HttpClient,
    LeaderboardUser,
    fetch_leaderboard,
    normal_score,
    valid_detail,
)


DATA = ROOT / "research" / "offbook_detection" / "data"
OLD_REFERENCE = DATA / "oq_elo_matchup200_reference_level22_1600plus_20260815"
CACHE_DETAILS = DATA / "oq_transformer_100000_20260813" / "acquisition" / "game_details.jsonl"
MATERIALIZED_GAMES = DATA / "oq_transformer_61145_20260813" / "games.csv"
PRIOR_SNAPSHOT_DETAILS = OLD_REFERENCE / "provenance" / "oq_snapshot_20260815" / "acquisition" / "game_details.jsonl"
DEFAULT_OUTPUT = DATA / "oq_elo_matchup400_reference_level22_1600plus_20260815"
MINIMUM_ELO = 1600
HISTORICAL_MAXIMUM_ELO = 2486
PRIOR_FORMAL_MAXIMUM_ELO = 2495
TOP_BIN_LOWER = 2400
BIN_WIDTH = 100
TARGET_PER_PAIR = 400
SEED = 20260815400
WORKERS = 20
REQUEST_TIMEOUT = 20.0
PLAYER_LIST_MAX_ATTEMPTS = 3
REFERENCE_TAG = "400"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


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
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_game_key(pair: tuple[int, int], game_id: str, source: str) -> str:
    value = f"{SEED}|{pair[0]}|{pair[1]}|{source}|{game_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def bin_lowers() -> list[int]:
    return list(range(MINIMUM_ELO, TOP_BIN_LOWER + 1, BIN_WIDTH))


def unordered_pairs() -> list[tuple[int, int]]:
    lowers = bin_lowers()
    return [(first, second) for index, first in enumerate(lowers) for second in lowers[index:]]


def rating_bin(value: Any, maximum: int) -> int | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rating) or rating < MINIMUM_ELO or rating > maximum:
        return None
    if rating >= TOP_BIN_LOWER:
        return TOP_BIN_LOWER
    return MINIMUM_ELO + int((rating - MINIMUM_ELO) // BIN_WIDTH) * BIN_WIDTH


def pair_for_players(players: Any, maximum: int) -> tuple[int, int] | None:
    if not isinstance(players, list) or len(players) != 2:
        return None
    bins = [rating_bin(player.get("oldR"), maximum) for player in players]
    if bins[0] is None or bins[1] is None:
        return None
    return tuple(sorted((bins[0], bins[1])))


def pair_for_detail(detail: dict[str, Any], maximum: int) -> tuple[int, int] | None:
    return pair_for_players(detail.get("players"), maximum)


def pair_label(lower: int, maximum: int) -> str:
    upper = maximum if lower == TOP_BIN_LOWER else lower + BIN_WIDTH
    return f"[{lower},{upper}]" if lower == TOP_BIN_LOWER else f"[{lower},{upper})"


def load_reference() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bundle = json.loads((OLD_REFERENCE / "selected_account_bundle.json").read_text(encoding="utf-8"))
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    index = bundle.get("index") if isinstance(bundle.get("index"), list) else []
    details_by_id = {str(row.get("id") or ""): row for row in details}
    index_by_id = {str(row.get("id") or ""): row for row in index}
    if len(details_by_id) != 6020 or "" in details_by_id or set(index_by_id) != set(details_by_id):
        raise ValueError("old Reference bundle is not the expected unique 6,020-game source")
    for game_id, detail in details_by_id.items():
        ok, reason = valid_detail(index_by_id[game_id], detail)
        if not ok:
            raise ValueError(f"old Reference game {game_id} failed contract validation: {reason}")
        if not replay_is_legal(detail):
            raise ValueError(f"old Reference game {game_id} failed legal replay validation")
    return details_by_id, index_by_id


def validate_local_detail_file(
    path: Path,
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], set[str], Counter[str]]:
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    ineligible_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for row in read_jsonl(path):
        counts["jsonlLines"] += 1
        game_id = str(row.get("game_id") or row.get("gameId") or "").strip()
        if not game_id:
            counts["excludedMissingGameId"] += 1
            continue
        reason = str(row.get("reason") or "unknown")
        if not row.get("valid"):
            counts[f"excludedCacheMarkedInvalid:{reason}"] += 1
            if reason not in {"request_failed_after_finite_retries"}:
                ineligible_ids.add(game_id)
            continue
        if game_id in records:
            counts["excludedDuplicateJsonlGameId"] += 1
            continue
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        ok, reason = valid_detail(summary, detail)
        if not ok:
            counts[f"excludedRevalidation:{reason}"] += 1
            ineligible_ids.add(game_id)
            continue
        if str(detail.get("gtype") or "") != "reversi" or detail.get("finished") is not True:
            counts["excludedWrongTypeOrUnfinished"] += 1
            ineligible_ids.add(game_id)
            continue
        players = detail.get("players") or []
        player_ids = [str(player.get("id") or "").strip() for player in players]
        if len(players) != 2 or not all(player_ids) or player_ids[0].casefold() == player_ids[1].casefold():
            counts["excludedInvalidPlayerIds"] += 1
            ineligible_ids.add(game_id)
            continue
        if not replay_is_legal(detail):
            counts["excludedIllegalReplay"] += 1
            ineligible_ids.add(game_id)
            continue
        records[game_id] = (summary, detail)
        counts["validUniqueGames"] += 1
    return records, ineligible_ids, counts


def load_valid_cache(maximum: int) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    sources = [
        ("rawCache", CACHE_DETAILS),
        ("priorSnapshot", PRIOR_SNAPSHOT_DETAILS),
    ]
    merged: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    source_audit: dict[str, Any] = {}
    known_ineligible_ids: set[str] = set()
    for source_name, path in sources:
        records, ineligible_ids, counts = validate_local_detail_file(path)
        known_ineligible_ids.update(ineligible_ids)
        duplicate_count = sum(game_id in merged for game_id in records)
        for game_id, record in records.items():
            merged.setdefault(game_id, record)
        source_audit[source_name] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "validatedUniqueCount": len(records), "duplicateAgainstEarlierSources": duplicate_count,
            "newAfterGlobalDedupCount": len(records) - duplicate_count,
            "validationCounts": dict(counts),
            "knownIneligibleGameIdCount": len(ineligible_ids),
        }

    reference, reference_index = load_reference()
    duplicate_count = sum(game_id in merged for game_id in reference)
    for game_id, detail in reference.items():
        merged.setdefault(game_id, (reference_index[game_id], detail))
    source_audit["currentReference"] = {
        "path": str(OLD_REFERENCE.resolve()),
        "validatedUniqueCount": len(reference), "duplicateAgainstEarlierSources": duplicate_count,
        "newAfterGlobalDedupCount": len(reference) - duplicate_count,
    }

    conflicts = known_ineligible_ids & set(reference)
    if conflicts:
        raise ValueError(f"validated Reference conflicts with locally known ineligible IDs: {sorted(conflicts)[:5]}")
    for game_id in known_ineligible_ids:
        merged.pop(game_id, None)

    with MATERIALIZED_GAMES.open("r", encoding="utf-8", newline="") as handle:
        materialized_ids = {
            str(row.get("game_id") or row.get("gameId") or row.get("id") or "").strip()
            for row in csv.DictReader(handle)
        }
    materialized_ids.discard("")
    source_audit["materializedDataset"] = {
        "path": str(MATERIALIZED_GAMES.resolve()), "sha256": sha256_file(MATERIALIZED_GAMES),
        "uniqueGameIdCount": len(materialized_ids),
        "withValidatedDetailInMergedSources": len(materialized_ids & set(merged)),
        "withoutLocallyAvailableFullDetail": len(materialized_ids - set(merged)),
        "note": "games.csv is an index/materialization; only IDs with a full locally validated detail are eligible",
    }
    source_audit["global"] = {
        "validatedUniqueCount": len(merged),
        "knownIneligibleGameIdCount": len(known_ineligible_ids),
    }
    return merged, source_audit


def compute_cache_selection(maximum: int) -> dict[str, Any]:
    reference, _ = load_reference()
    cache, validation = load_valid_cache(maximum)
    existing_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    cache_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    low_ids = []
    for game_id, detail in reference.items():
        pair = pair_for_detail(detail, maximum)
        if pair is None:
            low_ids.append(game_id)
        else:
            existing_by_pair[pair].add(game_id)
    for game_id, (_, detail) in cache.items():
        pair = pair_for_detail(detail, maximum)
        if pair is not None:
            cache_by_pair[pair].add(game_id)

    rows = []
    selected_ids = []
    for pair in unordered_pairs():
        existing = existing_by_pair[pair]
        available = cache_by_pair[pair]
        if not existing <= available:
            raise ValueError(f"old Reference contains games absent from the validated local union: {pair}")
        final_count = max(len(existing), min(TARGET_PER_PAIR, len(available)))
        candidates = sorted(
            available - existing,
            key=lambda game_id: (stable_game_key(pair, game_id, "cache"), game_id),
        )
        selected = candidates[: final_count - len(existing)]
        selected_ids.extend(selected)
        rows.append({
            "pairLowerA": pair[0],
            "pairLowerB": pair[1],
            "pairLabelA": pair_label(pair[0], maximum),
            "pairLabelB": pair_label(pair[1], maximum),
            "existingCount": len(existing),
            "cacheAvailableCount": len(available),
            "cacheNewAvailableCount": len(available - existing),
            "cacheSelectedCount": len(selected),
            "postCacheCount": len(existing) + len(selected),
            "remainingGapToTarget": max(0, TARGET_PER_PAIR - len(existing) - len(selected)),
            "cacheCapacityExhaustedBelowTarget": len(available) < TARGET_PER_PAIR,
            "existingGameIds": sorted(existing),
            "cacheSelectedGameIds": selected,
        })
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("cache selection contains duplicate game IDs")
    return {
        "maximumElo": maximum,
        "reference": reference,
        "cache": cache,
        "validation": validation,
        "lowIds": low_ids,
        "selectedIds": selected_ids,
        "rows": rows,
    }


def write_cache_audit(output: Path, result: dict[str, Any], name: str) -> None:
    rows = result["rows"]
    json_rows = rows
    csv_rows = [{key: value for key, value in row.items() if not key.endswith("GameIds")} for row in rows]
    audit_dir = output / "provenance" / "local_cache_audit"
    atomic_json(audit_dir / f"{name}.json", {
        "schema": f"oq-reference{REFERENCE_TAG}-cache-pair-audit-v1",
        "createdAtUtc": utc_now(),
        "minimumElo": MINIMUM_ELO,
        "maximumElo": result["maximumElo"],
        "binWidth": BIN_WIDTH,
        "targetPerUnorderedPair": TARGET_PER_PAIR,
        "seed": SEED,
        "cacheSource": str(CACHE_DETAILS.resolve()),
        "cacheSourceSha256": sha256_file(CACHE_DETAILS),
        "oldReference": str(OLD_REFERENCE.resolve()),
        "localSourceAudit": result["validation"],
        "summary": {
            "pairCount": len(rows),
            "referenceGameCount": len(result["reference"]),
            "mainMatrixGameCount": sum(row["existingCount"] for row in rows),
            "lowEloExtensionGameCount": len(result["lowIds"]),
            "cacheValidUniqueGameCount": len(result["cache"]),
            "newAfterReferenceDedupCount": len(set(result["cache"]) - set(result["reference"])),
            "cacheSelectedCount": len(result["selectedIds"]),
            "pairsAtTarget": sum(row["postCacheCount"] >= TARGET_PER_PAIR for row in rows),
            "pairsBelowTarget": sum(row["postCacheCount"] < TARGET_PER_PAIR for row in rows),
            "postCacheMainMatrixGameCount": sum(row["postCacheCount"] for row in rows),
            "postCacheReferenceGameCount": sum(row["postCacheCount"] for row in rows) + len(result["lowIds"]),
        },
        "pairs": json_rows,
    })
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def snapshot_dir(output: Path) -> Path:
    return output / "provenance" / "oq_snapshot_reference400_20260815"


def batch_manifest_path(output: Path) -> Path:
    return snapshot_dir(output) / "batch_manifest.json"


def load_batch(output: Path) -> dict[str, Any]:
    path = batch_manifest_path(output)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("batchId") != "oq-reference400-20260815-once":
            raise ValueError("conflicting OQ snapshot batch already exists")
        return value
    return {
        "schema": "oq-reference400-one-shot-batch-v1",
        "batchId": "oq-reference400-20260815-once",
        "createdAtUtc": utc_now(),
        "networkPolicy": {
            "leaderboardFullBatchesAllowed": 1,
            "playerListFullBatchesAllowed": 1,
            "playerListMaximumAttemptsPerUser": PLAYER_LIST_MAX_ATTEMPTS,
            "requestTimeoutSeconds": REQUEST_TIMEOUT,
            "workers": WORKERS,
            "resumePolicy": "resume missing items in this batch only; never rescan completed items",
        },
        "stages": {},
    }


def save_batch(output: Path, batch: dict[str, Any]) -> None:
    batch["updatedAtUtc"] = utc_now()
    atomic_json(batch_manifest_path(output), batch)


def command_local_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = compute_cache_selection(PRIOR_FORMAL_MAXIMUM_ELO)
    write_cache_audit(output, result, "cache_pair_audit_historical_max_2495")
    print(json.dumps({
        "ok": True,
        "cacheSelectedCount": len(result["selectedIds"]),
        "postCacheReferenceGameCount": sum(row["postCacheCount"] for row in result["rows"]) + len(result["lowIds"]),
    }, ensure_ascii=False, indent=2))
    return 0


def command_leaderboard(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    batch = load_batch(output)
    stage = batch["stages"].get("leaderboard") or {}
    if stage.get("complete"):
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0
    if stage.get("startedAtUtc") and not (snap / "acquisition" / "leaderboard_pages.jsonl").exists():
        raise RuntimeError("leaderboard stage was started but its page checkpoint is missing")
    stage.setdefault("startedAtUtc", utc_now())
    stage["batchOrdinal"] = 1
    prior_invocations = int(stage.get("invocationCount") or 0)
    if prior_invocations == 0 and (snap / "acquisition" / "leaderboard_pages.jsonl").exists():
        prior_invocations = 1
        stage.setdefault("failures", []).append({
            "atUtc": utc_now(),
            "invocation": 1,
            "error": "incomplete prior invocation; checkpoint inspection found missing page 24",
        })
    stage["invocationCount"] = prior_invocations + 1
    if stage["invocationCount"] > 1:
        stage["resumePolicyApplied"] = "same batch; cached successful pages skipped; missing pages only"
    batch["stages"]["leaderboard"] = stage
    save_batch(output, batch)
    client = HttpClient(REQUEST_TIMEOUT, True)
    try:
        users = fetch_leaderboard(client, snap, MINIMUM_ELO, WORKERS)
    except Exception as exc:
        stage.setdefault("failures", []).append({
            "atUtc": utc_now(),
            "invocation": stage["invocationCount"],
            "error": f"{type(exc).__name__}: {exc}",
        })
        stage["complete"] = False
        save_batch(output, batch)
        raise
    maximum = max(PRIOR_FORMAL_MAXIMUM_ELO, max(user.rating for user in users))
    stage.update({
        "complete": True,
        "completedAtUtc": utc_now(),
        "userCount": len(users),
        "snapshotLeaderboardMaximum": max(user.rating for user in users),
        "dynamicMaximumElo": maximum,
        "leaderboardCsvSha256": sha256_file(snap / "leaderboard.csv"),
        "leaderboardJsonSha256": sha256_file(snap / "leaderboard.json"),
        "rawPagesSha256": sha256_file(snap / "acquisition" / "leaderboard_pages.jsonl"),
    })
    save_batch(output, batch)
    dynamic = compute_cache_selection(maximum)
    write_cache_audit(output, dynamic, "cache_pair_audit_dynamic_max")
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def command_finalize_leaderboard(args: argparse.Namespace) -> int:
    """Freeze the attempted full batch after an objectively terminal page failure."""
    output = args.output.resolve()
    snap = snapshot_dir(output)
    batch = load_batch(output)
    stage = batch["stages"].get("leaderboard") or {}
    if stage.get("complete"):
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0
    pages: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(snap / "acquisition" / "leaderboard_pages.jsonl"):
        if isinstance(row.get("data"), dict):
            pages[int(row["page"])] = row["data"]
    boundary_candidates = []
    for page, data in pages.items():
        ratings = [int(float(user.get("rating", 0))) for user in data.get("users") or []]
        if not ratings or ratings[-1] < MINIMUM_ELO:
            boundary_candidates.append(page)
    if not boundary_candidates:
        raise ValueError("leaderboard boundary was never observed")
    boundary = min(boundary_candidates)
    missing = [page for page in range(boundary + 1) if page not in pages]
    if missing != [24]:
        raise ValueError(f"only the documented terminal page 24 failure may be frozen, got {missing}")
    retry_rows = list(read_jsonl(snap / "acquisition" / "leaderboard_retry_attempts.jsonl"))
    failed_transports = {
        str(row.get("attemptKind") or "") for row in retry_rows
        if int(row.get("page", -1)) == 24 and not row.get("ok")
    }
    if not {"final-extended-timeout", "protocol-response-diagnostic"} <= failed_transports:
        raise ValueError("page 24 finite retry evidence is incomplete")

    raw_users = []
    page_audit = []
    for page in range(boundary + 1):
        if page == 24:
            page_audit.append({
                "page": page, "terminalStatus": "failed_after_finite_retries",
                "count": None, "kept": None,
            })
            continue
        data = pages[page]
        users = data.get("users") or []
        ratings = [int(float(user.get("rating", 0))) for user in users]
        kept = 0
        for index, raw in enumerate(users):
            rating = int(float(raw.get("rating", 0)))
            if rating < MINIMUM_ELO:
                continue
            raw_users.append({
                "rank": int(data.get("start", page * len(users))) + index + 1,
                "id": str(raw.get("id") or "").strip().lower(),
                "name": str(raw.get("name") or ""), "rating": rating,
                "page": page, "index_on_page": index,
            })
            kept += 1
        page_audit.append({
            "page": page, "terminalStatus": "success", "start": data.get("start"),
            "count": len(users), "kept": kept,
            "first_rating": ratings[0] if ratings else None,
            "last_rating": ratings[-1] if ratings else None,
        })
    raw_users.sort(key=lambda row: (row["rank"], row["id"]))
    count = len(raw_users)
    for index, row in enumerate(raw_users):
        row["stratum"] = min((index * 10) // max(count, 1), 9) + 1
    with (snap / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "id", "name", "rating", "page", "index_on_page", "stratum",
        ])
        writer.writeheader()
        writer.writerows(raw_users)
    atomic_json(snap / "leaderboard.json", {
        "schema": "oq-reference400-leaderboard-terminal-status-snapshot-v1",
        "fetched_at": utc_now(), "cutoff_rating": MINIMUM_ELO, "count": count,
        "pages": page_audit, "boundaryPage": boundary,
        "terminalSuccessPageCount": boundary,
        "terminalFailurePages": [{
            "page": 24,
            "reason": "OQ returned no Socket.IO event after finite xhr-polling and websocket retries",
            "impact": "approximately 20 leaderboard accounts at ranks 481-500 are absent",
        }],
        "completenessPolicy": "every required page has a terminal success/failure status; no second full scan",
    })
    maximum_observed = max(row["rating"] for row in raw_users)
    maximum = max(PRIOR_FORMAL_MAXIMUM_ELO, maximum_observed)
    stage.update({
        "complete": True,
        "completeWithTerminalFailures": True,
        "completedAtUtc": utc_now(), "userCount": count,
        "snapshotLeaderboardMaximum": maximum_observed,
        "dynamicMaximumElo": maximum,
        "terminalFailurePages": [24],
        "leaderboardCsvSha256": sha256_file(snap / "leaderboard.csv"),
        "leaderboardJsonSha256": sha256_file(snap / "leaderboard.json"),
        "rawPagesSha256": sha256_file(snap / "acquisition" / "leaderboard_pages.jsonl"),
        "retryLogSha256": sha256_file(snap / "acquisition" / "leaderboard_retry_attempts.jsonl"),
    })
    save_batch(output, batch)
    dynamic = compute_cache_selection(maximum)
    write_cache_audit(output, dynamic, "cache_pair_audit_dynamic_max")
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def command_integrate_recovered_page(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    acquisition = snap / "acquisition"
    recovery_path = acquisition / "leaderboard_page_24_direct_recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if int(recovery.get("page", -1)) != 24 or not isinstance(recovery.get("data"), dict):
        raise ValueError("invalid recovered leaderboard page 24 payload")
    pages: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(acquisition / "leaderboard_pages.jsonl"):
        if isinstance(row.get("data"), dict):
            pages[int(row["page"])] = row["data"]
    if 24 not in pages:
        append_jsonl(acquisition / "leaderboard_pages.jsonl", recovery)
        pages[24] = recovery["data"]
    elif pages[24] != recovery["data"]:
        raise ValueError("recovered page 24 conflicts with an existing page payload")
    boundary_candidates = []
    for page, data in pages.items():
        ratings = [int(float(user.get("rating", 0))) for user in data.get("users") or []]
        if not ratings or ratings[-1] < MINIMUM_ELO:
            boundary_candidates.append(page)
    boundary = min(boundary_candidates)
    missing = [page for page in range(boundary + 1) if page not in pages]
    if missing:
        raise ValueError(f"recovered leaderboard still has missing pages: {missing}")
    old_ids = {row.user_id for row in load_leaderboard_users(snap)}
    rows = []
    page_audit = []
    for page in range(boundary + 1):
        data = pages[page]
        users = data.get("users") or []
        ratings = [int(float(user.get("rating", 0))) for user in users]
        kept = 0
        for index, raw in enumerate(users):
            rating = int(float(raw.get("rating", 0)))
            if rating < MINIMUM_ELO:
                continue
            rows.append({
                "rank": int(data.get("start", page * len(users))) + index + 1,
                "id": str(raw.get("id") or "").strip().lower(),
                "name": str(raw.get("name") or ""), "rating": rating,
                "page": page, "index_on_page": index,
            })
            kept += 1
        page_audit.append({
            "page": page, "terminalStatus": "success", "start": data.get("start"),
            "count": len(users), "kept": kept,
            "first_rating": ratings[0] if ratings else None,
            "last_rating": ratings[-1] if ratings else None,
        })
    rows.sort(key=lambda row: (row["rank"], row["id"]))
    count = len(rows)
    for index, row in enumerate(rows):
        row["stratum"] = min((index * 10) // max(count, 1), 9) + 1
    new_ids = {row["id"] for row in rows} - old_ids
    expected_new = {
        str(raw.get("id") or "").strip().lower()
        for raw in recovery["data"].get("users") or []
        if int(float(raw.get("rating", 0))) >= MINIMUM_ELO
    }
    if new_ids != expected_new:
        raise ValueError("recovered page account delta does not exactly match page 24")
    with (snap / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "id", "name", "rating", "page", "index_on_page", "stratum",
        ])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(snap / "leaderboard.json", {
        "schema": "oq-reference400-leaderboard-complete-snapshot-v1",
        "fetched_at": utc_now(), "cutoff_rating": MINIMUM_ELO, "count": count,
        "pages": page_audit, "boundaryPage": boundary,
        "terminalSuccessPageCount": boundary + 1, "terminalFailurePages": [],
        "recovery": {
            "page": 24,
            "authorization": "user-authorized retry after VPN was disabled",
            "transport": recovery.get("transport"),
            "recoveredAtUtc": recovery.get("fetched_at"),
            "newLeaderboardAccountCount": len(new_ids),
        },
        "completenessPolicy": "all required pages succeeded within the original batch plus the user-authorized failed-page retry",
    })
    batch = load_batch(output)
    leaderboard_stage = batch["stages"]["leaderboard"]
    leaderboard_stage.update({
        "complete": True, "completeWithTerminalFailures": False,
        "terminalFailurePages": [], "recoveredFailurePages": [24],
        "recoveryCompletedAtUtc": utc_now(), "userCount": count,
        "leaderboardCsvSha256": sha256_file(snap / "leaderboard.csv"),
        "leaderboardJsonSha256": sha256_file(snap / "leaderboard.json"),
        "rawPagesSha256": sha256_file(acquisition / "leaderboard_pages.jsonl"),
        "retryLogSha256": sha256_file(acquisition / "leaderboard_retry_attempts.jsonl"),
    })
    player_stage = batch["stages"].get("playerLists") or {}
    player_stage.update({
        "complete": False,
        "userAuthorizedRecovery": True,
        "additionalAuthorizedUserCount": len(new_ids),
        "additionalAuthorizedUserIds": sorted(new_ids),
    })
    batch["stages"]["playerLists"] = player_stage
    save_batch(output, batch)
    print(json.dumps({
        "ok": True, "leaderboardUserCount": count,
        "recoveredPage": 24, "newUserIds": sorted(new_ids),
    }, ensure_ascii=False, indent=2))
    return 0


def load_leaderboard_users(snap: Path) -> list[LeaderboardUser]:
    with (snap / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [LeaderboardUser(
        rank=int(row["rank"]), user_id=row["id"], name=row["name"],
        rating=int(row["rating"]), page=int(row["page"]),
        index_on_page=int(row["index_on_page"]), stratum=int(row["stratum"]) - 1,
    ) for row in rows]


def command_player_lists(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    batch = load_batch(output)
    leaderboard_stage = batch["stages"].get("leaderboard") or {}
    if not leaderboard_stage.get("complete"):
        raise RuntimeError("leaderboard must be frozen before player lists")
    stage = batch["stages"].get("playerLists") or {}
    if stage.get("complete"):
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0
    stage.setdefault("startedAtUtc", utc_now())
    stage["batchOrdinal"] = 1
    batch["stages"]["playerLists"] = stage
    save_batch(output, batch)

    users = load_leaderboard_users(snap)
    final_path = snap / "acquisition" / "player_lists.jsonl"
    attempts_path = snap / "acquisition" / "player_list_attempts.jsonl"
    final_by_user = {str(row.get("userId") or ""): row for row in read_jsonl(final_path)}
    attempts_by_user: Counter[str] = Counter(
        str(row.get("userId") or "") for row in read_jsonl(attempts_path)
    )
    user_by_id = {user.user_id: user for user in users}
    pending = [user for user in users if user.user_id not in final_by_user]
    client = HttpClient(REQUEST_TIMEOUT, True)

    while pending:
        eligible = [user for user in pending if attempts_by_user[user.user_id] < PLAYER_LIST_MAX_ATTEMPTS]
        if not eligible:
            break
        failures: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(client.json, f"{GAME_BASE_URL}/games/{GTYPE}/{user.user_id}.json"): user
                for user in eligible
            }
            for future in as_completed(futures):
                user = futures[future]
                attempt = attempts_by_user[user.user_id] + 1
                attempts_by_user[user.user_id] = attempt
                try:
                    payload = future.result()
                    if not isinstance(payload, dict) or not isinstance(payload.get("games", []), list):
                        raise ValueError("player-list payload is not an object with a games list")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failures[user.user_id] = error
                    append_jsonl(attempts_path, {
                        "at": utc_now(), "userId": user.user_id, "attempt": attempt,
                        "ok": False, "error": error,
                    })
                else:
                    row = {
                        "ok": True, "fetchedAtUtc": utc_now(), "userId": user.user_id,
                        "rank": user.rank, "rating": user.rating, "attempts": attempt,
                        "payload": payload,
                    }
                    append_jsonl(attempts_path, {
                        "at": row["fetchedAtUtc"], "userId": user.user_id,
                        "attempt": attempt, "ok": True,
                    })
                    append_jsonl(final_path, row)
                    final_by_user[user.user_id] = row
        pending = [user_by_id[user_id] for user_id in failures]

    for user in users:
        if user.user_id in final_by_user:
            continue
        row = {
            "ok": False, "fetchedAtUtc": utc_now(), "userId": user.user_id,
            "rank": user.rank, "rating": user.rating,
            "attempts": attempts_by_user[user.user_id],
            "error": "failed after finite retry limit",
        }
        append_jsonl(final_path, row)
        final_by_user[user.user_id] = row
    if set(final_by_user) != set(user_by_id):
        raise ValueError("player-list snapshot does not contain exactly one terminal row per leaderboard user")
    success = sum(bool(row.get("ok")) for row in final_by_user.values())
    failed = len(final_by_user) - success
    stage.update({
        "complete": True,
        "completedAtUtc": utc_now(),
        "expectedUserCount": len(users),
        "terminalUserCount": len(final_by_user),
        "successCount": success,
        "failureCount": failed,
        "snapshotSha256": sha256_file(final_path),
        "attemptLogSha256": sha256_file(attempts_path),
    })
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def command_candidate_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    batch = load_batch(output)
    player_stage = batch["stages"].get("playerLists") or {}
    if not player_stage.get("complete"):
        raise RuntimeError("player-list snapshot must be frozen before candidate audit")
    maximum = int((batch["stages"].get("leaderboard") or {})["dynamicMaximumElo"])
    cache_result = compute_cache_selection(maximum)
    reference_ids = set(cache_result["reference"])
    cache_ids = set(cache_result["cache"])
    cache_all_ids: set[str] = set()
    cache_known_invalid_ids: set[str] = set()
    for local_detail_path in (CACHE_DETAILS, PRIOR_SNAPSHOT_DETAILS):
        for row in read_jsonl(local_detail_path):
            game_id = str(row.get("game_id") or row.get("gameId") or "").strip()
            if not game_id:
                continue
            cache_all_ids.add(game_id)
            if not row.get("valid") and str(row.get("reason") or "") != "request_failed_after_finite_retries":
                cache_known_invalid_ids.add(game_id)
    post_cache_by_pair = {
        (row["pairLowerA"], row["pairLowerB"]): row for row in cache_result["rows"]
    }
    candidates: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    sources_by_game: dict[str, set[str]] = defaultdict(set)
    for player_row in read_jsonl(snap / "acquisition" / "player_lists.jsonl"):
        if not player_row.get("ok"):
            counts["failedPlayerLists"] += 1
            continue
        user_id = str(player_row["userId"])
        games = (player_row.get("payload") or {}).get("games") or []
        counts["listedSummaryRows"] += len(games)
        for game in games:
            game_id = str(game.get("id") or "").strip()
            if not game_id:
                counts["missingGameId"] += 1
                continue
            sources_by_game[game_id].add(user_id)
            if game_id in candidates:
                counts["duplicateSnapshotDiscovery"] += 1
                continue
            if game_id in reference_ids:
                counts["deduplicatedCurrentReference"] += 1
                continue
            if game_id in cache_ids:
                counts["deduplicatedValidatedCache"] += 1
                continue
            if game_id in cache_all_ids:
                counts["deduplicatedKnownInvalidOrDuplicateCache"] += 1
                continue
            if not normal_score(game):
                counts["excludedNotScore"] += 1
                continue
            pair = pair_for_players(game.get("players"), maximum)
            if pair is None:
                counts["excludedOutsideFormalEloRange"] += 1
                continue
            if post_cache_by_pair[pair]["postCacheCount"] >= TARGET_PER_PAIR:
                counts["excludedPairAlreadyFullFromCache"] += 1
                continue
            candidates[game_id] = {
                "gameId": game_id,
                "pairLowerA": pair[0], "pairLowerB": pair[1],
                "summary": game,
            }
            counts["eligibleUniqueSummaryCandidates"] += 1
    for game_id, row in candidates.items():
        row["discoveredFromUserIds"] = sorted(sources_by_game[game_id])
        pair = (row["pairLowerA"], row["pairLowerB"])
        row["stableSha256"] = stable_game_key(pair, game_id, "snapshot")
    ordered = sorted(candidates.values(), key=lambda row: (row["pairLowerA"], row["pairLowerB"], row["stableSha256"], row["gameId"]))
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_pair[(row["pairLowerA"], row["pairLowerB"])].append(row)
    pair_rows = []
    for pair in unordered_pairs():
        cache_row = post_cache_by_pair[pair]
        available = by_pair[pair]
        pair_rows.append({
            **{key: value for key, value in cache_row.items() if not key.endswith("GameIds")},
            "snapshotSummaryCandidateCount": len(available),
            "snapshotSummaryCandidateGameIds": [row["gameId"] for row in available],
            "maximumPossiblePostSnapshotCountBeforeDetailValidation": min(
                TARGET_PER_PAIR, cache_row["postCacheCount"] + len(available)
            ),
        })
    acquisition = snap / "acquisition"
    atomic_json(acquisition / "candidate_dedup_audit.json", {
        "schema": f"oq-reference{REFERENCE_TAG}-candidate-dedup-audit-v1",
        "createdAtUtc": utc_now(), "maximumElo": maximum,
        "cacheAllSeenGameIdCount": len(cache_all_ids),
        "cacheKnownInvalidGameIdCount": len(cache_known_invalid_ids),
        "counts": dict(counts), "candidateCount": len(ordered), "candidates": ordered,
    })
    atomic_json(acquisition / "pair_capacity_before_details.json", {
        "schema": f"oq-reference{REFERENCE_TAG}-pair-capacity-before-details-v1",
        "createdAtUtc": utc_now(), "pairs": pair_rows,
    })
    stage = {
        "complete": True, "completedAtUtc": utc_now(),
        "candidateCount": len(ordered),
        "candidateAuditSha256": sha256_file(acquisition / "candidate_dedup_audit.json"),
    }
    batch["stages"]["candidateAudit"] = stage
    save_batch(output, batch)
    print(json.dumps({**stage, "counts": dict(counts)}, ensure_ascii=False, indent=2))
    return 0


def validate_snapshot_detail(
    summary: dict[str, Any], detail: dict[str, Any], expected_pair: tuple[int, int], maximum: int
) -> tuple[bool, str]:
    game_id = str(summary.get("id") or "")
    if str(detail.get("id") or "") != game_id:
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
    summary_ids = [str(row.get("id") or "").strip().casefold() for row in summary_players]
    detail_ids = [str(row.get("id") or "").strip().casefold() for row in detail_players]
    if not all(detail_ids) or len(set(detail_ids)) != 2 or summary_ids != detail_ids:
        return False, "player_id_mismatch"
    if pair_for_detail(detail, maximum) != expected_pair:
        return False, "detail_pair_mismatch"
    if not replay_is_legal(detail):
        return False, "illegal_replay"
    return True, "ok"


def command_details(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    acquisition = snap / "acquisition"
    batch = load_batch(output)
    if not (batch["stages"].get("candidateAudit") or {}).get("complete"):
        raise RuntimeError("candidate audit must be frozen before detail acquisition")
    maximum = int(batch["stages"]["leaderboard"]["dynamicMaximumElo"])
    candidate_payload = json.loads((acquisition / "candidate_dedup_audit.json").read_text(encoding="utf-8"))
    candidates = candidate_payload.get("candidates") or []
    candidates_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_pair[(int(row["pairLowerA"]), int(row["pairLowerB"]))].append(row)
    for rows in candidates_by_pair.values():
        rows.sort(key=lambda row: (row["stableSha256"], row["gameId"]))
    cache_result = compute_cache_selection(maximum)
    cache_rows = {
        (int(row["pairLowerA"]), int(row["pairLowerB"])): row
        for row in cache_result["rows"]
    }
    detail_path = acquisition / "game_details.jsonl"
    attempt_path = acquisition / "game_detail_attempts.jsonl"
    terminal_by_id = {
        str(row.get("gameId") or ""): row for row in read_jsonl(detail_path)
        if row.get("gameId")
    }
    attempts_by_id: Counter[str] = Counter(
        str(row.get("gameId") or "") for row in read_jsonl(attempt_path)
    )
    write_lock = threading.Lock()
    client = HttpClient(REQUEST_TIMEOUT, True)

    def acquire_pair(pair: tuple[int, int]) -> dict[str, Any]:
        gap = max(0, TARGET_PER_PAIR - int(cache_rows[pair]["postCacheCount"]))
        valid_rows: list[dict[str, Any]] = []
        valid_verified_count = 0
        considered_ids: list[str] = []
        for candidate in candidates_by_pair.get(pair, []):
            game_id = str(candidate["gameId"])
            considered_ids.append(game_id)
            terminal = terminal_by_id.get(game_id)
            if terminal is None:
                detail = None
                last_error = None
                while attempts_by_id[game_id] < 3:
                    attempt = attempts_by_id[game_id] + 1
                    attempts_by_id[game_id] = attempt
                    at = utc_now()
                    try:
                        value = client.json(f"{GAME_BASE_URL}/game/{game_id}.json")
                        if not isinstance(value, dict):
                            raise ValueError("game detail payload is not an object")
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        with write_lock:
                            append_jsonl(attempt_path, {
                                "atUtc": at, "gameId": game_id, "attempt": attempt,
                                "ok": False, "error": last_error,
                            })
                    else:
                        detail = value
                        with write_lock:
                            append_jsonl(attempt_path, {
                                "atUtc": at, "gameId": game_id, "attempt": attempt, "ok": True,
                            })
                        break
                if detail is None:
                    terminal = {
                        "gameId": game_id, "fetchedAtUtc": utc_now(), "valid": False,
                        "reason": "request_failed_after_finite_retries", "error": last_error,
                        "pairLowerA": pair[0], "pairLowerB": pair[1],
                        "summary": candidate["summary"],
                    }
                else:
                    ok, reason = validate_snapshot_detail(candidate["summary"], detail, pair, maximum)
                    terminal = {
                        "gameId": game_id, "fetchedAtUtc": utc_now(), "valid": ok,
                        "reason": reason, "pairLowerA": pair[0], "pairLowerB": pair[1],
                        "stableSha256": candidate["stableSha256"],
                        "summary": candidate["summary"], "detail": detail,
                    }
                with write_lock:
                    append_jsonl(detail_path, terminal)
                    terminal_by_id[game_id] = terminal
            if terminal.get("valid"):
                valid_verified_count += 1
                if len(valid_rows) < gap:
                    valid_rows.append(terminal)
        all_candidates_considered = len(considered_ids) == len(candidates_by_pair.get(pair, []))
        return {
            "pairLowerA": pair[0], "pairLowerB": pair[1],
            "postCacheCount": cache_rows[pair]["postCacheCount"],
            "gapBeforeSnapshot": gap,
            "snapshotSummaryCandidateCount": len(candidates_by_pair.get(pair, [])),
            "snapshotDetailConsideredCount": len(considered_ids),
            "snapshotValidVerifiedCount": valid_verified_count,
            "snapshotInvalidCount": len(considered_ids) - valid_verified_count,
            "snapshotValidSelectedCount": len(valid_rows),
            "snapshotSelectedGameIds": [row["gameId"] for row in valid_rows],
            "finalCount": int(cache_rows[pair]["postCacheCount"]) + len(valid_rows),
            "remainingGap": max(0, gap - len(valid_rows)),
            "allSnapshotCandidatesConsidered": all_candidates_considered,
            "capacityExhaustedBelowTarget": all_candidates_considered and len(valid_rows) < gap,
        }

    active_pairs = [
        pair for pair in unordered_pairs()
        if int(cache_rows[pair]["postCacheCount"]) < TARGET_PER_PAIR
    ]
    results = []
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(active_pairs))) as executor:
        futures = {executor.submit(acquire_pair, pair): pair for pair in active_pairs}
        for future in as_completed(futures):
            results.append(future.result())
    completed_by_pair = {(row["pairLowerA"], row["pairLowerB"]): row for row in results}
    for pair in unordered_pairs():
        if pair in completed_by_pair:
            continue
        completed_by_pair[pair] = {
            "pairLowerA": pair[0], "pairLowerB": pair[1],
            "postCacheCount": cache_rows[pair]["postCacheCount"], "gapBeforeSnapshot": 0,
            "snapshotSummaryCandidateCount": 0, "snapshotDetailConsideredCount": 0,
            "snapshotValidVerifiedCount": 0, "snapshotInvalidCount": 0,
            "snapshotValidSelectedCount": 0, "snapshotSelectedGameIds": [],
            "finalCount": cache_rows[pair]["postCacheCount"], "remainingGap": 0,
            "allSnapshotCandidatesConsidered": False,
            "capacityExhaustedBelowTarget": False,
        }
    results = [completed_by_pair[pair] for pair in unordered_pairs()]
    selected_ids = [game_id for row in results for game_id in row["snapshotSelectedGameIds"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("snapshot detail selection contains duplicate game IDs")
    atomic_json(acquisition / "pair_capacity_after_details.json", {
        "schema": f"oq-reference{REFERENCE_TAG}-pair-capacity-after-details-v1",
        "createdAtUtc": utc_now(), "maximumElo": maximum,
        "snapshotSelectedCount": len(selected_ids), "pairs": results,
    })
    stage = {
        "complete": True, "completedAtUtc": utc_now(),
        "terminalDetailRecordCount": len(terminal_by_id),
        "snapshotSelectedCount": len(selected_ids),
        "pairsAtTarget": sum(row["finalCount"] >= TARGET_PER_PAIR for row in results),
        "pairsCapacityExhaustedBelowTarget": sum(row["capacityExhaustedBelowTarget"] for row in results),
        "remainingGapTotal": sum(row["remainingGap"] for row in results),
        "detailSnapshotSha256": sha256_file(detail_path),
        "detailAttemptLogSha256": sha256_file(attempt_path),
        "pairCapacityAuditSha256": sha256_file(acquisition / "pair_capacity_after_details.json"),
    }
    batch["stages"]["gameDetails"] = stage
    save_batch(output, batch)
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write headerless empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_materialize(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    snap = snapshot_dir(output)
    acquisition = snap / "acquisition"
    batch = load_batch(output)
    detail_stage = batch["stages"].get("gameDetails") or {}
    if not detail_stage.get("complete"):
        raise RuntimeError("detail acquisition must be complete before materialization")
    maximum = int(batch["stages"]["leaderboard"]["dynamicMaximumElo"])
    cache_result = compute_cache_selection(maximum)
    reference_details, reference_index = load_reference()
    cache_records = cache_result["cache"]
    cache_selected_ids = list(cache_result["selectedIds"])
    capacity = json.loads((acquisition / "pair_capacity_after_details.json").read_text(encoding="utf-8"))
    capacity_rows = capacity.get("pairs") or []
    snapshot_selected_ids = [
        str(game_id) for row in capacity_rows for game_id in row.get("snapshotSelectedGameIds") or []
    ]
    snapshot_records = {
        str(row.get("gameId") or ""): row for row in read_jsonl(acquisition / "game_details.jsonl")
        if row.get("valid")
    }
    if any(game_id not in snapshot_records for game_id in snapshot_selected_ids):
        raise ValueError("selected snapshot detail is missing from the validated detail snapshot")
    all_ids = [*reference_details, *cache_selected_ids, *snapshot_selected_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("materialized selection contains duplicate game IDs")

    details_by_id = dict(reference_details)
    index_by_id = dict(reference_index)
    source_by_id = {game_id: "existingReference" for game_id in reference_details}
    for game_id in cache_selected_ids:
        summary, detail = cache_records[game_id]
        details_by_id[game_id] = detail
        index_by_id[game_id] = summary
        source_by_id[game_id] = "validatedCacheExpansion"
    for game_id in snapshot_selected_ids:
        record = snapshot_records[game_id]
        details_by_id[game_id] = record["detail"]
        index_by_id[game_id] = record["summary"]
        source_by_id[game_id] = "uniqueSnapshotExpansion"
    ordered_details = sorted(
        details_by_id.values(), key=lambda detail: (str(detail.get("created") or ""), str(detail.get("id") or ""))
    )
    table_by_id = {str(detail["id"]): table for table, detail in enumerate(ordered_details, start=1)}
    rows = []
    for detail in ordered_details:
        game_id = str(detail["id"])
        row = selected_game_row(detail, source_by_id[game_id], MINIMUM_ELO, maximum, BIN_WIDTH)
        row["bundleTable"] = table_by_id[game_id]
        row["expectedEngineFile"] = f"engine_level22/game_0_{table_by_id[game_id]}_{game_id}.json"
        rows.append(row)
    row_by_id = {str(row["gameId"]): row for row in rows}
    main_rows = [row for row in rows if row["inMainMatrix"]]
    low_rows = [row for row in rows if row["partitionScope"] == "baseline_low_elo_extension"]
    if len(low_rows) != 116 or {row["gameId"] for row in low_rows} != {
        game_id for game_id in reference_details if pair_for_detail(reference_details[game_id], maximum) is None
    }:
        raise ValueError("low-Elo historical extension was not preserved exactly")

    existing_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    cache_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    snapshot_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in main_rows:
        pair = tuple(sorted((int(row["blackBinLower"]), int(row["whiteBinLower"]))))
        source = row["sourceKind"]
        target = existing_by_pair if source == "existingReference" else cache_by_pair if source == "validatedCacheExpansion" else snapshot_by_pair
        target[pair].append(str(row["gameId"]))
    capacity_by_pair = {(int(row["pairLowerA"]), int(row["pairLowerB"])): row for row in capacity_rows}
    unordered = []
    for pair in unordered_pairs():
        existing_ids = sorted(existing_by_pair[pair])
        cache_ids = sorted(cache_by_pair[pair])
        snapshot_ids = sorted(snapshot_by_pair[pair])
        merged = sorted([*existing_ids, *cache_ids, *snapshot_ids])
        cap = capacity_by_pair[pair]
        if len(merged) != int(cap["finalCount"]):
            raise ValueError(f"materialized pair count disagrees with frozen capacity audit: {pair}")
        unordered.append({
            "pairLowerA": pair[0], "pairLowerB": pair[1],
            "pairLabelA": pair_label(pair[0], maximum), "pairLabelB": pair_label(pair[1], maximum),
            "partitionKey": f"{pair_label(pair[0], maximum)}__{pair_label(pair[1], maximum)}",
            "partitionScope": "main_bilateral", "targetPerUnorderedPair": TARGET_PER_PAIR,
            "existingCount": len(existing_ids), "cacheAvailableCount": next(
                row["cacheAvailableCount"] for row in cache_result["rows"]
                if (row["pairLowerA"], row["pairLowerB"]) == pair
            ),
            "cacheSelectedCount": len(cache_ids),
            "snapshotSummaryCandidateCount": cap["snapshotSummaryCandidateCount"],
            "snapshotDetailConsideredCount": cap["snapshotDetailConsideredCount"],
            "snapshotSelectedCount": len(snapshot_ids), "finalCount": len(merged),
            "remainingGap": max(0, TARGET_PER_PAIR - len(merged)),
            "capacityExhaustedBelowTarget": cap["capacityExhaustedBelowTarget"],
            "existingGameIds": existing_ids, "cacheSelectedGameIds": cache_ids,
            "snapshotSelectedGameIds": snapshot_ids, "mergedGameIds": merged,
        })
    low_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in low_rows:
        pair = archive_pair_for_detail(details_by_id[str(row["gameId"])], MINIMUM_ELO, maximum, BIN_WIDTH)
        if pair is None:
            raise ValueError("low extension row has no archive pair")
        low_by_pair[pair].append(str(row["gameId"]))
    for pair in sorted(low_by_pair):
        game_ids = sorted(low_by_pair[pair])
        unordered.append({
            "pairLowerA": pair[0], "pairLowerB": pair[1],
            "pairLabelA": archive_label(pair[0], MINIMUM_ELO, maximum, BIN_WIDTH),
            "pairLabelB": archive_label(pair[1], MINIMUM_ELO, maximum, BIN_WIDTH),
            "partitionKey": f"{archive_label(pair[0], MINIMUM_ELO, maximum, BIN_WIDTH)}__{archive_label(pair[1], MINIMUM_ELO, maximum, BIN_WIDTH)}",
            "partitionScope": "baseline_low_elo_extension", "targetPerUnorderedPair": None,
            "existingCount": len(game_ids), "cacheAvailableCount": None,
            "cacheSelectedCount": 0, "snapshotSummaryCandidateCount": 0,
            "snapshotDetailConsideredCount": 0, "snapshotSelectedCount": 0,
            "finalCount": len(game_ids), "remainingGap": 0,
            "capacityExhaustedBelowTarget": None,
            "existingGameIds": game_ids, "cacheSelectedGameIds": [],
            "snapshotSelectedGameIds": [], "mergedGameIds": game_ids,
        })
    directed_keys = {(target, opponent) for target in bin_lowers() for opponent in bin_lowers()}
    for row in low_rows:
        directed_keys.add((int(row["blackBinLower"]), int(row["whiteBinLower"])))
        directed_keys.add((int(row["whiteBinLower"]), int(row["blackBinLower"])))
    directed = []
    for target, opponent in sorted(directed_keys):
        game_ids = sorted({
            str(row["gameId"]) for row in rows
            if row["partitionScope"] != "outside_unpartitioned" and (
                (row["blackBinLower"] == target and row["whiteBinLower"] == opponent)
                or (row["whiteBinLower"] == target and row["blackBinLower"] == opponent)
            )
        })
        target_label = archive_label(target, MINIMUM_ELO, maximum, BIN_WIDTH)
        opponent_label = archive_label(opponent, MINIMUM_ELO, maximum, BIN_WIDTH)
        directed.append({
            "targetBinLower": target, "targetBinLabel": target_label,
            "opponentBinLower": opponent, "opponentBinLabel": opponent_label,
            "partitionKey": f"{target_label}__vs__{opponent_label}",
            "partitionScope": "baseline_low_elo_extension" if target < MINIMUM_ELO or opponent < MINIMUM_ELO else "main_bilateral",
            "existingGameCount": sum(source_by_id[game_id] == "existingReference" for game_id in game_ids),
            "cacheSelectedGameCount": sum(source_by_id[game_id] == "validatedCacheExpansion" for game_id in game_ids),
            "snapshotSelectedGameCount": sum(source_by_id[game_id] == "uniqueSnapshotExpansion" for game_id in game_ids),
            "mergedGameCount": len(game_ids), "mergedGameIds": game_ids,
        })

    bundle = {
        "schema": f"oq-account-bundle-elo-matchup{REFERENCE_TAG}-expansion-v1",
        "account": f"elo_matchup{REFERENCE_TAG}_reference_multi_target",
        "fetchedAt": json.loads((snap / "leaderboard.json").read_text(encoding="utf-8"))["fetched_at"],
        "selection": {
            "policy": "retain existing Reference; cache first; then unique snapshot; stable SHA-256 order",
            "seed": SEED, "targetPerUnorderedPair": TARGET_PER_PAIR,
            "minimumElo": MINIMUM_ELO, "maximumElo": maximum, "binWidth": BIN_WIDTH,
            "existingReferenceGameCount": len(reference_details),
            "cacheSelectedCount": len(cache_selected_ids),
            "snapshotSelectedCount": len(snapshot_selected_ids),
            "mergedGameCount": len(rows), "gameIds": [str(detail["id"]) for detail in ordered_details],
        },
        "index": [index_by_id[str(detail["id"])] for detail in ordered_details],
        "details": ordered_details,
    }
    atomic_json(output / "selected_account_bundle.json", bundle)
    write_csv(output / "selected_games_with_partitions.csv", rows)
    atomic_json(output / "selected_games_with_partitions.json", {
        "schema": f"oq-matchup{REFERENCE_TAG}-selected-games-partitions-v1", "games": rows,
    })
    added_rows = [row for row in rows if row["sourceKind"] != "existingReference"]
    write_csv(output / "added_games.csv", added_rows)
    atomic_json(output / "added_games.json", {"schema": f"oq-matchup{REFERENCE_TAG}-added-games-v1", "games": added_rows})
    unordered_csv = [{key: value for key, value in row.items() if not key.endswith("GameIds")} for row in unordered]
    directed_csv = [{key: value for key, value in row.items() if not key.endswith("GameIds")} for row in directed]
    write_csv(output / "partitions_unordered.csv", unordered_csv)
    atomic_json(output / "partitions_unordered.json", {"schema": f"oq-matchup{REFERENCE_TAG}-unordered-partitions-v1", "partitions": unordered})
    write_csv(output / "partitions_directed.csv", directed_csv)
    atomic_json(output / "partitions_directed.json", {"schema": f"oq-matchup{REFERENCE_TAG}-directed-partitions-v1", "partitions": directed})
    audit = {
        "schema": f"oq-reference{REFERENCE_TAG}-selection-audit-v1", "ok": True, "createdAtUtc": utc_now(),
        "gameCount": len(rows), "uniqueGameIdCount": len(row_by_id),
        "mainMatrixGameCount": len(main_rows), "lowEloExtensionGameCount": len(low_rows),
        "existingReferenceGameCount": len(reference_details),
        "cacheSelectedCount": len(cache_selected_ids), "snapshotSelectedCount": len(snapshot_selected_ids),
        "formalPairCount": 45, "pairsAtTarget": sum(row["finalCount"] >= TARGET_PER_PAIR for row in unordered if row["partitionScope"] == "main_bilateral"),
        "pairsCapacityExhaustedBelowTarget": sum(bool(row["capacityExhaustedBelowTarget"]) for row in unordered if row["partitionScope"] == "main_bilateral"),
        "checks": {
            "allExistingReferenceGamesRetained": set(reference_details) <= set(row_by_id),
            "lowEloExtensionPreservedExactly": len(low_rows) == 116,
            "allGameIdsUnique": len(rows) == len(row_by_id),
            "allFormalPairsMeetFrozenCapacity": all(
                row["finalCount"] >= TARGET_PER_PAIR or row["capacityExhaustedBelowTarget"]
                for row in unordered if row["partitionScope"] == "main_bilateral"
            ),
        },
    }
    audit["ok"] = all(audit["checks"].values())
    atomic_json(output / "selection_audit.json", audit)
    atomic_json(output / "expansion_manifest.json", {
        "schema": f"oq-reference{REFERENCE_TAG}-expansion-manifest-v1", "createdAtUtc": utc_now(),
        "configuration": {
            "minimumElo": MINIMUM_ELO, "historicalMaximumElo": PRIOR_FORMAL_MAXIMUM_ELO,
            "dynamicMaximumElo": maximum, "topBinLower": TOP_BIN_LOWER,
            "binWidth": BIN_WIDTH, "targetPerUnorderedPair": TARGET_PER_PAIR, "seed": SEED,
        },
        "sources": {
            "oldReferenceDirectory": str(OLD_REFERENCE.resolve()),
            "oldReferenceBundleSha256": sha256_file(OLD_REFERENCE / "selected_account_bundle.json"),
            "cacheDetails": str(CACHE_DETAILS.resolve()), "cacheDetailsSha256": sha256_file(CACHE_DETAILS),
            "oneShotSnapshotDirectory": str(snap.resolve()),
            "oneShotBatchManifestSha256": sha256_file(batch_manifest_path(output)),
        },
        "selectionAudit": audit,
    })
    batch["stages"]["materializeSelection"] = {
        "complete": True, "completedAtUtc": utc_now(), "gameCount": len(rows),
        "mainMatrixGameCount": len(main_rows), "lowEloExtensionGameCount": len(low_rows),
        "selectionAuditSha256": sha256_file(output / "selection_audit.json"),
        "bundleSha256": sha256_file(output / "selected_account_bundle.json"),
    }
    save_batch(output, batch)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "local-audit", "leaderboard", "finalize-leaderboard",
            "integrate-recovered-page", "player-lists", "candidate-audit",
            "details",
            "materialize",
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers = {
        "local-audit": command_local_audit,
        "leaderboard": command_leaderboard,
        "finalize-leaderboard": command_finalize_leaderboard,
        "integrate-recovered-page": command_integrate_recovered_page,
        "player-lists": command_player_lists,
        "candidate-audit": command_candidate_audit,
        "details": command_details,
        "materialize": command_materialize,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
