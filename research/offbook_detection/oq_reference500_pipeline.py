#!/usr/bin/env python3
"""Resume-safe one-shot OQ Reference expansion toward 500 games per pair."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import oq_reference200_pipeline as pipeline


ROOT = HERE.parents[1]
DATA = ROOT / "research" / "offbook_detection" / "data"
CURRENT_REFERENCE = DATA / "oq_elo_matchup450_reference_level22_1600plus_20260820"
CURRENT_SNAPSHOT_DETAILS = (
    CURRENT_REFERENCE
    / "provenance"
    / "oq_snapshot_reference450_20260820"
    / "acquisition"
    / "game_details.jsonl"
)
SNAPSHOT_NAME = "oq_snapshot_reference500_20260821"
BATCH_ID = "oq-reference500-20260821-once"
OUTPUT_NAME = "oq_elo_matchup500_reference_level22_1600plus_20260821"


def load_current_reference() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bundle_path = CURRENT_REFERENCE / "selected_account_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    index = bundle.get("index") if isinstance(bundle.get("index"), list) else []
    details_by_id = {str(row.get("id") or ""): row for row in details}
    index_by_id = {str(row.get("id") or ""): row for row in index}
    if (
        not details
        or "" in details_by_id
        or len(details_by_id) != len(details)
        or set(index_by_id) != set(details_by_id)
    ):
        raise ValueError("current matchup450 bundle is not a unique, aligned source")
    for game_id, detail in details_by_id.items():
        ok, reason = pipeline.valid_detail(index_by_id[game_id], detail)
        if not ok:
            raise ValueError(f"current Reference game {game_id} failed contract validation: {reason}")
        if not pipeline.replay_is_legal(detail):
            raise ValueError(f"current Reference game {game_id} failed legal replay validation")
    return details_by_id, index_by_id


def snapshot_dir(output: Path) -> Path:
    return output / "provenance" / SNAPSHOT_NAME


def batch_manifest_path(output: Path) -> Path:
    return snapshot_dir(output) / "batch_manifest.json"


def load_batch(output: Path) -> dict[str, Any]:
    path = batch_manifest_path(output)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("batchId") != BATCH_ID:
            raise ValueError("conflicting OQ 500 snapshot batch already exists")
        return value
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "target matchup500 directory exists without the required batch manifest; refusing to overwrite"
        )
    return {
        "schema": "oq-reference500-one-shot-batch-v1",
        "batchId": BATCH_ID,
        "createdAtUtc": pipeline.utc_now(),
        "configuration": {
            "minimumElo": pipeline.MINIMUM_ELO,
            "historicalFormalEloMaximum": 2495,
            "targetPerUnorderedPair": 500,
            "selectionRandomSeed": 20260821500,
        },
        "networkPolicy": {
            "leaderboardFullBatchesAllowed": 1,
            "playerListFullBatchesAllowed": 1,
            "playerListMaximumAttemptsPerUser": pipeline.PLAYER_LIST_MAX_ATTEMPTS,
            "gameDetailMaximumAttemptsPerGame": 3,
            "requestTimeoutSeconds": pipeline.REQUEST_TIMEOUT,
            "workers": pipeline.WORKERS,
            "direct": True,
            "proxyDisabled": True,
            "httpClientMode": "urllib.build_opener(ProxyHandler({}))",
            "environmentProxyUse": False,
            "resumePolicy": "resume only missing pages/users/details in this batch; never rescan successful items",
        },
        "stages": {},
        "recovery": [],
    }


def configure() -> None:
    pipeline.OLD_REFERENCE = CURRENT_REFERENCE
    pipeline.CACHE_DETAILS = DATA / "oq_transformer_100000_20260813" / "acquisition" / "game_details.jsonl"
    pipeline.MATERIALIZED_GAMES = DATA / "oq_transformer_61145_20260813" / "games.csv"
    pipeline.PRIOR_SNAPSHOT_DETAILS = CURRENT_SNAPSHOT_DETAILS
    pipeline.DEFAULT_OUTPUT = DATA / OUTPUT_NAME
    pipeline.PRIOR_FORMAL_MAXIMUM_ELO = 2495
    pipeline.TOP_BIN_LOWER = 2400
    pipeline.TARGET_PER_PAIR = 500
    pipeline.SEED = 20260821500
    pipeline.REFERENCE_TAG = "500"
    pipeline.load_reference = load_current_reference
    pipeline.snapshot_dir = snapshot_dir
    pipeline.batch_manifest_path = batch_manifest_path
    pipeline.load_batch = load_batch


def finalize_leaderboard(args: Any) -> int:
    """Freeze a finite, terminal page failure without starting a second full scan."""
    output = Path(args.output).resolve()
    snap = snapshot_dir(output)
    acquisition = snap / "acquisition"
    batch = load_batch(output)
    stage = batch["stages"].get("leaderboard") or {}
    if stage.get("complete"):
        print(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0

    pages: dict[int, dict[str, Any]] = {}
    for row in pipeline.read_jsonl(acquisition / "leaderboard_pages.jsonl"):
        if isinstance(row.get("data"), dict):
            pages[int(row["page"])] = row["data"]
    boundary_candidates = []
    for page, data in pages.items():
        ratings = [int(float(user.get("rating", 0))) for user in data.get("users") or []]
        if not ratings or ratings[-1] < pipeline.MINIMUM_ELO:
            boundary_candidates.append(page)
    if not boundary_candidates:
        raise ValueError("leaderboard boundary was never observed")
    boundary = min(boundary_candidates)
    missing = [page for page in range(boundary + 1) if page not in pages]
    if not missing:
        raise ValueError("leaderboard is complete; rerun the normal leaderboard command")
    failure_text = " ".join(str(item.get("error") or "") for item in stage.get("failures") or [])
    if not all(f"{page}" in failure_text for page in missing):
        raise ValueError("terminal missing pages do not match persisted batch failures")

    terminal_failure_path = acquisition / "leaderboard_terminal_failures.jsonl"
    existing_terminal = {
        int(row.get("page", -1))
        for row in pipeline.read_jsonl(terminal_failure_path)
        if row.get("page") is not None
    }
    for page in missing:
        if page in existing_terminal:
            continue
        pipeline.append_jsonl(terminal_failure_path, {
            "atUtc": pipeline.utc_now(),
            "batchId": BATCH_ID,
            "page": page,
            "status": "failed_after_finite_retries",
            "retryPolicy": "five finite retry rounds within the same full leaderboard batch",
            "evidence": "leaderboard pipeline raised terminal missing-page error after cached successful pages were reused",
        })

    raw_users = []
    page_audit = []
    for page in range(boundary + 1):
        if page in missing:
            page_audit.append({
                "page": page,
                "terminalStatus": "failed_after_finite_retries",
                "count": None,
                "kept": None,
            })
            continue
        data = pages[page]
        users = data.get("users") or []
        ratings = [int(float(user.get("rating", 0))) for user in users]
        kept = 0
        for index, raw in enumerate(users):
            rating = int(float(raw.get("rating", 0)))
            if rating < pipeline.MINIMUM_ELO:
                continue
            raw_users.append({
                "rank": int(data.get("start", page * len(users))) + index + 1,
                "id": str(raw.get("id") or "").strip().lower(),
                "name": str(raw.get("name") or ""),
                "rating": rating,
                "page": page,
                "index_on_page": index,
            })
            kept += 1
        page_audit.append({
            "page": page,
            "terminalStatus": "success",
            "start": data.get("start"),
            "count": len(users),
            "kept": kept,
            "first_rating": ratings[0] if ratings else None,
            "last_rating": ratings[-1] if ratings else None,
        })
    raw_users.sort(key=lambda row: (row["rank"], row["id"]))
    count = len(raw_users)
    for index, row in enumerate(raw_users):
        row["stratum"] = min((index * 10) // max(count, 1), 9) + 1
    leaderboard_csv = snap / "leaderboard.csv"
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "id", "name", "rating", "page", "index_on_page", "stratum",
        ])
        writer.writeheader()
        writer.writerows(raw_users)
    leaderboard_json = snap / "leaderboard.json"
    pipeline.atomic_json(leaderboard_json, {
        "schema": "oq-reference500-leaderboard-terminal-status-snapshot-v1",
        "fetched_at": pipeline.utc_now(),
        "cutoff_rating": pipeline.MINIMUM_ELO,
        "count": count,
        "pages": page_audit,
        "boundaryPage": boundary,
        "terminalSuccessPageCount": boundary + 1 - len(missing),
        "terminalFailurePages": missing,
        "completenessPolicy": "every required page has a terminal success/failure status; no second full scan",
    })
    maximum_observed = max(row["rating"] for row in raw_users)
    maximum = max(pipeline.PRIOR_FORMAL_MAXIMUM_ELO, maximum_observed)
    stage.update({
        "complete": True,
        "completeWithTerminalFailures": True,
        "completedAtUtc": pipeline.utc_now(),
        "userCount": count,
        "snapshotLeaderboardMaximum": maximum_observed,
        "dynamicMaximumElo": maximum,
        "terminalFailurePages": missing,
        "leaderboardCsvSha256": pipeline.sha256_file(leaderboard_csv),
        "leaderboardJsonSha256": pipeline.sha256_file(leaderboard_json),
        "rawPagesSha256": pipeline.sha256_file(acquisition / "leaderboard_pages.jsonl"),
        "terminalFailureLogSha256": pipeline.sha256_file(terminal_failure_path),
    })
    batch["stages"]["leaderboard"] = stage
    pipeline.save_batch(output, batch)
    dynamic = pipeline.compute_cache_selection(maximum)
    pipeline.write_cache_audit(output, dynamic, "cache_pair_audit_dynamic_max")
    print(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0


def ensure_batch(output: Path) -> None:
    output = output.resolve()
    batch = load_batch(output)
    pipeline.save_batch(output, batch)


def main() -> int:
    configure()
    pipeline.command_finalize_leaderboard = finalize_leaderboard
    parsed = pipeline.build_parser().parse_args()
    ensure_batch(Path(parsed.output))
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
