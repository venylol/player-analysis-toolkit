#!/usr/bin/env python3
"""Concurrently pull a deduplicated bilateral OQ extension cohort."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pull_oq_reversi_5min_elo2000_games as base


MOVE_FIELDS = [
    "game_id", "mode", "gtype", "tcb", "created", "finalStatus", "move_index", "ply",
    "color_to_move", "player_id", "player_name", "player_seed_rank", "player_seed_rating",
    "opponent_id", "opponent_is_elo2000_plus", "move", "thinking_time_ms",
    "has_terminal_status", "move_status", "fetched_at",
]
SUMMARY_FIELDS = [
    "game_id", "mode", "gtype", "tcb", "created", "finalStatus", "color", "player_id",
    "player_name", "player_seed_rank", "player_seed_rating", "opponent_id",
    "opponent_is_elo2000_plus", "recorded_move_count", "total_thinking_time_ms", "fetched_at",
]
GAME_FIELDS = [
    "game_id", "mode", "gtype", "tcb", "created", "finalStatus", "length", "black_id",
    "black_name", "black_rating_in_seed", "white_id", "white_name", "white_rating_in_seed",
    "recorded_sides", "fetched_at",
]


def game_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["game_id"].strip() for row in csv.DictReader(handle) if row.get("game_id", "").strip()}


def baseline_counts(path: Path, rated: dict[str, base.RatedUser]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            black = row.get("black_id", "").strip().lower()
            white = row.get("white_id", "").strip().lower()
            if black in rated and white in rated:
                counts[black] += 1
                counts[white] += 1
    return counts


def atomic_progress(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", required=True, type=Path)
    parser.add_argument("--baseline-games", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-games", type=int, default=1000)
    parser.add_argument("--http-workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.15)
    parser.add_argument("--detail-batch", type=int, default=512)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if args.target_games <= 0 or args.http_workers <= 0 or args.detail_batch <= 0:
        raise ValueError("targets, workers, and detail batch must be positive")
    base.configure_http(args.direct)
    rated = base.load_rated_users(args.users.resolve())
    baseline_path = args.baseline_games.resolve()
    excluded = game_ids(baseline_path)
    counts = baseline_counts(baseline_path, rated)
    users = sorted(rated.values(), key=lambda user: (-counts.get(user.user_id, 0), user.rank, user.user_id))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    games_path = output / "games.csv"
    moves_path = output / "move_times.csv"
    summaries_path = output / "game_player_summaries.csv"
    progress_path = output / "progress.json"
    base.ensure_csv(games_path, GAME_FIELDS)
    base.ensure_csv(moves_path, MOVE_FIELDS)
    base.ensure_csv(summaries_path, SUMMARY_FIELDS)
    existing = game_ids(games_path)
    seen = excluded | existing
    if len(existing) >= args.target_games:
        print(json.dumps({"ok": True, "targetReached": True, "games": len(existing)}, indent=2))
        return 0

    lock = threading.Lock()
    stats: Counter[str] = Counter()
    failures: dict[str, list[str]] = {"users": [], "games": []}

    def fetch_user(user: base.RatedUser) -> tuple[str, list[dict[str, Any]]]:
        try:
            result = base.get_json(
                f"{base.BASE_URL}/games/{base.GTYPE}/{user.user_id}.json",
                args.retries, args.timeout, args.retry_delay,
            )
            games = result.get("games", []) if isinstance(result, dict) else []
            candidates = []
            local = set()
            for game in games:
                if not base.is_normal_score_game(game):
                    continue
                game_id = str(game.get("id", "")).strip()
                if not game_id or game_id in seen or game_id in local:
                    continue
                bilateral = base.both_players_are_ranked(game, rated)
                if bilateral is False:
                    continue
                local.add(game_id)
                candidates.append(game)
            with lock:
                stats["userListsOk"] += 1
                stats["summaryCandidates"] += len(candidates)
            return user.user_id, candidates
        except Exception as exc:
            with lock:
                stats["userListsFailed"] += 1
                failures["users"].append(f"{user.user_id}: {type(exc).__name__}: {exc}")
            return user.user_id, []

    print(f"fetching {len(users)} user lists with {args.http_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.http_workers, thread_name_prefix="oq-list") as pool:
        listed = dict(pool.map(fetch_user, users))

    # Preserve the baseline-count ordering and take one candidate per source user per round.
    cursors = {user.user_id: 0 for user in users}
    ordered: list[dict[str, Any]] = []
    ordered_ids = set()
    while True:
        added = 0
        for user in users:
            candidates = listed.get(user.user_id, [])
            while cursors[user.user_id] < len(candidates):
                game = candidates[cursors[user.user_id]]
                cursors[user.user_id] += 1
                game_id = str(game.get("id", "")).strip()
                if game_id in ordered_ids or game_id in seen:
                    continue
                ordered_ids.add(game_id)
                ordered.append(game)
                added += 1
                break
        if not added:
            break
    stats["deduplicatedCandidates"] = len(ordered)

    def fetch_detail(game: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        game_id = str(game.get("id", "")).strip()
        try:
            detail = base.get_json(
                f"{base.BASE_URL}/game/{game_id}.json",
                args.retries, args.timeout, args.retry_delay,
            )
            return game, detail, None
        except Exception as exc:
            return game, None, f"{game_id}: {type(exc).__name__}: {exc}"

    cursor = 0
    while len(existing) < args.target_games and cursor < len(ordered):
        batch = ordered[cursor : cursor + args.detail_batch]
        cursor += len(batch)
        print(f"detail batch through {cursor}/{len(ordered)}; accepted={len(existing)}/{args.target_games}", flush=True)
        with ThreadPoolExecutor(max_workers=args.http_workers, thread_name_prefix="oq-detail") as pool:
            results = list(pool.map(fetch_detail, batch))
        for game, detail, error in results:
            game_id = str(game.get("id", "")).strip()
            if error:
                stats["detailFailed"] += 1
                failures["games"].append(error)
                continue
            stats["detailOk"] += 1
            if game_id in seen:
                continue
            seen.add(game_id)
            if int(detail.get("tcb", 0) or 0) != base.EXPECTED_TCB:
                stats["wrongTimeControl"] += 1
                continue
            if base.both_players_are_ranked(detail, rated) is not True:
                stats["notBilateral"] += 1
                continue
            fetched_at = base.utc_now_iso()
            move_rows, summary_rows, game_row = base.extract_rows(game, detail, rated, fetched_at)
            if game_row is None or game_row.get("recorded_sides") != "black|white":
                stats["incompleteExtraction"] += 1
                continue
            base.append_rows(moves_path, MOVE_FIELDS, move_rows)
            base.append_rows(summaries_path, SUMMARY_FIELDS, summary_rows)
            base.append_rows(games_path, GAME_FIELDS, [game_row])
            existing.add(game_id)
            stats["gamesAdded"] += 1
            stats["moveRowsAdded"] += len(move_rows)
            if len(existing) >= args.target_games:
                break
        progress = {
            "schema": "oq-bilateral-extension-concurrent-progress-v1",
            "updatedAt": base.utc_now_iso(), "targetGames": args.target_games,
            "games": len(existing), "httpWorkers": args.http_workers,
            "candidateCursor": cursor, "candidateCount": len(ordered),
            "stats": dict(stats), "failures": failures,
        }
        atomic_progress(progress_path, progress)

    final = {
        "schema": "oq-bilateral-extension-concurrent-final-v1",
        "ok": len(existing) == args.target_games,
        "targetGames": args.target_games, "games": len(existing),
        "excludedBaselineGames": len(excluded), "httpWorkers": args.http_workers,
        "stats": dict(stats), "failures": failures,
        "paths": {"games": str(games_path), "moves": str(moves_path), "summaries": str(summaries_path)},
    }
    atomic_progress(output / "pull_manifest.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if final["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
