#!/usr/bin/env python3
"""
Pull Othello Quest 5-minute normal games for users rated 2000+.

Input:
  research/oq_reversi_5min_rating_2000_users.csv

Outputs:
  research/oq_reversi_5min_elo2000_games/move_times.csv
  research/oq_reversi_5min_elo2000_games/game_player_summaries.csv
  research/oq_reversi_5min_elo2000_games/games.csv
  research/oq_reversi_5min_elo2000_games/progress.json

Only normal 5-minute score games are used: finalStatus must start with
"SCORE:" and the game detail tcb must equal 300000 milliseconds.
Only the side whose player id is in the Elo/rating 2000+ input list is recorded.
If both players are in the list, both sides are recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from http.client import RemoteDisconnected
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen


GTYPE = "reversi"
MODE_LABEL = "reversi_5min"
EXPECTED_TCB = 300000
BASE_URL = "http://questgames.net"
USER_AGENT = "egaroucid-othello-quest-research/1.0"
HTTP_OPEN = urlopen


def configure_http(direct: bool) -> None:
    global HTTP_OPEN
    HTTP_OPEN = build_opener(ProxyHandler({})).open if direct else urlopen


@dataclass(frozen=True)
class RatedUser:
    rank: int
    user_id: str
    name: str
    rating: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_json(url: str, retries: int, timeout: float, delay: float) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with HTTP_OPEN(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, RemoteDisconnected, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def load_rated_users(path: Path) -> dict[str, RatedUser]:
    users: dict[str, RatedUser] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            user_id = row["id"].strip().lower()
            users[user_id] = RatedUser(
                rank=int(row["rank"]),
                user_id=user_id,
                name=row["name"],
                rating=int(row["rating"]),
            )
    return users


def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def load_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("game_id", "")).strip()
            for row in csv.DictReader(f)
            if str(row.get("game_id", "")).strip()
        }


def count_existing_expected_tcb_games(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(
            1
            for row in csv.DictReader(f)
            if str(row.get("game_id", "")).strip()
            and int(row.get("tcb", 0) or 0) == EXPECTED_TCB
        )


def load_existing_bilateral_counts(
    path: Path,
    rated_users: dict[str, RatedUser],
) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    bilateral_games = 0
    seen: set[str] = set()
    if not path.exists():
        return bilateral_games, counts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            game_id = str(row.get("game_id", "")).strip()
            if not game_id or game_id in seen:
                continue
            seen.add(game_id)
            if int(row.get("tcb", 0) or 0) != EXPECTED_TCB:
                continue
            if not str(row.get("finalStatus", "")).startswith("SCORE:"):
                continue
            black_id = str(row.get("black_id", "")).strip().lower()
            white_id = str(row.get("white_id", "")).strip().lower()
            if black_id not in rated_users or white_id not in rated_users:
                continue
            bilateral_games += 1
            counts[black_id] += 1
            counts[white_id] += 1
    return bilateral_games, counts


def both_players_are_ranked(game: dict[str, Any], rated_users: dict[str, RatedUser]) -> bool | None:
    players = game.get("players") or []
    if len(players) < 2:
        return None
    return player_id(players[0]) in rated_users and player_id(players[1]) in rated_users


def stratified_round_robin_users(
    users: list[RatedUser],
    strata: int,
    skip_strata: set[int],
) -> list[tuple[int, RatedUser]]:
    if strata <= 1:
        return [(0, user) for user in users] if 0 not in skip_strata else []
    buckets: list[list[RatedUser]] = [[] for _ in range(strata)]
    for index, user in enumerate(users):
        stratum = min(index * strata // max(len(users), 1), strata - 1)
        buckets[stratum].append(user)
    selected: list[tuple[int, RatedUser]] = []
    max_bucket_size = max((len(bucket) for bucket in buckets), default=0)
    for offset in range(max_bucket_size):
        for stratum, bucket in enumerate(buckets):
            if stratum in skip_strata or offset >= len(bucket):
                continue
            selected.append((stratum, bucket[offset]))
    return selected


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_users": [], "seen_games": []}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def player_id(player: dict[str, Any]) -> str:
    return str(player.get("id", "")).strip().lower()


def is_normal_score_game(game: dict[str, Any]) -> bool:
    return str(game.get("finalStatus", "")).startswith("SCORE:")


def side_label(side: int) -> str:
    return "black" if side == 0 else "white"


def sum_side_time(moves: list[dict[str, Any]], side: int) -> int:
    total = 0
    move_index = 0
    for move in moves:
        if "m" not in move:
            continue
        if move_index % 2 == side:
            total += int(move.get("t", 0) or 0)
        move_index += 1
    return total


def extract_rows(
    game: dict[str, Any],
    detail: dict[str, Any],
    rated_users: dict[str, RatedUser],
    fetched_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    players = detail.get("players") or game.get("players") or []
    if len(players) < 2:
        return [], [], None

    black_id = player_id(players[0])
    white_id = player_id(players[1])
    high_sides = []
    if black_id in rated_users:
        high_sides.append(0)
    if white_id in rated_users:
        high_sides.append(1)
    if not high_sides:
        return [], [], None

    position = detail.get("position") or {}
    moves = position.get("moves") or []
    if not isinstance(moves, list):
        return [], [], None

    game_id = str(game.get("id") or detail.get("id") or "")
    tcb = int(detail.get("tcb", 0) or 0)
    final_status = str(game.get("finalStatus", ""))
    created = str(game.get("created") or detail.get("created") or "")
    length = int(game.get("length", 0) or 0)

    game_row = {
        "game_id": game_id,
        "mode": MODE_LABEL,
        "gtype": GTYPE,
        "tcb": tcb,
        "created": created,
        "finalStatus": final_status,
        "length": length,
        "black_id": black_id,
        "black_name": players[0].get("name", ""),
        "black_rating_in_seed": rated_users[black_id].rating if black_id in rated_users else "",
        "white_id": white_id,
        "white_name": players[1].get("name", ""),
        "white_rating_in_seed": rated_users[white_id].rating if white_id in rated_users else "",
        "recorded_sides": "|".join(side_label(side) for side in high_sides),
        "fetched_at": fetched_at,
    }

    move_rows: list[dict[str, Any]] = []
    move_index = 0
    for move in moves:
        if "m" not in move:
            continue
        side = move_index % 2
        side_id = black_id if side == 0 else white_id
        if side_id in rated_users:
            rated = rated_users[side_id]
            move_rows.append(
                {
                    "game_id": game_id,
                    "mode": MODE_LABEL,
                    "gtype": GTYPE,
                    "tcb": tcb,
                    "created": created,
                    "finalStatus": final_status,
                    "move_index": move_index,
                    "ply": move_index + 1,
                    "color_to_move": side_label(side),
                    "player_id": side_id,
                    "player_name": players[side].get("name", ""),
                    "player_seed_rank": rated.rank,
                    "player_seed_rating": rated.rating,
                    "opponent_id": white_id if side == 0 else black_id,
                    "opponent_is_elo2000_plus": (white_id if side == 0 else black_id) in rated_users,
                    "move": move.get("m", ""),
                    "thinking_time_ms": int(move.get("t", 0) or 0),
                    "has_terminal_status": "s" in move,
                    "move_status": move.get("s", ""),
                    "fetched_at": fetched_at,
                }
            )
        move_index += 1

    summary_rows = []
    for side in high_sides:
        side_id = black_id if side == 0 else white_id
        rated = rated_users[side_id]
        summary_rows.append(
            {
                "game_id": game_id,
                "mode": MODE_LABEL,
                "gtype": GTYPE,
                "tcb": tcb,
                "created": created,
                "finalStatus": final_status,
                "color": side_label(side),
                "player_id": side_id,
                "player_name": players[side].get("name", ""),
                "player_seed_rank": rated.rank,
                "player_seed_rating": rated.rating,
                "opponent_id": white_id if side == 0 else black_id,
                "opponent_is_elo2000_plus": (white_id if side == 0 else black_id) in rated_users,
                "recorded_move_count": sum(1 for row in move_rows if row["player_id"] == side_id),
                "total_thinking_time_ms": sum_side_time(moves, side),
                "fetched_at": fetched_at,
            }
        )

    return move_rows, summary_rows, game_row


def run_balanced_bilateral_pull(
    args: argparse.Namespace,
    rated_users: dict[str, RatedUser],
    move_path: Path,
    summary_path: Path,
    games_path: Path,
    progress_path: Path,
    move_fields: list[str],
    summary_fields: list[str],
    game_fields: list[str],
) -> int:
    minimum_games = int(args.balanced_bilateral_min_existing_games)
    target_games = int(args.target_bilateral_game_count)
    if minimum_games <= 0:
        raise ValueError("--balanced-bilateral-min-existing-games must be positive")
    if target_games <= 0:
        raise ValueError("--target-bilateral-game-count must be positive in balanced bilateral mode")

    progress = load_progress(progress_path)
    completed_users = set(progress.get("completed_users", []))
    seen_games = set(progress.get("seen_games", []))
    existing_game_ids = load_existing_game_ids(games_path)
    seen_games.update(existing_game_ids)
    bilateral_before, existing_counts = load_existing_bilateral_counts(games_path, rated_users)
    eligible_users = sorted(
        (
            rated_user
            for rated_user in rated_users.values()
            if existing_counts.get(rated_user.user_id, 0) >= minimum_games
        ),
        key=lambda rated_user: (
            -existing_counts.get(rated_user.user_id, 0),
            rated_user.rank,
            rated_user.user_id,
        ),
    )

    stats = progress.setdefault("stats", {})
    stats.setdefault("mode", MODE_LABEL)
    stats.setdefault("gtype", GTYPE)
    stats.setdefault("expected_tcb", EXPECTED_TCB)
    stats.setdefault("normal_score_games_seen", 0)
    stats.setdefault("game_details_recorded", 0)
    stats.setdefault("move_rows_recorded", 0)
    stats.setdefault("summary_rows_recorded", 0)
    started_at = utc_now_iso()
    stats["last_started_at"] = started_at
    run_stats: dict[str, Any] = {
        "mode": "balanced-bilateral-ranked",
        "started_at": started_at,
        "minimum_existing_bilateral_games": minimum_games,
        "eligible_users": len(eligible_users),
        "eligible_user_order": [rated_user.user_id for rated_user in eligible_users],
        "bilateral_games_before_run": bilateral_before,
        "target_bilateral_game_count": target_games,
        "users_listed": 0,
        "user_list_fetch_errors": 0,
        "candidate_games_after_summary_filter": 0,
        "game_detail_fetches": 0,
        "game_detail_fetch_errors": 0,
        "non_five_minute_details_skipped": 0,
        "non_bilateral_details_skipped": 0,
        "games_added": 0,
        "move_rows_added": 0,
        "summary_rows_added": 0,
        "rounds_started": 0,
        "new_games_by_source_user": {},
        "failed_user_ids": [],
        "failed_game_ids": [],
    }
    stats["last_run"] = run_stats

    if bilateral_before >= target_games:
        run_stats["target_reached"] = True
        run_stats["bilateral_games_after_run"] = bilateral_before
        run_stats["completed_at"] = utc_now_iso()
        stats["completed_at"] = run_stats["completed_at"]
        save_progress(progress_path, progress)
        print(json.dumps(run_stats, ensure_ascii=False, indent=2))
        return 0

    candidates: dict[str, list[dict[str, Any]]] = {}
    for index, rated_user in enumerate(eligible_users, start=1):
        print(
            f"[{index}/{len(eligible_users)}] list {rated_user.user_id} "
            f"existing_bilateral_games={existing_counts.get(rated_user.user_id, 0)}"
        )
        try:
            game_list = get_json(
                f"{BASE_URL}/games/{GTYPE}/{rated_user.user_id}.json",
                args.retries,
                args.timeout,
                args.delay,
            )
        except Exception as exc:
            if not args.continue_on_fetch_error:
                raise
            run_stats["user_list_fetch_errors"] += 1
            run_stats["failed_user_ids"].append(rated_user.user_id)
            run_stats["last_fetch_error"] = f"{type(exc).__name__}: {exc}"
            print(f"skip user after fetch error: {rated_user.user_id}: {exc}")
            continue
        run_stats["users_listed"] += 1
        listed_ids: set[str] = set()
        user_candidates: list[dict[str, Any]] = []
        games = game_list.get("games", []) if isinstance(game_list, dict) else []
        for game in games:
            if not is_normal_score_game(game):
                continue
            game_id = str(game.get("id", "")).strip()
            if not game_id or game_id in listed_ids or game_id in seen_games:
                continue
            listed_ids.add(game_id)
            summary_bilateral = both_players_are_ranked(game, rated_users)
            if summary_bilateral is False:
                continue
            user_candidates.append(game)
        candidates[rated_user.user_id] = user_candidates
        run_stats["candidate_games_after_summary_filter"] += len(user_candidates)
        time.sleep(args.delay)

    cursors = {rated_user.user_id: 0 for rated_user in eligible_users}
    current_bilateral_games = bilateral_before
    source_counts: dict[str, int] = defaultdict(int)
    examined_this_run: set[str] = set()

    while current_bilateral_games < target_games:
        run_stats["rounds_started"] += 1
        added_this_round = 0
        candidates_remaining = False
        for rated_user in eligible_users:
            if current_bilateral_games >= target_games:
                break
            user_id = rated_user.user_id
            user_candidates = candidates.get(user_id, [])
            while cursors[user_id] < len(user_candidates):
                candidates_remaining = True
                game = user_candidates[cursors[user_id]]
                cursors[user_id] += 1
                game_id = str(game.get("id", "")).strip()
                if not game_id or game_id in seen_games or game_id in examined_this_run:
                    continue
                examined_this_run.add(game_id)
                try:
                    detail = get_json(
                        f"{BASE_URL}/game/{game_id}.json",
                        args.retries,
                        args.timeout,
                        args.delay,
                    )
                except Exception as exc:
                    if not args.continue_on_fetch_error:
                        raise
                    run_stats["game_detail_fetch_errors"] += 1
                    run_stats["failed_game_ids"].append(game_id)
                    run_stats["last_fetch_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"skip game after fetch error: {game_id}: {exc}")
                    continue
                run_stats["game_detail_fetches"] += 1
                seen_games.add(game_id)
                if int(detail.get("tcb", 0) or 0) != EXPECTED_TCB:
                    run_stats["non_five_minute_details_skipped"] += 1
                    continue
                if both_players_are_ranked(detail, rated_users) is not True:
                    run_stats["non_bilateral_details_skipped"] += 1
                    continue

                fetched_at = utc_now_iso()
                move_rows, summary_rows, game_row = extract_rows(
                    game,
                    detail,
                    rated_users,
                    fetched_at,
                )
                if game_row is None or game_row.get("recorded_sides") != "black|white":
                    run_stats["non_bilateral_details_skipped"] += 1
                    continue
                append_rows(move_path, move_fields, move_rows)
                append_rows(summary_path, summary_fields, summary_rows)
                append_rows(games_path, game_fields, [game_row])
                existing_game_ids.add(game_id)
                current_bilateral_games += 1
                source_counts[user_id] += 1
                added_this_round += 1
                run_stats["games_added"] += 1
                run_stats["move_rows_added"] += len(move_rows)
                run_stats["summary_rows_added"] += len(summary_rows)
                run_stats["new_games_by_source_user"] = dict(sorted(source_counts.items()))
                run_stats["bilateral_games_after_latest_record"] = current_bilateral_games
                stats["game_details_recorded"] += 1
                stats["move_rows_recorded"] += len(move_rows)
                stats["summary_rows_recorded"] += len(summary_rows)
                stats["last_updated_at"] = utc_now_iso()
                run_stats["last_updated_at"] = stats["last_updated_at"]
                progress["completed_users"] = sorted(completed_users)
                progress["seen_games"] = sorted(seen_games)
                save_progress(progress_path, progress)
                print(
                    f"record {game_id}: source={user_id} "
                    f"bilateral={current_bilateral_games}/{target_games}"
                )
                time.sleep(args.delay)
                break

        if added_this_round == 0:
            if not candidates_remaining:
                break
            # Remaining entries were duplicates or failed the hard filters.
            if all(cursors[user.user_id] >= len(candidates.get(user.user_id, [])) for user in eligible_users):
                break

    completed_at = utc_now_iso()
    run_stats["bilateral_games_after_run"] = current_bilateral_games
    run_stats["target_reached"] = current_bilateral_games >= target_games
    run_stats["stopped_early_candidates_exhausted"] = current_bilateral_games < target_games
    run_stats["new_games_by_source_user"] = dict(sorted(source_counts.items()))
    run_stats["completed_at"] = completed_at
    stats["completed_at"] = completed_at
    stats["last_updated_at"] = completed_at
    progress["completed_users"] = sorted(completed_users)
    progress["seen_games"] = sorted(seen_games)
    save_progress(progress_path, progress)
    print(json.dumps(run_stats, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", default="research/oq_reversi_5min_rating_2000_users.csv")
    parser.add_argument("--out-dir", default="research/oq_reversi_5min_elo2000_games")
    parser.add_argument("--limit-users", type=int, default=0, help="debug limit; 0 means all users")
    parser.add_argument("--start-after-user", default="", help="skip users through this user id before starting")
    parser.add_argument("--delay", type=float, default=0.12, help="delay between HTTP requests")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--target-game-count",
        type=int,
        default=0,
        help="stop after this many recorded SCORE games with tcb=300000; 0 means no target",
    )
    parser.add_argument(
        "--target-bilateral-game-count",
        type=int,
        default=0,
        help="stop when this many strict games have both players in the current rating list",
    )
    parser.add_argument(
        "--balanced-bilateral-min-existing-games",
        type=int,
        default=0,
        help="activate bilateral round-robin mode for players with at least this many existing bilateral games",
    )
    parser.add_argument(
        "--sampling-strata",
        type=int,
        default=1,
        help="split ranked users into this many equal-size contiguous strata and scan them round-robin",
    )
    parser.add_argument(
        "--skip-strata",
        default="",
        help="comma-separated 1-based strata to skip, for example 1 or 1,3",
    )
    parser.add_argument(
        "--initial-stratum-game-counts",
        default="",
        help="comma-separated existing new-game counts for each sampling stratum",
    )
    parser.add_argument(
        "--stratum-game-targets",
        default="",
        help="comma-separated cumulative new-game targets; stop adding from a stratum when its target is reached",
    )
    parser.add_argument(
        "--rescan-completed",
        action="store_true",
        help="list users even if they were completed in a previous run; existing game ids are still skipped before detail fetch",
    )
    parser.add_argument("--direct", action="store_true", help="ignore environment and Windows application proxy settings")
    parser.add_argument(
        "--continue-on-fetch-error",
        action="store_true",
        help="record exhausted user-list or game-detail request errors and continue with the next candidate",
    )
    args = parser.parse_args()
    configure_http(args.direct)
    if args.balanced_bilateral_min_existing_games > 0 and args.target_game_count > 0:
        raise ValueError("balanced bilateral mode cannot be combined with --target-game-count")

    user_path = Path(args.users)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rated_users = load_rated_users(user_path)
    selected_users = list(rated_users.values())
    if args.start_after_user:
        start_after_user = args.start_after_user.strip().lower()
        selected_users = list(
            selected_users[
                next(
                    (
                        idx + 1
                        for idx, rated_user in enumerate(selected_users)
                        if rated_user.user_id == start_after_user
                    ),
                    0,
                ) :
            ]
        )
    if args.limit_users > 0:
        selected_users = selected_users[: args.limit_users]
    sampling_strata = max(1, int(args.sampling_strata))
    try:
        skip_strata = {
            int(text.strip()) - 1
            for text in str(args.skip_strata).split(",")
            if text.strip()
        }
    except ValueError as exc:
        raise ValueError("--skip-strata must be a comma-separated list of 1-based integers") from exc
    invalid_skip_strata = sorted(value + 1 for value in skip_strata if value < 0 or value >= sampling_strata)
    if invalid_skip_strata:
        raise ValueError(f"--skip-strata outside 1..{sampling_strata}: {invalid_skip_strata}")
    try:
        stratum_game_counts = (
            [int(text.strip()) for text in str(args.initial_stratum_game_counts).split(",")]
            if str(args.initial_stratum_game_counts).strip()
            else [0] * sampling_strata
        )
        stratum_game_targets = (
            [int(text.strip()) for text in str(args.stratum_game_targets).split(",")]
            if str(args.stratum_game_targets).strip()
            else []
        )
    except ValueError as exc:
        raise ValueError("stratum game counts and targets must be comma-separated integers") from exc
    if len(stratum_game_counts) != sampling_strata:
        raise ValueError(f"--initial-stratum-game-counts requires {sampling_strata} values")
    if stratum_game_targets and len(stratum_game_targets) != sampling_strata:
        raise ValueError(f"--stratum-game-targets requires {sampling_strata} values")
    if any(value < 0 for value in [*stratum_game_counts, *stratum_game_targets]):
        raise ValueError("stratum game counts and targets cannot be negative")
    if stratum_game_targets and any(
        count > target for count, target in zip(stratum_game_counts, stratum_game_targets)
    ):
        raise ValueError("an initial stratum count cannot exceed its target")
    stratified_users = stratified_round_robin_users(selected_users, sampling_strata, skip_strata)

    move_fields = [
        "game_id",
        "mode",
        "gtype",
        "tcb",
        "created",
        "finalStatus",
        "move_index",
        "ply",
        "color_to_move",
        "player_id",
        "player_name",
        "player_seed_rank",
        "player_seed_rating",
        "opponent_id",
        "opponent_is_elo2000_plus",
        "move",
        "thinking_time_ms",
        "has_terminal_status",
        "move_status",
        "fetched_at",
    ]
    summary_fields = [
        "game_id",
        "mode",
        "gtype",
        "tcb",
        "created",
        "finalStatus",
        "color",
        "player_id",
        "player_name",
        "player_seed_rank",
        "player_seed_rating",
        "opponent_id",
        "opponent_is_elo2000_plus",
        "recorded_move_count",
        "total_thinking_time_ms",
        "fetched_at",
    ]
    game_fields = [
        "game_id",
        "mode",
        "gtype",
        "tcb",
        "created",
        "finalStatus",
        "length",
        "black_id",
        "black_name",
        "black_rating_in_seed",
        "white_id",
        "white_name",
        "white_rating_in_seed",
        "recorded_sides",
        "fetched_at",
    ]

    move_path = out_dir / "move_times.csv"
    summary_path = out_dir / "game_player_summaries.csv"
    games_path = out_dir / "games.csv"
    progress_path = out_dir / "progress.json"
    ensure_csv(move_path, move_fields)
    ensure_csv(summary_path, summary_fields)
    ensure_csv(games_path, game_fields)

    if args.balanced_bilateral_min_existing_games > 0:
        return run_balanced_bilateral_pull(
            args,
            rated_users,
            move_path,
            summary_path,
            games_path,
            progress_path,
            move_fields,
            summary_fields,
            game_fields,
        )

    progress = load_progress(progress_path)
    completed_users = set(progress.get("completed_users", []))
    seen_games = set(progress.get("seen_games", []))
    seen_games.update(load_existing_game_ids(games_path))
    strict_game_count = count_existing_expected_tcb_games(games_path)
    stats = progress.setdefault("stats", {})
    stats.setdefault("mode", MODE_LABEL)
    stats.setdefault("gtype", GTYPE)
    stats.setdefault("expected_tcb", EXPECTED_TCB)
    stats.setdefault("normal_score_games_seen", 0)
    stats.setdefault("game_details_recorded", 0)
    stats.setdefault("move_rows_recorded", 0)
    stats.setdefault("summary_rows_recorded", 0)
    stats["last_started_at"] = utc_now_iso()
    run_stats = {
        "started_at": stats["last_started_at"],
        "selected_users": len(stratified_users),
        "sampling_strata": sampling_strata,
        "skip_strata_1_based": sorted(value + 1 for value in skip_strata),
        "initial_stratum_game_counts": list(stratum_game_counts),
        "stratum_game_targets": list(stratum_game_targets),
        "rescan_completed": bool(args.rescan_completed),
        "users_listed": 0,
        "users_skipped_completed": 0,
        "normal_score_games_seen": 0,
        "existing_games_skipped": 0,
        "detail_fetches": 0,
        "game_details_recorded": 0,
        "move_rows_recorded": 0,
        "summary_rows_recorded": 0,
        "non_five_minute_details_skipped": 0,
        "strict_five_minute_games_before_run": strict_game_count,
        "target_game_count": int(args.target_game_count),
        "user_list_fetch_errors": 0,
        "game_detail_fetch_errors": 0,
        "failed_user_ids": [],
        "failed_game_ids": [],
    }
    stats["last_run"] = run_stats

    target_reached = bool(args.target_game_count > 0 and strict_game_count >= args.target_game_count)
    for idx, (stratum, rated_user) in enumerate(stratified_users, start=1):
        if target_reached:
            break
        if stratum_game_targets and stratum_game_counts[stratum] >= stratum_game_targets[stratum]:
            continue
        if rated_user.user_id in completed_users and not args.rescan_completed:
            run_stats["users_skipped_completed"] += 1
            continue
        list_url = f"{BASE_URL}/games/{GTYPE}/{rated_user.user_id}.json"
        print(
            f"[{idx}/{len(stratified_users)}] stratum={stratum + 1}/{sampling_strata} "
            f"list {rated_user.user_id} rating={rated_user.rating}"
        )
        try:
            game_list = get_json(list_url, args.retries, args.timeout, args.delay)
        except Exception as exc:
            if not args.continue_on_fetch_error:
                raise
            run_stats["user_list_fetch_errors"] += 1
            run_stats["failed_user_ids"].append(rated_user.user_id)
            run_stats["last_fetch_error"] = f"{type(exc).__name__}: {exc}"
            stats["last_updated_at"] = utc_now_iso()
            run_stats["last_updated_at"] = stats["last_updated_at"]
            save_progress(progress_path, progress)
            print(f"skip user after fetch error: {rated_user.user_id}: {exc}")
            continue
        run_stats["users_listed"] += 1
        games = game_list.get("games", []) if isinstance(game_list, dict) else []
        for game in games:
            if stratum_game_targets and stratum_game_counts[stratum] >= stratum_game_targets[stratum]:
                break
            if not is_normal_score_game(game):
                continue
            stats["normal_score_games_seen"] += 1
            run_stats["normal_score_games_seen"] += 1
            game_id = str(game.get("id", ""))
            if not game_id or game_id in seen_games:
                run_stats["existing_games_skipped"] += 1
                continue
            detail_url = f"{BASE_URL}/game/{game_id}.json"
            try:
                detail = get_json(detail_url, args.retries, args.timeout, args.delay)
            except Exception as exc:
                if not args.continue_on_fetch_error:
                    raise
                run_stats["game_detail_fetch_errors"] += 1
                run_stats["failed_game_ids"].append(game_id)
                run_stats["last_fetch_error"] = f"{type(exc).__name__}: {exc}"
                stats["last_updated_at"] = utc_now_iso()
                run_stats["last_updated_at"] = stats["last_updated_at"]
                save_progress(progress_path, progress)
                print(f"skip game after fetch error: {game_id}: {exc}")
                continue
            run_stats["detail_fetches"] += 1
            if int(detail.get("tcb", 0) or 0) != EXPECTED_TCB:
                seen_games.add(game_id)
                run_stats["non_five_minute_details_skipped"] += 1
                progress["seen_games"] = sorted(seen_games)
                stats["last_updated_at"] = utc_now_iso()
                run_stats["last_updated_at"] = stats["last_updated_at"]
                save_progress(progress_path, progress)
                time.sleep(args.delay)
                continue
            fetched_at = utc_now_iso()
            move_rows, summary_rows, game_row = extract_rows(game, detail, rated_users, fetched_at)
            if game_row is None:
                seen_games.add(game_id)
                continue
            append_rows(move_path, move_fields, move_rows)
            append_rows(summary_path, summary_fields, summary_rows)
            append_rows(games_path, game_fields, [game_row])
            seen_games.add(game_id)
            stats["game_details_recorded"] += 1
            stats["move_rows_recorded"] += len(move_rows)
            stats["summary_rows_recorded"] += len(summary_rows)
            run_stats["game_details_recorded"] += 1
            run_stats["move_rows_recorded"] += len(move_rows)
            run_stats["summary_rows_recorded"] += len(summary_rows)
            strict_game_count += 1
            stratum_game_counts[stratum] += 1
            stats["strict_five_minute_games_recorded"] = strict_game_count
            run_stats["strict_five_minute_games_after_latest_record"] = strict_game_count
            run_stats["stratum_game_counts"] = list(stratum_game_counts)
            progress["completed_users"] = sorted(completed_users)
            progress["seen_games"] = sorted(seen_games)
            stats["last_updated_at"] = utc_now_iso()
            run_stats["last_updated_at"] = stats["last_updated_at"]
            save_progress(progress_path, progress)
            if args.target_game_count > 0 and strict_game_count >= args.target_game_count:
                target_reached = True
                break
            time.sleep(args.delay)

        if target_reached:
            break
        completed_users.add(rated_user.user_id)
        progress["completed_users"] = sorted(completed_users)
        progress["seen_games"] = sorted(seen_games)
        stats["last_completed_user"] = rated_user.user_id
        stats["last_updated_at"] = utc_now_iso()
        run_stats["last_updated_at"] = stats["last_updated_at"]
        save_progress(progress_path, progress)
        time.sleep(args.delay)

    stats["completed_at"] = utc_now_iso()
    stats["strict_five_minute_games_recorded"] = strict_game_count
    run_stats["strict_five_minute_games_after_run"] = strict_game_count
    run_stats["target_reached"] = target_reached
    run_stats["stratum_game_counts"] = list(stratum_game_counts)
    run_stats["stratum_targets_reached"] = bool(
        stratum_game_targets
        and all(count >= target for count, target in zip(stratum_game_counts, stratum_game_targets))
    )
    run_stats["completed_at"] = stats["completed_at"]
    progress["completed_users"] = sorted(completed_users)
    progress["seen_games"] = sorted(seen_games)
    save_progress(progress_path, progress)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
