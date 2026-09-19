#!/usr/bin/env python3
"""Acquire and materialize the OQ corpus for the offline Transformer plan.

The network stages use twenty continuously replenished worker threads by default.
Only normal five-minute score games with a complete two-sided move clock are kept.
Existing one-sided-clock games are deliberately excluded rather than repaired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


GTYPE = "reversi"
TCB_MS = 300_000
LEADERBOARD_URL = "http://questgames.net:3002/socket.io/1"
LEADERBOARD_EVENT = "25764636"
GAME_BASE_URL = "http://questgames.net"
USER_AGENT = "player-analysis-toolkit-transformer-dataset/1.0"
MOVE_RE = re.compile(r"^[a-h][1-8]$")
DEFAULT_OUTPUT = Path("research/offbook_detection/data/oq_transformer_100000_20260813")
DEFAULT_BASE = Path(
    "research/tcn_loss_model/data/"
    "oq_elo2000_5min_bilateral_10000_source_only_20260804/handoff"
)
DEFAULT_EXTENSION = Path(
    "research/tcn_loss_model/outputs/oq_bilateral_extension_1200_hint_source_20260804"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_split(game_id: str) -> str:
    bucket = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            yield value


def repair_jsonl_preserving_original(path: Path) -> int:
    """Remove interleaved lines after a duplicate-process incident, retaining evidence."""
    if not path.exists():
        return 0
    valid_lines: list[str] = []
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(value, dict):
                valid_lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            else:
                invalid += 1
    if invalid == 0:
        return 0
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(f"{path.stem}.concurrent-corrupt-{stamp}{path.suffix}")
    os.replace(path, backup)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(valid_lines)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"repaired {path}: dropped={invalid} preserved_original={backup}", flush=True)
    return invalid


class HttpClient:
    def __init__(self, timeout: float, direct: bool) -> None:
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}) if direct else ProxyHandler())
        self._local = threading.local()

    def request_text(self, url: str, data: bytes | None = None) -> str:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "text/plain;charset=UTF-8"
        request = Request(url, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, RemoteDisconnected, ConnectionResetError) as exc:
            raise RuntimeError(f"request failed: {url}: {type(exc).__name__}: {exc}") from exc

    def json(self, url: str) -> Any:
        return json.loads(self.request_text(url))

    def leaderboard_page(self, page: int) -> dict[str, Any]:
        stamp = int(time.time() * 1000)
        handshake = self.request_text(f"{LEADERBOARD_URL}/?t={stamp}")
        session_id = handshake.split(":", 1)[0].strip()
        if not session_id:
            raise RuntimeError(f"leaderboard handshake returned no session id: {handshake[:100]!r}")
        poll = f"{LEADERBOARD_URL}/xhr-polling/{session_id}"
        self.request_text(f"{poll}?t={int(time.time() * 1000)}")
        packet = {
            "name": LEADERBOARD_EVENT,
            "args": [{"gtype": GTYPE, "page": str(page)}],
        }
        encoded = "5:::" + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        self.request_text(
            f"{poll}?t={int(time.time() * 1000)}",
            data=encoded.encode("utf-8"),
        )
        payload = self.request_text(f"{poll}?t={int(time.time() * 1000)}")
        if not payload.startswith("5:::"):
            raise RuntimeError(f"unexpected leaderboard payload: {payload[:100]!r}")
        decoded = json.loads(payload[4:])
        arguments = decoded.get("args") or []
        if decoded.get("name") != LEADERBOARD_EVENT or not arguments:
            raise RuntimeError("unexpected leaderboard event")
        result = arguments[0]
        if not isinstance(result, dict):
            raise RuntimeError("leaderboard result is not an object")
        return result


@dataclass(frozen=True)
class LeaderboardUser:
    rank: int
    user_id: str
    name: str
    rating: int
    page: int
    index_on_page: int
    stratum: int = -1


def run_dynamic(
    items: Iterable[Any],
    worker: Callable[[Any], Any],
    on_result: Callable[[Any, Any], bool],
    workers: int,
) -> list[tuple[Any, str]]:
    """Run a replenished worker pool; return failures without blocking a batch."""
    iterator = iter(items)
    failures: list[tuple[Any, str]] = []
    stop = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, Any] = {}
        for _ in range(workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(worker, item)] = item
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                item = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # isolated network task failure
                    failures.append((item, f"{type(exc).__name__}: {exc}"))
                else:
                    if on_result(item, result):
                        stop = True
                if not stop:
                    try:
                        next_item = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        pending[executor.submit(worker, next_item)] = next_item
    return failures


def fetch_leaderboard(client: HttpClient, output: Path, cutoff: int, workers: int) -> list[LeaderboardUser]:
    csv_path = output / "leaderboard.csv"
    json_path = output / "leaderboard.json"
    page_cache_path = output / "acquisition" / "leaderboard_pages.jsonl"
    if csv_path.exists() and json_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return [
                LeaderboardUser(
                    rank=int(row["rank"]), user_id=row["id"], name=row["name"],
                    rating=int(row["rating"]), page=int(row["page"]),
                    index_on_page=int(row["index_on_page"]), stratum=int(row["stratum"]) - 1,
                )
                for row in csv.DictReader(handle)
            ]

    pages: dict[int, dict[str, Any]] = {
        int(row["page"]): row["data"] for row in read_jsonl(page_cache_path)
        if "page" in row and isinstance(row.get("data"), dict)
    }
    boundary: int | None = None
    for cached_page, cached_data in pages.items():
        cached_users = cached_data.get("users") or []
        cached_ratings = [int(float(user.get("rating", 0))) for user in cached_users]
        if not cached_ratings or cached_ratings[-1] < cutoff:
            boundary = cached_page if boundary is None else min(boundary, cached_page)

    def page_numbers() -> Iterator[int]:
        page = 0
        while page < 10_000:
            if page not in pages:
                yield page
            page += 1

    def accept(page: int, data: dict[str, Any]) -> bool:
        nonlocal boundary
        raw_users = data.get("users") or []
        pages[page] = data
        append_jsonl(page_cache_path, {"page": page, "fetched_at": utc_now(), "data": data})
        ratings = [int(float(user.get("rating", 0))) for user in raw_users]
        if not ratings or ratings[-1] < cutoff:
            boundary = page if boundary is None else min(boundary, page)
        print(f"leaderboard page={page} count={len(raw_users)} last={ratings[-1] if ratings else None}", flush=True)
        return boundary is not None

    failures = [] if boundary is not None else run_dynamic(page_numbers(), client.leaderboard_page, accept, workers)
    if boundary is None:
        raise RuntimeError("leaderboard cutoff boundary was not found")
    missing = [page for page in range(boundary + 1) if page not in pages]
    for retry_round in range(5):
        if not missing:
            break
        def accept_retry(page: int, data: dict[str, Any]) -> bool:
            pages[page] = data
            append_jsonl(page_cache_path, {"page": page, "fetched_at": utc_now(), "data": data})
            return False
        retry_failures = run_dynamic(missing, client.leaderboard_page, accept_retry, workers)
        missing = [page for page, _ in retry_failures]
    if missing:
        raise RuntimeError(f"leaderboard pages failed after retries: {missing[:20]}")

    users: list[LeaderboardUser] = []
    page_audit = []
    for page in range(boundary + 1):
        data = pages[page]
        raw_users = data.get("users") or []
        ratings = [int(float(user.get("rating", 0))) for user in raw_users]
        kept = 0
        for index, raw_user in enumerate(raw_users):
            rating = int(float(raw_user.get("rating", 0)))
            if rating < cutoff:
                continue
            users.append(LeaderboardUser(
                rank=int(data.get("start", page * len(raw_users))) + index + 1,
                user_id=str(raw_user.get("id", "")).strip().lower(),
                name=str(raw_user.get("name", "")), rating=rating, page=page,
                index_on_page=index,
            ))
            kept += 1
        page_audit.append({
            "page": page, "start": data.get("start"), "count": len(raw_users), "kept": kept,
            "first_rating": ratings[0] if ratings else None,
            "last_rating": ratings[-1] if ratings else None,
        })
    users.sort(key=lambda user: (user.rank, user.user_id))
    count = len(users)
    users = [
        LeaderboardUser(**{**asdict(user), "stratum": min((index * 10) // max(count, 1), 9)})
        for index, user in enumerate(users)
    ]
    output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["rank", "id", "name", "rating", "page", "index_on_page", "stratum"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for user in users:
            writer.writerow({
                "rank": user.rank, "id": user.user_id, "name": user.name,
                "rating": user.rating, "page": user.page,
                "index_on_page": user.index_on_page, "stratum": user.stratum + 1,
            })
    atomic_json(json_path, {
        "schema": "oq-transformer-leaderboard-v1", "fetched_at": utc_now(),
        "cutoff_rating": cutoff, "count": count, "strata": 10,
        "pages": page_audit, "request_failures_seen": len(failures),
    })
    return users


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def existing_complete_ids(base: Path, extension: Path) -> tuple[set[str], set[str], set[str]]:
    base_ids = {
        row["game_id"] for row in read_csv_rows(base / "games.csv")
        if row.get("recorded_sides") == "black|white"
    }
    extension_ids = {
        row["game_id"] for row in read_csv_rows(extension / "games.csv")
        if row.get("recorded_sides") == "black|white"
    }
    overlap = base_ids & extension_ids
    if overlap:
        raise ValueError(f"existing complete cohorts overlap: {len(overlap)}")
    base_ids = clock_valid_existing_ids(base, base_ids)
    extension_ids = clock_valid_existing_ids(extension, extension_ids)
    return base_ids | extension_ids, base_ids, extension_ids


def clock_valid_existing_ids(directory: Path, candidate_ids: set[str]) -> set[str]:
    totals = {game_id: {"black": 0, "white": 0} for game_id in candidate_ids}
    for row in read_csv_rows(directory / "raw_nodes_with_pass.csv"):
        if row["game_id"] in totals:
            totals[row["game_id"]][row["side_to_move"]] += int(row["actual_thinking_time_ms"])
    return {
        game_id for game_id, clocks in totals.items()
        if clocks["black"] <= TCB_MS and clocks["white"] <= TCB_MS
    }


def stratified_user_order(users: list[LeaderboardUser], completed: set[str]) -> Iterator[LeaderboardUser]:
    buckets = [[user for user in users if user.stratum == stratum and user.user_id not in completed] for stratum in range(10)]
    for offset in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if offset < len(bucket):
                yield bucket[offset]


def normal_score(game: dict[str, Any]) -> bool:
    return str(game.get("finalStatus", "")).startswith("SCORE:")


def load_acquisition_state(acquisition: Path) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    completed_users: set[str] = set()
    candidates: dict[str, dict[str, Any]] = {}
    valid_details: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(acquisition / "player_lists.jsonl"):
        # A successful empty list and a timed-out/unavailable list are both final
        # for this acquisition run. Do not hold the corpus target hostage to one
        # inactive account or repeatedly spend its 20-second timeout.
        if row.get("user_id"):
            completed_users.add(str(row["user_id"]))
        if row.get("ok"):
            for candidate in row.get("candidates", []):
                game_id = str(candidate.get("game", {}).get("id", ""))
                if game_id:
                    candidates.setdefault(game_id, candidate)
    for row in read_jsonl(acquisition / "game_details.jsonl"):
        game_id = str(row.get("game_id", ""))
        if game_id and row.get("valid"):
            valid_details.setdefault(game_id, row)
    return completed_users, candidates, valid_details


def valid_detail(summary: dict[str, Any], detail: dict[str, Any]) -> tuple[bool, str]:
    if not normal_score(summary):
        return False, "not_score"
    if int(detail.get("tcb", 0) or 0) != TCB_MS:
        return False, "not_five_minute"
    players = detail.get("players") or summary.get("players") or []
    moves = (detail.get("position") or {}).get("moves") or []
    if len(players) != 2 or not isinstance(moves, list) or not moves:
        return False, "missing_players_or_moves"
    clock_totals = [0, 0]
    for index, move in enumerate(moves):
        if "m" not in move or "t" not in move:
            return False, "incomplete_two_sided_clock"
        try:
            clock_totals[index % 2] += int(move["t"])
        except (TypeError, ValueError):
            return False, "invalid_clock_value"
    if max(clock_totals) > TCB_MS:
        return False, "clock_overrun"
    return True, "ok"


def record_has_valid_clock(record: dict[str, Any]) -> bool:
    return valid_detail(record["summary"], record["detail"])[0]


def acquire_new_games(
    client: HttpClient,
    output: Path,
    users: list[LeaderboardUser],
    excluded_ids: set[str],
    needed: int,
    workers: int,
) -> dict[str, dict[str, Any]]:
    acquisition = output / "acquisition"
    acquisition.mkdir(parents=True, exist_ok=True)
    player_path = acquisition / "player_lists.jsonl"
    detail_path = acquisition / "game_details.jsonl"
    failed_path = acquisition / "failures.jsonl"
    repair_jsonl_preserving_original(detail_path)
    completed_users, candidates, valid_details = load_acquisition_state(acquisition)
    candidates = {game_id: row for game_id, row in candidates.items() if game_id not in excluded_ids}
    valid_details = {game_id: row for game_id, row in valid_details.items() if game_id not in excluded_ids}
    valid_details = {
        game_id: row for game_id, row in valid_details.items() if record_has_valid_clock(row)
    }
    detail_attempted: set[str] = {
        str(row.get("game_id", "")) for row in read_jsonl(detail_path) if row.get("game_id")
    }

    while len(valid_details) < needed:
        candidate_gap = needed - len({*valid_details, *candidates})
        if candidate_gap > 0:
            ordered = list(stratified_user_order(users, completed_users))
            if not ordered:
                raise RuntimeError(
                    f"all leaderboard users exhausted with {len(valid_details)}/{needed} valid new games"
                )

            def fetch_list(user: LeaderboardUser) -> dict[str, Any]:
                return client.json(f"{GAME_BASE_URL}/games/{GTYPE}/{user.user_id}.json")

            def accept_list(user: LeaderboardUser, payload: dict[str, Any]) -> bool:
                raw_games = payload.get("games", []) if isinstance(payload, dict) else []
                found = []
                for game in raw_games:
                    game_id = str(game.get("id", "")).strip()
                    if not game_id or game_id in excluded_ids or game_id in candidates or game_id in valid_details:
                        continue
                    if not normal_score(game):
                        continue
                    candidate = {
                        "game": game, "source_user_id": user.user_id,
                        "source_rank": user.rank, "source_rating": user.rating,
                        "source_stratum": user.stratum + 1,
                    }
                    candidates[game_id] = candidate
                    found.append(candidate)
                completed_users.add(user.user_id)
                append_jsonl(player_path, {
                    "ok": True, "fetched_at": utc_now(), "user_id": user.user_id,
                    "rank": user.rank, "rating": user.rating, "stratum": user.stratum + 1,
                    "listed_games": len(raw_games), "candidates": found,
                })
                print(
                    f"list user={user.user_id} stratum={user.stratum + 1} "
                    f"new={len(found)} candidates={len(candidates)} valid={len(valid_details)}/{needed}",
                    flush=True,
                )
                return len({*valid_details, *candidates}) >= needed

            failures = run_dynamic(ordered, fetch_list, accept_list, workers)
            for user, error in failures:
                completed_users.add(user.user_id)
                append_jsonl(player_path, {
                    "ok": False, "fetched_at": utc_now(), "user_id": user.user_id,
                    "rank": user.rank, "rating": user.rating, "stratum": user.stratum + 1,
                    "listed_games": 0, "candidates": [], "error": error,
                })
                append_jsonl(failed_path, {
                    "stage": "player_list", "at": utc_now(), "user_id": user.user_id,
                    "rank": user.rank, "stratum": user.stratum + 1, "error": error,
                })
            if not candidates:
                raise RuntimeError("player-list discovery produced no candidates")

        pending_candidates = [
            (game_id, row) for game_id, row in candidates.items()
            if game_id not in valid_details and game_id not in detail_attempted
        ]
        if not pending_candidates:
            # Invalid candidates have been consumed; discover another balanced tranche.
            candidates = dict(valid_details)
            continue

        def fetch_detail(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
            game_id, _ = item
            return client.json(f"{GAME_BASE_URL}/game/{game_id}.json")

        def accept_detail(item: tuple[str, dict[str, Any]], detail: dict[str, Any]) -> bool:
            game_id, candidate = item
            detail_attempted.add(game_id)
            ok, reason = valid_detail(candidate["game"], detail)
            record = {
                "game_id": game_id, "valid": ok, "reason": reason, "fetched_at": utc_now(),
                **{key: candidate[key] for key in (
                    "source_user_id", "source_rank", "source_rating", "source_stratum"
                )},
                "summary": candidate["game"], "detail": detail,
            }
            append_jsonl(detail_path, record)
            if ok:
                valid_details[game_id] = record
            print(f"detail game={game_id} {reason} valid={len(valid_details)}/{needed}", flush=True)
            return len(valid_details) >= needed

        failures = run_dynamic(pending_candidates, fetch_detail, accept_detail, workers)
        for item, error in failures:
            game_id, _ = item
            append_jsonl(failed_path, {
                "stage": "game_detail", "at": utc_now(), "game_id": game_id, "error": error,
            })
        # Failed details remain unattempted and are retried after other work; successful
        # invalid details remain excluded by detail_attempted.
        if failures and len(valid_details) < needed:
            for item, _ in failures:
                detail_attempted.discard(item[0])

        atomic_json(acquisition / "progress.json", {
            "schema": "oq-transformer-acquisition-progress-v1", "updated_at": utc_now(),
            "completed_users": len(completed_users), "candidates": len(candidates),
            "valid_new_games": len(valid_details), "needed_new_games": needed,
        })

    selected = dict(sorted(valid_details.items())[:needed])
    return selected


class OthelloBoard:
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self) -> None:
        self.board = [["-" for _ in range(8)] for _ in range(8)]
        self.board[3][3], self.board[3][4] = "O", "X"
        self.board[4][3], self.board[4][4] = "X", "O"
        self.current = "X"

    @staticmethod
    def opponent(color: str) -> str:
        return "O" if color == "X" else "X"

    def captures(self, row: int, column: int, color: str) -> list[tuple[int, int]]:
        if self.board[row][column] != "-":
            return []
        other = self.opponent(color)
        result: list[tuple[int, int]] = []
        for delta_row, delta_column in self.directions:
            scan_row, scan_column = row + delta_row, column + delta_column
            line: list[tuple[int, int]] = []
            while 0 <= scan_row < 8 and 0 <= scan_column < 8 and self.board[scan_row][scan_column] == other:
                line.append((scan_row, scan_column))
                scan_row += delta_row
                scan_column += delta_column
            if line and 0 <= scan_row < 8 and 0 <= scan_column < 8 and self.board[scan_row][scan_column] == color:
                result.extend(line)
        return result

    def legal_moves(self) -> list[str]:
        return [
            f"{chr(97 + column)}{row + 1}" for row in range(8) for column in range(8)
            if self.captures(row, column, self.current)
        ]

    def text(self) -> str:
        return "".join(cell for row in self.board for cell in row)

    def apply(self, move: str) -> None:
        move = move.strip().lower()
        if move == "-":
            if self.legal_moves():
                raise ValueError("pass with a legal move")
            self.current = self.opponent(self.current)
            return
        if not MOVE_RE.fullmatch(move):
            raise ValueError(f"bad move: {move}")
        row, column = int(move[1]) - 1, ord(move[0]) - 97
        flips = self.captures(row, column, self.current)
        if not flips:
            raise ValueError(f"illegal move {move} for {self.current}")
        self.board[row][column] = self.current
        for flip_row, flip_column in flips:
            self.board[flip_row][flip_column] = self.current
        self.current = self.opponent(self.current)


GAME_FIELDS = [
    "game_id", "source_cohort", "mode", "gtype", "tcb", "created", "final_status",
    "node_count", "black_id", "black_name", "white_id", "white_name",
    "source_user_id", "source_rank", "source_rating", "source_stratum", "split",
]
NODE_FIELDS = [
    "game_id", "source_cohort", "node_index", "strict_ply", "actor_color", "actor_id",
    "actor_name", "opponent_id", "is_pass", "move", "thinking_time_ms",
    "black_remaining_time_ms_after", "white_remaining_time_ms_after", "board_before",
    "n_legal_moves", "legal_moves", "tcb", "time_control_id", "split",
]


def materialize_existing(
    game_writer: csv.DictWriter,
    node_writer: csv.DictWriter,
    directory: Path,
    accepted_ids: set[str],
    cohort: str,
) -> tuple[int, int, int]:
    games = {row["game_id"]: row for row in read_csv_rows(directory / "games.csv") if row["game_id"] in accepted_ids}
    nodes_by_game: dict[str, list[dict[str, str]]] = {game_id: [] for game_id in games}
    for row in read_csv_rows(directory / "raw_nodes_with_pass.csv"):
        if row["game_id"] in nodes_by_game:
            nodes_by_game[row["game_id"]].append(row)
    node_count = pass_count = 0
    for game_id in sorted(games):
        game = games[game_id]
        rows = sorted(nodes_by_game[game_id], key=lambda row: int(row["move_index"]))
        if not rows:
            raise ValueError(f"existing game has no nodes: {game_id}")
        remaining = {"black": TCB_MS, "white": TCB_MS}
        game_writer.writerow({
            "game_id": game_id, "source_cohort": cohort, "mode": "reversi_5min",
            "gtype": GTYPE, "tcb": TCB_MS, "created": game["created"],
            "final_status": game["finalStatus"], "node_count": len(rows),
            "black_id": game["black_id"], "black_name": game["black_name"],
            "white_id": game["white_id"], "white_name": game["white_name"],
            "source_user_id": "", "source_rank": "", "source_rating": "",
            "source_stratum": "", "split": stable_split(game_id),
        })
        for row in rows:
            side = row["side_to_move"]
            elapsed = int(row["actual_thinking_time_ms"])
            remaining[side] -= elapsed
            if remaining[side] < 0:
                raise ValueError(f"negative remaining clock in existing game {game_id}")
            black_id, white_id = game["black_id"].lower(), game["white_id"].lower()
            actor_id = black_id if side == "black" else white_id
            opponent_id = white_id if side == "black" else black_id
            is_pass = int(row["is_pass_record"])
            node_writer.writerow({
                "game_id": game_id, "source_cohort": cohort,
                "node_index": int(row["move_index"]), "strict_ply": int(row["global_placement_ply"]),
                "actor_color": side, "actor_id": actor_id,
                "actor_name": game["black_name"] if side == "black" else game["white_name"],
                "opponent_id": opponent_id, "is_pass": is_pass, "move": row["actual_move"],
                "thinking_time_ms": elapsed,
                "black_remaining_time_ms_after": remaining["black"],
                "white_remaining_time_ms_after": remaining["white"],
                "board_before": row["board"], "n_legal_moves": row["n_legal_moves"],
                "legal_moves": row["legal_moves"], "tcb": TCB_MS,
                "time_control_id": "300000_no_increment", "split": stable_split(game_id),
            })
            node_count += 1
            pass_count += is_pass
    return len(games), node_count, pass_count


def materialize_new(
    game_writer: csv.DictWriter,
    node_writer: csv.DictWriter,
    records: dict[str, dict[str, Any]],
) -> tuple[int, int, int]:
    node_count = pass_count = 0
    for game_id, record in sorted(records.items()):
        summary, detail = record["summary"], record["detail"]
        players = detail.get("players") or summary.get("players")
        moves = (detail.get("position") or {}).get("moves")
        black_id = str(players[0].get("id", "")).strip().lower()
        white_id = str(players[1].get("id", "")).strip().lower()
        created = str(summary.get("created") or detail.get("created") or "")
        final_status = str(summary.get("finalStatus", ""))
        game_writer.writerow({
            "game_id": game_id, "source_cohort": "new_elo1600_20260813",
            "mode": "reversi_5min", "gtype": GTYPE, "tcb": TCB_MS,
            "created": created, "final_status": final_status, "node_count": len(moves),
            "black_id": black_id, "black_name": players[0].get("name", ""),
            "white_id": white_id, "white_name": players[1].get("name", ""),
            "source_user_id": record["source_user_id"], "source_rank": record["source_rank"],
            "source_rating": record["source_rating"], "source_stratum": record["source_stratum"],
            "split": stable_split(game_id),
        })
        board = OthelloBoard()
        remaining = {"black": TCB_MS, "white": TCB_MS}
        strict_ply = 0
        for node_index, move_record in enumerate(moves):
            move = str(move_record["m"]).strip().lower()
            side = "black" if board.current == "X" else "white"
            legal = board.legal_moves()
            is_pass = int(move == "-")
            if is_pass:
                if legal:
                    raise ValueError(f"pass with legal moves at {(game_id, node_index)}")
                pass_count += 1
            else:
                if move not in legal:
                    raise ValueError(f"illegal move at {(game_id, node_index)}: {move}")
                strict_ply += 1
            elapsed = int(move_record["t"])
            remaining[side] -= elapsed
            if remaining[side] < 0:
                raise ValueError(f"negative remaining clock in new game {game_id}")
            actor_index = 0 if side == "black" else 1
            actor_id = black_id if side == "black" else white_id
            opponent_id = white_id if side == "black" else black_id
            node_writer.writerow({
                "game_id": game_id, "source_cohort": "new_elo1600_20260813",
                "node_index": node_index, "strict_ply": strict_ply, "actor_color": side,
                "actor_id": actor_id, "actor_name": players[actor_index].get("name", ""),
                "opponent_id": opponent_id, "is_pass": is_pass, "move": move,
                "thinking_time_ms": elapsed,
                "black_remaining_time_ms_after": remaining["black"],
                "white_remaining_time_ms_after": remaining["white"],
                "board_before": board.text(), "n_legal_moves": len(legal),
                "legal_moves": " ".join(legal), "tcb": TCB_MS,
                "time_control_id": "300000_no_increment", "split": stable_split(game_id),
            })
            board.apply(move)
            node_count += 1
        if strict_ply > 60:
            raise ValueError(f"more than 60 placements in {game_id}")
    return len(records), node_count, pass_count


def materialize(
    output: Path,
    base: Path,
    extension: Path,
    base_ids: set[str],
    extension_ids: set[str],
    new_records: dict[str, dict[str, Any]],
    target: int,
) -> dict[str, Any]:
    games_path, nodes_path = output / "games.csv", output / "nodes.csv"
    split_path = output / "split_manifest.csv"
    with games_path.open("w", encoding="utf-8", newline="") as games_handle, nodes_path.open(
        "w", encoding="utf-8", newline=""
    ) as nodes_handle:
        game_writer = csv.DictWriter(games_handle, fieldnames=GAME_FIELDS)
        node_writer = csv.DictWriter(nodes_handle, fieldnames=NODE_FIELDS)
        game_writer.writeheader(); node_writer.writeheader()
        base_shape = materialize_existing(game_writer, node_writer, base, base_ids, "existing_base_bilateral")
        extension_shape = materialize_existing(
            game_writer, node_writer, extension, extension_ids, "existing_extension_bilateral"
        )
        new_shape = materialize_new(game_writer, node_writer, new_records)
    total_games = base_shape[0] + extension_shape[0] + new_shape[0]
    if total_games != target:
        raise ValueError(f"materialized {total_games} games, expected {target}")
    split_counts = {"train": 0, "validation": 0, "test": 0}
    with games_path.open("r", encoding="utf-8", newline="") as source, split_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=["game_id", "split"])
        writer.writeheader()
        seen: set[str] = set()
        for row in csv.DictReader(source):
            if row["game_id"] in seen:
                raise ValueError(f"duplicate final game id: {row['game_id']}")
            seen.add(row["game_id"])
            split_counts[row["split"]] += 1
            writer.writerow({"game_id": row["game_id"], "split": row["split"]})
    manifest = {
        "schema": "oq-small-transformer-change-point-dataset-v1", "ok": True,
        "created_at": utc_now(), "target_games": target, "games": total_games,
        "selection": {
            "game_type": GTYPE, "tcb_ms": TCB_MS, "normal_score_only": True,
            "leaderboard_cutoff": 1600, "leaderboard_strata": 10,
            "opponent_may_be_unranked": True, "complete_two_sided_clock_required": True,
            "old_one_sided_clock_games_excluded": True,
        },
        "cohorts": {
            "existing_base_bilateral": {"games": base_shape[0], "nodes": base_shape[1], "passes": base_shape[2]},
            "existing_extension_bilateral": {"games": extension_shape[0], "nodes": extension_shape[1], "passes": extension_shape[2]},
            "new_elo1600_20260813": {"games": new_shape[0], "nodes": new_shape[1], "passes": new_shape[2]},
        },
        "splits": split_counts,
        "node_contract": {
            "board_before": "64 cells before the decision; X is black and O is white",
            "strict_ply": "increments on placements only; pass retains the previous value",
            "remaining_time": "post-decision TCB minus cumulative side thinking time",
            "target_views": "construct two views per game by mapping actor_id to each target player",
        },
        "files": {},
    }
    for path in (games_path, nodes_path, split_path, output / "leaderboard.csv", output / "leaderboard.json"):
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-source", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--extension-source", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--target-games", type=int, default=20_000)
    parser.add_argument("--cutoff-rating", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--direct", action="store_true", default=True)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument(
        "--acquire-only", action="store_true",
        help="stop after enough valid game details are cached; do not replace final tables",
    )
    args = parser.parse_args()
    if args.workers != 20:
        raise ValueError("this acquisition is frozen to exactly 20 worker threads")
    if args.timeout != 20.0:
        raise ValueError("this acquisition is frozen to a 20-second request timeout")
    if args.cutoff_rating != 1600:
        raise ValueError("this dataset contract is frozen to Elo >= 1600")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete_ids, base_ids, extension_ids = existing_complete_ids(args.base_source, args.extension_source)
    needed = args.target_games - len(complete_ids)
    if needed <= 0:
        raise ValueError("existing complete games already meet or exceed the target")
    client = HttpClient(args.timeout, args.direct)
    users = fetch_leaderboard(client, args.output_dir, args.cutoff_rating, args.workers)
    if args.materialize_only:
        _, _, new_records = load_acquisition_state(args.output_dir / "acquisition")
        if len(new_records) < needed:
            raise ValueError(f"only {len(new_records)}/{needed} valid new games are cached")
        new_records = dict(sorted(new_records.items())[:needed])
    else:
        new_records = acquire_new_games(
            client, args.output_dir, users, complete_ids, needed, args.workers
        )
    if args.acquire_only:
        print(json.dumps({
            "ok": True, "stage": "acquisition_complete", "target_games": args.target_games,
            "existing_complete_games": len(complete_ids), "valid_new_games": len(new_records),
        }, ensure_ascii=False, indent=2))
        return 0
    manifest = materialize(
        args.output_dir, args.base_source, args.extension_source,
        base_ids, extension_ids, new_records, args.target_games,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; completed JSONL records remain resumable", file=sys.stderr)
        raise SystemExit(130)
