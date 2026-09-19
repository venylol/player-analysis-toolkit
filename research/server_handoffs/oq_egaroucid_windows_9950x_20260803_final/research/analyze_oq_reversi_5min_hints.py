#!/usr/bin/env python3
"""
UNSAFE DELIVERY LINEAGE — DO NOT USE THIS FILE TO RESUME THE OLD RUN.

Its hint6 worker scheduling can interleave ``setboard`` and ``hint`` calls on the
same persistent engine, so a response can be attached to the wrong position.  The
2026-08-04 10,000-game delivery audit proved widespread illegal hint6 moves.  Keep
this file only for provenance.  The implementation below now documents the required
atomic position transaction and per-search board provenance for future successors,
but this historical entry point remains hard-disabled in ``main``.

Every future hint row must retain independent native Console response boards in
``hint1_board_setboard`` and ``hint6_board_setboard``, plus the corresponding
``*_request_board_setboard`` values, until the complete model-ready artifact passes
final acceptance.  Intermediate engine artifacts must not be deleted or recycled
earlier.

Analyze already collected OQ 5-minute games with Egaroucid hints.

By default, analyze each Elo2000+ player's move position from
research/oq_reversi_5min_elo2000_games/move_times.csv. With
--all-game-positions, incrementally fill every source position for both players:
  - n_legal_moves is calculated locally from the board, not from engine output.
  - level 2 hint 1 is searched with no book by default.
  - level 18 hint 6 is searched with book enabled.
  - nodes are recorded for both hint runs. Book hits may legitimately have 0 nodes.
  - existing (game_id, move_index) rows are reused instead of recalculated.
  - pass source rows are retained with empty hints; the adjacent playable
    same-side position supplies the score needed by downstream pass labels.

The script is resumable via progress.json in the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import threading
import time
from http.client import RemoteDisconnected
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen


MODE_LABEL = "reversi_5min"
GTYPE = "reversi"
EXPECTED_TCB = 300000
BASE_URL = "http://questgames.net"
USER_AGENT = "egaroucid-othello-quest-research/1.0"
HTTP_OPEN = urlopen


def configure_http(direct: bool) -> None:
    global HTTP_OPEN
    HTTP_OPEN = build_opener(ProxyHandler({})).open if direct else urlopen
MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)
HINT_ROW_RE = re.compile(r"^\|")


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


class OthelloBoard:
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self) -> None:
        self.board = [["-" for _ in range(8)] for _ in range(8)]
        self.board[3][3] = "O"
        self.board[3][4] = "X"
        self.board[4][3] = "X"
        self.board[4][4] = "O"
        self.current = "X"

    @staticmethod
    def opponent(color: str) -> str:
        return "O" if color == "X" else "X"

    def captures(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        if self.board[row][col] != "-":
            return []
        opponent = self.opponent(color)
        out: list[tuple[int, int]] = []
        for dr, dc in self.directions:
            rr = row + dr
            cc = col + dc
            line: list[tuple[int, int]] = []
            while 0 <= rr < 8 and 0 <= cc < 8 and self.board[rr][cc] == opponent:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < 8 and 0 <= cc < 8 and self.board[rr][cc] == color:
                out.extend(line)
        return out

    def legal_moves(self, color: str | None = None) -> list[str]:
        use_color = color or self.current
        moves: list[str] = []
        for row in range(8):
            for col in range(8):
                if self.captures(row, col, use_color):
                    moves.append(chr(ord("a") + col) + str(row + 1))
        return moves

    def normalize_turn(self) -> None:
        if self.legal_moves(self.current):
            return
        other = self.opponent(self.current)
        if self.legal_moves(other):
            self.current = other

    def apply_move(self, move: str) -> str:
        text = move.strip().lower()
        if text == "-":
            side = self.current
            if self.legal_moves(side):
                raise ValueError(f"pass is illegal for {side}: legal moves are available")
            self.current = self.opponent(self.current)
            return side
        if not MOVE_RE.match(text):
            raise ValueError(f"bad move: {move}")
        row = int(text[1]) - 1
        col = ord(text[0]) - ord("a")
        flips = self.captures(row, col, self.current)
        if not flips:
            raise ValueError(f"illegal move {move} for {self.current}")
        side = self.current
        self.board[row][col] = self.current
        for rr, cc in flips:
            self.board[rr][cc] = self.current
        self.current = self.opponent(self.current)
        return side

    def to_setboard_str(self) -> str:
        chars = []
        for row in range(8):
            for col in range(8):
                chars.append(self.board[row][col])
        chars.append(self.current)
        return "".join(chars)


class PersistentEngine:
    def __init__(
        self,
        engine_exe: Path,
        level: int,
        threads: int,
        hash_level: int,
        stderr_log: Path,
        use_book: bool,
    ) -> None:
        if not engine_exe.exists():
            raise FileNotFoundError(f"Egaroucid console not found: {engine_exe}")
        args = [
            str(engine_exe),
            "-l",
            str(level),
            "-t",
            str(threads),
            "-hash",
            str(hash_level),
            "-noautocacheclear",
        ]
        if not use_book:
            args.append("-nobook")
        # Do not add -q or -noboard.  Both must be absent so every hint response
        # contains Egaroucid's native board echo, which is parsed and retained as
        # independent evidence of the position actually searched.
        self.stderr_handle = stderr_log.open("a", encoding="utf-8")
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._buffer = ""
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            args,
            cwd=str(engine_exe.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            encoding="utf-8",
            errors="replace",
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open Egaroucid pipes")
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        self._wait_for_prompt(timeout=60.0)

    def _read_output(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                chunk = self.proc.stdout.read(1)
                if chunk == "":
                    break
                self._output_queue.put(chunk)
        finally:
            self._output_queue.put(None)

    @staticmethod
    def _prompt_index(text: str) -> int:
        match = re.search(r"(?:\A|\r?\n)>\s", text)
        return match.end() if match else -1

    @staticmethod
    def _strip_prompt(text: str) -> str:
        return re.sub(r"(?:\A|\r?\n)>\s\Z", "", text)

    def _wait_for_prompt(self, timeout: float) -> str:
        deadline = time.time() + timeout
        while True:
            idx = self._prompt_index(self._buffer)
            if idx >= 0:
                out = self._buffer[:idx]
                self._buffer = self._buffer[idx:]
                return self._strip_prompt(out)
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Egaroucid prompt")
            try:
                chunk = self._output_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Egaroucid exited with code {self.proc.returncode}")
                continue
            if chunk is None:
                raise RuntimeError("Egaroucid output stream closed")
            self._buffer += chunk

    def _command_unlocked(self, text: str, timeout: float) -> str:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text.rstrip() + "\n")
        self.proc.stdin.flush()
        output = self._wait_for_prompt(timeout=timeout)
        try:
            self.stderr_handle.write(output)
            self.stderr_handle.flush()
        except Exception:
            pass
        return output

    def command(self, text: str, timeout: float) -> str:
        with self._lock:
            return self._command_unlocked(text, timeout)

    def hint_for_board(
        self, board: str, n: int, timeout: float
    ) -> tuple[list[dict[str, Any]], str]:
        """Run one indivisible setboard+hint transaction on this engine."""
        with self._lock:
            self._command_unlocked(f"setboard {board}", timeout=30.0)
            output = self._command_unlocked(f"hint {n}", timeout=timeout)
        return parse_hint_output(output), parse_console_board_state(output)

    def setboard(self, board: str, timeout: float = 30.0) -> None:
        self.command(f"setboard {board}", timeout=timeout)

    def hint(self, n: int, timeout: float) -> list[dict[str, Any]]:
        output = self.command(f"hint {n}", timeout=timeout)
        return parse_hint_output(output)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.stderr_handle.close()


def parse_hint_output(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        text = line.strip()
        if not HINT_ROW_RE.match(text):
            continue
        parts = [part.strip() for part in text.split("|")[1:-1]]
        if len(parts) < 7 or parts[0] == "Level":
            continue
        score_text = parts[3].replace("+", "")
        try:
            score = int(score_text)
        except ValueError:
            score = None
        try:
            nodes = int(parts[5])
        except ValueError:
            nodes = None
        rows.append(
            {
                "level_text": parts[0],
                "depth": parts[1],
                "move": parts[2].lower(),
                "score": score,
                "time": parts[4],
                "nodes": nodes,
                "nps": parts[6],
                "is_book": parts[0].lower() == "book",
            }
        )
    return rows


def parse_console_board_state(output: str) -> str:
    """Parse the native 8x8 board and side-to-move echoed after a hint command."""
    board_rows: dict[int, str] = {}
    side = ""
    for line in output.splitlines():
        match = re.match(
            r"^\s*([1-8])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+"
            r"([.XO])\s+([.XO])\s+([.XO])\s+([.XO])(?:\s+.*)?$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            row_index = int(match.group(1))
            board_rows[row_index] = "".join(match.groups()[1:]).upper().replace(".", "-")
        side_match = re.search(r"\b(BLACK|WHITE)\s+to\s+move\b", line, flags=re.IGNORECASE)
        if side_match:
            parsed = "X" if side_match.group(1).upper() == "BLACK" else "O"
            if side and side != parsed:
                raise RuntimeError("conflicting side-to-move values in Console board echo")
            side = parsed
    if set(board_rows) != set(range(1, 9)) or not side:
        raise RuntimeError(f"hint response lacks one complete native Console board echo: {output!r}")
    return "".join(board_rows[index] for index in range(1, 9)) + side


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


def validate_engine_row_provenance(rows: list[dict[str, Any]]) -> None:
    """Reject completed placement rows whose searches lack exact board provenance."""
    for row in rows:
        if str(row.get("actual_move", "")) == "-":
            continue
        key = (row.get("game_id"), row.get("move_index"))
        board = str(row.get("board", ""))
        hint1_request_board = str(row.get("hint1_request_board_setboard", ""))
        hint1_board = str(row.get("hint1_board_setboard", ""))
        hint6_request_board = str(row.get("hint6_request_board_setboard", ""))
        hint6_board = str(row.get("hint6_board_setboard", ""))
        if (
            not board
            or hint1_request_board != board
            or hint1_board != board
            or hint6_request_board != board
            or hint6_board != board
        ):
            raise RuntimeError(
                f"engine board provenance mismatch for {key}: "
                f"board={board!r} hint1_request={hint1_request_board!r} "
                f"hint1_console={hint1_board!r} hint6_request={hint6_request_board!r} "
                f"hint6_console={hint6_board!r}"
            )
        legal_moves = set(str(row.get("legal_moves", "")).split())
        hint1_move = str(row.get("hint1_move", "")).lower()
        if hint1_move not in legal_moves:
            raise RuntimeError(f"illegal hint1 move for {key}: {hint1_move!r}")
        expected_hint6 = min(6, len(legal_moves))
        for rank in range(1, expected_hint6 + 1):
            move = str(row.get(f"hint6_{rank}_move", "")).lower()
            if move not in legal_moves:
                raise RuntimeError(f"missing or illegal hint6 rank {rank} for {key}: {move!r}")


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_games": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def load_games(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_target_moves(path: Path) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != MODE_LABEL or row.get("gtype") != GTYPE:
                continue
            if str(row.get("move", "")).strip() == "-":
                continue
            game_id = row.get("game_id", "")
            if not game_id:
                continue
            try:
                move_index = int(row["move_index"])
            except (KeyError, ValueError):
                continue
            targets.setdefault(game_id, set()).add(move_index)
    return targets


def load_all_game_position_indices(games: list[dict[str, str]]) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = {}
    for game in games:
        game_id = str(game.get("game_id", ""))
        if not game_id:
            continue
        try:
            length = int(game.get("length", 0) or 0)
        except ValueError:
            continue
        if length > 0:
            targets[game_id] = set(range(length))
    return targets


def load_existing_move_indices(path: Path) -> dict[str, set[int]]:
    existing: dict[str, set[int]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            game_id = str(row.get("game_id", ""))
            if not game_id:
                continue
            try:
                move_index = int(row["move_index"])
            except (KeyError, ValueError):
                continue
            existing.setdefault(game_id, set()).add(move_index)
    return existing


def load_games_completed_in_rows(path: Path, target_moves: dict[str, set[int]]) -> set[str]:
    if not path.exists():
        return set()
    existing: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            game_id = row.get("game_id", "")
            if game_id not in target_moves:
                continue
            try:
                move_index = int(row["move_index"])
            except (KeyError, ValueError):
                continue
            existing.setdefault(game_id, set()).add(move_index)
    return {
        game_id
        for game_id, expected_move_indices in target_moves.items()
        if expected_move_indices and expected_move_indices.issubset(existing.get(game_id, set()))
    }


def load_target_move_keys(path: Path) -> dict[str, set[tuple[int, str]]]:
    targets: dict[str, set[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != MODE_LABEL or row.get("gtype") != GTYPE:
                continue
            if str(row.get("move", "")).strip() == "-":
                continue
            game_id = row.get("game_id", "")
            player_id = str(row.get("player_id", "")).lower()
            if not game_id or not player_id:
                continue
            try:
                move_index = int(row["move_index"])
            except (KeyError, ValueError):
                continue
            targets.setdefault(game_id, set()).add((move_index, player_id))
    return targets


def build_rows_for_game(
    game: dict[str, str],
    detail: dict[str, Any],
    target_move_indices: set[int],
    engine_hint1: PersistentEngine,
    hint1_level: int,
    timeout_hint1: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    moves = ((detail.get("position") or {}).get("moves") or [])
    board = OthelloBoard()
    rows: list[dict[str, Any]] = []
    hint6_jobs: list[dict[str, Any]] = []
    game_id = game["game_id"]
    players = detail.get("players") or []
    black_id = str(game.get("black_id") or (players[0].get("id") if len(players) > 0 else "")).lower()
    white_id = str(game.get("white_id") or (players[1].get("id") if len(players) > 1 else "")).lower()

    for move_index, move in enumerate(m for m in moves if "m" in m):
        setboard = board.to_setboard_str()
        side = "black" if board.current == "X" else "white"
        player_id = black_id if side == "black" else white_id
        actual_move = str(move.get("m", "")).lower()

        if move_index not in target_move_indices:
            board.apply_move(actual_move)
            continue

        legal_moves = board.legal_moves()
        n_legal = len(legal_moves)
        if actual_move != "-" and n_legal == 0:
            # Terminal or malformed non-pass state.
            break
        base = {
            "game_id": game_id,
            "mode": MODE_LABEL,
            "gtype": GTYPE,
            "tcb": detail.get("tcb", game.get("tcb", "")),
            "created": game.get("created", detail.get("created", "")),
            "finalStatus": game.get("finalStatus", ""),
            "move_index": move_index,
            "ply": move_index + 1,
            "side_to_move": side,
            "player_id": player_id,
            "actual_move": actual_move,
            "actual_thinking_time_ms": int(move.get("t", 0) or 0),
            "board": setboard,
            "n_legal_moves": n_legal,
            "legal_moves": " ".join(legal_moves),
            "analyzed_at": utc_now_iso(),
        }

        if actual_move == "-":
            rows.append(base)
            board.apply_move(actual_move)
            continue

        hint1, hint1_console_board = engine_hint1.hint_for_board(
            setboard, 1, timeout=timeout_hint1
        )

        h1 = hint1[0] if hint1 else {}
        row = {
            **base,
            "hint1_request_board_setboard": setboard,
            "hint1_board_setboard": hint1_console_board,
            "hint1_level": hint1_level,
            "hint1_move": h1.get("move", ""),
            "hint1_score": h1.get("score", ""),
            "hint1_nodes": h1.get("nodes", ""),
            "hint1_depth": h1.get("depth", ""),
            "hint1_is_book": h1.get("is_book", ""),
        }
        rows.append(row)
        hint6_jobs.append({"row_index": len(rows) - 1, "board": setboard})
        board.apply_move(actual_move)
    return rows, hint6_jobs


def build_position_task_rows_for_game(
    game: dict[str, str],
    detail: dict[str, Any],
    target_move_indices: set[int],
) -> list[dict[str, Any]]:
    moves = ((detail.get("position") or {}).get("moves") or [])
    board = OthelloBoard()
    rows: list[dict[str, Any]] = []
    game_id = game["game_id"]
    players = detail.get("players") or []
    black_id = str(game.get("black_id") or (players[0].get("id") if len(players) > 0 else "")).lower()
    white_id = str(game.get("white_id") or (players[1].get("id") if len(players) > 1 else "")).lower()

    for move_index, move in enumerate(m for m in moves if "m" in m):
        setboard = board.to_setboard_str()
        side = "black" if board.current == "X" else "white"
        player_id = black_id if side == "black" else white_id
        actual_move = str(move.get("m", "")).lower()

        if move_index not in target_move_indices:
            board.apply_move(actual_move)
            continue

        legal_moves = board.legal_moves()
        n_legal = len(legal_moves)
        if actual_move != "-" and n_legal == 0:
            break

        rows.append(
            {
                "game_id": game_id,
                "mode": MODE_LABEL,
                "gtype": GTYPE,
                "tcb": detail.get("tcb", game.get("tcb", "")),
                "created": game.get("created", detail.get("created", "")),
                "finalStatus": game.get("finalStatus", ""),
                "move_index": move_index,
                "ply": move_index + 1,
                "side_to_move": side,
                "player_id": player_id,
                "actual_move": actual_move,
                "actual_thinking_time_ms": int(move.get("t", 0) or 0),
                "board": setboard,
                "n_legal_moves": n_legal,
                "legal_moves": " ".join(legal_moves),
                "task_created_at": utc_now_iso(),
            }
        )
        board.apply_move(actual_move)
    return rows


def add_hint6_results(
    rows: list[dict[str, Any]],
    hint6_jobs: list[dict[str, Any]],
    engines: list[PersistentEngine],
    timeout_l18: float,
) -> None:
    if not hint6_jobs:
        return
    if not engines:
        raise RuntimeError("no level18 workers are available")

    def run_job(
        job: dict[str, Any], engine: PersistentEngine
    ) -> tuple[int, str, str, list[dict[str, Any]]]:
        board = str(job["board"])
        hints, console_board = engine.hint_for_board(board, 6, timeout=timeout_l18)
        return (
            int(job["row_index"]),
            board,
            console_board,
            hints,
        )

    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = [
            executor.submit(run_job, job, engines[index % len(engines)])
            for index, job in enumerate(hint6_jobs)
        ]
        for future in as_completed(futures):
            row_index, hint6_request_board, hint6_console_board, hint6 = future.result()
            row = rows[row_index]
            row["hint6_request_board_setboard"] = hint6_request_board
            row["hint6_board_setboard"] = hint6_console_board
            for hint_index in range(1, 7):
                hint = hint6[hint_index - 1] if len(hint6) >= hint_index else {}
                row[f"hint6_{hint_index}_move"] = hint.get("move", "")
                row[f"hint6_{hint_index}_score"] = hint.get("score", "")
                row[f"hint6_{hint_index}_nodes"] = hint.get("nodes", "")
                row[f"hint6_{hint_index}_depth"] = hint.get("depth", "")
                row[f"hint6_{hint_index}_is_book"] = hint.get("is_book", "")


@dataclass
class EnginePair:
    hint1: PersistentEngine
    l18_workers: list[PersistentEngine]
    restart_count: int = 0
    positions_since_restart: int = 0

    def close(self) -> None:
        self.hint1.close()
        for worker in self.l18_workers:
            worker.close()


def open_engine_pair(args: argparse.Namespace, out_dir: Path) -> EnginePair:
    suffix = ""
    existing_restarts = int(getattr(args, "_engine_restart_count", 0) or 0)
    if existing_restarts:
        suffix = f".restart{existing_restarts}"
    l18_workers = []
    for worker_idx in range(max(1, int(args.level18_workers))):
        l18_workers.append(
            PersistentEngine(
                Path(args.engine),
                18,
                args.level18_threads,
                args.hash_level,
                out_dir / f"egaroucid_level18_book_worker{worker_idx + 1}{suffix}.log",
                use_book=True,
            )
        )
    return EnginePair(
        hint1=PersistentEngine(
            Path(args.engine),
            args.hint1_level,
            args.level6_threads,
            args.hash_level,
            out_dir / f"egaroucid_level{args.hint1_level}_hint1_nobook{suffix}.log",
            use_book=False,
        ),
        l18_workers=l18_workers,
        restart_count=existing_restarts,
        positions_since_restart=0,
    )


def main() -> int:
    raise RuntimeError(
        "UNSAFE PARALLEL HINT6 ANALYZER IS DISABLED: setboard+hint was not atomic; "
        "keep for provenance only"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="research/oq_reversi_5min_elo2000_games/games.csv")
    parser.add_argument("--target-moves", default="research/oq_reversi_5min_elo2000_games/move_times.csv")
    parser.add_argument(
        "--all-game-positions",
        action="store_true",
        help="incrementally analyze both players' positions in every selected game; pass rows are retained without engine hints",
    )
    parser.add_argument("--out-dir", default="research/oq_reversi_5min_elo2000_hints")
    parser.add_argument("--engine", default="C:/Users/MeroAF/Desktop/比赛编排/Egaroucid_for_Console_7_8_1_Windows_AVX512_AMD/Egaroucid_for_Console_7_8_1_AVX512_AMD.exe")
    parser.add_argument("--hint1-level", type=int, default=2)
    parser.add_argument("--level6-threads", type=int, default=1)
    parser.add_argument("--level18-workers", type=int, default=4)
    parser.add_argument("--level18-threads", type=int, default=16)
    parser.add_argument("--hash-level", type=int, default=25)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--write-batch-games", type=int, default=100, help="Flush CSV/progress after this many games")
    parser.add_argument("--engine-restart-positions", type=int, default=2000, help="Restart both engines after this many analyzed positions; 0 disables")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-http", type=float, default=20.0)
    parser.add_argument("--timeout-l6", type=float, default=60.0)
    parser.add_argument("--timeout-l18", type=float, default=240.0)
    parser.add_argument("--export-positions-only", action="store_true", help="Export board positions for the native analyzer and exit")
    parser.add_argument("--positions-out", default="research/oq_reversi_5min_elo2000_hints_tasks/position_tasks.csv")
    parser.add_argument("--direct", action="store_true", help="ignore environment and Windows application proxy settings")
    parser.add_argument(
        "--follow-until-game-count",
        type=int,
        default=0,
        help="keep discovering appended input games until this many tcb=300000 games exist and are analyzed",
    )
    parser.add_argument("--follow-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    configure_http(args.direct)
    if args.limit_games > 0 and args.follow_until_game_count > 0:
        raise ValueError("--limit-games cannot be combined with --follow-until-game-count")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    rows_path = out_dir / "position_hints.csv"

    fields = [
        "game_id", "mode", "gtype", "tcb", "created", "finalStatus",
        "move_index", "ply", "side_to_move", "player_id",
        "actual_move", "actual_thinking_time_ms",
        "board", "n_legal_moves", "legal_moves",
        "hint1_request_board_setboard", "hint1_board_setboard",
        "hint1_level", "hint1_move", "hint1_score", "hint1_nodes", "hint1_depth", "hint1_is_book",
        "hint6_request_board_setboard", "hint6_board_setboard",
    ]
    for i in range(1, 7):
        fields += [f"hint6_{i}_move", f"hint6_{i}_score", f"hint6_{i}_nodes", f"hint6_{i}_depth", f"hint6_{i}_is_book"]
    fields.append("analyzed_at")
    ensure_csv(rows_path, fields)

    games = load_games(Path(args.games))
    target_moves = (
        load_all_game_position_indices(games)
        if args.all_game_positions
        else load_target_moves(Path(args.target_moves))
    )
    games = [game for game in games if game.get("game_id") in target_moves]
    if args.limit_games > 0:
        games = games[: args.limit_games]

    if args.export_positions_only:
        target_move_keys = load_target_move_keys(Path(args.target_moves))
        task_fields = [
            "game_id", "mode", "gtype", "tcb", "created", "finalStatus",
            "move_index", "ply", "side_to_move", "player_id",
            "actual_move", "actual_thinking_time_ms",
            "board", "n_legal_moves", "legal_moves", "task_created_at",
        ]
        task_path = Path(args.positions_out)
        task_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_csv(task_path, task_fields)
        existing: set[tuple[str, int, str]] = set()
        with task_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    existing.add((row["game_id"], int(row["move_index"]), str(row["player_id"]).lower()))
                except (KeyError, ValueError):
                    continue
        written = 0
        for idx, game in enumerate(games, start=1):
            game_id = game["game_id"]
            expected_keys = {(game_id, move_index, player_id) for move_index, player_id in target_move_keys.get(game_id, set())}
            if expected_keys and expected_keys.issubset(existing):
                continue
            print(f"[{idx}/{len(games)}] export positions {game_id}")
            detail = get_json(f"{BASE_URL}/game/{game_id}.json", args.retries, args.timeout_http, args.delay)
            if int(detail.get("tcb", 0) or 0) != EXPECTED_TCB:
                continue
            final_status = str(game.get("finalStatus") or detail.get("finalStatus") or "")
            if not final_status.startswith("SCORE:"):
                continue
            rows = []
            for row in build_position_task_rows_for_game(game, detail, target_moves[game_id]):
                key = (str(row["game_id"]), int(row["move_index"]), str(row["player_id"]))
                if key not in existing:
                    existing.add(key)
                    rows.append(row)
            append_rows(task_path, task_fields, rows)
            written += len(rows)
            if idx % 20 == 0:
                print(f"exported_rows={written}")
            time.sleep(args.delay)
        print(f"exported_rows={written}")
        return 0

    progress = load_progress(progress_path)
    completed_from_rows = load_games_completed_in_rows(rows_path, target_moves)
    completed_games = (
        completed_from_rows
        if args.all_game_positions
        else set(progress.get("completed_games", [])) | completed_from_rows
    )
    existing_move_indices = load_existing_move_indices(rows_path)
    stats = progress.setdefault("stats", {})
    stats.setdefault("mode", MODE_LABEL)
    stats.setdefault("gtype", GTYPE)
    stats["hint1_level"] = args.hint1_level
    stats["hint1_uses_book"] = False
    stats["level6_uses_book"] = False
    stats["level18_uses_book"] = True
    stats["level6_threads"] = args.level6_threads
    stats["level18_workers"] = args.level18_workers
    stats["level18_threads_per_worker"] = args.level18_threads
    stats["hash_level"] = args.hash_level
    stats["position_scope"] = "all-game-positions" if args.all_game_positions else "elo2000-seed-player-positions"
    stats["write_batch_games"] = args.write_batch_games
    stats["engine_restart_positions"] = args.engine_restart_positions
    stats.setdefault("engine_restart_count", 0)
    stats.setdefault("games_completed", 0)
    stats.setdefault("positions_analyzed", 0)
    stats["last_started_at"] = utc_now_iso()

    setattr(args, "_engine_restart_count", int(stats.get("engine_restart_count", 0) or 0))
    engine_pair = open_engine_pair(args, out_dir)
    pending_rows: list[dict[str, Any]] = []
    pending_hint6_jobs: list[dict[str, Any]] = []
    pending_completed_games: list[str] = []
    write_batch_games = max(1, args.write_batch_games)

    def restart_engines() -> None:
        nonlocal engine_pair
        engine_pair.close()
        stats["engine_restart_count"] = int(stats.get("engine_restart_count", 0)) + 1
        setattr(args, "_engine_restart_count", int(stats["engine_restart_count"]))
        engine_pair = open_engine_pair(args, out_dir)

    def add_pending_hint6_results() -> None:
        job_index = 0
        restart_positions = int(args.engine_restart_positions)
        while job_index < len(pending_hint6_jobs):
            if restart_positions > 0:
                remaining = restart_positions - engine_pair.positions_since_restart
                if remaining <= 0:
                    restart_engines()
                    remaining = restart_positions
                jobs = pending_hint6_jobs[job_index : job_index + remaining]
            else:
                jobs = pending_hint6_jobs[job_index:]
            add_hint6_results(pending_rows, jobs, engine_pair.l18_workers, args.timeout_l18)
            engine_pair.positions_since_restart += len(jobs)
            job_index += len(jobs)
            if restart_positions > 0 and engine_pair.positions_since_restart >= restart_positions:
                restart_engines()
        pending_hint6_jobs.clear()

    def flush_pending() -> None:
        if not pending_completed_games:
            return
        add_pending_hint6_results()
        validate_engine_row_provenance(pending_rows)
        append_rows(rows_path, fields, pending_rows)
        for completed_game_id in pending_completed_games:
            completed_games.add(completed_game_id)
        stats["games_completed"] = int(stats.get("games_completed", 0)) + len(pending_completed_games)
        engine_position_count = sum(str(row.get("actual_move", "")) != "-" for row in pending_rows)
        pass_row_count = len(pending_rows) - engine_position_count
        stats["positions_analyzed"] = int(stats.get("positions_analyzed", 0)) + engine_position_count
        stats["source_rows_written"] = int(stats.get("source_rows_written", 0)) + len(pending_rows)
        stats["pass_rows_retained"] = int(stats.get("pass_rows_retained", 0)) + pass_row_count
        stats["last_completed_game"] = pending_completed_games[-1]
        stats["last_updated_at"] = utc_now_iso()
        progress["completed_games"] = sorted(completed_games)
        save_progress(progress_path, progress)
        pending_rows.clear()
        pending_completed_games.clear()

    try:
        while True:
            all_input_games = load_games(Path(args.games))
            target_moves = (
                load_all_game_position_indices(all_input_games)
                if args.all_game_positions
                else load_target_moves(Path(args.target_moves))
            )
            games = [game for game in all_input_games if game.get("game_id") in target_moves]
            if args.limit_games > 0:
                games = games[: args.limit_games]
            pending_games = [game for game in games if game.get("game_id") not in completed_games]

            for idx, game in enumerate(pending_games, start=1):
                game_id = game["game_id"]
                print(f"[{idx}/{len(pending_games)}] analyze {game_id}")
                detail = get_json(f"{BASE_URL}/game/{game_id}.json", args.retries, args.timeout_http, args.delay)
                if int(detail.get("tcb", 0) or 0) != EXPECTED_TCB:
                    print(f"skip {game_id}: tcb={detail.get('tcb')}")
                    pending_completed_games.append(game_id)
                    if len(pending_completed_games) >= write_batch_games:
                        flush_pending()
                    continue
                final_status = str(game.get("finalStatus") or detail.get("finalStatus") or "")
                if not final_status.startswith("SCORE:"):
                    print(f"skip {game_id}: finalStatus={final_status}")
                    pending_completed_games.append(game_id)
                    if len(pending_completed_games) >= write_batch_games:
                        flush_pending()
                    continue
                missing_move_indices = target_moves[game_id] - existing_move_indices.get(game_id, set())
                rows, hint6_jobs = build_rows_for_game(
                    game,
                    detail,
                    missing_move_indices,
                    engine_pair.hint1,
                    args.hint1_level,
                    args.timeout_l6,
                )
                row_offset = len(pending_rows)
                for job in hint6_jobs:
                    pending_hint6_jobs.append({**job, "row_index": int(job["row_index"]) + row_offset})
                pending_rows.extend(rows)
                existing_move_indices.setdefault(game_id, set()).update(
                    int(row["move_index"]) for row in rows
                )
                pending_completed_games.append(game_id)
                if len(pending_completed_games) >= write_batch_games:
                    flush_pending()
                time.sleep(args.delay)

            flush_pending()
            if args.follow_until_game_count <= 0:
                break

            strict_input_count = sum(
                int(game.get("tcb", 0) or 0) == EXPECTED_TCB
                and str(game.get("finalStatus", "")).startswith("SCORE:")
                for game in all_input_games
            )
            eligible_ids = {
                str(game.get("game_id", ""))
                for game in games
                if int(game.get("tcb", 0) or 0) == EXPECTED_TCB
                and str(game.get("finalStatus", "")).startswith("SCORE:")
            }
            stats["follow_strict_input_games"] = strict_input_count
            stats["follow_eligible_games"] = len(eligible_ids)
            stats["follow_pending_games"] = len(eligible_ids - completed_games)
            stats["follow_target_game_count"] = int(args.follow_until_game_count)
            stats["last_updated_at"] = utc_now_iso()
            progress["completed_games"] = sorted(completed_games)
            save_progress(progress_path, progress)
            if strict_input_count >= args.follow_until_game_count and eligible_ids.issubset(completed_games):
                break
            print(
                f"follow waiting: strict_games={strict_input_count}/{args.follow_until_game_count} "
                f"eligible_pending={len(eligible_ids - completed_games)}"
            )
            time.sleep(max(0.2, float(args.follow_poll_seconds)))
    finally:
        try:
            flush_pending()
        finally:
            engine_pair.close()

    stats["completed_at"] = utc_now_iso()
    progress["completed_games"] = sorted(completed_games)
    save_progress(progress_path, progress)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
