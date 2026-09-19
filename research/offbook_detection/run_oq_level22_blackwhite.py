#!/usr/bin/env python3
"""Run or resume the self-contained Level22 cache for the BW Reference.

Existing baseline engine files are independently validated and remapped into
the new directory.  Only newly selected game IDs are sent to Egaroucid.  The
new reference never reads the old engine directory after remapping completes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from pull_oq_transformer_dataset import OthelloBoard as SourceBoard
from oq_blackwhite_contract import load_expansion_config, resolve_repo_path


DATA = ROOT / "research" / "offbook_detection" / "data"
DEFAULT_REFERENCE = DATA / "oq_elo_matchup600_blackwhite_reference_level22_1600plus_20260911"
# The formal runtime baseline is the self-contained 600 chain.  Historical
# expansion configs may still override this with their old input, but a direct
# invocation against the current reference must never require the archived
# 550 directory.
CURRENT_REFERENCE = DEFAULT_REFERENCE
DEFAULT_ENGINE = Path(r"C:\Users\MeroAF\Desktop\比赛编排\Egaroucid_for_Console_7_8_1_Windows_AVX512_AMD\Egaroucid_for_Console_7_8_1_AVX512_AMD.exe")
MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)
WORKERS = 12
THREADS = 16
LEVEL = 22
HASH_LEVEL = 25
WLD_FROM_PLY = 39
LEVEL22_CONFIG: dict[str, Any] | None = None
LEVEL22_CONFIG_PATH: Path | None = None
LEVEL22_CONFIG_SHA256: str | None = None


def apply_level22_config(config_path: Path | None) -> dict[str, Any] | None:
    global LEVEL22_CONFIG, LEVEL22_CONFIG_PATH, LEVEL22_CONFIG_SHA256
    global CURRENT_REFERENCE, DEFAULT_REFERENCE, DEFAULT_ENGINE
    global WORKERS, THREADS, LEVEL, HASH_LEVEL, WLD_FROM_PLY
    if config_path is None:
        return None
    config, resolved_path, config_sha = load_expansion_config(config_path)
    LEVEL22_CONFIG = config
    LEVEL22_CONFIG_PATH = resolved_path
    LEVEL22_CONFIG_SHA256 = config_sha
    CURRENT_REFERENCE = Path(config["paths"]["baselineSourceReference"])
    DEFAULT_REFERENCE = Path(config["paths"]["sourceOutputDirectory"])
    DEFAULT_ENGINE = Path(config["paths"]["enginePath"])
    level22 = config["level22"]
    WORKERS = int(level22["workers"])
    THREADS = int(level22["threadsPerWorker"])
    LEVEL = int(level22["level"])
    HASH_LEVEL = int(level22["hash"])
    WLD_FROM_PLY = int(level22["wldFromPlyInclusive"])
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_moves(detail: dict[str, Any]) -> list[dict[str, Any]]:
    moves = (detail.get("position") or {}).get("moves") or []
    if not isinstance(moves, list):
        raise ValueError(f"source move list is not a list: {detail.get('id')}")
    return [item for item in moves if isinstance(item, dict)]


class Board:
    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    def __init__(self) -> None:
        self.board = [["-" for _ in range(8)] for _ in range(8)]
        self.board[3][3] = "O"
        self.board[3][4] = "X"
        self.board[4][3] = "X"
        self.board[4][4] = "O"
        self.current = "X"
        self.normalize_turn()

    @staticmethod
    def opponent(color: str) -> str:
        return "O" if color == "X" else "X"

    def captures(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        if self.board[row][col] != "-":
            return []
        opponent = self.opponent(color)
        out: list[tuple[int, int]] = []
        for dr, dc in self.directions:
            rr, cc = row + dr, col + dc
            line: list[tuple[int, int]] = []
            while 0 <= rr < 8 and 0 <= cc < 8 and self.board[rr][cc] == opponent:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < 8 and 0 <= cc < 8 and self.board[rr][cc] == color:
                out.extend(line)
        return out

    def legal_moves(self, color: str | None = None) -> list[tuple[int, int]]:
        use_color = color or self.current
        return [(r, c) for r in range(8) for c in range(8) if self.captures(r, c, use_color)]

    def normalize_turn(self) -> None:
        if self.legal_moves(self.current):
            return
        other = self.opponent(self.current)
        if self.legal_moves(other):
            self.current = other

    def apply_move(self, move: str) -> str:
        text = move.strip().lower()
        if not MOVE_RE.fullmatch(text):
            raise ValueError(f"invalid move: {move}")
        row, col = int(text[1]) - 1, ord(text[0]) - ord("a")
        flips = self.captures(row, col, self.current)
        if not flips:
            raise ValueError(f"illegal move {move} for {self.current}")
        side = self.current
        self.board[row][col] = side
        for rr, cc in flips:
            self.board[rr][cc] = side
        self.current = self.opponent(self.current)
        self.normalize_turn()
        return side

    def to_setboard_str(self) -> str:
        return "".join(self.board[row][col] for row in range(8) for col in range(8)) + self.current

    def final_scores(self) -> tuple[int, int]:
        black = sum(value == "X" for row in self.board for value in row)
        white = sum(value == "O" for row in self.board for value in row)
        empty = 64 - black - white
        if black > white:
            return black + empty, white
        if white > black:
            return black, white + empty
        return black + empty // 2, white + empty // 2


class PersistentEngine:
    def __init__(self, executable: Path, stderr_path: Path) -> None:
        self.executable = executable.resolve()
        self.stderr_handle = stderr_path.open("a", encoding="utf-8")
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self.buffer = ""
        self.lock = threading.Lock()
        args = [str(self.executable), "-q", "-noboard", "-l", str(LEVEL), "-t", str(THREADS), "-hash", str(HASH_LEVEL), "-noautocacheclear"]
        self.proc = subprocess.Popen(args, cwd=str(self.executable.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0, encoding="utf-8", errors="replace")
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open Egaroucid pipes")
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        self._wait_prompt()

    def _read_output(self) -> None:
        assert self.proc.stdout is not None
        while True:
            chunk = self.proc.stdout.read(1)
            if chunk == "":
                self.output_queue.put(None)
                return
            self.output_queue.put(chunk)

    @staticmethod
    def prompt_index(text: str) -> int:
        match = re.search(r"(?:\A|\r?\n)>\s", text)
        return match.end() if match else -1

    def _wait_prompt(self, timeout: float = 60.0) -> str:
        deadline = time.time() + timeout
        while True:
            index = self.prompt_index(self.buffer)
            if index >= 0:
                output, self.buffer = self.buffer[:index], self.buffer[index:]
                return re.sub(r"(?:\A|\r?\n)>\s\Z", "", output)
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Egaroucid prompt")
            try:
                chunk = self.output_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Egaroucid exited with code {self.proc.returncode}")
                continue
            if chunk is None:
                raise RuntimeError("Egaroucid output stream closed")
            self.buffer += chunk

    def command(self, text: str, timeout: float = 120.0) -> str:
        with self.lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(text.rstrip() + "\n")
            self.proc.stdin.flush()
            output = self._wait_prompt(timeout)
            self.stderr_handle.write(output)
            self.stderr_handle.flush()
            return output

    def setboard(self, board: str) -> None:
        self.command(f"setboard {board}")

    def play(self, move: str) -> None:
        self.command(f"play {move}")

    def hint(self) -> dict[str, Any]:
        output = self.command("hint 1")
        lines = [line.strip() for line in output.splitlines() if line.strip().startswith("|")]
        if len(lines) < 2:
            raise RuntimeError(f"unexpected Egaroucid hint output: {output!r}")
        body = [part.strip() for part in lines[1].split("|")[1:-1]]
        if len(body) < 4:
            raise RuntimeError(f"unexpected Egaroucid hint row: {lines[1]!r}")
        return {"bestMove": body[2].lower(), "bestEval": int(body[3].replace("+", "")), "depth": body[1]}

    def metadata(self, sha256: str) -> dict[str, Any]:
        return {"name": "Egaroucid for Console", "path": str(self.executable), "sha256": sha256, "level": LEVEL, "threads": THREADS, "hash": HASH_LEVEL, "book": "enabled-default"}

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


def timed_events(detail: dict[str, Any], black: dict[str, Any], white: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    engine_ply = 0
    for source_index, item in enumerate(source_moves(detail)):
        source_color = "black" if source_index % 2 == 0 else "white"
        move = str(item.get("m") or "").strip().lower()
        if MOVE_RE.fullmatch(move):
            event_type = "move"
            engine_ply += 1
            event_engine_ply: int | None = engine_ply
        elif move == "-":
            event_type = "pass"
            event_engine_ply = None
        else:
            event_type = "terminal_event"
            event_engine_ply = None
        player = black if source_color == "black" else white
        try:
            thinking = int(item.get("t")) if item.get("t") is not None else None
        except (TypeError, ValueError):
            thinking = None
        events.append({"sourceMoveIndex": source_index, "turnNumber": source_index + 1, "enginePly": event_engine_ply, "eventType": event_type, "move": move or None, "playerColor": source_color, "playerName": player.get("name", player.get("id", "")), "playerAccount": player.get("id", ""), "thinkingTimeMs": thinking, "status": item.get("s"), "delay": item.get("delay"), "bestMove": None, "bestEval": None, "actualEval": None, "lossPositive": None, "lossClipped": None, "lossSignedUser": None, "engineJudge": None, "boardBefore": None, "legalMoveCount": None, "bestDepth": None, "nextDepth": None})
    return events


def analyze_game(detail: dict[str, Any], engine: PersistentEngine, engine_sha: str) -> dict[str, Any]:
    players = detail.get("players") or []
    black, white = players[0], players[1]
    events = timed_events(detail, black, white)
    move_events = [event for event in events if event["eventType"] == "move"]
    board = Board()
    engine.setboard(board.to_setboard_str())
    nodes: list[dict[str, Any]] = []
    for ply, event in enumerate(move_events, start=1):
        move = str(event["move"])
        side_before = board.current
        board_before = board.to_setboard_str()
        legal_count = len(board.legal_moves())
        player_color = "black" if side_before == "X" else "white"
        if event["playerColor"] != player_color:
            raise RuntimeError(f"source turn mismatch at {detail.get('id')}:{event['sourceMoveIndex']}")
        current_hint = engine.hint()
        board.apply_move(move)
        engine.play(move)
        if not board.legal_moves("X") and not board.legal_moves("O"):
            black_score, white_score = board.final_scores()
            margin = black_score - white_score
            actual_eval = margin if side_before == "X" else -margin
            next_depth = "End"
        else:
            next_hint = engine.hint()
            actual_eval = int(next_hint["bestEval"]) if board.current == side_before else -int(next_hint["bestEval"])
            next_depth = next_hint.get("depth", "")
        best_eval = int(current_hint["bestEval"])
        loss_positive = best_eval - actual_eval
        loss_clipped = max(0, loss_positive)
        node = {"ply": ply, "plyGroup": math.ceil(ply / 2), "move": move, "playerColor": player_color, "playerName": black.get("name", black.get("id", "")) if player_color == "black" else white.get("name", white.get("id", "")), "playerAccount": black.get("id", "") if player_color == "black" else white.get("id", ""), "sourceMoveIndex": event["sourceMoveIndex"], "thinkingTimeMs": event["thinkingTimeMs"], "boardBefore": board_before, "legalMoveCount": legal_count, "bestMove": current_hint["bestMove"], "bestEval": best_eval, "actualEval": actual_eval, "lossPositive": loss_positive, "lossClipped": loss_clipped, "lossSignedUser": actual_eval - best_eval, "engineJudge": "Mistake" if loss_clipped >= 4 else "Disagree" if loss_clipped > 0 else "", "bestDepth": current_hint.get("depth", ""), "nextDepth": next_depth}
        nodes.append(node)
        for key in ("bestMove", "bestEval", "actualEval", "lossPositive", "lossClipped", "lossSignedUser", "engineJudge", "boardBefore", "legalMoveCount", "bestDepth", "nextDepth"):
            event[key] = node[key]
    players_summary = []
    for color, player in (("black", black), ("white", white)):
        values = [float(node["lossClipped"]) for node in nodes if node["playerColor"] == color]
        players_summary.append({"key": str(player.get("id") or "").strip().casefold(), "name": player.get("name", player.get("id", "")), "account": player.get("id", ""), "color": color, "ftdSide": color, "nodeCount": len(values), "totalLoss": round(sum(values), 3), "averageLoss": round(sum(values) / len(values), 3) if values else None})
    return {"schema": "ega-game-analysis-v1", "analyzedAt": utc_now(), "round": 0, "table": None, "gameId": str(detail.get("id") or ""), "source": "oq-account-bundle", "black": {"name": black.get("name", black.get("id", "")), "account": black.get("id", "")}, "white": {"name": white.get("name", white.get("id", "")), "account": white.get("id", "")}, "ftdBlack": {"name": black.get("name", black.get("id", "")), "account": black.get("id", "")}, "ftdWhite": {"name": white.get("name", white.get("id", "")), "account": white.get("id", "")}, "actualSideByFtdSide": {"black": "black", "white": "white"}, "engine": engine.metadata(engine_sha), "sourceEventCount": len(events), "moveCount": len(move_events), "passCount": sum(event["eventType"] == "pass" for event in events), "terminalEventCount": sum(event["eventType"] == "terminal_event" for event in events), "events": events, "nodes": nodes, "players": players_summary, "created": detail.get("created"), "finalStatus": detail.get("finalStatus"), "isTournamentGame": detail.get("isTournamentGame"), "tournament": detail.get("tournament"), "bundleWorkerId": None}


def validate_engine_game(game: dict[str, Any], detail: dict[str, Any], engine_sha: str) -> None:
    if str(game.get("gameId") or "") != str(detail.get("id") or ""):
        raise ValueError("engine gameId mismatch")
    source = source_moves(detail)
    coords = [(index, item) for index, item in enumerate(source) if MOVE_RE.fullmatch(str(item.get("m") or ""))]
    nodes = game.get("nodes") if isinstance(game.get("nodes"), list) else []
    events = game.get("events") if isinstance(game.get("events"), list) else []
    if len(nodes) != len(coords) or len(events) != len(source) or game.get("moveCount") != len(coords):
        raise ValueError("engine source/event/node count mismatch")
    for ply, (node, (source_index, item)) in enumerate(zip(nodes, coords, strict=True), start=1):
        if node.get("ply") != ply or int(node.get("sourceMoveIndex", -1)) != source_index or str(node.get("move") or "").lower() != str(item.get("m") or "").lower() or node.get("thinkingTimeMs") != item.get("t"):
            raise ValueError(f"engine move/time provenance mismatch at ply {ply}")
        for field in ("bestEval", "actualEval", "lossClipped"):
            if not isinstance(node.get(field), (int, float)) or isinstance(node.get(field), bool) or not math.isfinite(float(node[field])):
                raise ValueError(f"non-finite {field}")
    for index, (source_event, event) in enumerate(zip(source, events, strict=True)):
        if event.get("sourceMoveIndex") != index or event.get("thinkingTimeMs") != source_event.get("t"):
            raise ValueError(f"engine event provenance mismatch at {index}")
    contract = game.get("engine") or {}
    if contract.get("level") != LEVEL or contract.get("threads") != THREADS or contract.get("hash") != HASH_LEVEL or contract.get("book") != "enabled-default" or str(contract.get("sha256") or "").casefold() != engine_sha.casefold():
        raise ValueError("engine contract mismatch")


def load_inputs(reference: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    bundle = read_json(reference / "selected_account_bundle.json")
    details = bundle.get("details") or []
    rows = read_json(reference / "selected_games_with_partitions.json").get("games") or []
    by_detail = {str(detail.get("id") or ""): detail for detail in details}
    by_row = {str(row.get("gameId") or ""): row for row in rows}
    if len(by_detail) != len(details) or set(by_detail) != set(by_row):
        raise ValueError("source bundle and selected partition rows disagree")
    return by_detail, by_row, rows


def validate_existing_reference(reference: Path, details: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]], engine_sha: str) -> dict[str, Any]:
    """Validate the current self-contained reference without an old source.

    The 600 reference already contains every Level22 file, including the files
    inherited from the historical 550 expansion.  Once that historical input
    is archived, rerunning the remap step would be both unnecessary and
    destructive to the provenance story.  This path therefore validates the
    existing 600 artifact in place and performs no writes.
    """

    engine_dir = reference / "engine_level22"
    progress = read_json(engine_dir / "progress.json")
    if progress.get("complete") is not True or int(progress.get("completedCount", -1)) != len(details):
        raise ValueError("current 600 reference Level22 progress is incomplete")

    index_rows = read_json(reference / "engine_game_index.json").get("games") or []
    index = {str(row.get("gameId") or ""): row for row in index_rows}
    if len(index) != len(index_rows) or set(index) != set(details):
        raise ValueError("current 600 Level22 engine index does not match the source")
    if len({str(row.get("engineFile") or "") for row in index_rows}) != len(index_rows):
        raise ValueError("current 600 Level22 engine files are not one-to-one")

    for game_id, row in index.items():
        engine_file = (reference / str(row.get("engineFile") or "")).resolve()
        if reference not in engine_file.parents or not engine_file.is_file():
            raise ValueError(f"current 600 Level22 file is missing or outside the reference: {game_id}")
        if sha256_file(engine_file).casefold() != str(row.get("engineFileSha256") or "").casefold():
            raise ValueError(f"current 600 Level22 file SHA mismatch: {game_id}")
        validate_engine_game(read_json(engine_file), details[game_id], engine_sha)

    completion = read_json(reference / "reference_completion_audit.json")
    runner = read_json(engine_dir / "runner_audit.json")
    if completion.get("ok") is not True or runner.get("ok") is not True:
        raise ValueError("current 600 Level22 completion or runner audit is not ok=true")
    if int(runner.get("allSelectedGameCount", -1)) != len(details):
        raise ValueError("current 600 Level22 runner count does not match the source")

    return {
        "gameCount": len(details),
        "remappedBaseline": 0,
        "newGameCount": int(runner.get("newGameCount", -1)),
        "reusedBaselineGameCount": int(runner.get("reusedBaselineGameCount", -1)),
        "baselineMode": "current_600_self_contained",
        "completionAudit": completion,
    }


def remap_existing(reference: Path, old_reference: Path, details: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]], engine_sha: str) -> dict[str, Any]:
    old_bundle = read_json(old_reference / "selected_account_bundle.json")
    old_details = {str(detail.get("id") or ""): detail for detail in old_bundle.get("details") or []}
    old_index = {str(row.get("gameId") or ""): row for row in (read_json(old_reference / "engine_game_index.json").get("games") or [])}
    if not set(old_details) <= set(details) or set(old_index) != set(old_details):
        raise ValueError("new source does not retain exactly the configured baseline input required for engine reuse")
    engine_dir = reference / "engine_level22"
    engine_dir.mkdir(parents=True, exist_ok=True)
    remapped = 0
    for game_id in sorted(old_details):
        if old_details[game_id] != details[game_id]:
            raise ValueError(f"retained detail changed: {game_id}")
        source = old_reference / str(old_index[game_id]["engineFile"])
        if sha256_file(source).casefold() != str(old_index[game_id]["engineFileSha256"]).casefold():
            raise ValueError(f"old engine SHA mismatch: {game_id}")
        game = read_json(source)
        validate_engine_game(game, details[game_id], engine_sha)
        target = reference / str(rows[game_id]["expectedEngineFile"])
        if target.exists():
            validate_engine_game(read_json(target), details[game_id], engine_sha)
        else:
            game["table"] = int(rows[game_id]["bundleTable"])
            atomic_json(target, game)
        remapped += 1
    manifest = {"schema": "oq-reference-blackwhite-level22-cache-remap-v2", "ok": True, "createdAtUtc": utc_now(), "oldReference": str(old_reference.resolve()), "newReference": str(reference.resolve()), "configSha256": LEVEL22_CONFIG_SHA256, "engineSha256": engine_sha, "contract": {"level": LEVEL, "workers": WORKERS, "threadsPerConsole": THREADS, "hash": HASH_LEVEL, "book": "enabled-default", "wldFromPlyInclusive": WLD_FROM_PLY}, "sourceGameCount": len(old_details), "remappedGameCount": remapped, "checks": {"allCurrentMatchup500GamesRetained": True, "sourceDetailsIdentical": True, "oldEngineIndexHashesVerified": True, "moveAndTimeProvenanceVerified": True, "engineContractVerified": True}}
    atomic_json(engine_dir / "cache_remap_manifest.json", manifest)
    return manifest


def run_new_games(reference: Path, details: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]], engine_path: Path, engine_sha: str, progress: dict[str, Any]) -> dict[str, Any]:
    engine_dir = reference / "engine_level22"
    all_ids = sorted(details)
    # The per-game atomic files are the authoritative resume state.  A prior
    # process can have advanced the checkpoint before a file was fully
    # written, so do not trust a checkpoint ID until its file passes the full
    # engine contract validation below.  Rebuilding this set also prevents a
    # corrupt/partial file from being mistaken for a completed game.
    completed: set[str] = set()
    new_ids = [game_id for game_id in all_ids if rows[game_id].get("sourceKind") != "existingReference"]
    # Per-game atomic files are authoritative even if an interruption happens
    # between a game write and the next progress checkpoint.
    for game_id in all_ids:
        target = reference / str(rows[game_id]["expectedEngineFile"])
        if not target.is_file():
            continue
        try:
            validate_engine_game(json.loads(target.read_text(encoding="utf-8")), details[game_id], engine_sha)
        except Exception:
            continue
        completed.add(game_id)
    pending = [game_id for game_id in new_ids if game_id not in completed]
    lock = threading.Lock()
    errors: list[dict[str, Any]] = list(progress.get("errors") or [])

    def save_progress(force: bool = False) -> None:
        progress["completedGameIds"] = sorted(completed)
        progress["completedCount"] = len(completed)
        progress["totalCount"] = len(all_ids)
        progress["updatedAt"] = utc_now()
        progress["pendingNewCount"] = len([item for item in new_ids if item not in completed])
        progress["complete"] = len(completed) == len(all_ids)
        progress["errors"] = errors
        atomic_json(engine_dir / "progress.json", progress)

    save_progress(True)

    def worker(worker_id: int, assigned: list[str]) -> dict[str, Any]:
        stderr = engine_dir / f"worker_{worker_id:02d}.log"
        engine = PersistentEngine(engine_path, stderr)
        local_done = 0
        try:
            for game_id in assigned:
                target = reference / str(rows[game_id]["expectedEngineFile"])
                if game_id in completed:
                    continue
                last_error = None
                for attempt in range(1, 3):
                    try:
                        result = analyze_game(details[game_id], engine, engine_sha)
                        validate_engine_game(result, details[game_id], engine_sha)
                        result["table"] = int(rows[game_id]["bundleTable"])
                        result["bundleWorkerId"] = worker_id
                        atomic_json(target, result)
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt == 2:
                            with lock:
                                errors.append({"gameId": game_id, "workerId": worker_id, "attempts": attempt, "terminal": True, "error": last_error, "atUtc": utc_now()})
                    else:
                        with lock:
                            completed.add(game_id)
                        local_done += 1
                        break
                if game_id not in completed and last_error:
                    continue
                if local_done % 5 == 0:
                    with lock:
                        save_progress()
        finally:
            engine.close()
        return {"workerId": worker_id, "completed": local_done, "assigned": len(assigned)}

    assignments = [pending[index::WORKERS] for index in range(WORKERS)]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(worker, index, assignment) for index, assignment in enumerate(assignments)]
        for future in as_completed(futures):
            results.append(future.result())
            with lock:
                save_progress()
    save_progress(True)
    if errors or len(completed) != len(all_ids):
        raise RuntimeError(f"Level22 did not complete: completed={len(completed)}/{len(all_ids)} terminalErrors={len(errors)}")
    return {"workers": sorted(results, key=lambda row: row["workerId"]), "completed": len(completed), "newGameCount": len(new_ids)}


def build_wld_and_indexes(reference: Path, details: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]], engine_sha: str, engine_path: Path) -> dict[str, Any]:
    engine_dir = reference / "engine_level22"
    index_rows: list[dict[str, Any]] = []
    wld_rows: list[dict[str, Any]] = []
    node_count = 0
    event_count = 0
    for game_id in sorted(details):
        target = reference / str(rows[game_id]["expectedEngineFile"])
        game = read_json(target)
        validate_engine_game(game, details[game_id], engine_sha)
        file_hash = sha256_file(target)
        node_count += len(game["nodes"])
        event_count += len(game["events"])
        index_rows.append({"gameId": game_id, "bundleTable": int(rows[game_id]["bundleTable"]), "sourceKind": rows[game_id]["sourceKind"], "partitionScope": rows[game_id]["partitionScope"], "blackWhiteCellKey": rows[game_id].get("blackWhiteCellKey"), "unorderedPartitionKey": rows[game_id].get("unorderedPartitionKey"), "blackTargetDirectedPartition": rows[game_id].get("blackTargetDirectedPartition"), "whiteTargetDirectedPartition": rows[game_id].get("whiteTargetDirectedPartition"), "engineFile": str(target.relative_to(reference)).replace("\\", "/"), "engineFileSha256": file_hash, "nodeCount": len(game["nodes"]), "eventCount": len(game["events"])})
        players = details[game_id].get("players") or []
        for color, player in (("black", players[0]), ("white", players[1])):
            total = sum(float(node.get("lossClipped") or 0) for node in game["nodes"] if node.get("playerColor") == color and int(node.get("ply") or 0) >= WLD_FROM_PLY)
            wld_rows.append({"game_id": game_id, "player_id": str(player.get("id") or ""), "side": color, "engine_wld_loss_total_from_ply39": round(total, 6)})
    wld_rows.sort(key=lambda row: (row["game_id"], row["side"]))
    atomic_json(engine_dir / "engine_wld_loss_totals_from_ply39.json", {"schema": "player-engine-wld-loss-totals-v1", "wldFromPly": WLD_FROM_PLY, "globalPlacementPlyPolicy": "pass-free actual placement ply; boundary is inclusive", "gamePlayerTotals": wld_rows, "playerTotals": [], "configSha256": LEVEL22_CONFIG_SHA256})
    write_csv(engine_dir / "engine_wld_loss_totals_by_game_player_from_ply39.csv", wld_rows)
    atomic_json(reference / "engine_game_index.json", {"schema": "oq-reference-engine-game-index-v2", "games": index_rows, "configSha256": LEVEL22_CONFIG_SHA256})
    write_csv(reference / "engine_game_index.csv", [{key: value for key, value in row.items()} for row in index_rows])
    by_cell: defaultdict[str, list[str]] = defaultdict(list)
    by_unordered: defaultdict[str, list[str]] = defaultdict(list)
    by_directed: defaultdict[str, list[str]] = defaultdict(list)
    for row in index_rows:
        if row.get("partitionScope") == "main_bilateral":
            by_cell[str(row.get("blackWhiteCellKey"))].append(row["gameId"])
            by_unordered[str(row.get("unorderedPartitionKey"))].append(row["gameId"])
            by_directed[str(row.get("blackTargetDirectedPartition"))].append(row["gameId"])
            by_directed[str(row.get("whiteTargetDirectedPartition"))].append(row["gameId"])
    def partition_index(values: dict[str, list[str]], dimension: str) -> dict[str, Any]:
        return {"schema": "oq-reference-partition-engine-index-v2", "partitionDimension": dimension, "partitions": [{"partitionKey": key, "gameCount": len(sorted(set(ids))), "gameIds": sorted(set(ids)), "engineFiles": [str(rows[game_id]["expectedEngineFile"]) for game_id in sorted(set(ids))]} for key, ids in sorted(values.items())]}
    atomic_json(reference / "partitions_black_white_engine_index.json", partition_index(by_cell, "black_white_directed"))
    atomic_json(reference / "partitions_unordered_engine_index.json", partition_index(by_unordered, "unordered_legacy_summary"))
    atomic_json(reference / "partitions_directed_engine_index.json", partition_index(by_directed, "target_opponent_compatibility"))
    partition_audit = {"schema": "oq-reference-partition-engine-index-audit-v2", "ok": True, "verifiedAtUtc": utc_now(), "configSha256": LEVEL22_CONFIG_SHA256, "gameCount": len(index_rows), "existingReferenceGameCount": sum(row["sourceKind"] == "existingReference" for row in index_rows), "cacheSelectedGameCount": sum(row["sourceKind"] == "validatedCacheExpansion" for row in index_rows), "snapshotSelectedGameCount": sum(row["sourceKind"] == "uniqueSnapshotExpansion" for row in index_rows), "addedGameCount": sum(row["sourceKind"] != "existingReference" for row in index_rows), "mainMatrixGameCount": sum(row["partitionScope"] == "main_bilateral" for row in index_rows), "lowEloExtensionGameCount": sum(row["partitionScope"] == "baseline_low_elo_extension" for row in index_rows), "outsideUnpartitionedGameCount": sum(row["partitionScope"] == "outside_unpartitioned" for row in index_rows), "uniqueEngineFileCount": len({row["engineFile"] for row in index_rows}), "engineDirectory": str(engine_dir.resolve()), "partitionEngineIndexes": [str((reference / name).resolve()) for name in ("partitions_black_white_engine_index.json", "partitions_unordered_engine_index.json", "partitions_directed_engine_index.json")], "checks": {"oneEngineFilePerGame": len(index_rows) == len({row["gameId"] for row in index_rows}) == len({row["engineFile"] for row in index_rows}), "blackWhiteIndexIsSourceDimension": True, "unorderedAndDirectedAreDerivedViews": True}}
    atomic_json(reference / "partition_engine_index_audit.json", partition_audit)
    contract = {"level": LEVEL, "workers": WORKERS, "threadsPerConsole": THREADS, "hash": HASH_LEVEL, "book": "enabled-default", "wldFromPlyInclusive": WLD_FROM_PLY, "enginePath": str(engine_path), "engineSha256": engine_sha, "sourceBundle": str((reference / "selected_account_bundle.json").resolve()), "sourceBundleSha256": sha256_file(reference / "selected_account_bundle.json")}
    audit = {"schema": "ega-level22-parallel-audit-v2", "ok": True, "createdAtUtc": utc_now(), "configSha256": LEVEL22_CONFIG_SHA256, "gameCount": len(index_rows), "workerCount": WORKERS, "threadsPerConsole": THREADS, "engineLevel": LEVEL, "hashLevel": HASH_LEVEL, "book": "enabled-default", "enginePath": str(engine_path), "engineSha256": engine_sha, "sourceBundle": contract["sourceBundle"], "sourceBundleSha256": contract["sourceBundleSha256"], "nodeCount": node_count, "eventCount": event_count, "wldGameCount": len({row["game_id"] for row in wld_rows}), "contract": contract, "checks": ["reused baseline Level22 files were validated before remapping", "new game files were produced at Level22 with 12 workers and 16 threads per console", "WLD ply39 boundary is inclusive", "source-to-engine move/time provenance is complete"]}
    summary = {"schema": "ega-level22-parallel-summary-v2", "ok": True, "createdAtUtc": utc_now(), "configSha256": LEVEL22_CONFIG_SHA256, "gameCount": len(index_rows), "workerCount": WORKERS, "threadsPerConsole": THREADS, "engineLevel": LEVEL, "hashLevel": HASH_LEVEL, "book": "enabled-default", "nodeCount": node_count, "eventCount": event_count, "wldGameCount": len({row["game_id"] for row in wld_rows})}
    atomic_json(engine_dir / "audit.json", audit)
    atomic_json(engine_dir / "summary.json", summary)
    atomic_json(engine_dir / "runner_audit.json", {"schema": "oq-reference-blackwhite-level22-runner-audit-v2", "ok": True, "createdAtUtc": utc_now(), "configSha256": LEVEL22_CONFIG_SHA256, "contract": contract, "newGameCount": sum(row["sourceKind"] != "existingReference" for row in index_rows), "reusedBaselineGameCount": sum(row["sourceKind"] == "existingReference" for row in index_rows), "allSelectedGameCount": len(index_rows), "wldFromPlyInclusive": WLD_FROM_PLY})
    return {"indexRows": index_rows, "wldRows": wld_rows, "nodeCount": node_count, "eventCount": event_count, "audit": audit}


def write_completion_artifacts(
    reference: Path,
    details: dict[str, dict[str, Any]],
    built: dict[str, Any],
    engine_path: Path,
    engine_sha: str,
) -> dict[str, Any]:
    """Write the source completion certificate and self-excluded SHA manifest."""

    progress_path = reference / "engine_level22" / "progress.json"
    progress = read_json(progress_path)
    game_count = len(details)
    wld_game_count = len({str(row.get("game_id") or "") for row in built["wldRows"]})
    contract = {
        "level": LEVEL,
        "workers": WORKERS,
        "threadsPerConsole": THREADS,
        "hash": HASH_LEVEL,
        "book": "enabled-default",
        "wldFromPlyInclusive": WLD_FROM_PLY,
        "enginePath": str(engine_path.resolve()),
        "engineSha256": engine_sha,
        "sourceBundle": str((reference / "selected_account_bundle.json").resolve()),
        "sourceBundleSha256": sha256_file(reference / "selected_account_bundle.json"),
    }
    completion = {
        "schema": "oq-reference-level22-completion-audit-v2",
        "ok": bool(
            progress.get("complete") is True
            and int(progress.get("completedCount", -1)) == game_count
            and len(built["indexRows"]) == game_count
            and len({str(row.get("gameId") or "") for row in built["indexRows"]}) == game_count
            and len({str(row.get("engineFile") or "") for row in built["indexRows"]}) == game_count
            and wld_game_count == game_count
        ),
        "verifiedAtUtc": utc_now(),
        "configSha256": LEVEL22_CONFIG_SHA256,
        "gameCount": game_count,
        "gameFileCount": len(built["indexRows"]),
        "nodeCount": built["nodeCount"],
        "eventCount": built["eventCount"],
        "wldGameCount": wld_game_count,
        "engineDirectory": str((reference / "engine_level22").resolve()),
        "progressSnapshot": {
            "complete": progress.get("complete"),
            "completedCount": progress.get("completedCount"),
            "totalCount": progress.get("totalCount"),
            "updatedAt": progress.get("updatedAt"),
            "authority": "complete" if progress.get("complete") else "incomplete",
        },
        "contract": contract,
        "checks": [
            "exactly one complete game_*.json per selected gameId",
            "continuous pass-free global actual ply and exact move/sourceMoveIndex/thinkingTimeMs provenance",
            "finite bestEval, actualEval, and lossClipped at every coordinate move",
            "summary, runner audit, exact game files, engine contract, and source hashes agree",
            "WLD ply39 output covers every selected game with unique game/side rows",
        ],
    }
    completion_path = reference / "reference_completion_audit.json"
    atomic_json(completion_path, completion)
    final_manifest = reference / "final_sha256_manifest.json"
    files = []
    for path in sorted(item for item in reference.rglob("*") if item.is_file() and item != final_manifest):
        files.append({
            "path": path.relative_to(reference).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    atomic_json(final_manifest, {
        "schema": "oq-reference-final-sha256-manifest-v2",
        "createdAtUtc": utc_now(),
        "configSha256": LEVEL22_CONFIG_SHA256,
        "referenceDirectory": str(reference.resolve()),
        "fileCount": len(files),
        "files": files,
        "selfHashPolicy": "final_sha256_manifest.json is excluded to avoid recursive self-hash",
    })
    return {"completion": completion, "manifestSha256": sha256_file(final_manifest), "fileCount": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--old-reference", type=Path, default=CURRENT_REFERENCE)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--config", type=Path, help="UTF-8 configuration-driven Level22 contract")
    args = parser.parse_args()
    reference_was_default = args.reference == DEFAULT_REFERENCE
    old_reference_was_default = args.old_reference == CURRENT_REFERENCE
    engine_was_default = args.engine == DEFAULT_ENGINE
    apply_level22_config(args.config)
    if LEVEL22_CONFIG is not None and reference_was_default:
        args.reference = Path(LEVEL22_CONFIG["paths"]["sourceOutputDirectory"])
    if LEVEL22_CONFIG is not None and old_reference_was_default:
        args.old_reference = Path(LEVEL22_CONFIG["paths"]["baselineSourceReference"])
    if LEVEL22_CONFIG is not None and engine_was_default:
        args.engine = Path(LEVEL22_CONFIG["paths"]["enginePath"])
    reference = args.reference.resolve()
    old_reference = args.old_reference.resolve()
    engine_path = args.engine.resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"source directory is missing: {reference}")
    if reference != DEFAULT_REFERENCE and not old_reference.is_dir():
        raise FileNotFoundError(f"configured baseline directory is missing: {old_reference}")
    engine_sha = sha256_file(engine_path)
    expected_engine_sha = str((LEVEL22_CONFIG or {}).get("engineSha256") or "8c4541a70b3a7878a9d914effc861d0b873b433ba025c65b90efa0225147c16e")
    if engine_sha.casefold() != expected_engine_sha.casefold():
        raise ValueError("unexpected Egaroucid executable SHA-256")
    details, rows, row_list = load_inputs(reference)
    # The current formal reference is already complete and self-contained.  A
    # historical expansion config can still mention its 550 input, but that
    # input is no longer needed to validate or resume the existing 600 cache.
    if reference == DEFAULT_REFERENCE:
        validation = validate_existing_reference(reference, details, rows, engine_sha)
        print(json.dumps({"ok": True, "reference": str(reference), **validation}, ensure_ascii=False, indent=2))
        return 0
    engine_dir = reference / "engine_level22"
    engine_dir.mkdir(parents=True, exist_ok=True)
    remap = remap_existing(reference, old_reference, details, rows, engine_sha)
    progress_path = engine_dir / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {"schema": "oq-reference-level22-progress-v2", "complete": False, "completedGameIds": [], "errors": [], "startedAt": utc_now(), "totalCount": len(details), "configSha256": LEVEL22_CONFIG_SHA256, "contract": {"level": LEVEL, "workers": WORKERS, "threadsPerConsole": THREADS, "hash": HASH_LEVEL, "book": "enabled-default", "wldFromPlyInclusive": WLD_FROM_PLY, "engineSha256": engine_sha}}
    if LEVEL22_CONFIG_SHA256 is not None and progress.get("configSha256") != LEVEL22_CONFIG_SHA256:
        raise ValueError("Level22 progress belongs to a different configuration contract")
    progress["configSha256"] = LEVEL22_CONFIG_SHA256
    progress["contract"] = {"level": LEVEL, "workers": WORKERS, "threadsPerConsole": THREADS, "hash": HASH_LEVEL, "book": "enabled-default", "wldFromPlyInclusive": WLD_FROM_PLY, "engineSha256": engine_sha}
    run = run_new_games(reference, details, rows, engine_path, engine_sha, progress)
    built = build_wld_and_indexes(reference, details, rows, engine_sha, engine_path)
    completion = write_completion_artifacts(reference, details, built, engine_path, engine_sha)
    print(json.dumps({"ok": True, "reference": str(reference), "gameCount": len(details), "remappedBaseline": remap["remappedGameCount"], "newGameCount": run["newGameCount"], "nodeCount": built["nodeCount"], "eventCount": built["eventCount"], "wldGameCount": len({row["game_id"] for row in built["wldRows"]}), "completionAudit": completion["completion"], "finalManifestSha256": completion["manifestSha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
