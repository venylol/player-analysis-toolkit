#!/usr/bin/env python3
"""Freeze OQ games whose actual-placement nodes all have level18 hint6 scores.

The export retains explicit pass rows, but labels each actual placement against the
next actual-placement position. A pass between them therefore produces a same-side
transition and uses current_best - next_same_side_best. Existing engine rows are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_oq_reversi_5min_hints import OthelloBoard


EXPECTED_TCB = 300000
INPUT_POLICY = "uniform-no-current-player-loss-history-v1"
SEVERITY_NAMES = ("class_zero", "class_1_3", "class_4_9", "class_ge10")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(game_id: str) -> str:
    bucket = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def severity_class(loss: int) -> int:
    if loss == 0:
        return 0
    if loss <= 3:
        return 1
    if loss <= 9:
        return 2
    return 3


def int_or_blank(value: Any) -> int | str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"expected integer-valued engine field, got {text!r}")
    return int(number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True)
    parser.add_argument("--moves", required=True)
    parser.add_argument("--hints", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-games", type=int, default=0)
    args = parser.parse_args()

    games_path = Path(args.games)
    moves_path = Path(args.moves)
    hints_path = Path(args.hints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games: dict[str, dict[str, str]] = {}
    for row in read_csv(games_path):
        game_id = row.get("game_id", "").strip()
        if not game_id or row.get("tcb") != str(EXPECTED_TCB):
            continue
        if not row.get("finalStatus", "").startswith("SCORE:"):
            continue
        games[game_id] = row

    known_passes: dict[str, set[int]] = defaultdict(set)
    with moves_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id not in games:
                continue
            move_index = int(row["move_index"])
            if row.get("move", "").strip() == "-":
                known_passes[game_id].add(move_index)

    hinted_indices: dict[str, set[int]] = defaultdict(set)
    with hints_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id not in games:
                continue
            move_index = int(row["move_index"])
            if row.get("actual_move", "").strip() == "-":
                known_passes[game_id].add(move_index)
            elif row.get("hint6_1_score", "").strip():
                hinted_indices[game_id].add(move_index)

    selected_game_ids: list[str] = []
    for game_id, game in games.items():
        length = int(game.get("length", 0) or 0)
        expected = set(range(length))
        if expected.issubset(hinted_indices.get(game_id, set()) | known_passes.get(game_id, set())):
            selected_game_ids.append(game_id)
    selected_game_ids.sort()
    if args.expected_games and len(selected_game_ids) != args.expected_games:
        raise RuntimeError(
            f"expected {args.expected_games} complete games, found {len(selected_game_ids)}"
        )

    selected_set = set(selected_game_ids)
    move_rows_by_game: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with moves_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id in selected_set:
                move_rows_by_game[game_id][int(row["move_index"])] = row
    hint_rows_by_game: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with hints_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id not in selected_set:
                continue
            move_index = int(row["move_index"])
            existing = hint_rows_by_game[game_id].get(move_index)
            if existing is None or (not existing.get("hint6_1_score") and row.get("hint6_1_score")):
                hint_rows_by_game[game_id][move_index] = row

    selected_games = [games[game_id] for game_id in selected_game_ids]
    raw_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    pass_rows = 0
    negative_raw_loss = 0

    for game_id in selected_game_ids:
        game = games[game_id]
        length = int(game["length"])
        hints = hint_rows_by_game.get(game_id, {})
        moves = move_rows_by_game.get(game_id, {})
        board = OthelloBoard()
        source_rows: list[dict[str, Any]] = []
        placement_ply = 0

        for move_index in range(length):
            hint = hints.get(move_index, {})
            move_meta = moves.get(move_index, {})
            actual_move = str(hint.get("actual_move") or move_meta.get("move") or "").strip().lower()
            if not actual_move:
                raise RuntimeError(f"{game_id} move_index={move_index}: actual move is unavailable")
            setboard = board.to_setboard_str()
            board64 = setboard[:64]
            side_to_move = "black" if board.current == "X" else "white"
            player_id = str(
                hint.get("player_id")
                or move_meta.get("player_id")
                or (game.get("black_id") if side_to_move == "black" else game.get("white_id"))
                or ""
            ).strip().lower()
            if not player_id:
                raise RuntimeError(f"{game_id} move_index={move_index}: player id is unavailable")
            is_pass = actual_move == "-"
            if is_pass:
                pass_rows += 1
            else:
                placement_ply += 1
            hint_board = str(hint.get("board", ""))
            if hint_board and hint_board[:64] != board64:
                raise RuntimeError(f"{game_id} move_index={move_index}: replay board differs from hint board")
            n_legal_moves = len(board.legal_moves())
            legal_moves = " ".join(board.legal_moves())
            row: dict[str, Any] = {
                "game_id": game_id,
                "mode": game.get("mode", "reversi_5min"),
                "gtype": game.get("gtype", "reversi"),
                "tcb": EXPECTED_TCB,
                "created": game.get("created", ""),
                "finalStatus": game.get("finalStatus", ""),
                "move_index": move_index,
                "ply": move_index + 1,
                "source_ply_including_pass": move_index + 1,
                "global_placement_ply": placement_ply,
                "side_to_move": side_to_move,
                "player_id": player_id,
                "actual_move": actual_move,
                "actual_thinking_time_ms": int(
                    hint.get("actual_thinking_time_ms")
                    or move_meta.get("thinking_time_ms")
                    or 0
                ),
                "board": board64,
                "board_setboard": setboard,
                "n_legal_moves": n_legal_moves,
                "legal_moves": legal_moves,
                "is_pass_record": int(is_pass),
                "hint1_level": hint.get("hint1_level", ""),
                "hint1_move": hint.get("hint1_move", ""),
                "hint1_score": int_or_blank(hint.get("hint1_score")),
                "hint1_nodes": int_or_blank(hint.get("hint1_nodes")),
                "hint1_depth": hint.get("hint1_depth", ""),
                "hint1_is_book": hint.get("hint1_is_book", ""),
                "split": stable_split(game_id),
                "input_policy": INPUT_POLICY,
            }
            for hint_index in range(1, 7):
                for suffix in ("move", "score", "nodes", "depth", "is_book"):
                    name = f"hint6_{hint_index}_{suffix}"
                    value = hint.get(name, "")
                    row[name] = int_or_blank(value) if suffix in {"score", "nodes"} else value
            source_rows.append(row)
            board.apply_move(actual_move)

        nonpass_positions = [index for index, row in enumerate(source_rows) if not row["is_pass_record"]]
        for placement_index, source_index in enumerate(nonpass_positions):
            row = source_rows[source_index]
            next_source_index = (
                nonpass_positions[placement_index + 1]
                if placement_index + 1 < len(nonpass_positions)
                else None
            )
            if next_source_index is None:
                next_row = None
                raw_loss: int | str = ""
                disc_loss: int | str = ""
                severity: int | str = ""
                label_available = 0
            else:
                next_row = source_rows[next_source_index]
                current_best = int(row["hint6_1_score"])
                next_best = int(next_row["hint6_1_score"])
                same_side = row["side_to_move"] == next_row["side_to_move"]
                raw_loss = current_best - next_best if same_side else current_best + next_best
                disc_loss = max(0, raw_loss)
                severity = severity_class(disc_loss)
                label_available = 1
                if raw_loss < 0:
                    negative_raw_loss += 1
            row.update(
                {
                    "next_nonpass_move_index": "" if next_row is None else next_row["move_index"],
                    "next_nonpass_source_ply": "" if next_row is None else next_row["source_ply_including_pass"],
                    "next_side_to_move": "" if next_row is None else next_row["side_to_move"],
                    "next_best_score": "" if next_row is None else next_row["hint6_1_score"],
                    "child_pass_count": "" if next_row is None else next_source_index - source_index - 1,
                    "has_consecutive_child": int(next_row is not None),
                    "child_continuity_ok": int(next_row is not None),
                    "same_side_after_move": int(
                        next_row is not None and row["side_to_move"] == next_row["side_to_move"]
                    ),
                    "raw_loss": raw_loss,
                    "disc_loss": disc_loss,
                    "severity_class": severity,
                    "severity_class_name": "" if severity == "" else SEVERITY_NAMES[int(severity)],
                    "label_zero": "" if disc_loss == "" else int(disc_loss == 0),
                    "label_ge4": "" if disc_loss == "" else int(disc_loss >= 4),
                    "label_ge10": "" if disc_loss == "" else int(disc_loss >= 10),
                    "label_available": label_available,
                    "label_formula": "" if next_row is None else (
                        "current_best-next_same_side_best"
                        if row["side_to_move"] == next_row["side_to_move"]
                        else "current_best+next_opponent_best"
                    ),
                }
            )
            decision_rows.append(dict(row))

        raw_rows.extend(source_rows)
        split_rows.append({"game_id": game_id, "split": stable_split(game_id)})
        context_rows.extend(
            {
                "game_id": row["game_id"],
                "ply": row["source_ply_including_pass"],
                "move_index": row["move_index"],
                "split": row["split"],
                "input_policy": INPUT_POLICY,
                "label_quality": "complete-nonpass-level18-hint6-pass-skipped-to-next-placement-v1",
            }
            for row in source_rows
        )

    raw_fields = [
        "game_id", "mode", "gtype", "tcb", "created", "finalStatus",
        "move_index", "ply", "source_ply_including_pass", "global_placement_ply",
        "side_to_move", "player_id", "actual_move", "actual_thinking_time_ms",
        "board", "board_setboard", "n_legal_moves", "legal_moves", "is_pass_record",
        "hint1_level", "hint1_move", "hint1_score", "hint1_nodes", "hint1_depth", "hint1_is_book",
    ]
    for hint_index in range(1, 7):
        raw_fields.extend(
            f"hint6_{hint_index}_{suffix}"
            for suffix in ("move", "score", "nodes", "depth", "is_book")
        )
    raw_fields.extend(("split", "input_policy"))
    decision_fields = raw_fields + [
        "next_nonpass_move_index", "next_nonpass_source_ply", "next_side_to_move",
        "next_best_score", "child_pass_count", "has_consecutive_child",
        "child_continuity_ok", "same_side_after_move", "raw_loss", "disc_loss",
        "severity_class", "severity_class_name", "label_zero", "label_ge4",
        "label_ge10", "label_available", "label_formula",
    ]

    games_out = output_dir / "games.csv"
    raw_out = output_dir / "raw_nodes_with_pass.csv"
    decisions_out = output_dir / "decision_labels.csv"
    context_out = output_dir / "context_metadata.csv"
    split_out = output_dir / "split_manifest.csv"
    write_csv(games_out, list(selected_games[0].keys()), selected_games)
    write_csv(raw_out, raw_fields, raw_rows)
    write_csv(decisions_out, decision_fields, decision_rows)
    write_csv(
        context_out,
        ["game_id", "ply", "move_index", "split", "input_policy", "label_quality"],
        context_rows,
    )
    write_csv(split_out, ["game_id", "split"], split_rows)

    labelled = [row for row in decision_rows if row["label_available"]]
    splits = defaultdict(int)
    for row in split_rows:
        splits[row["split"]] += 1
    manifest: dict[str, Any] = {
        "schema": "oq-elo2000-5min-bilateral-hint6-handoff-v1",
        "created_at": utc_now_iso(),
        "source": {
            "games": str(games_path.resolve()),
            "moves": str(moves_path.resolve()),
            "hints": str(hints_path.resolve()),
        },
        "selection": {
            "strict_tcb": EXPECTED_TCB,
            "normal_score_only": True,
            "actual_placement_hint6_1_complete": True,
            "games": len(selected_game_ids),
        },
        "counts": {
            "raw_rows": len(raw_rows),
            "decision_nodes": len(decision_rows),
            "labelled_nodes": len(labelled),
            "pass_rows": pass_rows,
            "players": len({row["player_id"] for row in decision_rows}),
            "negative_raw_loss_nodes": negative_raw_loss,
            "splits": dict(splits),
        },
        "label_definition": {
            "normal": "current_hint6_1_score + next_opponent_actual_placement_hint6_1_score",
            "pass": "current_hint6_1_score - next_same_side_actual_placement_hint6_1_score",
            "pass_rows": "retained for source continuity; not decision nodes; no engine search required",
            "disc_loss": "max(0, raw_loss)",
            "classes": {"0": "class_zero", "1-3": "class_1_3", "4-9": "class_4_9", ">=10": "class_ge10"},
        },
        "input_policy": INPUT_POLICY,
        "model_ready_status": "raw-and-precomputed-label handoff; 362-feature/CNN NPZ still must be materialized and validated",
        "files": {},
    }
    for path in (games_out, raw_out, decisions_out, context_out, split_out):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
