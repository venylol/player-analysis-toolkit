"""Frozen dataset contract and whole-game canonicalization for stage 2."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from research.offbook_detection.transformer_legal_pretrain.transforms import transform_flat


FIRST_MOVE_TRANSFORM = {"d3": 7, "c4": 2, "f5": 0, "e6": 5}
SQUARES = tuple(f"{column}{row}" for row in range(1, 9) for column in "abcdefgh")
SQUARE_TO_INDEX = {square: index for index, square in enumerate(SQUARES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def transform_square(square: str, transform_id: int) -> str:
    if square not in SQUARE_TO_INDEX:
        raise ValueError(f"invalid board square: {square!r}")
    marker = torch.zeros(64, dtype=torch.uint8)
    marker[SQUARE_TO_INDEX[square]] = 1
    transformed = transform_flat(marker, transform_id)
    return SQUARES[int(torch.nonzero(transformed, as_tuple=False).item())]


def canonical_transform_id(first_black_move: str) -> int:
    try:
        transform_id = FIRST_MOVE_TRANSFORM[first_black_move]
    except KeyError as error:
        raise ValueError(f"unsupported first black move: {first_black_move!r}") from error
    mapped = transform_square(first_black_move, transform_id)
    if mapped != "f5":
        raise AssertionError(f"canonicalization contract failed: {first_black_move} -> {mapped}")
    return transform_id


def board_string_to_black_white_codes(board: str) -> torch.Tensor:
    if len(board) != 64 or any(cell not in "-XO" for cell in board):
        raise ValueError("board_before must contain exactly 64 characters from '-XO'")
    return torch.tensor([0 if cell == "-" else 1 if cell == "X" else 2 for cell in board], dtype=torch.uint8)


def canonical_target_boards(board_strings: Sequence[str], transform_id: int) -> torch.Tensor:
    """Return [2, N, 64] codes; view 0 targets black and view 1 targets white."""
    fixed_color = torch.stack([board_string_to_black_white_codes(board) for board in board_strings])
    canonical = transform_flat(fixed_color, transform_id)
    white_target = torch.where(canonical == 1, 2, torch.where(canonical == 2, 1, canonical))
    return torch.stack((canonical, white_target))


@dataclass(frozen=True)
class GameNodes:
    game_id: str
    split: str
    black_id: str
    white_id: str
    transform_id: int
    board_before: tuple[str, ...]
    node_index: np.ndarray
    strict_ply: np.ndarray
    actor_is_black: np.ndarray
    is_pass: np.ndarray
    thinking_time_ms: np.ndarray
    black_remaining_time_ms_after: np.ndarray
    white_remaining_time_ms_after: np.ndarray
    time_control_id: str

    @property
    def node_count(self) -> int:
        return len(self.board_before)


def _rows_to_game(rows: list[dict[str, str]]) -> GameNodes:
    if not rows:
        raise ValueError("cannot construct an empty game")
    first = rows[0]
    game_id = first["game_id"]
    for expected, row in enumerate(rows):
        if row["game_id"] != game_id:
            raise ValueError("rows from different games were mixed")
        if int(row["node_index"]) != expected:
            raise ValueError(f"non-contiguous node_index in game {game_id}")
        if row["split"] != first["split"] or row["time_control_id"] != first["time_control_id"]:
            raise ValueError(f"split or time control changes within game {game_id}")
    if first["actor_color"] != "black" or first["is_pass"] != "0":
        raise ValueError(f"game {game_id} does not begin with a black placement")
    black_rows = [row for row in rows if row["actor_color"] == "black"]
    white_rows = [row for row in rows if row["actor_color"] == "white"]
    black_id = black_rows[0]["actor_id"]
    white_id = white_rows[0]["actor_id"]
    if any(row["actor_id"] != black_id for row in black_rows):
        raise ValueError(f"black actor changes within game {game_id}")
    if any(row["actor_id"] != white_id for row in white_rows):
        raise ValueError(f"white actor changes within game {game_id}")
    return GameNodes(
        game_id=game_id,
        split=first["split"],
        black_id=black_id,
        white_id=white_id,
        transform_id=canonical_transform_id(first["move"]),
        board_before=tuple(row["board_before"] for row in rows),
        node_index=np.asarray([int(row["node_index"]) for row in rows], dtype=np.int16),
        strict_ply=np.asarray([int(row["strict_ply"]) for row in rows], dtype=np.int16),
        actor_is_black=np.asarray([row["actor_color"] == "black" for row in rows], dtype=np.bool_),
        is_pass=np.asarray([row["is_pass"] == "1" for row in rows], dtype=np.bool_),
        thinking_time_ms=np.asarray([int(row["thinking_time_ms"]) for row in rows], dtype=np.int32),
        black_remaining_time_ms_after=np.asarray(
            [int(row["black_remaining_time_ms_after"]) for row in rows], dtype=np.int32
        ),
        white_remaining_time_ms_after=np.asarray(
            [int(row["white_remaining_time_ms_after"]) for row in rows], dtype=np.int32
        ),
        time_control_id=first["time_control_id"],
    )


def iter_games(nodes_csv: Path) -> Iterator[GameNodes]:
    """Stream nodes.csv, which is contractually grouped by game and node index."""
    with nodes_csv.resolve(strict=True).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "game_id", "node_index", "strict_ply", "actor_color", "actor_id", "is_pass", "move",
            "thinking_time_ms", "black_remaining_time_ms_after", "white_remaining_time_ms_after",
            "board_before", "time_control_id", "split",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"nodes.csv is missing columns: {sorted(required - set(reader.fieldnames or ())) }")
        rows: list[dict[str, str]] = []
        current_game_id: str | None = None
        seen_game_ids: set[str] = set()
        for row in reader:
            game_id = row["game_id"]
            if current_game_id is not None and game_id != current_game_id:
                yield _rows_to_game(rows)
                seen_game_ids.add(current_game_id)
                rows = []
                if game_id in seen_game_ids:
                    raise ValueError(f"game {game_id} appears in multiple blocks")
            rows.append(row)
            current_game_id = game_id
        if rows:
            yield _rows_to_game(rows)

