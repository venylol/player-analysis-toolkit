"""Canonical fixed-color board-token construction.

Active toolkit semantics are X=black and O=white for every board snapshot.
The archived snapshot-side-to-move experiment is not used by this module.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


BOARD_PERSPECTIVE = "legacy_fixed_color"
LEGACY_BOARD_PERSPECTIVE = BOARD_PERSPECTIVE
ARCHIVED_SNAPSHOT_BOARD_PERSPECTIVE = "snapshot_side_to_move_v1"
SUPPORTED_BOARD_PERSPECTIVES = frozenset({BOARD_PERSPECTIVE})
BOARD_CONTEXT_NAMES = ("current", "prev_opponent", "prev_own")
BOARD_TOKEN_IDS = {"padding": 0, "empty": 1, "black_X": 2, "white_O": 3}


def normalize_side_to_move(side_to_move: Any) -> str:
    """Return a strict canonical side; missing/unknown values are contract errors."""
    if side_to_move is None:
        raise ValueError("side_to_move is missing")
    side = str(side_to_move).strip().casefold()
    if side not in {"black", "white"}:
        raise ValueError(f"invalid side_to_move: {side_to_move!r}")
    return side


def normalize_snapshot_tokens(tokens: np.ndarray, side_to_move: Any) -> np.ndarray:
    """Map fixed black/white tokens to the snapshot's local mover/opponent tokens."""
    side = normalize_side_to_move(side_to_move)
    source = np.asarray(tokens)
    if np.any((source < 0) | (source > 3)):
        raise ValueError("board tokens must use only padding=0, empty=1, black=2, white=3")
    result = source.copy()
    if side == "white":
        black = source == 2
        white = source == 3
        result[black] = 3
        result[white] = 2
    return result


def encode_fixed_color_board(board: Any) -> np.ndarray:
    """Encode a raw fixed-color 64-square board without changing coordinates."""
    text = str(board).strip().upper()
    if len(text) != 64 or any(ch not in {"X", "O", "-", "."} for ch in text):
        raise ValueError(f"invalid fixed-color 8x8 board: {board!r}")
    encoded = np.ones(64, dtype=np.uint8)
    for index, token in enumerate(text):
        if token == "X":
            encoded[index] = 2
        elif token == "O":
            encoded[index] = 3
    return encoded


def encode_snapshot_board(board: Any, side_to_move: Any) -> np.ndarray:
    return normalize_snapshot_tokens(encode_fixed_color_board(board), side_to_move)


def _move_token(move: Any) -> int:
    text = str(move).strip().casefold()
    if text in {"", "-", "none", "nan"}:
        return 0
    if len(text) != 2 or text[0] not in "abcdefgh" or text[1] not in "12345678":
        raise ValueError(f"invalid move coordinate: {move!r}")
    return (int(text[1]) - 1) * 8 + (ord(text[0]) - ord("a")) + 1


def _snapshot_groups(
    frame: Any,
    game_ids: Iterable[Any],
    *,
    required_columns: set[str],
) -> tuple[list[str], dict[str, Any]]:
    required = {"game_id", "side_to_move", "board"} | required_columns
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"board context frame is missing columns: {missing}")
    requested_games = [str(game_id) for game_id in game_ids]
    if len(requested_games) != len(set(requested_games)):
        raise ValueError("game_ids contains duplicates")
    selected = frame.loc[frame["game_id"].astype(str).isin(requested_games)]
    grouped = {str(game_id): view for game_id, view in selected.groupby("game_id", sort=False)}
    absent = [game_id for game_id in requested_games if game_id not in grouped]
    if absent:
        raise ValueError(f"requested games are absent from board context frame: {absent[:5]}")
    if not requested_games or max((len(grouped[game_id]) for game_id in requested_games), default=0) <= 0:
        raise ValueError("cannot build board contexts from an empty selection")
    return requested_games, grouped


def make_board_context_token_sequences(
    frame: Any,
    game_ids: Iterable[Any],
    *,
    return_source_positions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build fixed-color board tokens when move-coordinate context is not needed."""
    requested_games, grouped = _snapshot_groups(frame, game_ids, required_columns=set())
    max_len = max(len(grouped[game_id]) for game_id in requested_games)
    boards = np.zeros((len(requested_games), max_len, 3, 64), dtype=np.uint8)
    sources = np.full((len(requested_games), max_len, 3), -1, dtype=np.int16)
    for game_index, game_id in enumerate(requested_games):
        view = grouped[game_id]
        sides = [normalize_side_to_move(value) for value in view["side_to_move"]]
        fixed_color_boards = [encode_fixed_color_board(board) for board in view["board"]]
        for position, side in enumerate(sides):
            boards[game_index, position, 0] = fixed_color_boards[position]
            sources[game_index, position, 0] = position
            previous_opponent = None
            previous_own = None
            for previous in range(position - 1, -1, -1):
                if previous_own is None and sides[previous] == side:
                    previous_own = previous
                if previous_opponent is None and sides[previous] != side:
                    previous_opponent = previous
                if previous_opponent is not None and previous_own is not None:
                    break
            for context, source_position in ((1, previous_opponent), (2, previous_own)):
                if source_position is None:
                    continue
                boards[game_index, position, context] = fixed_color_boards[source_position]
                sources[game_index, position, context] = source_position
    if return_source_positions:
        return boards, sources
    return boards


def make_board_context_sequences(
    frame: Any,
    game_ids: Iterable[Any],
    _workers: int = 1,
    *,
    return_source_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all three contexts with X=black and O=white.

    Recorded sides select previous-own and previous-opponent source rows, including
    pass-safe histories. They never change the fixed-color token meaning.
    ``_workers`` is accepted for compatibility with the historical builder.
    """
    requested_games, grouped = _snapshot_groups(frame, game_ids, required_columns={"actual_move"})
    boards, sources = make_board_context_token_sequences(
        frame, requested_games, return_source_positions=True
    )
    max_len = boards.shape[1]
    moves = np.zeros((len(requested_games), max_len, 3), dtype=np.uint8)
    for game_index, game_id in enumerate(requested_games):
        view = grouped[game_id]
        move_tokens = [_move_token(move) for move in view["actual_move"]]
        for position in range(len(view)):
            for context in (1, 2):
                source_position = int(sources[game_index, position, context])
                if source_position >= 0:
                    moves[game_index, position, context] = move_tokens[source_position]
    if return_source_positions:
        return boards, moves, sources
    return boards, moves


def require_board_perspective(value: Any, expected: str) -> str:
    actual = str(value).strip() if value is not None else ""
    if actual != expected:
        raise ValueError(
            f"board perspective mismatch: expected {expected!r}, got {actual or '<missing>'!r}"
        )
    return actual
