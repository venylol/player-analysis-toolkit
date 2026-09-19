"""Extract real pass positions from the retained 11,200-game OQ dataset.

The model-ready NPZ excludes explicit pass rows.  The authoritative placement
metadata retains the original move_index, so a gap of exactly one row between
two placements identifies the intervening pass.  The board before the later
placement is also the board seen by the passing player.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


EXPECTED_GAMES = 11_200
EXPECTED_CONTEXT_ROWS = 671_066
EXPECTED_PASSES = 10_998
EXPECTED_SPLIT_COUNTS = {"train": 8_755, "validation": 1_121, "test": 1_122}
CELL_TO_CODE = {"-": 0, "X": 1, "O": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pack_board(board: str) -> np.ndarray:
    if len(board) != 64 or any(cell not in CELL_TO_CODE for cell in board):
        raise ValueError("board must contain exactly 64 characters from '-', 'X', 'O'")
    packed = np.zeros(16, dtype=np.uint8)
    for index, cell in enumerate(board):
        packed[index // 4] |= CELL_TO_CODE[cell] << ((index % 4) * 2)
    return packed


def legal_moves_bitboard(board: str) -> int:
    """Return legal X moves with square bit i matching board character i."""
    if len(board) != 64 or any(cell not in CELL_TO_CODE for cell in board):
        raise ValueError("board must contain exactly 64 characters from '-', 'X', 'O'")
    result = 0
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for square, cell in enumerate(board):
        if cell != "-":
            continue
        row, col = divmod(square, 8)
        for dr, dc in directions:
            r, c = row + dr, col + dc
            seen_opponent = False
            while 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == "O":
                seen_opponent = True
                r += dr
                c += dc
            if seen_opponent and 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == "X":
                result |= 1 << square
                break
    return result


def normalize_for_side_to_move(fixed_color_board: str, side_to_move: str) -> str:
    side = side_to_move.strip().lower()
    if side == "black":
        return fixed_color_board
    if side == "white":
        return fixed_color_board.translate(str.maketrans("XO", "OX"))
    raise ValueError(f"unsupported side_to_move: {side_to_move!r}")


def extract(metadata_csv: Path, split_npz: Path) -> dict[str, np.ndarray]:
    with np.load(split_npz, allow_pickle=False) as data:
        game_ids = data["game_id"].tolist()
        splits = data["split"].tolist()
    if len(game_ids) != EXPECTED_GAMES or len(set(game_ids)) != EXPECTED_GAMES:
        raise ValueError("split NPZ does not contain exactly 11,200 unique games")
    split_by_game = dict(zip(game_ids, splits, strict=True))

    packed_boards: list[np.ndarray] = []
    legal_masks: list[int] = []
    pass_game_ids: list[str] = []
    pass_move_indices: list[int] = []
    pass_sides: list[str] = []
    pass_splits: list[str] = []
    previous_move_index: dict[str, int] = {}
    seen_games: set[str] = set()
    context_rows = 0

    csv.field_size_limit(sys.maxsize)
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context_rows += 1
            game_id = row["game_id"]
            move_index = int(row["move_index"])
            seen_games.add(game_id)
            previous = previous_move_index.get(game_id)
            if previous is not None:
                gap = move_index - previous - 1
                if gap not in (0, 1):
                    raise ValueError(f"unexpected move_index gap {gap} in game {game_id}")
                if gap == 1:
                    next_side = row["side_to_move"].strip().lower()
                    if next_side not in ("black", "white"):
                        raise ValueError(f"unsupported side_to_move: {next_side!r}")
                    pass_side = "white" if next_side == "black" else "black"
                    pass_board = normalize_for_side_to_move(row["board"], pass_side)
                    next_board = normalize_for_side_to_move(row["board"], next_side)
                    pass_legal = legal_moves_bitboard(pass_board)
                    if pass_legal != 0:
                        raise ValueError(f"inferred pass has legal moves in game {game_id}, index {move_index - 1}")
                    if legal_moves_bitboard(next_board) == 0:
                        raise ValueError(f"placement after pass has no legal moves in game {game_id}, index {move_index}")
                    packed_boards.append(pack_board(pass_board))
                    legal_masks.append(pass_legal)
                    pass_game_ids.append(game_id)
                    pass_move_indices.append(move_index - 1)
                    pass_sides.append(pass_side)
                    pass_splits.append(split_by_game[game_id])
            previous_move_index[game_id] = move_index

    if context_rows != EXPECTED_CONTEXT_ROWS or len(seen_games) != EXPECTED_GAMES:
        raise ValueError(
            f"unexpected metadata shape: rows={context_rows}, games={len(seen_games)}"
        )
    if len(packed_boards) != EXPECTED_PASSES:
        raise ValueError(f"expected {EXPECTED_PASSES} passes, found {len(packed_boards)}")
    actual_split_counts = {
        split: pass_splits.count(split) for split in EXPECTED_SPLIT_COUNTS
    }
    if actual_split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected pass split counts: {actual_split_counts}")

    return {
        "packed_boards": np.stack(packed_boards).astype(np.uint8, copy=False),
        "legal_masks": np.asarray(legal_masks, dtype=np.uint64),
        "game_id": np.asarray(pass_game_ids, dtype="<U64"),
        "move_index": np.asarray(pass_move_indices, dtype=np.int16),
        "side_to_move": np.asarray(pass_sides, dtype="<U5"),
        "split": np.asarray(pass_splits, dtype="<U10"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--split-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    metadata_csv = args.metadata_csv.resolve(strict=True)
    split_npz = args.split_npz.resolve(strict=True)
    output_npz = args.output_npz.resolve()
    manifest_path = args.manifest.resolve()
    if output_npz.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing supplement or manifest")

    arrays = extract(metadata_csv, split_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    split_counts = {
        split: int((arrays["split"] == split).sum()) for split in EXPECTED_SPLIT_COUNTS
    }
    manifest = {
        "schema": "transformer-legal-pass-supplement-v1",
        "source": "real pass nodes inferred from authoritative 11,200-game placement metadata",
        "boardOrder": "a1,b1,...,h8",
        "boardEncoding": "2-bit cells: empty=0, current-side X=1, opponent O=2",
        "legalTarget": "uint64 bitboard; all records verified equal to zero",
        "records": int(arrays["packed_boards"].shape[0]),
        "uniqueBoards": int(np.unique(arrays["packed_boards"], axis=0).shape[0]),
        "splitCounts": split_counts,
        "metadataCsv": str(metadata_csv),
        "metadataCsvSha256": sha256_file(metadata_csv),
        "splitNpz": str(split_npz),
        "splitNpzSha256": sha256_file(split_npz),
        "outputNpz": str(output_npz),
        "outputNpzSha256": sha256_file(output_npz),
        "ruleChecks": {
            "allPassActorsHaveZeroLegalMoves": True,
            "allFollowingPlacementActorsHaveLegalMoves": True,
            "allMoveIndexGapsExactlyOne": True,
        },
        "encoding": "UTF-8",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
