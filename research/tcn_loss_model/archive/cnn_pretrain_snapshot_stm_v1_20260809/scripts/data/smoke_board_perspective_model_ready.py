#!/usr/bin/env python3
"""Generate and validate a two-game snapshot-perspective model-ready smoke bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_perspective import BOARD_PERSPECTIVE, make_board_context_sequences
from src.data_contract import validate_model_ready_npz
from src.feature_policy import INPUT_POLICY


def board(black: int, white: int) -> str:
    values = ["-"] * 64
    values[black] = "X"
    values[white] = "O"
    return "".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "base" / "tcn_board_cnn_time_model_best.pt",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frames = {
        "normal": pd.DataFrame([
            {"game_id": "normal", "side_to_move": "black", "board": board(0, 1), "actual_move": "a1"},
            {"game_id": "normal", "side_to_move": "white", "board": board(2, 3), "actual_move": "b1"},
            {"game_id": "normal", "side_to_move": "black", "board": board(4, 5), "actual_move": "c1"},
        ]),
        "pass": pd.DataFrame([
            {"game_id": "pass", "side_to_move": "black", "board": board(8, 9), "actual_move": "a2"},
            {"game_id": "pass", "side_to_move": "white", "board": board(10, 11), "actual_move": "b2"},
            {"game_id": "pass", "side_to_move": "white", "board": board(12, 13), "actual_move": "c2"},
            {"game_id": "pass", "side_to_move": "black", "board": board(14, 15), "actual_move": "d2"},
        ]),
    }
    frame = pd.concat(frames.values(), ignore_index=True)
    games = np.asarray(["normal", "pass"])
    board_tokens, board_moves, sources = make_board_context_sequences(
        frame, games, return_source_positions=True
    )
    shape = board_tokens.shape[:2]
    valid_lengths = [3, 4]
    valid = np.zeros(shape, dtype=bool)
    sides = np.full(shape, "", dtype="U8")
    players = np.full(shape, "", dtype="U8")
    ply = np.zeros(shape, dtype=np.int16)
    for game_index, (game_id, length) in enumerate(zip(games, valid_lengths, strict=True)):
        view = frames[str(game_id)]
        valid[game_index, :length] = True
        sides[game_index, :length] = view["side_to_move"].to_numpy()
        players[game_index, :length] = "smoke"
        ply[game_index, :length] = np.arange(1, length + 1)
    zeros = np.zeros(shape, dtype=np.float32)
    arrays = {
        "X": np.zeros((*shape, 362), dtype=np.float32),
        "board_tokens": board_tokens,
        "board_move_tokens": board_moves,
        "current_hint_tokens": np.zeros((*shape, 6), dtype=np.uint8),
        "current_hint_values": np.zeros((*shape, 4), dtype=np.float32),
        "prev_own_hint_values": np.zeros((*shape, 2), dtype=np.float32),
        "actual_thinking_time_ms": np.where(valid, 1000, 0).astype(np.float32),
        "disc_loss": zeros.copy(), "raw_loss": zeros.copy(),
        "severity_class": np.zeros(shape, dtype=np.int8),
        "label_zero": valid.astype(np.int8),
        "label_ge4": np.zeros(shape, dtype=np.int8),
        "label_ge10": np.zeros(shape, dtype=np.int8),
        "move_index": np.where(valid, ply - 1, -1).astype(np.int16),
        "source_ply_including_pass": ply.copy(),
        "label_available": valid.copy(),
        "has_consecutive_child": valid.copy(),
        "child_continuity_ok": valid.copy(),
        "same_side_after_move": np.zeros(shape, dtype=bool),
        "current_score": zeros.copy(), "actual_move_score": zeros.copy(),
        "wld_class": np.zeros(shape, dtype=np.int8), "wld_loss": zeros.copy(),
        "wld_label_available": np.zeros(shape, dtype=bool),
        "mask": valid, "game_id": games, "player_id": players,
        "global_placement_ply": ply, "side_to_move": sides,
        "split": np.asarray(["train", "validation"]),
        "input_features": np.asarray(checkpoint["input_features"]),
        "board_cnn_channels": np.asarray(checkpoint["board_encoding"]["cnn_channels"]),
        "preprocessing_sha256": np.asarray("smoke-preprocessing"),
        "input_policy": np.asarray(INPUT_POLICY),
        "board_perspective": np.asarray(BOARD_PERSPECTIVE),
        "board_feature_revision": np.asarray("board_context_snapshot_side_to_move_v1__23_planes_v1"),
    }
    npz = output / "model_ready_snapshot_stm_v1_smoke.npz"
    np.savez_compressed(npz, **arrays)
    validation = validate_model_ready_npz(
        npz,
        expected_input_features=list(checkpoint["input_features"]),
        expected_board_channels=list(checkpoint["board_encoding"]["cnn_channels"]),
        expected_preprocessing_sha256="smoke-preprocessing",
        expected_board_perspective=BOARD_PERSPECTIVE,
    )
    mapping = {
        "schema": "board-perspective-human-mapping-smoke-v1",
        "boardPerspective": BOARD_PERSPECTIVE,
        "normalCurrentRow2": {"side": "black", "sources": sources[0, 2].tolist()},
        "normalWhiteCurrentRow1": {
            "side": "white", "sources": sources[0, 1].tolist(),
            "current": "row1 raw white O -> local X",
            "prevOpponent": "row0 raw black X -> local X",
            "prevOwn": "missing -> all-zero padding",
        },
        "passCurrentRow3": {
            "side": "black", "sources": sources[1, 3].tolist(),
            "current": "row3 black -> local X",
            "prevOpponent": "row2 white -> local X (raw O)",
            "prevOwn": "row0 black -> local X (raw X)",
        },
        "missingContextsAllZero": bool(np.all(board_tokens[:, 0, 1:] == 0)),
        "coordinatesUnchanged": {
            "passRow2ActualMove": "c2",
            "encodedHistoryAtPassRow3PrevOpponent": int(board_moves[1, 3, 1]),
            "expectedMoveToken": 11,
        },
        "validation": validation,
    }
    if mapping["coordinatesUnchanged"]["encodedHistoryAtPassRow3PrevOpponent"] != 11:
        raise AssertionError("history move coordinate changed")
    (output / "human_mapping_check.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
