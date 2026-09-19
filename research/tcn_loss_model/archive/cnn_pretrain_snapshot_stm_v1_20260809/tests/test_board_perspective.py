from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.backbone import BoardCNNEncoder, ModelConfig
from src.board_perspective import (
    BOARD_PERSPECTIVE,
    encode_fixed_color_board,
    make_board_context_sequences,
    normalize_snapshot_tokens,
    require_board_perspective,
)


def _board(*pieces: tuple[int, str]) -> str:
    values = ["-"] * 64
    for index, token in pieces:
        values[index] = token
    return "".join(values)


def _legal_for_x(tokens: np.ndarray) -> set[int]:
    grid = tokens.reshape(8, 8)
    legal: set[int] = set()
    for row in range(8):
        for column in range(8):
            if grid[row, column] != 1:
                continue
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                r, c = row + dr, column + dc
                seen_o = False
                while 0 <= r < 8 and 0 <= c < 8 and grid[r, c] == 3:
                    seen_o = True
                    r, c = r + dr, c + dc
                if seen_o and 0 <= r < 8 and 0 <= c < 8 and grid[r, c] == 2:
                    legal.add(row * 8 + column)
                    break
    return legal


class SnapshotPerspectiveTests(unittest.TestCase):
    def test_black_unchanged_and_white_swaps_only_occupied_tokens(self):
        source = np.asarray([0, 1, 2, 3, 2, 3], dtype=np.uint8)
        np.testing.assert_array_equal(normalize_snapshot_tokens(source, "black"), source)
        np.testing.assert_array_equal(
            normalize_snapshot_tokens(source, "white"),
            np.asarray([0, 1, 3, 2, 3, 2], dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "side_to_move"):
            normalize_snapshot_tokens(source, "")

    def test_each_context_uses_its_source_row_side_including_pass_pattern(self):
        # Rows 1 and 2 are both white-to-move: black passed between the two
        # placement snapshots. A ply-parity implementation would get row 2 wrong.
        frame = pd.DataFrame([
            {"game_id": "pass", "side_to_move": "black", "board": _board((0, "X"), (1, "O")), "actual_move": "a1"},
            {"game_id": "pass", "side_to_move": "white", "board": _board((2, "X"), (3, "O")), "actual_move": "b1"},
            {"game_id": "pass", "side_to_move": "white", "board": _board((4, "X"), (5, "O")), "actual_move": "c1"},
            {"game_id": "pass", "side_to_move": "black", "board": _board((6, "X"), (7, "O")), "actual_move": "d1"},
        ])
        tokens, moves, sources = make_board_context_sequences(
            frame, ["pass"], return_source_positions=True
        )
        np.testing.assert_array_equal(sources[0, 3], np.asarray([3, 2, 0]))
        self.assertEqual(tokens[0, 3, 0, 6], 2)  # current black is local X
        self.assertEqual(tokens[0, 3, 1, 5], 2)  # historical white is local X
        self.assertEqual(tokens[0, 3, 1, 4], 3)  # historical black is local O
        self.assertEqual(tokens[0, 3, 2, 0], 2)  # historical black is local X
        self.assertEqual(int(moves[0, 3, 1]), 3)
        self.assertTrue(np.all(tokens[0, 0, 1:] == 0))
        self.assertTrue(np.all(moves[0, 0] == 0))

    def test_normal_alternation_current_previous_opponent_and_previous_own(self):
        frame = pd.DataFrame([
            {"game_id": "normal", "side_to_move": side, "board": _board((i, "X"), (i + 8, "O")), "actual_move": f"a{i + 1}"}
            for i, side in enumerate(("black", "white", "black"))
        ])
        _tokens, _moves, sources = make_board_context_sequences(
            frame, ["normal"], return_source_positions=True
        )
        np.testing.assert_array_equal(sources[0, 2], np.asarray([2, 1, 0]))

    def test_local_x_legal_moves_equal_raw_snapshot_mover_moves(self):
        # Standard initial Othello position in fixed black(X)/white(O) colors.
        raw = encode_fixed_color_board(_board((27, "O"), (28, "X"), (35, "X"), (36, "O")))
        black_expected = _legal_for_x(raw)
        white_expected = _legal_for_x(normalize_snapshot_tokens(raw, "white"))
        self.assertEqual(_legal_for_x(normalize_snapshot_tokens(raw, "black")), black_expected)
        self.assertEqual(_legal_for_x(normalize_snapshot_tokens(raw, "white")), white_expected)
        self.assertEqual(black_expected, {19, 26, 37, 44})
        self.assertEqual(white_expected, {20, 29, 34, 43})

    def test_white_current_stem_planes_are_empty_white_black(self):
        frame = pd.DataFrame([{
            "game_id": "white", "side_to_move": "white",
            "board": _board((0, "X"), (1, "O")), "actual_move": "b1",
        }])
        tokens, moves = make_board_context_sequences(frame, ["white"])
        encoder = BoardCNNEncoder(ModelConfig())
        planes, _, _ = encoder.build_input_planes(
            torch.from_numpy(tokens), torch.from_numpy(moves),
            torch.zeros((1, 1, 6), dtype=torch.uint8),
            torch.zeros((1, 1, 4)), torch.zeros((1, 1, 2)),
        )
        self.assertEqual(float(planes[0, 0, 0, 2]), 1.0)  # empty
        self.assertEqual(float(planes[0, 1, 0, 1]), 1.0)  # raw white -> local X
        self.assertEqual(float(planes[0, 2, 0, 0]), 1.0)  # raw black -> local O

    def test_formal_personal_and_new_control_share_identical_builder_output(self):
        frame = pd.DataFrame([
            {"game_id": "g", "side_to_move": "black", "board": _board((0, "X")), "actual_move": "a1", "hint6_1_move": "c3", "hint6_1_score": 7, "disc_loss": 4},
            {"game_id": "g", "side_to_move": "white", "board": _board((1, "O")), "actual_move": "b1", "hint6_1_move": "d4", "hint6_1_score": -5, "disc_loss": 2},
        ])
        unchanged = frame.copy(deep=True)
        formal = make_board_context_sequences(frame, ["g"])
        personal = make_board_context_sequences(frame.copy(), ["g"])
        control = make_board_context_sequences(frame.iloc[:].copy(), ["g"])
        for left, right in ((formal, personal), (formal, control)):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
        pd.testing.assert_frame_equal(frame, unchanged)

    def test_legacy_and_snapshot_contract_cannot_mix(self):
        self.assertEqual(require_board_perspective(BOARD_PERSPECTIVE, BOARD_PERSPECTIVE), BOARD_PERSPECTIVE)
        with self.assertRaisesRegex(ValueError, "board perspective mismatch"):
            require_board_perspective("legacy_fixed_color", BOARD_PERSPECTIVE)
        with self.assertRaisesRegex(ValueError, "board perspective mismatch"):
            require_board_perspective(None, BOARD_PERSPECTIVE)


if __name__ == "__main__":
    unittest.main()
