from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_RESEARCH = (
    ROOT
    / "data"
    / "oq_elo2000_5min_bilateral_10000_source_only_20260804"
    / "source_snapshot"
    / "official_research"
)
BOARD_MODEL = OFFICIAL_RESEARCH / "train_causal_transformer_board_model.py"
if BOARD_MODEL.is_file():
    if str(OFFICIAL_RESEARCH) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_RESEARCH))
    import train_causal_transformer_board_model as board_model  # noqa: E402
else:
    board_model = None


@unittest.skipIf(board_model is None, "private source snapshot is not included")
class FixedColorBoardStateTests(unittest.TestCase):
    def test_x_is_black_and_o_is_white_even_when_white_moves(self) -> None:
        board = "XO" + "-" * 62
        self.assertEqual(board_model.encode_board64(board)[:3].tolist(), [2, 3, 1])

        records = [(["white"], [board], ["a1"])]
        contexts, _moves = board_model.build_board_context_chunk((records, 1))
        self.assertEqual(contexts[0, 0, 0, :3].tolist(), [2, 3, 1])

    def test_published_token_contract_is_fixed_color(self) -> None:
        self.assertEqual(
            board_model.BOARD_ENCODING["token_ids"],
            {"padding": 0, "empty": 1, "X": 2, "O": 3},
        )


if __name__ == "__main__":
    unittest.main()
