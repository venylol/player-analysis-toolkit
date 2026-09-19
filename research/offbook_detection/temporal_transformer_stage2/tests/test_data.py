from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.data import (  # noqa: E402
    FIRST_MOVE_TRANSFORM,
    canonical_target_boards,
    transform_square,
)


class DataContractTest(unittest.TestCase):
    def test_all_legal_openings_map_to_f5(self) -> None:
        self.assertEqual(
            {move: transform_square(move, transform_id) for move, transform_id in FIRST_MOVE_TRANSFORM.items()},
            {"d3": "f5", "c4": "f5", "f5": "f5", "e6": "f5"},
        )

    def test_one_transform_is_reused_for_every_board_and_target_view(self) -> None:
        boards = [
            "---------------------------OX------XO---------------------------",
            "-------------------X-------XX------XO---------------------------",
        ]
        result = canonical_target_boards(boards, FIRST_MOVE_TRANSFORM["d3"])
        self.assertEqual(result.shape, (2, 2, 64))
        self.assertTrue(
            torch.equal(result[1], torch.where(result[0] == 1, 2, torch.where(result[0] == 2, 1, result[0])))
        )
        self.assertEqual(torch.bincount(result[0, 0].long(), minlength=3).tolist(), [60, 2, 2])
        self.assertEqual(torch.bincount(result[1, 0].long(), minlength=3).tolist(), [60, 2, 2])


if __name__ == "__main__":
    unittest.main()
