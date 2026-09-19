from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.model import (  # noqa: E402
    TemporalTransformer,
    TemporalTransformerConfig,
)


class TemporalModelTest(unittest.TestCase):
    def test_all_three_tasks_share_one_bidirectional_encoder(self) -> None:
        model = TemporalTransformer(TemporalTransformerConfig(dropout=0.0))
        batch_size, length = 3, 7
        batch = {
            "board_embedding": torch.randn(batch_size, length, 96),
            "thinking_time": torch.randn(batch_size, length),
            "target_remaining_time": torch.randn(batch_size, length),
            "opponent_remaining_time": torch.randn(batch_size, length),
            "strict_ply": torch.arange(1, length + 1).repeat(batch_size, 1).float(),
            "node_index": torch.arange(length).repeat(batch_size, 1),
            "actor_is_target": torch.randint(0, 2, (batch_size, length)).bool(),
            "actor_is_black": torch.randint(0, 2, (batch_size, length)).bool(),
            "is_pass": torch.zeros(batch_size, length).bool(),
            "time_control_index": torch.zeros(batch_size).long(),
            "padding_mask": torch.zeros(batch_size, length).bool(),
            "lengths": torch.full((batch_size,), length),
        }
        encoded, thinking, remaining, board = model.recover(
            batch, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])
        )
        self.assertEqual(encoded.shape, (batch_size, length, 128))
        self.assertEqual(thinking.shape, (batch_size,))
        self.assertEqual(remaining.shape, (batch_size, 2))
        self.assertEqual(board.shape, (batch_size, 96))


if __name__ == "__main__":
    unittest.main()
