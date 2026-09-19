from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.evaluate_test import (  # noqa: E402
    MetricAccumulator,
    length_bin,
)


class DeliveryTest(unittest.TestCase):
    def test_length_bins_cover_boundary_values(self) -> None:
        self.assertEqual(length_bin(9), "le_59")
        self.assertEqual(length_bin(59), "le_59")
        self.assertEqual(length_bin(60), "60")
        self.assertEqual(length_bin(61), "61")
        self.assertEqual(length_bin(62), "ge_62")

    def test_metric_accumulator_uses_per_element_mse(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.sequences = 2
        accumulator.add("thinking_time_mse", 4.0, 2)
        accumulator.add("both_remaining_times_mse", 8.0, 4)
        accumulator.add("board_embedding_mse", 384.0, 192)
        result = accumulator.result()
        self.assertEqual(result["thinking_time_mse"], 2.0)
        self.assertEqual(result["both_remaining_times_mse"], 2.0)
        self.assertEqual(result["board_embedding_mse"], 2.0)
        self.assertEqual(result["selection_metric_sum_mse"], 6.0)


if __name__ == "__main__":
    unittest.main()
