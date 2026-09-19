from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research" / "offbook_detection" / "fit_monotonic_evidence.py"
SPEC = importlib.util.spec_from_file_location("fit_monotonic_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def board_tokens(occupied: int) -> np.ndarray:
    values = np.ones(64, dtype=np.int8)
    values[:occupied] = 2
    return values


class MonotonicEvidenceFitTests(unittest.TestCase):
    def test_strict_move_ply_uses_parent_occupied_minus_three(self) -> None:
        self.assertEqual(MODULE.strict_move_ply(board_tokens(4)), 1)
        self.assertEqual(MODULE.strict_move_ply(board_tokens(5)), 2)
        self.assertEqual(MODULE.strict_book_node_ply("X" * 5 + "-" * 59), 1)

    def test_step_time_ratio_uses_initial_total_time(self) -> None:
        self.assertEqual(MODULE.step_time_ratio(3_000, 300_000), 0.01)
        self.assertEqual(MODULE.step_time_ratio(600, 60_000), 0.01)
        with self.assertRaises(ValueError):
            MODULE.step_time_ratio(1, 0)
        with self.assertRaises(ValueError):
            MODULE.step_time_ratio(61_000, 60_000)

    def test_log_scaled_time_ratio_uses_one_percent_clock_scale(self) -> None:
        self.assertEqual(MODULE.log_scaled_time_ratio(0), 0)
        self.assertAlmostEqual(MODULE.log_scaled_time_ratio(0.01), math.log(2))
        self.assertAlmostEqual(MODULE.log_scaled_time_ratio(0.05), math.log(6))
        with self.assertRaises(ValueError):
            MODULE.log_scaled_time_ratio(-0.01)

    def test_frequency_features_follow_frozen_missing_rules(self) -> None:
        book = MODULE.GlobalFrequencyBook(
            lookup={(1, "child"): 10, (2, "known"): 20},
            frequency_sum_by_ply={2: 100},
            max_ply=30,
            min_count=5,
            metadata={},
        )

        exact = book.features("child", "known", 2)
        self.assertEqual(exact.parent_child_ratio, 1.0)
        self.assertEqual(exact.global_ratio, 0.2)
        self.assertEqual(exact.global_rarity, -math.log(0.2))
        self.assertFalse(exact.child_frequency_imputed)

        missing_child = book.features("child", "missing", 2)
        self.assertEqual(missing_child.parent_child_ratio, 0.4)
        self.assertEqual(missing_child.child_frequency, 4)
        self.assertEqual(missing_child.same_ply_frequency_sum, 104)
        self.assertEqual(missing_child.global_ratio, 4 / 104)
        self.assertTrue(missing_child.child_frequency_imputed)

        missing_parent = book.features("missing-parent", "known", 2)
        self.assertTrue(math.isnan(missing_parent.parent_child_ratio))
        self.assertEqual(missing_parent.global_ratio, 0.2)

        after_coverage = book.features("child", "known", 31)
        self.assertTrue(math.isnan(after_coverage.parent_child_ratio))
        self.assertTrue(math.isnan(after_coverage.global_rarity))

    def test_curve_is_monotonic_and_differences_match_h(self) -> None:
        frame = pd.DataFrame({
            "account": ["p"] * 3,
            "game_id": ["g"] * 3,
            "target_decision_number": [1, 2, 3],
        })
        probabilities = np.asarray([0.1, 0.2, 0.05])
        result = MODULE.add_curve_columns(frame, probabilities)
        h = -np.log1p(-probabilities)
        self.assertTrue(np.allclose(result["h_increment"], h))
        self.assertTrue(np.allclose(result["d_cumulative"], np.cumsum(h)))
        self.assertTrue(np.all(np.diff(result["d_cumulative"]) >= 0))
        self.assertTrue(np.allclose(result["a_second_difference"], np.diff(h, prepend=h[0])))


if __name__ == "__main__":
    unittest.main()
