from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "offbook_detection"))

from build_matchup30_level22_expansion import (
    archive_bin_for_rating,
    archive_pair_for_detail,
    pair_for_detail,
    select_expansion,
)


class Matchup30ExpansionTests(unittest.TestCase):
    def test_baseline_above_target_is_retained_without_additions(self) -> None:
        pair = (1600, 1700)
        available = {pair: {f"g{i}" for i in range(50)}}
        baseline = {pair: {f"g{i}" for i in range(35)}}
        selected, coverage = select_expansion(available, baseline, [pair], target=30, seed=1)
        self.assertEqual(selected[pair], [])
        self.assertEqual(coverage[0]["mergedGameCount"], 35)

    def test_sparse_pair_fills_to_target(self) -> None:
        pair = (1600, 1800)
        available = {pair: {f"g{i}" for i in range(40)}}
        baseline = {pair: {f"g{i}" for i in range(18)}}
        selected, coverage = select_expansion(available, baseline, [pair], target=30, seed=2)
        self.assertEqual(len(selected[pair]), 12)
        self.assertEqual(coverage[0]["mergedGameCount"], 30)

    def test_objective_capacity_below_target_is_exhausted(self) -> None:
        pair = (2300, 2400)
        available = {pair: {"a", "b", "c", "d", "e"}}
        baseline = {pair: {"a", "b"}}
        selected, coverage = select_expansion(available, baseline, [pair], target=30, seed=3)
        self.assertEqual(set(selected[pair]), {"c", "d", "e"})
        self.assertEqual(coverage[0]["fillTarget"], 5)
        self.assertEqual(coverage[0]["shortfallAfterExpansion"], 0)

    def test_selection_is_seed_reproducible(self) -> None:
        pair = (1700, 2200)
        available = {pair: {f"g{i}" for i in range(60)}}
        baseline = {pair: {f"g{i}" for i in range(11)}}
        first, _ = select_expansion(available, baseline, [pair], target=30, seed=20260814)
        second, _ = select_expansion(available, baseline, [pair], target=30, seed=20260814)
        self.assertEqual(first, second)

    def test_pair_uses_both_old_r_values_and_is_unordered(self) -> None:
        detail = {"players": [{"oldR": 2250}, {"oldR": 1650}]}
        self.assertEqual(pair_for_detail(detail, 1600, 2486, 100), (1600, 2200))
        detail["players"][1]["oldR"] = 1500
        self.assertIsNone(pair_for_detail(detail, 1600, 2486, 100))

    def test_below_cutoff_baseline_uses_natural_100_point_archive_bin(self) -> None:
        self.assertEqual(archive_bin_for_rating(1598.084, 1600, 2486, 100), 1500)
        self.assertEqual(archive_bin_for_rating(769.269, 1600, 2486, 100), 700)
        detail = {"players": [{"oldR": 1675}, {"oldR": 769.269}]}
        self.assertEqual(archive_pair_for_detail(detail, 1600, 2486, 100), (700, 1600))


if __name__ == "__main__":
    unittest.main()
