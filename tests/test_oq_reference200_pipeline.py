from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "offbook_detection"))

from oq_reference200_pipeline import (
    HISTORICAL_MAXIMUM_ELO,
    bin_lowers,
    pair_for_players,
    rating_bin,
    stable_game_key,
    unordered_pairs,
)


class OqReference200PipelineTests(unittest.TestCase):
    def test_fixed_nine_bins_make_exactly_45_unordered_pairs(self) -> None:
        self.assertEqual(bin_lowers(), list(range(1600, 2500, 100)))
        self.assertEqual(len(unordered_pairs()), 45)

    def test_dynamic_maximum_expands_top_bin_without_2500_bin(self) -> None:
        self.assertEqual(rating_bin(2486, HISTORICAL_MAXIMUM_ELO), 2400)
        self.assertIsNone(rating_bin(2487, HISTORICAL_MAXIMUM_ELO))
        self.assertEqual(rating_bin(2521, 2521), 2400)
        self.assertNotIn(2500, bin_lowers())

    def test_pair_is_unordered_and_requires_both_ratings_in_scope(self) -> None:
        players = [{"oldR": 2250}, {"oldR": 1650}]
        self.assertEqual(pair_for_players(players, 2486), (1600, 2200))
        players[1]["oldR"] = 1599
        self.assertIsNone(pair_for_players(players, 2486))

    def test_stable_order_key_is_reproducible_and_source_qualified(self) -> None:
        pair = (1600, 2200)
        self.assertEqual(stable_game_key(pair, "game", "cache"), stable_game_key(pair, "game", "cache"))
        self.assertNotEqual(stable_game_key(pair, "game", "cache"), stable_game_key(pair, "game", "snapshot"))


if __name__ == "__main__":
    unittest.main()
