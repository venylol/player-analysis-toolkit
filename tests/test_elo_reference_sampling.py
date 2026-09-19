from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "offbook_detection"))

from build_elo_reference import Candidate, bin_for_rating, bin_lowers, bin_upper, choose_candidates


def candidate(game: int, player: str, side: str, lower: int = 1600) -> Candidate:
    return Candidate(
        game_id=f"g{game:03d}", created="2026-01-01T00:00:00Z", lower=lower,
        upper=1700, target_id=player, target_side=side, target_old_r=1650,
        opponent_id=f"o{game:03d}", opponent_old_r=1500,
        source_dataset="test", tie_key=f"{game:03d}",
    )


class EloReferenceSamplingTests(unittest.TestCase):
    def test_bins_include_truncated_maximum(self) -> None:
        self.assertEqual(bin_lowers(1600, 2486, 100), list(range(1600, 2500, 100)))
        self.assertEqual(bin_for_rating(1699.999, 1600, 2486, 100), 1600)
        self.assertEqual(bin_for_rating(1700, 1600, 2486, 100), 1700)
        self.assertEqual(bin_for_rating(2486, 1600, 2486, 100), 2400)
        self.assertIsNone(bin_for_rating(2486.001, 1600, 2486, 100))
        self.assertEqual(bin_upper(2400, 2486, 100), 2486)

    def test_full_bin_is_balanced_and_game_ids_are_unique(self) -> None:
        pool = []
        for index in range(50):
            pool.append(candidate(index, f"b{index}", "black"))
            pool.append(candidate(index, f"w{index}", "white"))
        selected, _ = choose_candidates({1600: pool}, [1600], target_per_bin=40)
        games = [item.game_id for item, _, _ in selected]
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(set(games)), 40)
        self.assertEqual(sum(item.target_side == "black" for item, _, _ in selected), 20)
        self.assertEqual(sum(item.target_side == "white" for item, _, _ in selected), 20)

    def test_repeat_relaxation_rotates_accounts(self) -> None:
        pool = []
        game = 0
        for player in ("a", "b"):
            for side in ("black", "white"):
                for _ in range(4):
                    pool.append(candidate(game, player, side))
                    game += 1
        selected, counts = choose_candidates({1600: pool}, [1600], target_per_bin=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(item.target_side == "black" for item, _, _ in selected), 4)
        self.assertEqual(sum(item.target_side == "white" for item, _, _ in selected), 4)
        self.assertLessEqual(max(counts[1600].values()) - min(counts[1600].values()), 1)
        self.assertTrue(any(phase == "round_robin_repeat" for _, phase, _ in selected))


if __name__ == "__main__":
    unittest.main()
