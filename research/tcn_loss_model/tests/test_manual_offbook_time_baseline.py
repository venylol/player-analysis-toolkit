from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_manual_offbook_time_baseline import threshold_scan


class ManualOffbookTimeBaselineTests(unittest.TestCase):
    def test_threshold_scan_retains_no_offbook_game_without_target_nodes(self) -> None:
        nodes = pd.DataFrame([
            {
                "account": "player",
                "game_id": "anchored",
                "manual_judgment": "offbook",
                "is_manual_anchor": True,
                "target_decision_number": 1,
                "score": 2.0,
            }
        ])
        games = pd.DataFrame([
            {"account": "player", "game_id": "anchored", "manual_judgment": "offbook"},
            {"account": "player", "game_id": "empty", "manual_judgment": "no_offbook"},
        ])

        result = threshold_scan(nodes, games, "score", (1.0,))[0]

        self.assertEqual(result["gamesWithCandidate"], 1)
        self.assertEqual(result["exactAnchorRate"], 1.0)
        self.assertEqual(result["noOffbookSpecificity"], 1.0)
        self.assertEqual(result["candidateGameRate"], 0.5)


if __name__ == "__main__":
    unittest.main()
