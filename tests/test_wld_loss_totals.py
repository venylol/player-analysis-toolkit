from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from player_analysis_toolkit.analysis_core import (
    ENGINE_WLD_TOTAL_FIELD,
    PREDICTED_WLD_TOTAL_FIELD,
    actual_move_score_from_next_best,
    aggregate_loss_games,
    cluster_bootstrap_engine_wld_per_game_difference,
    engine_wld_totals_by_game_player,
    loss_game_summary,
    predicted_wld_totals_by_game_player,
    wld_loss_from_scores,
)


def node(
    ply: int,
    player: str,
    side: str,
    before: float,
    actual: float,
    *,
    source_index: int | None = None,
) -> dict[str, object]:
    return {
        "ply": ply,
        "sourceMoveIndex": ply - 1 if source_index is None else source_index,
        "move": "d3",
        "playerAccount": player,
        "playerColor": side,
        "bestEval": before,
        "actualEval": actual,
        "lossClipped": 0,
    }


class WldFormulaTests(unittest.TestCase):
    def test_rank_drops(self) -> None:
        self.assertEqual(wld_loss_from_scores(1, 0), 0.5)  # Win -> Draw
        self.assertEqual(wld_loss_from_scores(0, -1), 0.5)  # Draw -> Loss
        self.assertEqual(wld_loss_from_scores(1, -1), 1.0)  # Win -> Loss
        self.assertEqual(wld_loss_from_scores(0, 0), 0.0)
        self.assertEqual(wld_loss_from_scores(-1, 0), 0.0)  # improvement

    def test_normal_turn_flips_next_score(self) -> None:
        actual = actual_move_score_from_next_best(-1, same_side_after_pass=False)
        self.assertEqual(actual, 1.0)
        self.assertEqual(wld_loss_from_scores(1, actual), 0.0)

    def test_same_side_after_pass_does_not_flip(self) -> None:
        actual = actual_move_score_from_next_best(-1, same_side_after_pass=True)
        self.assertEqual(actual, -1.0)
        self.assertEqual(wld_loss_from_scores(1, actual), 1.0)


class EngineWldAggregationTests(unittest.TestCase):
    def test_default_summary_is_unchanged(self) -> None:
        game = {"gameId": "g", "nodes": [node(39, "a", "black", 1, -1)]}
        self.assertNotIn(ENGINE_WLD_TOTAL_FIELD, loss_game_summary(game))
        self.assertNotIn(ENGINE_WLD_TOTAL_FIELD, aggregate_loss_games([game]))

    def test_ply_38_excluded_and_39_included_with_pass_gap(self) -> None:
        game = {
            "gameId": "g",
            "nodes": [
                node(38, "a", "black", 1, -1),
                node(39, "a", "black", 1, 0, source_index=40),
            ],
        }
        summary = loss_game_summary(game, 39)
        self.assertEqual(summary[ENGINE_WLD_TOTAL_FIELD], 0.5)

    def test_game_sides_and_players_only_receive_their_own_moves(self) -> None:
        games = [
            {
                "gameId": "g1",
                "nodes": [
                    node(39, "alice", "black", 1, -1),
                    node(40, "bob", "white", 1, 0),
                ],
            },
            {
                "gameId": "g2",
                "nodes": [node(39, "alice", "white", 0, -1)],
            },
        ]
        totals = engine_wld_totals_by_game_player(games, 39)
        by_game = {
            (row["game_id"], row["player_id"]): row[ENGINE_WLD_TOTAL_FIELD]
            for row in totals["gamePlayerTotals"]
        }
        by_player = {
            row["player_id"]: row[ENGINE_WLD_TOTAL_FIELD]
            for row in totals["playerTotals"]
        }
        self.assertEqual(by_game, {("g1", "alice"): 1.0, ("g1", "bob"): 0.5, ("g2", "alice"): 0.5})
        self.assertEqual(by_player, {"alice": 1.5, "bob": 0.5})

    def test_regular_totals_have_no_per_move_or_unrequested_metrics(self) -> None:
        totals = engine_wld_totals_by_game_player(
            [{"gameId": "g", "nodes": [node(39, "棋手甲", "black", 1, 0)]}],
            39,
        )
        public_text = repr(totals)
        for forbidden in ("before_rank", "after_rank", "average", "reversal", "wld_loss_rate"):
            self.assertNotIn(forbidden, public_text)

    def test_whole_game_wld_bootstrap_uses_game_means(self) -> None:
        reported = [{"gameId": "r", "nodes": [node(39, "a", "black", 0, 0)]}]
        controls = [
            {"gameId": "c1", "nodes": [node(39, "a", "black", 1, -1)]},
            {"gameId": "c2", "nodes": [node(39, "a", "black", 1, -1)]},
        ]
        self.assertEqual(
            cluster_bootstrap_engine_wld_per_game_difference(reported, controls, 20, 7, 39),
            [-1.0, -1.0],
        )


class PredictedWldAggregationTests(unittest.TestCase):
    def test_expected_wld_loss_sums_by_game_and_player(self) -> None:
        rows = [
            {"game_id": "g", "player_id": "甲", "side": "black", "global_placement_ply": 38, "wld_applicable": True, "expected_wld_loss": 1},
            {"game_id": "g", "player_id": "甲", "side": "black", "global_placement_ply": 39, "wld_applicable": True, "expected_wld_loss": 0.25},
            {"game_id": "g", "player_id": "甲", "side": "black", "global_placement_ply": 41, "wld_applicable": False, "expected_wld_loss": ""},
            {"game_id": "g", "player_id": "乙", "side": "white", "global_placement_ply": 40, "wld_applicable": True, "expected_wld_loss": 0.5},
            {"game_id": "h", "player_id": "甲", "side": "white", "global_placement_ply": 39, "wld_applicable": True, "expected_wld_loss": 0.75},
        ]
        totals = predicted_wld_totals_by_game_player(rows, 39)
        by_game = {
            (row["game_id"], row["player_id"]): row[PREDICTED_WLD_TOTAL_FIELD]
            for row in totals["gamePlayerTotals"]
        }
        by_player = {
            row["player_id"]: row[PREDICTED_WLD_TOTAL_FIELD]
            for row in totals["playerTotals"]
        }
        self.assertEqual(by_game, {("g", "甲"): 0.25, ("g", "乙"): 0.5, ("h", "甲"): 0.75})
        self.assertEqual(by_player, {"甲": 1.0, "乙": 0.5})

    def test_missing_required_model_field_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields.*expected_wld_loss"):
            predicted_wld_totals_by_game_player(
                [{"game_id": "g", "player_id": "p", "global_placement_ply": 39, "wld_applicable": True}],
                39,
            )


if __name__ == "__main__":
    unittest.main()
