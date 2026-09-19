from __future__ import annotations

import unittest

from research.offbook_detection.first_long_think_v2.core import RuleConfig, detect_game_views


def metadata() -> dict[str, str]:
    return {
        "game_id": "game",
        "black_id": "black-player",
        "white_id": "white-player",
        "recorded_sides": "black|white",
        "split": "test",
    }


def rows(times: list[float]) -> list[dict[str, str]]:
    result = []
    for index, thinking_time in enumerate(times):
        color = "black" if index % 2 == 0 else "white"
        result.append({
            "game_id": "game",
            "move_index": str(index),
            "global_placement_ply": str(index + 1),
            "side_to_move": color,
            "player_id": f"{color}-player",
            "is_pass_record": "0",
            "actual_move": "a1",
            "actual_thinking_time_ms": str(thinking_time),
        })
    return result


class EvaluationCutoffRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuleConfig(5, 38, 1.75, 6.0)

    def test_time_anchor_before_evaluation_cutoff_wins(self) -> None:
        game_rows = rows([100, 100, 100, 100, 176, 100, 100, 100])
        scores = {ply: 0.0 for ply in range(1, 9)}
        scores[7] = 7.0
        black, _ = detect_game_views(metadata(), game_rows, scores, self.config)
        self.assertEqual(black["anchor"]["anchor_source"], "time_rule_before_evaluation_cutoff")
        self.assertEqual(black["anchor"]["anchor_strict_ply"], 5)
        self.assertEqual(black["evaluation_cutoff"]["strict_ply"], 7)

    def test_evaluation_cutoff_is_fallback_anchor(self) -> None:
        game_rows = rows([100] * 8)
        scores = {ply: 0.0 for ply in range(1, 9)}
        scores[7] = -7.0
        black, _ = detect_game_views(metadata(), game_rows, scores, self.config)
        self.assertEqual(black["anchor"]["anchor_source"], "absolute_evaluation_cutoff")
        self.assertEqual(black["anchor"]["anchor_strict_ply"], 7)
        self.assertEqual(black["anchor"]["current_score"], -7.0)

    def test_time_spike_at_cutoff_node_is_not_analyzed(self) -> None:
        game_rows = rows([100, 100, 100, 100, 100, 100, 1000, 100])
        scores = {ply: 0.0 for ply in range(1, 9)}
        scores[7] = 8.0
        black, _ = detect_game_views(metadata(), game_rows, scores, self.config)
        self.assertEqual(black["anchor"]["anchor_source"], "absolute_evaluation_cutoff")
        self.assertEqual(black["anchor"]["anchor_strict_ply"], 7)

    def test_score_equal_to_six_does_not_cross_strict_threshold(self) -> None:
        game_rows = rows([100] * 8)
        scores = {ply: 0.0 for ply in range(1, 9)}
        scores[7] = 6.0
        black, _ = detect_game_views(metadata(), game_rows, scores, self.config)
        self.assertEqual(black["result"], "no_anchor")
        self.assertIsNone(black["evaluation_cutoff"])

    def test_only_recorded_player_views_are_emitted(self) -> None:
        game_metadata = metadata()
        game_metadata["recorded_sides"] = "white"
        records = detect_game_views(
            game_metadata,
            rows([100] * 6),
            {ply: 0.0 for ply in range(1, 7)},
            self.config,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_color"], "white")

    def test_pass_row_player_id_is_not_treated_as_placement_identity(self) -> None:
        game_rows = rows([100] * 6)
        game_rows.insert(4, {
            "game_id": "game",
            "move_index": "4",
            "global_placement_ply": "4",
            "side_to_move": "black",
            "player_id": "white-player",
            "is_pass_record": "1",
            "actual_move": "-",
            "actual_thinking_time_ms": "0",
        })
        for move_index, row in enumerate(game_rows):
            row["move_index"] = str(move_index)
        records = detect_game_views(
            metadata(), game_rows, {ply: 0.0 for ply in range(1, 7)}, self.config
        )
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
