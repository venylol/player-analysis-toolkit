from __future__ import annotations

import unittest

from research.offbook_detection.first_long_think_v1.core import (
    RuleConfig,
    contiguous_ply_bins,
    detect_game_views,
    stratified_review_sample,
)


def metadata(game_id: str = "game") -> dict[str, str]:
    return {
        "game_id": game_id,
        "black_id": f"black-{game_id}",
        "white_id": f"white-{game_id}",
        "split": "test",
    }


def rows(times: list[float], game_id: str = "game") -> list[dict[str, str]]:
    result = []
    for index, thinking_time in enumerate(times):
        color = "black" if index % 2 == 0 else "white"
        result.append({
            "game_id": game_id,
            "node_index": str(index),
            "strict_ply": str(index + 1),
            "actor_color": color,
            "actor_id": f"{color}-{game_id}",
            "is_pass": "0",
            "move": "a1",
            "thinking_time_ms": str(thinking_time),
        })
    return result


class DetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuleConfig(min_ply=5, max_ply=38, multiplier=1.75)

    def test_first_strictly_greater_same_player_node_is_anchor(self) -> None:
        game_rows = rows([100, 100, 100, 100, 176, 175])
        black, white = detect_game_views(metadata(), game_rows, self.config)
        self.assertEqual(black["result"], "anchor")
        self.assertEqual(black["anchor"]["anchor_strict_ply"], 5)
        self.assertEqual(black["anchor"]["anchor_target_decision"], 3)
        self.assertEqual(black["anchor"]["prior_time_median_ms"], 100)
        self.assertEqual(black["anchor"]["prior_observation_count"], 2)
        self.assertEqual(white["result"], "no_anchor")

    def test_equality_does_not_qualify(self) -> None:
        black, white = detect_game_views(metadata(), rows([100, 100, 100, 100, 175, 175]), self.config)
        self.assertEqual(black["result"], "no_anchor")
        self.assertEqual(white["result"], "no_anchor")

    def test_nodes_after_cap_do_not_qualify(self) -> None:
        values = [100.0] * 40
        values[38] = 10_000.0  # strict ply 39, black
        black, _ = detect_game_views(metadata(), rows(values), self.config)
        self.assertEqual(black["result"], "no_anchor")


class SamplingTest(unittest.TestCase):
    def test_thirty_anchors_plus_two_no_anchor_use_distinct_games(self) -> None:
        records = []
        for ply in range(5, 39):
            for index in range(4):
                game_id = f"anchor-{ply}-{index}"
                records.append({
                    "game_id": game_id,
                    "target_color": "black",
                    "result": "anchor",
                    "anchor": {"anchor_strict_ply": ply},
                })
        for index in range(4):
            records.append({
                "game_id": f"none-{index}",
                "target_color": "white",
                "result": "no_anchor",
                "anchor": None,
            })
        selected, bins = stratified_review_sample(
            records,
            min_ply=5,
            max_ply=38,
            bin_count=10,
            anchors_per_bin=3,
            no_anchor_count=2,
            seed=42,
        )
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({row["game_id"] for row in selected}), 32)
        self.assertEqual(sum(row["result"] == "anchor" for row in selected), 30)
        self.assertEqual(sum(row["result"] == "no_anchor" for row in selected), 2)
        self.assertEqual([row["selected"] for row in bins], [3] * 10)
        self.assertEqual(contiguous_ply_bins(5, 38, 10)[0], (5, 8))
        self.assertEqual(contiguous_ply_bins(5, 38, 10)[-1], (36, 38))


if __name__ == "__main__":
    unittest.main()
