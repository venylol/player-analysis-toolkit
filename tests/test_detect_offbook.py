from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "detect_offbook.py"
SPEC = importlib.util.spec_from_file_location("detect_offbook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def node(ply: int, time_ms: float, score: float, *, account: str = "target") -> dict:
    return {
        "ply": ply,
        "move": "d3",
        "playerAccount": account,
        "playerColor": "black" if account == "target" else "white",
        "thinkingTimeMs": time_ms,
        "bestEval": score,
    }


def game(target_nodes: list[dict]) -> dict:
    nodes = []
    by_ply = {item["ply"]: item for item in target_nodes}
    for ply in range(1, max(by_ply) + 1):
        nodes.append(by_ply.get(ply, node(ply, 100, 0, account="opponent")))
    return {"gameId": "g1", "nodes": nodes}


def detect(source: dict, account: str = "target", time_limit_ms: float = 300000) -> dict:
    return MODULE.detect_game(source, account, time_limit_ms)


class DetectOffbookTests(unittest.TestCase):
    def test_first_time_anchor_over_5_5_seconds(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 5500, 0), node(7, 5501, 0),
        ]))
        self.assertEqual(result["algorithmLabel"], "offbook")
        self.assertEqual(result["offBookPly"], 7)
        self.assertEqual(result["anchorSource"], "time_rule_without_evaluation_cutoff")

    def test_time_equal_to_5_5_seconds_does_not_trigger(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 5500, 0),
        ]))
        self.assertEqual(result["algorithmLabel"], "no_offbook")
        self.assertIsNone(result["offBookPly"])

    def test_score_equal_to_positive_or_negative_six_does_not_trigger(self) -> None:
        for score in (6.0, -6.0):
            with self.subTest(score=score):
                result = detect(game([
                    node(1, 100, 0), node(3, 100, 0), node(5, 100, score),
                ]))
                self.assertEqual(result["algorithmLabel"], "no_offbook")
                self.assertIsNone(result["offBookPly"])

    def test_absolute_score_over_six_is_fallback_anchor(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 100, -7),
        ]))
        self.assertEqual(result["offBookPly"], 5)
        self.assertEqual(result["anchorSource"], "absolute_evaluation_cutoff")

    def test_time_search_stops_strictly_before_evaluation_cutoff(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 100, 0), node(7, 1000, 7),
        ]))
        self.assertEqual(result["offBookPly"], 7)
        self.assertEqual(result["anchorSource"], "absolute_evaluation_cutoff")

    def test_time_anchor_before_evaluation_cutoff_wins(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 5501, 0), node(7, 100, 7),
        ]))
        self.assertEqual(result["offBookPly"], 5)
        self.assertEqual(result["anchorSource"], "time_rule_before_evaluation_cutoff")

    def test_there_is_no_maximum_ply(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 100, 0), node(60, 5501, 0),
        ]))
        self.assertEqual(result["offBookPly"], 60)

    def test_time_threshold_grows_logarithmically_with_time_limit(self) -> None:
        self.assertAlmostEqual(MODULE.time_threshold_ms(300000), 5500.0)
        self.assertAlmostEqual(MODULE.time_threshold_ms(600000), 6166.391, places=3)
        self.assertAlmostEqual(MODULE.time_threshold_ms(1200000), 6833.582, places=3)
        self.assertAlmostEqual(MODULE.time_threshold_ms(1500000), 7048.467, places=3)

    def test_fast_threshold_grows_logarithmically_with_time_limit(self) -> None:
        self.assertAlmostEqual(MODULE.fast_threshold_ms(300000), 2000.0)
        self.assertAlmostEqual(MODULE.fast_threshold_ms(600000), 2242.324, places=3)

    def test_three_consecutive_quick_moves_reject_time_candidate(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0),
            node(5, 6000, 0), node(7, 1000, 0), node(9, 1000, 0), node(11, 1000, 0),
            node(13, 6000, 0), node(15, 3000, 0), node(17, 3000, 0), node(19, 3000, 0),
        ]))
        self.assertEqual(result["offBookPly"], 13)
        self.assertEqual(result["algorithmEvidence"]["rejectedCandidates"][0]["ply"], 5)

    def test_two_consecutive_quick_moves_do_not_reject_candidate(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0),
            node(5, 6000, 0), node(7, 1000, 0), node(9, 1000, 0),
            node(11, 2500, 0), node(13, 1000, 0),
        ]))
        self.assertEqual(result["offBookPly"], 5)
        self.assertEqual(result["postFastCheck"]["status"], "passed")

    def test_rejected_evaluation_cutoff_yields_no_offbook(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0),
            node(5, 100, 7), node(7, 1000, 0), node(9, 1000, 0), node(11, 1000, 0),
        ]))
        self.assertEqual(result["algorithmLabel"], "no_offbook")
        self.assertEqual(result["algorithmEvidence"]["noAnchorReason"], "all_candidates_rejected_by_post_fast_check")

    def test_insufficient_following_moves_keeps_candidate(self) -> None:
        result = detect(game([
            node(1, 100, 0), node(3, 100, 0), node(5, 6000, 0), node(7, 1000, 0),
        ]))
        self.assertEqual(result["offBookPly"], 5)
        self.assertEqual(result["postFastCheck"]["status"], "insufficient")

    def test_game_with_no_target_placement_still_gets_no_offbook_label(self) -> None:
        source = {
            "gameId": "one-move",
            "black": {"account": "opponent"},
            "white": {"account": "target"},
            "nodes": [node(1, 1, 0, account="opponent")],
        }
        result = detect(source)
        self.assertEqual(result["targetColor"], "white")
        self.assertEqual(result["targetMoveCount"], 0)
        self.assertEqual(result["algorithmLabel"], "no_offbook")
        self.assertIsNone(result["offBookPly"])


if __name__ == "__main__":
    unittest.main()
