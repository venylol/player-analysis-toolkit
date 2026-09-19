from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPTS = TOOLKIT_ROOT / "scripts" / "analysis"
SRC_ROOT = TOOLKIT_ROOT / "src"
for path in (SRC_ROOT, ANALYSIS_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import offbook_analysis as offbook


def game(game_id: str, phase_losses: list[float | None]) -> dict:
    plies = (1, 31, 48, 54)
    nodes = [
        {
            "ply": ply,
            "move": "d3",
            "playerAccount": "target",
            "playerColor": "black",
            "lossClipped": loss,
            "thinkingTimeMs": 100,
        }
        for ply, loss in zip(plies, phase_losses)
        if loss is not None
    ]
    return {
        "gameId": game_id,
        "round": 1,
        "color": "black",
        "isTournamentGame": False,
        "nodes": nodes,
    }


class OffBookPhaseBootstrapTests(unittest.TestCase):
    def test_fixed_boundaries_cover_one_to_sixty_once(self) -> None:
        covered = [
            ply
            for phase in offbook.POST_OFFBOOK_PHASES
            for ply in range(phase["start"], phase["stop"])
        ]
        self.assertEqual(covered, list(range(1, 61)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_segment_includes_anchor_and_excludes_pre_anchor_nodes(self) -> None:
        source_nodes = [
            {
                "ply": ply,
                "move": move,
                "playerAccount": "target" if ply % 2 else "opponent",
                "playerColor": "black" if ply % 2 else "white",
                "sourceMoveIndex": ply - 1,
                "lossClipped": float(ply),
                "thinkingTimeMs": ply * 100,
            }
            for ply, move in enumerate(("d3", "c3", "c4", "c5", "d6"), start=1)
        ]
        marks = {
            "schema": "player-offbook-algorithm-records-v1",
            "mode": "target",
            "labeledBy": "algorithm",
            "account": "target",
            "records": [{
                "gameId": "g1", "judgment": "offbook", "algorithmLabel": "offbook", "offBookPly": 3,
            }],
        }
        engine_games = [{"gameId": "g1", "nodes": source_nodes}]
        with patch.object(offbook, "read_json", return_value=marks), patch.object(
            offbook, "load_engine_games", return_value=engine_games
        ):
            result = offbook.segment_member(
                {"account": "target", "marks": "marks.json", "engineDirectory": "eg"},
                "target",
            )
        self.assertEqual([node["ply"] for node in result["postGames"][0]["nodes"]], [3, 5])
        self.assertEqual(result["postGames"][0]["excludedPreOffBookMoveCount"], 1)

    def test_pass_source_gap_does_not_consume_global_placement_ply(self) -> None:
        source_nodes = [
            {"ply": 1, "move": "d3", "sourceMoveIndex": 0},
            {"ply": 2, "move": "c3", "sourceMoveIndex": 2},
        ]
        target_game = {"gameId": "pass-game", "source": {"nodes": source_nodes}}
        audit = offbook.audit_global_placement_ply(target_game)
        self.assertEqual(audit["sourceMoveIndexGapCount"], 1)
        self.assertEqual(audit["maximumPly"], 2)

    def test_ply_audit_rejects_values_above_sixty(self) -> None:
        source_nodes = [
            {"ply": ply, "move": "d3", "sourceMoveIndex": ply - 1}
            for ply in range(1, 62)
        ]
        with self.assertRaisesRegex(ValueError, "exceeds the Othello maximum"):
            offbook.audit_global_placement_ply(
                {"gameId": "invalid-long-game", "source": {"nodes": source_nodes}}
            )

    def test_one_game_spans_phases_but_draw_occurs_once_per_group_per_repetition(self) -> None:
        reported = [game("r1", [1, 2, 3, 4]), game("r2", [2, 3, 4, 5])]
        controls = [game("c1", [5, 4, 3, 2]), game("c2", [4, 3, 2, 1]), game("c3", [3, 3, 3, 3])]
        calls: list[int] = []
        original = offbook.draw_game_cluster_indices

        def recording_draw(rng, size):
            calls.append(size)
            return original(rng, size)

        with patch.object(offbook, "draw_game_cluster_indices", side_effect=recording_draw):
            by_phase, _ = offbook.post_offbook_phase_comparison(
                reported, controls, repetitions=20, seed=7
            )
        self.assertEqual(calls, [2, 3] * 20)
        self.assertTrue(by_phase["bootstrapPolicy"]["sameSampledGameListsSharedAcrossAllPhases"])
        self.assertTrue(by_phase["bootstrapPolicy"]["sameSampledGameListsSharedAcrossAllMetrics"])
        for phase in offbook.POST_OFFBOOK_PHASES:
            self.assertEqual(by_phase[phase["key"]]["reported"]["contributingIndependentGameCount"], 2)

    def test_ge10_is_parallel_to_ge4_in_all_fixed_phases(self) -> None:
        reported = [game("r1", [3, 4, 9, 10]), game("r2", [10, 9, 4, 3])]
        controls = [game("c1", [0, 3, 4, 10]), game("c2", [4, 4, 9, 9])]
        by_phase, combined = offbook.post_offbook_phase_comparison(
            reported, controls, repetitions=200, seed=41
        )
        for phase in offbook.POST_OFFBOOK_PHASES:
            output = by_phase[phase["key"]]
            for arm in ("reported", "control"):
                self.assertLessEqual(
                    output[arm]["lossAtLeast10Count"],
                    output[arm]["lossAtLeast4Count"],
                )
                self.assertLessEqual(
                    output[arm]["lossAtLeast10Rate"],
                    output[arm]["lossAtLeast4Rate"],
                )
            self.assertIn("lossAtLeast10Rate", output["reportedMinusControl"])
        for scheme in combined["weightSchemes"].values():
            metrics = list(scheme["metrics"])
            self.assertEqual(
                metrics.index("lossAtLeast10Rate"),
                metrics.index("lossAtLeast4Rate") + 1,
            )

    def test_seed_is_reproducible(self) -> None:
        reported = [game("r1", [0, 3, 6, 9]), game("r2", [8, 5, 2, 1])]
        controls = [game("c1", [1, 2, 3, 4]), game("c2", [4, 3, 2, 1])]
        first = offbook.post_offbook_phase_comparison(reported, controls, 200, 1234)
        second = offbook.post_offbook_phase_comparison(reported, controls, 200, 1234)
        self.assertEqual(first, second)

    def test_missing_phase_is_null_and_combined_weights_are_not_renormalized(self) -> None:
        reported = [game("r1", [None, 2, 3, 4])]
        controls = [game("c1", [1, 2, 3, 4])]
        by_phase, combined = offbook.post_offbook_phase_comparison(
            reported, controls, repetitions=50, seed=3
        )
        early = by_phase["ply1To30"]["reportedMinusControl"]["gameWeightedMeanLoss"]
        self.assertIsNone(early["estimate"])
        self.assertIsNone(early["clusterBootstrap95CI"])
        for scheme in combined["weightSchemes"].values():
            metric = scheme["metrics"]["gameWeightedMeanLoss"]
            self.assertIsNone(metric["reportedMinusControl"])
            self.assertIsNone(metric["clusterBootstrap95CI"])
            self.assertEqual(metric["successfulRepetitions"], 0)

    def test_combined_interval_comes_from_within_repetition_combination(self) -> None:
        reported = [
            game("r1", [0, 10, 0, 10]),
            game("r2", [10, 0, 10, 0]),
        ]
        controls = [game("c1", [2, 2, 2, 2]), game("c2", [2, 2, 2, 2])]
        by_phase, combined = offbook.post_offbook_phase_comparison(
            reported, controls, repetitions=2000, seed=91
        )
        combined_ci = combined["weightSchemes"]["equalPhase"]["metrics"][
            "gameWeightedMeanLoss"
        ]["clusterBootstrap95CI"]
        averaged_endpoints = [
            sum(
                0.25
                * by_phase[phase["key"]]["reportedMinusControl"]["gameWeightedMeanLoss"][
                    "clusterBootstrap95CI"
                ][endpoint]
                for phase in offbook.POST_OFFBOOK_PHASES
            )
            for endpoint in (0, 1)
        ]
        self.assertEqual(combined_ci, [3.0, 3.0])
        self.assertNotEqual(combined_ci, averaged_endpoints)

    def test_existing_model_units_remain_present_and_compatible(self) -> None:
        full_games = [
            game("r", [1, 2, 3, 4]),
            game("c", [2, 3, 4, 5]),
        ]
        post_games = [
            {**item, "offBookPly": 1, "excludedPreOffBookMoveCount": 0}
            for item in full_games
        ]
        segment = {
            "fullGames": full_games,
            "postGames": post_games,
            "plyAudits": [
                {
                    "gameId": item["gameId"],
                    "maximumPly": 54,
                    "sourceMoveIndexGapCount": 0,
                }
                for item in full_games
            ],
        }
        config = {
            "dataset": {"account": "target"},
            "reportedGameIds": ["r"],
            "bootstrap": 20,
            "modelBootstrap": 1,
            "seed": 5,
        }
        with patch.object(offbook, "segment_member", return_value=segment), patch.object(
            offbook, "two_part_model", return_value={"fixture": True}
        ):
            result = offbook.offbook_model(config)
        self.assertEqual(result["schema"], "player-offbook-segment-model-v1")
        for key in ("fullGame", "postOffBookInclusive"):
            self.assertEqual(set(result[key]), {"comparison", "clusterAwareTwoPartModel"})
            self.assertEqual(result[key]["clusterAwareTwoPartModel"], {"fixture": True})
        self.assertIn("postOffBookByPhase", result)
        self.assertIn("postOffBookPhaseCombined", result)


if __name__ == "__main__":
    unittest.main()
