from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TOOLKIT_ROOT / "src"
REVIEW_SCRIPTS = TOOLKIT_ROOT / "scripts" / "review"
for path in (SRC_ROOT, REVIEW_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from player_analysis_toolkit import analysis_core
import generate_review_checklist as checklist


PHASE_SCRIPT = (
    TOOLKIT_ROOT
    / "scripts"
    / "analysis"
    / "analyze_oq_loss_phase_boundaries.py"
)
SPEC = importlib.util.spec_from_file_location("oq_loss_phase_boundaries", PHASE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
phase_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase_analysis)


def game(game_id: str, losses: list[float], probabilities: bool = False) -> dict:
    nodes = []
    for ply, loss in enumerate(losses, start=1):
        node = {
            "ply": ply,
            "lossClipped": loss,
            "playerAccount": "target",
            "playerColor": "black",
        }
        if probabilities:
            node["probability_loss_ge4"] = 0.25
            node["probability_loss_ge10"] = 0.10
        nodes.append(node)
    return {
        "gameId": game_id,
        "round": 1,
        "color": "black",
        "isTournamentGame": False,
        "nodes": nodes,
    }


class LossGe10StatisticsTests(unittest.TestCase):
    def test_oversized_exact_combination_uses_reproducible_capped_sampling(self) -> None:
        universe = [{"gameId": f"g{index}"} for index in range(30)]
        for index, row in enumerate(universe):
            row["nodes"] = [{"lossClipped": float(index)}]
        selected = {f"g{index}" for index in range(15)}
        result = analysis_core.exact_combination_position(
            universe, selected, maximum_combinations=100, seed=7
        )
        repeated = analysis_core.exact_combination_position(
            universe, selected, maximum_combinations=100, seed=7
        )
        self.assertEqual(result, repeated)
        self.assertEqual(result["status"], "monte_carlo")
        self.assertEqual(result["combinationCount"], 155_117_520)
        self.assertEqual(result["sampledCombinationCount"], 100)
        self.assertEqual(result["seed"], 7)

    def test_exact_combination_default_cap_is_one_million(self) -> None:
        self.assertEqual(analysis_core.EXACT_COMBINATION_ENUMERATION_LIMIT, 1_000_000)

    def test_inclusive_threshold_boundaries(self) -> None:
        expected = {
            3: {"loss_ge4": False, "loss_ge10": False},
            4: {"loss_ge4": True, "loss_ge10": False},
            9: {"loss_ge4": True, "loss_ge10": False},
            10: {"loss_ge4": True, "loss_ge10": True},
        }
        for raw_loss, flags in expected.items():
            with self.subTest(raw_loss=raw_loss):
                self.assertEqual(analysis_core.loss_threshold_flags(raw_loss), flags)
        self.assertEqual(analysis_core.disc_loss(-3), 0.0)

    def test_ge10_count_and_rate_never_exceed_ge4(self) -> None:
        summary = analysis_core.aggregate_loss_games([game("g", [0, 3, 4, 9, 10, 20])])
        self.assertLessEqual(summary["lossAtLeast10Count"], summary["lossAtLeast4Count"])
        self.assertLessEqual(summary["lossAtLeast10Rate"], summary["lossAtLeast4Rate"])
        per_game = summary["games"][0]
        self.assertLessEqual(per_game["lossAtLeast10Count"], per_game["lossAtLeast4Count"])
        self.assertLessEqual(per_game["lossAtLeast10Rate"], per_game["lossAtLeast4Rate"])

    def test_model_probability_absence_is_explicit_and_not_zero_filled(self) -> None:
        summary = analysis_core.aggregate_loss_games([game("g", [4, 10])])
        for key in ("lossGe4", "lossGe10"):
            probability = summary["modelProbability"][key]
            self.assertEqual(probability["status"], "unavailable")
            self.assertEqual(probability["unavailableReason"], "模型概率不可用")
            self.assertIsNone(probability["expectedNodeCount"])
            self.assertIsNone(probability["actualMinusExpectedNodeCount"])

    def test_model_probability_expected_and_actual_metrics(self) -> None:
        summary = analysis_core.aggregate_loss_games(
            [game("g", [0, 4, 10, 20], probabilities=True)]
        )
        ge4 = summary["modelProbability"]["lossGe4"]
        ge10 = summary["modelProbability"]["lossGe10"]
        self.assertEqual(ge4["expectedNodeCount"], 1.0)
        self.assertEqual(ge4["actualNodeCount"], 3)
        self.assertEqual(ge4["actualRateMinusMeanProbability"], 0.5)
        self.assertEqual(ge10["expectedNodeCount"], 0.4)
        self.assertEqual(ge10["actualNodeCount"], 2)
        self.assertEqual(ge10["actualRateMinusMeanProbability"], 0.4)

    def test_threshold_bootstrap_draws_once_for_both_metrics(self) -> None:
        reported = [game("r1", [4]), game("r2", [10])]
        controls = [game("c1", [0]), game("c2", [4])]

        class RecordingRandom:
            instance = None

            def __init__(self, seed):
                self.calls = []
                RecordingRandom.instance = self

            def randrange(self, size):
                self.calls.append(size)
                return (len(self.calls) - 1) % size

        with patch.object(analysis_core.random, "Random", RecordingRandom):
            result = analysis_core.cluster_bootstrap_threshold_rate_differences(
                reported, controls, repetitions=5, seed=123
            )
        self.assertEqual(RecordingRandom.instance.calls, [2, 2, 2, 2] * 5)
        self.assertEqual(set(result), {4, 10})

    def test_fixed_seed_reproduces_threshold_intervals(self) -> None:
        reported = [game("r1", [4, 9]), game("r2", [10, 20])]
        controls = [game("c1", [0, 3]), game("c2", [4, 10])]
        first = analysis_core.cluster_bootstrap_threshold_rate_differences(
            reported, controls, repetitions=200, seed=20260803
        )
        second = analysis_core.cluster_bootstrap_threshold_rate_differences(
            reported, controls, repetitions=200, seed=20260803
        )
        self.assertEqual(first, second)

    def test_zero_loss_bootstrap_uses_whole_games(self) -> None:
        reported = [game("r1", [0, 0]), game("r2", [0])]
        controls = [game("c1", [4]), game("c2", [10, 10])]
        self.assertEqual(
            analysis_core.cluster_bootstrap_zero_loss_rate_difference(
                reported, controls, repetitions=20, seed=7
            ),
            [1.0, 1.0],
        )

    def test_json_csv_and_markdown_keep_ge10_adjacent_and_consistent(self) -> None:
        aggregate = analysis_core.aggregate_loss_games([game("g", [3, 4, 9, 10])])
        keys = list(aggregate)
        self.assertEqual(keys.index("lossAtLeast10Count"), keys.index("lossAtLeast4Count") + 1)
        self.assertEqual(keys.index("lossAtLeast10Rate"), keys.index("lossAtLeast4Rate") + 1)
        serialized = json.loads(json.dumps(aggregate, ensure_ascii=False))
        self.assertEqual(serialized["lossAtLeast4Count"], 3)
        self.assertEqual(serialized["lossAtLeast10Count"], 1)

        frame = pd.DataFrame(
            {
                "disc_loss": [3.0, 4.0, 9.0, 10.0],
                "raw_loss": [3.0, 4.0, 9.0, 10.0],
                "game_id": ["g"] * 4,
                "player_id": ["p"] * 4,
            }
        )
        row = phase_analysis.summarize_loss_frame(frame)
        csv_header = pd.DataFrame([row]).to_csv(index=False).splitlines()[0].split(",")
        self.assertEqual(csv_header.index("loss_ge10_count"), csv_header.index("loss_ge4_count") + 1)
        self.assertEqual(csv_header.index("loss_ge10_rate"), csv_header.index("loss_ge4_rate") + 1)
        self.assertEqual(row["loss_ge4_count"], aggregate["lossAtLeast4Count"])
        self.assertEqual(row["loss_ge10_count"], aggregate["lossAtLeast10Count"])

        report_row = checklist.summarize_loss_stats(aggregate, "fixture")
        report_row.update({"group": "fixture", "segment": "整局"})
        markdown = checklist.render_loss_control_table([report_row])
        header = markdown.splitlines()[0]
        self.assertLess(header.index("子损≥4节点数"), header.index("子损≥10节点数"))
        self.assertLess(header.index("子损≥4比例"), header.index("子损≥10比例"))
        self.assertIn("| 3 | 1 | 75.0% | 25.0% |", markdown)

    def test_stage_bootstrap_ge4_and_ge10_use_same_whole_game_weights(self) -> None:
        nodes = pd.DataFrame(
            {
                "game_id": ["g1"] * 4 + ["g2"] * 4,
                "player_id": ["p1"] * 4 + ["p2"] * 4,
                "ply": [1, 31, 48, 54] * 2,
                "disc_loss": [4.0, 10.0, 4.0, 10.0, 0.0, 4.0, 0.0, 4.0],
                "raw_loss": [4.0, 10.0, 4.0, 10.0, 0.0, 4.0, 0.0, 4.0],
            }
        )
        candidates = [{"k": 4, "boundaries": [30, 47, 53]}]
        stage_table = phase_analysis.summarize_candidate_stages(nodes, candidates, 60)
        shared_weights = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        result = phase_analysis.add_stage_bootstrap_intervals(
            stage_table, nodes, candidates, ["g1", "g2"], shared_weights, 60
        )
        for row in result.itertuples(index=False):
            if np.isfinite(row.loss_ge4_rate_boot_q025):
                self.assertLessEqual(
                    row.loss_ge10_rate_boot_q025, row.loss_ge4_rate_boot_q025
                )
                self.assertLessEqual(
                    row.loss_ge10_rate_boot_q975, row.loss_ge4_rate_boot_q975
                )


if __name__ == "__main__":
    unittest.main()
