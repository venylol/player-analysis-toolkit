import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "modeling"
    / "evaluate_personal_tcn_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_personal_tcn_ensemble", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PersonalBootstrapTests(unittest.TestCase):
    def test_reported_selection_allows_missing_optional_offbook_anchor(self):
        MODULE.validate_reported_selection(
            ["anchored", "no-anchor"], {"anchored": 9}, {"anchored", "no-anchor"}
        )

    def test_reported_selection_rejects_unaccounted_test_game(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            MODULE.validate_reported_selection(
                ["anchored"], {"anchored": 9}, {"anchored", "missing"}
            )

    def test_draws_are_deterministic_and_shared_shapes(self):
        members, games = MODULE.bootstrap_draws(100, 12, 2, 123)
        members_again, games_again = MODULE.bootstrap_draws(100, 12, 2, 123)
        np.testing.assert_array_equal(members, members_again)
        np.testing.assert_array_equal(games, games_again)
        self.assertEqual(members.shape, (100, 12))
        self.assertEqual(games.shape, (100, 2))

    def test_control_calibration_reports_adapted_expected_wld_loss(self):
        severity_targets = np.array([0, 3])
        wld_targets = np.array([0, 2])
        wld_losses = np.array([0.0, 1.0])
        applicable = np.array([False, True])
        base_severity = np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]])
        adapted_severity = base_severity.copy()
        base_wld = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]])
        adapted_wld = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        report = MODULE.summarize_control_adaptation(
            severity_targets, wld_targets, wld_losses, applicable,
            base_severity, adapted_severity, base_wld, adapted_wld, 2,
        )
        self.assertEqual(report["wldApplicableNodes"], 1)
        self.assertEqual(report["hardDecisionMatchRates"]["wldThreeClassExact"]["before"], 0.0)
        self.assertEqual(report["hardDecisionMatchRates"]["wldThreeClassExact"]["after"], 1.0)
        wld = report["probabilityRateCalibration"]["expectedWldLoss"]
        self.assertAlmostEqual(wld["before"], 0.25)
        self.assertAlmostEqual(wld["after"], 0.8)

    def test_reported_prediction_summary_returns_wld_metrics(self):
        frame = pd.DataFrame({
            "game_id": ["g"], "wld_applicable": [True],
        })
        probabilities = {
            "zero": np.array([[0.7], [0.5]]),
            "ge4": np.array([[0.2], [0.4]]),
            "ge10": np.array([[0.1], [0.2]]),
        }
        wld = np.array([[0.25], [0.75]])
        member_draws = np.array([[0, 1]])
        game_draws = np.array([[0]])
        report = MODULE.summarize_predicted_group(
            frame, probabilities, wld, ["g"], member_draws, game_draws
        )
        self.assertAlmostEqual(report["pointEstimates"]["expected_wld_loss"], 0.5)
        self.assertEqual(report["wldApplicableNodes"], 1)

    def test_reported_game_without_ply39_wld_nodes_has_zero_wld_contribution(self):
        frame = pd.DataFrame({
            "game_id": ["short", "long"], "wld_applicable": [False, True],
        })
        probabilities = {
            "zero": np.array([[0.6, 0.6]]),
            "ge4": np.array([[0.2, 0.2]]),
            "ge10": np.array([[0.1, 0.1]]),
        }
        wld = np.array([[0.9, 0.5]])
        member_draws = np.array([[0]])
        game_draws = np.array([[0, 1]])
        report = MODULE.summarize_predicted_group(
            frame, probabilities, wld, ["short", "long"], member_draws, game_draws
        )
        self.assertAlmostEqual(report["pointEstimates"]["expected_wld_loss"], 0.25)
        self.assertEqual(report["perGame"]["expected_wld_loss"][0]["nodes"], 0)
        self.assertTrue(report["perGame"]["expected_wld_loss"][0]["noApplicableNodes"])

if __name__ == "__main__":
    unittest.main()
