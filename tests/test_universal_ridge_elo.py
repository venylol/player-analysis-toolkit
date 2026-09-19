from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "train_universal_ridge_elo.py"
SPEC = importlib.util.spec_from_file_location("train_universal_ridge_elo", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UniversalRidgeEloTests(unittest.TestCase):
    def test_target_steps_are_post_offbook_and_target_only(self) -> None:
        game = {
            "nodes": [
                {"ply": 5, "playerAccount": "p", "actualEval": 1, "bestEval": 2, "boardBefore": "-" * 50 + "X" * 14},
                {"ply": 7, "playerAccount": "other", "actualEval": 3, "bestEval": 4, "boardBefore": "-" * 48 + "X" * 16},
                {"ply": 9, "playerAccount": "P", "actualEval": -5, "bestEval": 6, "boardBefore": "-" * 46 + "X" * 18},
            ]
        }
        self.assertEqual(MODULE.target_steps(game, "p", 7), [(-5.0, 6.0, 46.0)])

    def test_aggregate_features_uses_population_standard_deviation(self) -> None:
        features = MODULE.aggregate_features([(1, 3, 50), (3, 7, 40)])
        self.assertEqual(features, [2.0, 1.0, 5.0, 2.0, 45.0, 5.0])

    def test_polynomial_feature_count_is_27(self) -> None:
        model = MODULE.build_model(1.0)
        poly = model.named_steps["poly"]
        poly.fit([[0.0] * 6])
        self.assertEqual(len(poly.get_feature_names_out(MODULE.RAW_FEATURE_NAMES)), 27)


if __name__ == "__main__":
    unittest.main()
