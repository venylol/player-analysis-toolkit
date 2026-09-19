import copy
import math
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from player_analysis_toolkit import sentinel_elo as elo


def conditional_row(
    index: int,
    color: str,
    scope: str,
    *,
    target_elo: float | None = None,
    opponent_elo: float | None = None,
    game_id: str | None = None,
    target_id: str | None = None,
) -> dict:
    return {
        "schema": elo.SCHEMA_CONDITIONAL_RECORD,
        "recordId": f"record-{color}-{scope}-{index:03d}",
        "gameId": game_id or f"game-{color}-{scope}-{index:03d}",
        "targetPlayerId": target_id or f"player-{index:03d}",
        "opponentPlayerId": f"opponent-{index:03d}",
        "targetColor": color,
        "scope": scope,
        "targetOldR": float(target_elo if target_elo is not None else 1600 + (index % 30) * 20),
        "opponentOldR": float(opponent_elo if opponent_elo is not None else 1700 + (index % 15) * 15),
        "referenceZ1": (index - 15) / 10.0,
        "referenceZ2": (index - 15) / 8.0,
        "referenceZ3": (index - 15) / 6.0,
        **{f"phase{phase}": {"x": index % 3, "n": 5} for phase in range(1, 5)},
    }


def synthetic_reference() -> tuple[list[dict], dict, dict]:
    config = copy.deepcopy(elo.default_v3_config())
    records = [
        conditional_row(index, color, scope)
        for color in elo.COLORS
        for scope in elo.METRICS_SCOPES
        for index in range(30)
    ]
    scales = {
        f"{color}|{scope}": {
            "color": color,
            "scope": scope,
            "recordCount": 30,
            "selfEloSd": 180.0,
            "opponentEloSd": 160.0,
            "phase1ZSd": 1.0,
            "phase2ZSd": 1.0,
            "phase3ZSd": 1.0,
        }
        for color in elo.COLORS
        for scope in elo.METRICS_SCOPES
    }
    manifest = {
        "schema": elo.SCHEMA_CONDITIONAL_REFERENCE,
        "algorithmVersion": elo.ALGORITHM_VERSION_V2,
        "configSha256": elo.canonical_sha256(elo.conditional_config_for_v3(config)),
        "referenceFeaturePolicy": config["referenceFeaturePolicy"],
        "scales": scales,
    }
    return records, manifest, config


class SentinelEloV3Tests(unittest.TestCase):
    def test_versioned_contract_and_aligned_search_points(self) -> None:
        config = elo.validate_v3_config(elo.default_v3_config())
        self.assertEqual(config["searchSteps"], [40, 20, 10, 5, 2, 1])
        self.assertEqual(elo.elo_grid_v3(config)[-1], 2500)
        self.assertEqual(
            elo.aligned_elo_points_v3(1600, 2500, 40),
            list(range(1600, 2481, 40)) + [2500],
        )
        self.assertEqual(
            elo.aligned_elo_points_v3(1603, 1611, 2),
            [1603, 1604, 1606, 1608, 1610, 1611],
        )
        self.assertEqual(
            elo.aligned_elo_points_v3(1603, 1611, 5),
            [1603, 1605, 1610, 1611],
        )

    def test_basin_discovery_keeps_multiple_platform_and_boundary_basins(self) -> None:
        points = [
            {"elo": 1600, "score": 0.0},
            {"elo": 1601, "score": 1.0},
            {"elo": 1602, "score": 2.0},
            {"elo": 1603, "score": 1.0},
            {"elo": 1604, "score": 0.0},
            {"elo": 1605, "score": 0.0},
            {"elo": 1606, "score": 1.0},
            {"elo": 1607, "score": 2.0},
            {"elo": 1608, "score": 1.0},
            {"elo": 1609, "score": 0.0},
        ]
        basins = elo._v3_discover_basins(
            points,
            resolution=1,
            formal_minimum=1600,
            formal_maximum=1609,
            tolerance=1e-12,
        )
        self.assertEqual([basin["minimumEloPoints"] for basin in basins], [[1600], [1604, 1605], [1609]])
        self.assertTrue(basins[0]["boundaryTrend"])
        self.assertTrue(basins[1]["internalMinimum"])
        self.assertTrue(basins[2]["boundaryTrend"])
        self.assertEqual(
            elo._v3_merge_intervals([(1600, 1610), (1608, 1620), (1700, 1701)]),
            [{"lower": 1600, "upper": 1620}, {"lower": 1700, "upper": 1701}],
        )

    def test_sparse_threshold_never_turns_sample_gap_into_integer_interval(self) -> None:
        intervals = elo.intervals_for_score_threshold_v3(
            [
                {"elo": 1600, "score": 0.5},
                {"elo": 1620, "score": 0.5},
                {"elo": 1640, "score": 2.0},
            ],
            1.0,
        )
        self.assertEqual(len(intervals), 2)
        self.assertEqual([(row["lower"], row["upper"]) for row in intervals], [(1600, 1600), (1620, 1620)])

    def test_global_knn_matches_bruteforce_with_account_and_game_exclusion(self) -> None:
        records, manifest, config = synthetic_reference()
        samples = [
            {
                "account": "player-003",
                "targetColor": "black",
                "scope": "full_game",
                "stage": 1,
                "targetGameIds": ["game-black-full_game-004"],
                "trialElo": 1825,
                "opponentElo": 1780,
            },
            {
                "account": "PLAYER-017",
                "targetColor": "white",
                "scope": "post_offbook_inclusive",
                "stage": 2,
                "targetGameIds": ["game-white-post_offbook_inclusive-009"],
                "trialElo": 1960,
                "opponentElo": 1830,
                "previousZ": 0.25,
            },
            {
                "account": "absent-account",
                "targetColor": "black",
                "scope": "full_game",
                "stage": 3,
                "targetGameIds": [],
                "trialElo": 2100,
                "opponentElo": 1810,
                "previousZ": -0.5,
            },
            {
                "account": "player-024",
                "targetColor": "white",
                "scope": "full_game",
                "stage": 4,
                "targetGameIds": ["game-white-full_game-024"],
                "trialElo": 1700,
                "opponentElo": 1900,
                "previousZ": 0.75,
            },
        ]
        report = elo.audit_global_knn_consistency_v3(records, manifest, config, samples)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mismatchCount"], 0)
        self.assertEqual(report["globalTreeBuildCount"], 16)
        self.assertGreater(report["queryCount"], 0)

    def test_same_distance_boundary_uses_stable_game_key(self) -> None:
        records = [
            conditional_row(0, "black", "full_game", target_elo=1600, opponent_elo=1700),
            *[
                conditional_row(
                    index,
                    "black",
                    "full_game",
                    target_elo=1700,
                    opponent_elo=1700,
                    game_id=f"tie-{index:02d}",
                )
                for index in range(1, 20)
            ],
        ]
        _, _, config = synthetic_reference()
        scales = {
            "selfEloSd": 100.0,
            "opponentEloSd": 100.0,
            "phase1ZSd": 1.0,
            "phase2ZSd": 1.0,
            "phase3ZSd": 1.0,
        }
        pool = elo._GlobalConditionalPoolV3(records, scales, 1, config, "black|full_game")
        result = pool.nearest("absent", [], 1600, 1700)
        self.assertTrue(result["ok"])
        self.assertEqual(result["K"], elo.neighbor_count_v3(result["N_allowed"]))
        expected = [records[0]["gameId"]] + sorted(row["gameId"] for row in records[1:])[: result["K"] - 1]
        self.assertEqual(sorted(result["records"][index]["gameId"] for index in range(result["K"])), expected)
        self.assertGreaterEqual(pool.query_ball_count, 1)

    def test_adaptive_search_keeps_two_valleys_and_known_probe_isolated(self) -> None:
        records, manifest, config = synthetic_reference()
        index = elo.GlobalConditionalKNNIndexV3(records, manifest, config)

        class FakeEvaluator:
            def __init__(self, target_records, account, global_index, cfg):
                self.target_records = list(target_records)
                self.account = account
                self.index = global_index
                self.config = cfg
                self.cache = {}
                self.search_point_elos = set()
                self.fit_count = 0
                self.scoring_seconds = 0.0
                self.hard_failure_reasons = Counter()
                self.query_count_before = global_index.query_count
                self.tree_query_count_before = global_index.tree_query_count
                self.boundary_query_count_before = global_index.boundary_expansion_count
                self._tree_build_was_reused = not global_index.begin_account(account)

            @staticmethod
            def score(elo_value):
                return min((elo_value - 1750) ** 2, (elo_value - 2250) ** 2) / 10000.0

            def evaluate(self, elo_value, *, include_in_search=False):
                elo_value = int(elo_value)
                if include_in_search:
                    self.search_point_elos.add(elo_value)
                if elo_value not in self.cache:
                    score = self.score(elo_value)
                    self.cache[elo_value] = {
                        "elo": elo_value,
                        "score": score,
                        "meanNegativeLogLikelihood": score,
                        "meanConditionalZ": 0.0,
                        "validTargetGameCount": 1,
                        "failureReasons": [],
                        "failureReasonCounts": {},
                        "gameDiagnostics": [{
                            "gameId": "target",
                            "ok": True,
                            "gameNegativeLogLikelihood": score,
                            "meanConditionalZ": 0.0,
                            "phaseDiagnostics": [],
                        }],
                    }
                    self.fit_count += 4
                return self.cache[elo_value]

            def evaluate_many(self, elos, *, include_in_search=False):
                return [self.evaluate(elo_value, include_in_search=include_in_search) for elo_value in elos]

        target = [{"gameId": "target", "targetColor": "black", "scope": "full_game"}]
        with patch.object(elo, "_V3ScoreEvaluator", FakeEvaluator):
            curve = elo.score_candidate_curve_v3(
                target,
                records,
                manifest,
                target_account="target",
                config=config,
                known_elo=1855.5,
                index=index,
            )
        self.assertFalse(curve["fallbackToFullGrid"], curve["fallbackReasons"])
        self.assertLess(curve["evaluatedPointCount"], 901)
        self.assertEqual(curve["searchSteps"], [40, 20, 10, 5, 2, 1])
        self.assertEqual(curve["bestGridPoint"], 1750)
        basin_minima = {
            elo_value
            for basin in curve["discoveredBasins"]
            for elo_value in basin.get("minimumEloPoints", [])
        }
        self.assertIn(1750, basin_minima)
        self.assertIn(2250, basin_minima)
        self.assertNotIn(1855, curve["evaluatedEloPoints"])
        self.assertNotIn(1856, curve["evaluatedEloPoints"])
        self.assertEqual(curve["knownEloProbe"]["participatesInMinimumJ"], False)
        self.assertEqual(curve["knownEloProbe"]["evaluatedEloPoints"], [1855, 1856])

    def test_monotone_boundary_search_falls_back_to_complete_integer_grid(self) -> None:
        records, manifest, config = synthetic_reference()
        index = elo.GlobalConditionalKNNIndexV3(records, manifest, config)

        class MonotoneEvaluator:
            def __init__(self, target_records, account, global_index, cfg):
                self.cache = {}
                self.search_point_elos = set()
                self.index = global_index
                self.target_records = list(target_records)
                self.account = account
                self.config = cfg
                self.fit_count = 0
                self.scoring_seconds = 0.0
                self.hard_failure_reasons = Counter()
                self.query_count_before = global_index.query_count
                self.tree_query_count_before = global_index.tree_query_count
                self.boundary_query_count_before = global_index.boundary_expansion_count
                self._tree_build_was_reused = not global_index.begin_account(account)

            def evaluate(self, elo_value, *, include_in_search=False):
                elo_value = int(elo_value)
                if include_in_search:
                    self.search_point_elos.add(elo_value)
                self.cache.setdefault(elo_value, {
                    "elo": elo_value,
                    "score": float(elo_value),
                    "meanNegativeLogLikelihood": float(elo_value),
                    "meanConditionalZ": 0.0,
                    "validTargetGameCount": 1,
                    "failureReasons": [],
                    "failureReasonCounts": {},
                    "gameDiagnostics": [],
                })
                return self.cache[elo_value]

            def evaluate_many(self, elos, *, include_in_search=False):
                return [self.evaluate(elo_value, include_in_search=include_in_search) for elo_value in elos]

        with patch.object(elo, "_V3ScoreEvaluator", MonotoneEvaluator):
            curve = elo.score_candidate_curve_v3(
                [{"gameId": "target", "targetColor": "black", "scope": "full_game"}],
                records,
                manifest,
                target_account="target",
                config=config,
                index=index,
            )
        self.assertTrue(curve["fallbackToFullGrid"])
        self.assertIn("minimum_region_not_bracketed_on_lower_side", curve["fallbackReasons"])
        self.assertEqual(curve["evaluatedPointCount"], 901)
        self.assertTrue(curve["isFullGrid"])

    def test_v3_fit_failure_is_terminal_and_not_hidden_by_grid_fallback(self) -> None:
        records, manifest, config = synthetic_reference()
        pool = elo._GlobalConditionalPoolV3(
            records[:30], manifest["scales"]["black|full_game"], 1, config, "black|full_game"
        )
        target = dict(records[0], targetPlayerId="target", opponentPlayerId="other", gameId="target-game")
        failed = {"ok": False, "fitStatus": "optimizer_not_converged"}
        with patch.object(elo, "fit_weighted_beta_binomial", return_value=failed):
            result = elo.score_conditional_phase_v3(
                target, 1, 1800, pool,
                account="target", target_game_ids=["target-game"], config=config,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "beta_binomial_fit_failed")
        self.assertTrue(result["fitAttempted"])


if __name__ == "__main__":
    unittest.main()
