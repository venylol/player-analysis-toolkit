from __future__ import annotations

import copy
import inspect
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from player_analysis_toolkit import sentinel_elo as elo


def source_record(index: int, color: str, scope: str) -> dict:
    phases = {}
    for phase in range(1, 5):
        phases[f"phase{phase}"] = {
            "lossGe4Count": (index + phase) % 6,
            "validLossNodeCount": 10 + phase,
        }
    metrics = {
        "completeFourPhase": True,
        "equalPhaseGameGe4Rate": 0.2,
        **{
            f"phase{phase}": {
                "lossGe4Count": phases[f"phase{phase}"]["lossGe4Count"],
                "validLossNodeCount": phases[f"phase{phase}"]["validLossNodeCount"],
            }
            for phase in range(1, 5)
        },
    }
    return {
        "schema": elo.SCHEMA_DIRECTED,
        "gameId": f"game-{color}-{scope}-{index:03d}",
        "created": f"2026-08-{(index % 28) + 1:02d}T00:00:00Z",
        "targetPlayerId": f"player-{index:03d}",
        "opponentPlayerId": f"opponent-{index:03d}",
        "targetColor": color,
        "targetOldR": 1600.0 + index * 18.0,
        "opponentOldR": 1800.0 + index * 11.0,
        "formalReferenceEligible": True,
        "algorithmLabel": "offbook" if scope == "post_offbook_inclusive" else "no_offbook",
        "metrics": {
            "full_game": copy.deepcopy(metrics),
            "post_offbook_inclusive": copy.deepcopy(metrics),
        },
    }


def v4_rows(count: int = 12) -> tuple[list[dict], dict]:
    rows = []
    scales = {}
    for color in elo.COLORS:
        for scope in elo.METRICS_SCOPES:
            key = f"{color}|{scope}"
            sources = [source_record(index, color, scope) for index in range(count)]
            prepared, pool_scales = elo.anscombe_base_records_v4(sources)
            for row_index, row in enumerate(prepared):
                row["referenceZ1"] = (row_index - 10) / 5.0
                row["referenceZ2"] = (row_index - 10) / 7.0
                row["referenceZ3"] = (row_index - 10) / 9.0
                row["referenceZ4"] = (row_index - 10) / 11.0
            pool_scales[f"{color}|{scope}"]["phase1ZSd"] = 1.0
            pool_scales[f"{color}|{scope}"]["phase2ZSd"] = 1.0
            pool_scales[f"{color}|{scope}"]["phase3ZSd"] = 1.0
            rows.extend(prepared)
            scales[key] = pool_scales[key]
    manifest = {
        "schema": elo.SCHEMA_ANScombe_REFERENCE,
        "algorithmVersion": elo.ALGORITHM_VERSION_V4,
        "scales": scales,
    }
    return rows, manifest


class SentinelEloV4Tests(unittest.TestCase):
    def test_anscombe_endpoints_and_sampling_variance(self) -> None:
        values = [
            elo.anscombe_transform(0, 30),
            elo.anscombe_transform(30, 30),
            elo.anscombe_transform(7, 19),
        ]
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(elo.anscombe_sampling_variance(1), elo.anscombe_sampling_variance(10))
        self.assertGreater(elo.anscombe_sampling_variance(10), elo.anscombe_sampling_variance(100))

    def test_closed_form_statistics_match_hand_calculation(self) -> None:
        result = elo.anscombe_weighted_predictive_statistics(
            [1.0, 3.0], [0.2, 0.4], [1.0, 3.0], 2.0, 0.5
        )
        self.assertTrue(result["ok"])
        u1, u2 = 0.25, 0.75
        mu = u1 * 1.0 + u2 * 3.0
        c_value = 1.0 - (u1 * u1 + u2 * u2)
        observed = (u1 * (1.0 - mu) ** 2 + u2 * (3.0 - mu) ** 2) / c_value
        sampling = (u1 * (1.0 - u1) * 0.2 + u2 * (1.0 - u2) * 0.4) / c_value
        between = max(0.0, observed - sampling)
        mean_variance = u1 * u1 * (between + 0.2) + u2 * u2 * (between + 0.4)
        predictive = between + 0.5 + mean_variance
        target_z = (2.0 - mu) / math.sqrt(predictive)
        score = 0.5 * (math.log(2.0 * math.pi * predictive) + target_z ** 2)
        for key, expected in {
            "C": c_value,
            "localMeanY": mu,
            "observedVariance": observed,
            "samplingVariance": sampling,
            "betweenGameVariance": between,
            "meanEstimateVariance": mean_variance,
            "predictiveVariance": predictive,
            "targetZ": target_z,
            "negativeLogPredictiveDensity": score,
        }.items():
            self.assertAlmostEqual(result[key], expected, places=12, msg=key)

    def test_sampling_variance_dominates_between_game_variance(self) -> None:
        result = elo.anscombe_weighted_predictive_statistics(
            [1.0, 1.001], [10.0, 10.0], [1.0, 1.0], 1.0, 0.1
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["betweenGameVariance"], 0.0)

    def test_ceil_k_contract(self) -> None:
        self.assertEqual(elo.neighbor_count_v4(8), math.ceil(8 ** (2.0 / 3.0)))
        self.assertEqual(elo.neighbor_count_v4(9), math.ceil(9 ** (2.0 / 3.0)))
        self.assertEqual(elo.neighbor_count_v4(8), 4)

    def test_feature_dimensions_are_adjacent_phase_only(self) -> None:
        rows, manifest = v4_rows()
        config = elo.default_v4_config()
        index = elo.GlobalAnscombeKNNIndexV4(rows, manifest, config)
        self.assertEqual(index.pool("black", "full_game", 1).features.shape[1], 2)
        self.assertEqual(index.pool("black", "full_game", 2).features.shape[1], 3)
        self.assertEqual(index.pool("black", "full_game", 3).features.shape[1], 3)
        self.assertEqual(index.pool("black", "full_game", 4).features.shape[1], 3)
        row = next(row for row in rows if row["targetColor"] == "black" and row["scope"] == "full_game")
        stage3_row_index = index.pool("black", "full_game", 3).records.index(row)
        self.assertNotEqual(
            index.pool("black", "full_game", 3).features[stage3_row_index, 2],
            row["referenceZ1"] / manifest["scales"]["black|full_game"]["phase2ZSd"]
            if row["referenceZ1"] is not None else 0.0,
        )

    def test_preparation_is_sequential_and_excludes_own_source_game(self) -> None:
        sources = [source_record(index, color, scope)
                   for color in elo.COLORS for scope in elo.METRICS_SCOPES for index in range(12)]
        config = copy.deepcopy(elo.default_v4_config())
        with tempfile.TemporaryDirectory() as directory:
            manifest = elo.prepare_anscombe_reference_v4(
                sources,
                Path(directory),
                config=config,
                reference_manifest_sha256="source-sha",
            )
            self.assertTrue(manifest["singleFrozenCache"])
            self.assertFalse(manifest["sourceLevel22Rerun"])
            self.assertTrue(all(
                manifest["zDistributions"][f"{color}|{scope}|phase{phase}"]["finiteValueCount"] == 24
                for color in elo.COLORS for scope in elo.METRICS_SCOPES for phase in range(1, 5)
            ))
            self.assertTrue(all(
                row.get("referenceZ1") is not None
                and row.get("referenceZ2") is not None
                and row.get("referenceZ3") is not None
                and row.get("referenceZ4") is not None
                for row in elo.read_jsonl(Path(directory) / config["conditionalReferenceRecords"])
            ))
            self.assertEqual(manifest["referenceModelContractSha256"], elo.canonical_sha256(elo.v4_reference_model_contract(config)))

    def test_global_tree_equals_stable_bruteforce_with_ties_and_exclusions(self) -> None:
        rows, manifest = v4_rows(12)
        config = elo.default_v4_config()
        index = elo.GlobalAnscombeKNNIndexV4(rows, manifest, config)
        pool = index.pool("black", "full_game", 1)
        account = "not-in-reference"
        target_game_ids = [pool.records[0]["gameId"]]
        actual = pool.nearest(account, target_game_ids, 1700.0, 1900.0)
        brute = elo._v4_bruteforce_nearest(
            pool.records, pool.scales, 1, config,
            account=account, target_game_ids=target_game_ids,
            trial_elo=1700.0, opponent_elo=1900.0,
        )
        self.assertEqual(actual["neighborRecordIds"], brute["neighborRecordIds"])
        self.assertEqual(actual["K"], brute["K"])
        self.assertAlmostEqual(actual["boundaryDistance"], brute["boundaryDistance"])
        self.assertEqual(actual["neighborSetSha256"], brute["neighborSetSha256"])
        audit = elo.audit_global_knn_consistency_v4(
            rows, manifest, config, [{
                "targetColor": "black", "scope": "full_game", "stage": 1,
                "account": account, "targetGameIds": target_game_ids,
                "trialElo": 1700.0, "opponentElo": 1900.0,
            }]
        )
        self.assertTrue(audit["passed"])

    def test_formal_v4_hot_path_has_no_optimizer_call(self) -> None:
        source = inspect.getsource(elo.score_anscombe_phase_v4) + inspect.getsource(elo.score_candidate_curve_v4)
        self.assertNotIn("fit_weighted_beta_binomial", source)
        self.assertNotIn("beta_binomial", source.casefold())

    def test_target_selection_is_fixed_before_candidate_search(self) -> None:
        records = [source_record(index, "black", "full_game") for index in range(10)]
        records[0]["metrics"]["full_game"]["phase2"]["validLossNodeCount"] = 0
        selection = elo.select_target_records_v4(records)
        self.assertEqual(selection["selectedCount"], 9)
        self.assertEqual(selection["status"], "insufficient_target_games")
        self.assertIn("phase2_invalid_count", {row["reason"] for row in selection["excluded"]})

    def test_directed_matrix_has_explicit_orientation_and_unordered_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_games_with_partitions.json"
            path.write_text(
                '{"games":['
                '{"gameId":"g1","inMainMatrix":true,"blackOldR":1601,"whiteOldR":1701},'
                '{"gameId":"g2","inMainMatrix":true,"blackOldR":1701,"whiteOldR":1601}'
                ']}', encoding="utf-8"
            )
            result = elo.directed_elo_bucket_matrix_v4(directory)
            self.assertEqual(result["topLeftLabel"], "黑棋\\白棋")
            self.assertEqual(result["matrix"][0][1], 1)
            self.assertEqual(result["matrix"][1][0], 1)
            self.assertEqual(result["directedGameCount"], 2)
            self.assertEqual(result["unordered45Total"], 2)

    def test_adaptive_search_retains_multiple_basins_and_known_probe_isolated(self) -> None:
        config = elo.default_v4_config()
        scales = {
            f"{color}|{scope}": {
                "selfEloSd": 1.0,
                "opponentEloSd": 1.0,
                "phase1ZSd": 1.0,
                "phase2ZSd": 1.0,
                "phase3ZSd": 1.0,
            }
            for color in elo.COLORS for scope in elo.METRICS_SCOPES
        }
        manifest = {
            "schema": elo.SCHEMA_ANScombe_REFERENCE,
            "algorithmVersion": elo.ALGORITHM_VERSION_V4,
            "scales": scales,
        }

        class FakeEvaluator:
            def __init__(self, target_records, account, index, config):
                self.cache = {}
                self.search_point_elos = set()
                self.index = index
                self.scoring_seconds = 0.0
                self.closed_form_statistics_seconds = 0.0
                self.phase_knn_seconds = {f"phase{i}": 0.0 for i in range(1, 5)}
                self.query_count_before = index.query_count
                self.tree_query_count_before = index.tree_query_count
                self.boundary_query_count_before = index.boundary_expansion_count
                self._tree_build_was_reused = False

            def evaluate(self, elo_point, *, include_in_search=False):
                elo_point = int(elo_point)
                if include_in_search:
                    self.search_point_elos.add(elo_point)
                if elo_point not in self.cache:
                    score = min((elo_point - 1783) ** 2, (elo_point - 2267) ** 2)
                    self.cache[elo_point] = {
                        "elo": elo_point,
                        "score": float(score),
                        "minimumJ": float(score),
                        "validGameCount": 1,
                        "targetGameCount": 1,
                        "failureReasons": {},
                        "gameDiagnostics": [],
                        "ok": True,
                    }
                return self.cache[elo_point]

            def evaluate_many(self, elos, *, include_in_search=False):
                for elo_point in elos:
                    self.evaluate(elo_point, include_in_search=include_in_search)

        with patch.object(elo, "_V4ScoreEvaluator", FakeEvaluator):
            result = elo.score_candidate_curve_v4(
                [], [], manifest, target_account="target", config=config, known_elo=2000
            )
        self.assertIn(result["bestGridPoint"], {1783, 2267})
        self.assertGreaterEqual(len(result["discoveredBasins"]), 2)
        self.assertFalse(result["knownEloProbe"]["participatesInMinimumJ"])
        self.assertIn(2000, result["knownEloProbe"]["evaluatedEloPoints"])
        self.assertEqual(result["searchSteps"], [40, 20, 10, 5, 2, 1])

    def test_target_game_process_sharding_preserves_scores_and_neighbor_order(self) -> None:
        reference_rows, manifest = v4_rows(12)
        targets = []
        for index in range(2):
            target = source_record(index + 1, "black", "full_game")
            target["gameId"] = f"target-game-{index}"
            target["targetPlayerId"] = "target-player"
            target["opponentPlayerId"] = f"target-opponent-{index}"
            targets.append(target)

        serial_index = elo.GlobalAnscombeKNNIndexV4(
            reference_rows, manifest, elo.default_v4_config()
        )
        serial = elo._V4ScoreEvaluator(
            targets, "target-player", serial_index, elo.default_v4_config()
        )
        serial.target_game_workers = 1
        serial_result = serial.evaluate(1800)

        parallel_index = elo.GlobalAnscombeKNNIndexV4(
            reference_rows, manifest, elo.default_v4_config()
        )
        parallel = elo._V4ScoreEvaluator(
            targets, "target-player", parallel_index, elo.default_v4_config()
        )
        parallel.target_game_workers = 4
        try:
            parallel_result = parallel.evaluate(1800)
        finally:
            parallel.close()

        self.assertAlmostEqual(serial_result["score"], parallel_result["score"], places=12)
        self.assertEqual(
            parallel_result["parallelization"]["mode"],
            "target_game_process_pool",
        )
        self.assertEqual(parallel_result["parallelization"]["workers"], 4)
        self.assertGreaterEqual(parallel_result["parallelization"]["actualWorkers"], 1)
        self.assertTrue(all(
            worker_pid != os.getpid()
            for worker_pid in parallel_result["parallelization"]["workerProcessIds"]
        ))
        self.assertEqual(
            [row["gameId"] for row in serial_result["gameDiagnostics"]],
            [row["gameId"] for row in parallel_result["gameDiagnostics"]],
        )
        for serial_game, parallel_game in zip(
            serial_result["gameDiagnostics"], parallel_result["gameDiagnostics"], strict=True
        ):
            self.assertAlmostEqual(serial_game["gameScore"], parallel_game["gameScore"], places=12)
            for serial_phase, parallel_phase in zip(
                serial_game["phaseDiagnostics"], parallel_game["phaseDiagnostics"], strict=True
            ):
                self.assertEqual(
                    serial_phase["neighborRecordIds"], parallel_phase["neighborRecordIds"]
                )
                self.assertAlmostEqual(
                    serial_phase["negativeLogPredictiveDensity"],
                    parallel_phase["negativeLogPredictiveDensity"],
                    places=12,
                )


    def test_target_game_worker_limit_rejects_32(self) -> None:
        reference_rows, manifest = v4_rows(12)
        target = source_record(1, "black", "full_game")
        target["gameId"] = "target-game"
        index = elo.GlobalAnscombeKNNIndexV4(
            reference_rows, manifest, elo.default_v4_config()
        )
        evaluator = elo._V4ScoreEvaluator(
            [target], "target-player", index, elo.default_v4_config()
        )
        evaluator.target_game_workers = 32
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            evaluator.evaluate(1800)

    def test_calibration_contract_freezes_two_pass_and_worker_shape(self) -> None:
        config = elo.validate_v4_config(elo.default_v4_config())
        contract = elo.v4_calibration_contract(config)
        self.assertEqual(contract["workers"], 16)
        self.assertEqual(contract["taskUnit"], "one_player_account")
        self.assertEqual(contract["chunksize"], 1)
        self.assertEqual(contract["referenceQueryWorkersPerWorker"], 1)
        self.assertIn("calibration_accounts_only_produce_T95", contract["twoPass"])
        self.assertEqual(elo.v4_search_contract(config)["steps"], [40, 20, 10, 5, 2, 1])
        self.assertEqual(elo.DEFAULT_TARGET_GAME_WORKERS_V4, 4)
        self.assertEqual(elo.MAXIMUM_TARGET_GAME_WORKERS_V4, 4)


if __name__ == "__main__":
    unittest.main()
