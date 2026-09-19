from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from player_analysis_toolkit import sentinel_elo as elo


def node(ply: int, color: str = "black", account: str | None = None, loss: float = 0.0) -> dict:
    return {
        "ply": ply,
        "playerColor": color,
        "playerAccount": account or color,
        "lossPositive": loss,
        "lossClipped": loss,
    }


def metrics(rate: float, *, phases: tuple[float, float, float, float] | None = None) -> dict:
    rates = phases or (rate, rate, rate, rate)
    return {
        "analysisStartPly": 1,
        "validLossNodeCount": 4,
        "phase1": {"validLossNodeCount": 1, "lossGe4Count": 0, "lossGe4Rate": rates[0]},
        "phase2": {"validLossNodeCount": 1, "lossGe4Count": 0, "lossGe4Rate": rates[1]},
        "phase3": {"validLossNodeCount": 1, "lossGe4Count": 0, "lossGe4Rate": rates[2]},
        "phase4": {"validLossNodeCount": 1, "lossGe4Count": 0, "lossGe4Rate": rates[3]},
        "completeFourPhase": True,
        "equalPhaseGameGe4Rate": rate,
        "scopeAvailable": True,
    }


def reference(
    game_id: str,
    rate: float,
    *,
    color: str = "black",
    target_elo: float = 1650,
    opponent_elo: float = 1750,
    target_id: str | None = None,
    opponent_id: str | None = None,
    scope: str = "full_game",
) -> dict:
    full = metrics(rate)
    post = metrics(rate)
    return {
        "schema": elo.SCHEMA_DIRECTED,
        "gameId": game_id,
        "created": "2026-08-01T00:00:00Z",
        "targetPlayerId": target_id or f"target-{game_id}",
        "opponentPlayerId": opponent_id or f"opponent-{game_id}",
        "targetColor": color,
        "targetOldR": target_elo,
        "targetNewR": target_elo + 1,
        "opponentOldR": opponent_elo,
        "opponentNewR": opponent_elo + 1,
        "formalReferenceEligible": True,
        "algorithmLabel": "offbook" if scope == "post_offbook_inclusive" else "no_offbook",
        "metrics": {"full_game": full, "post_offbook_inclusive": post},
    }


class SentinelEloTests(unittest.TestCase):
    def test_v2_beta_binomial_hand_probability_and_mid_cdf(self) -> None:
        probability = elo.beta_binomial_probability(1, 3, 4, 16)
        self.assertAlmostEqual(probability, 0.35324675324675325, places=12)
        result = elo.beta_binomial_mid_cdf_z(1, 3, 4, 16)
        self.assertAlmostEqual(result["PExact"], probability, places=12)
        self.assertAlmostEqual(result["midCdf"], 0.706493506493507, places=12)
        self.assertTrue(math.isfinite(result["z"]))

    def test_v2_same_rate_different_count_has_different_evidence(self) -> None:
        small = elo.beta_binomial_mid_cdf_z(1, 3, 4, 16)
        large = elo.beta_binomial_mid_cdf_z(5, 15, 4, 16)
        self.assertNotAlmostEqual(small["z"], large["z"], places=8)

    def test_v2_weighted_mle_is_deterministic_and_audits_boundaries(self) -> None:
        observations = [(0, 5, 1.0), (1, 5, 2.0), (2, 5, 1.0), (1, 6, 0.5)]
        first = elo.fit_weighted_beta_binomial(observations)
        second = elo.fit_weighted_beta_binomial(observations)
        self.assertTrue(first["ok"])
        self.assertEqual(first["fitStatus"], second["fitStatus"])
        self.assertAlmostEqual(first["m"], second["m"], places=12)
        self.assertIn("kappaUpper", first["optimizerBoundary"])
        failed = elo.fit_weighted_beta_binomial([(2, 1, 1.0)])
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["fitStatus"], "invalid_observation")

    def test_v2_binomial_limit(self) -> None:
        m = 0.2
        actual = elo.beta_binomial_probability(2, 7, m * 1e10, (1 - m) * 1e10)
        expected = math.comb(7, 2) * m**2 * (1 - m)**5
        self.assertAlmostEqual(actual, expected, places=14)

    def test_v2_standardized_distance_uses_only_adjacent_previous_z(self) -> None:
        actual = elo.standardized_elo_distance(
            1800, 1700, 1900, 1700, 100, 200,
            reference_previous_z=2, target_previous_z=1,
            previous_z_sd=2, previous_z_weight=1,
        )
        self.assertAlmostEqual(actual, math.sqrt(1 + 1 + 0.25))

    def test_v2_curve_classification_uses_nll_minimum(self) -> None:
        internal = elo._nll_curve_stats_v2([
            {"elo": 1600, "meanNegativeLogLikelihood": 3.0},
            {"elo": 1601, "meanNegativeLogLikelihood": 1.0},
            {"elo": 1602, "meanNegativeLogLikelihood": 2.0},
        ])
        self.assertEqual(internal["status"], "valid")
        above = elo._nll_curve_stats_v2([
            {"elo": 1600, "meanNegativeLogLikelihood": 3.0},
            {"elo": 1601, "meanNegativeLogLikelihood": 2.0},
            {"elo": 1602, "meanNegativeLogLikelihood": 1.0},
        ])
        self.assertEqual(above["status"], "above_reference_range")
        below = elo._nll_curve_stats_v2([
            {"elo": 1600, "meanNegativeLogLikelihood": 1.0},
            {"elo": 1601, "meanNegativeLogLikelihood": 2.0},
            {"elo": 1602, "meanNegativeLogLikelihood": 3.0},
        ])
        self.assertEqual(below["status"], "below_reference_range")

    def test_v2_account_exclusion_removes_both_directed_sides(self) -> None:
        rows = [
            {"gameId": "g", "targetPlayerId": "Target", "opponentPlayerId": "other"},
            {"gameId": "g", "targetPlayerId": "other", "opponentPlayerId": "TARGET"},
            {"gameId": "safe", "targetPlayerId": "a", "opponentPlayerId": "b"},
        ]
        allowed, excluded = elo._allowed_conditional_records(rows, " target ", [])
        self.assertEqual(excluded, {"g"})
        self.assertEqual([row["gameId"] for row in allowed], ["safe"])

    def test_v2_sensitivity_comparison_audits_neighbor_changes(self) -> None:
        unified = elo.TargetEstimate({
            "status": "calibration_unavailable",
            "estimatedElo": 1800,
            "selectedGameCount": 10,
            "databaseCalibrated95Intervals": [],
            "phaseDiagnostics": [
                {"gameId": "g", "phase": 1, "neighborSetSha256": "a"},
                {"gameId": "g", "phase": 2, "neighborSetSha256": "b"},
            ],
        }, {}, ())
        rebuilt = elo.TargetEstimate({
            "status": "calibration_unavailable",
            "estimatedElo": 1812,
            "selectedGameCount": 10,
            "databaseCalibrated95Intervals": [],
            "phaseDiagnostics": [
                {"gameId": "g", "phase": 1, "neighborSetSha256": "a"},
                {"gameId": "g", "phase": 2, "neighborSetSha256": "c"},
            ],
        }, {}, ())
        result = elo.reference_z_sensitivity_comparison_v2(
            " Account ", unified, rebuilt,
            unified_manifest_sha256="1" * 64,
            rebuilt_manifest_sha256="2" * 64,
            rebuild_audit={"excludedSourceGameCount": 3},
        )
        self.assertEqual(result["account"], "account")
        self.assertEqual(result["estimatedEloDifference"], 12.0)
        self.assertEqual(result["changedNeighborSetCount"], 1)
        self.assertEqual(result["changedNeighborSetFraction"], 0.5)
        self.assertFalse(result["intervalComparisonAvailable"])

    def test_v2_sensitivity_rebuild_filters_source_game_before_prepare(self) -> None:
        rows = [
            {"gameId": "leak", "targetPlayerId": "target", "opponentPlayerId": "x"},
            {"gameId": "leak", "targetPlayerId": "x", "opponentPlayerId": "target"},
            {"gameId": "safe", "targetPlayerId": "a", "opponentPlayerId": "b"},
        ]
        captured: list[dict] = []
        config = elo.default_v2_config()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def fake_prepare(reference_rows, output_dir, **_kwargs):
                captured.extend(reference_rows)
                elo.atomic_write_json(
                    Path(output_dir) / config["conditionalReferenceManifest"],
                    {"schema": elo.SCHEMA_CONDITIONAL_REFERENCE},
                )
                return {"recordsSha256": "3" * 64}

            with patch.object(elo, "prepare_conditional_reference_v2", side_effect=fake_prepare):
                _manifest, audit = elo.rebuild_conditional_reference_for_account_v2(
                    rows, "TARGET", output, config=config,
                )
        self.assertEqual([row["gameId"] for row in captured], ["safe"])
        self.assertEqual(audit["excludedSourceGameCount"], 1)
        self.assertEqual(audit["remainingDirectedRecordCount"], 1)

    def test_v2_rejects_v1_calibration_artifact(self) -> None:
        config = elo.default_v2_config()
        manifest = {
            "schema": elo.SCHEMA_CONDITIONAL_REFERENCE,
            "algorithmVersion": elo.ALGORITHM_VERSION_V2,
            "configSha256": elo.canonical_sha256(config),
            "scales": {},
        }
        with self.assertRaisesRegex(ValueError, "v1 calibration artifacts"):
            elo.estimate_database_calibrated_range_v2(
                "u", [], [], manifest, config=config,
                calibration={"schema": elo.SCHEMA_CALIBRATION},
                conditional_manifest_sha256="x",
            )

    def test_v2_formal_calibration_rejects_non_16_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers=16"):
            elo.calibrate_global_interval_v2(
                [], {"details": []}, [], {}, ROOT / "tmp-never-created",
                config=elo.default_v2_config(),
                conditional_records_path=ROOT / "missing-records",
                conditional_manifest_path=ROOT / "missing-manifest",
                directed_records_sha256="0" * 64,
                parallel_workers=1,
            )

    def test_v2_tail_clipping_and_extreme_parameters_are_finite(self) -> None:
        near_zero = elo.beta_binomial_mid_cdf_z(0, 60, 1e-8, 10.0)
        near_one = elo.beta_binomial_mid_cdf_z(60, 60, 10.0, 1e-8)
        tiny_kappa = elo.beta_binomial_probability(1, 3, 2e-8, 8e-8)
        self.assertTrue(math.isfinite(near_zero["z"]))
        self.assertTrue(math.isfinite(near_one["z"]))
        self.assertGreaterEqual(near_zero["clippedMidCdf"], elo.Z_CDF_CLIP)
        self.assertLessEqual(near_one["clippedMidCdf"], 1 - elo.Z_CDF_CLIP)
        self.assertTrue(math.isfinite(tiny_kappa))

    def test_v2_optimizer_failure_is_not_silently_replaced(self) -> None:
        special, _optimize, tree = elo._scipy_v2()
        failed_optimize = SimpleNamespace(minimize=lambda *args, **kwargs: SimpleNamespace(
            success=False, status=2, message="forced failure", nit=1,
            fun=math.inf, x=[0.0, 0.0],
        ))
        with patch.object(elo, "_scipy_v2", return_value=(special, failed_optimize, tree)):
            result = elo.fit_weighted_beta_binomial([(1, 5, 1.0), (2, 5, 1.0)])
        self.assertFalse(result["ok"])
        self.assertEqual(result["fitStatus"], "optimizer_not_converged")

    def test_v2_stage_pool_uses_only_immediately_previous_z_and_full_pool(self) -> None:
        rows = []
        for index in range(27):
            rows.append({
                "schema": elo.SCHEMA_CONDITIONAL_RECORD,
                "recordId": f"r{index:02d}",
                "gameId": f"g{index:02d}",
                "targetPlayerId": f"p{index}",
                "opponentPlayerId": f"o{index}",
                "targetColor": "black",
                "scope": "full_game",
                "targetOldR": 1600 + index * 10,
                "opponentOldR": 1700 + index * 5,
                "referenceZ1": index / 10,
                "referenceZ2": 10 + index / 5,
                "referenceZ3": -index / 7,
                **{f"phase{phase}": {"x": index % 3, "n": 5} for phase in range(1, 5)},
            })
        scales = {
            "selfEloSd": 100.0, "opponentEloSd": 100.0,
            "phase1ZSd": 2.0, "phase2ZSd": 4.0, "phase3ZSd": 3.0,
        }
        cfg = elo.default_v2_config()
        phase1 = elo._ConditionalPool(rows, scales, 1, cfg)
        phase2 = elo._ConditionalPool(rows, scales, 2, cfg)
        phase3 = elo._ConditionalPool(rows, scales, 3, cfg)
        self.assertEqual(len(phase1.records), 27)
        self.assertEqual(len(phase2.records), 27)
        self.assertEqual(phase1.k, phase2.k)
        self.assertGreater(len(phase2.records), phase1.k)
        self.assertEqual(phase3.features.shape[1], 3)
        self.assertAlmostEqual(phase3.features[4, 2], rows[4]["referenceZ2"] / scales["phase2ZSd"])
        self.assertNotAlmostEqual(phase3.features[4, 2], rows[4]["referenceZ1"] / scales["phase1ZSd"])

        no_previous_cfg = {**cfg, "previousZWeight": 0}
        rows_without_z = [
            {key: value for key, value in row.items() if not key.startswith("referenceZ")}
            for row in rows
        ]
        phase3_unconditional = elo._ConditionalPool(
            rows_without_z, scales, 3, no_previous_cfg
        )
        self.assertEqual(len(phase3_unconditional.records), 27)
        self.assertEqual(phase3_unconditional.features.shape[1], 2)
        self.assertIsNone(phase3_unconditional.previous_key)

    def test_v1_comparison_can_use_slim_conditional_reference_rows(self) -> None:
        rows = []
        for index in range(27):
            rows.append({
                "schema": elo.SCHEMA_CONDITIONAL_RECORD,
                "recordId": f"r{index:02d}",
                "gameId": f"g{index:02d}",
                "targetPlayerId": f"p{index}",
                "opponentPlayerId": f"o{index}",
                "targetColor": "black",
                "scope": "full_game",
                "targetOldR": 1600 + index * 10,
                "opponentOldR": 1700 + index * 5,
                **{f"phase{phase}": {"x": index % 3, "n": 5} for phase in range(1, 5)},
            })
        target = reference("target", 0.4, target_id="new-account")
        curve = elo.score_candidate_curve(
            [target], rows, target_account="new-account", config=elo.default_v2_config()
        )
        self.assertTrue(any(point.get("score") is not None for point in curve["points"]))

    def test_v2_conditional_loader_rejects_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = elo.default_v2_config()
            records = root / config["conditionalReferenceRecords"]
            records.write_text("", encoding="utf-8")
            manifest = {
                "schema": elo.SCHEMA_CONDITIONAL_REFERENCE,
                "algorithmVersion": elo.ALGORITHM_VERSION_V2,
                "configSha256": "wrong",
                "recordsSha256": elo.sha256_file(records),
                "recordCount": 0,
            }
            (root / config["conditionalReferenceManifest"]).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config SHA-256 mismatch"):
                elo.load_conditional_reference_v2(root, config=config)

    def test_fixed_phase_boundaries_and_equal_phase_average(self) -> None:
        nodes = [
            node(1, loss=0), node(30, loss=4),
            node(31, loss=4), node(47, loss=0),
            node(48, loss=4), node(53, loss=0),
            node(54, loss=4), node(60, loss=0),
            node(2, "white", "white", loss=99),
        ]
        result = elo.phase_metrics_for_scope(nodes, "black", "black", "full_game", None)
        self.assertEqual([result[f"phase{i}"]["validLossNodeCount"] for i in range(1, 5)], [2, 2, 2, 2])
        self.assertEqual([result[f"phase{i}"]["lossGe4Count"] for i in range(1, 5)], [1, 1, 1, 1])
        self.assertEqual(result["equalPhaseGameGe4Rate"], 0.5)

    def test_anchor_is_inclusive_and_no_anchor_post_scope_is_unavailable(self) -> None:
        nodes = [node(30), node(31), node(48), node(54)]
        anchored = elo.phase_metrics_for_scope(nodes, "black", "black", "post_offbook_inclusive", 30)
        self.assertEqual(anchored["analysisStartPly"], 30)
        self.assertEqual(anchored["validLossNodeCount"], 4)
        unavailable = elo.phase_metrics_for_scope(nodes, "black", "black", "post_offbook_inclusive", None)
        self.assertFalse(unavailable["scopeAvailable"])
        self.assertEqual(unavailable["unavailableReason"], "no_offbook_anchor")

    def test_empty_phase_makes_whole_game_incomplete(self) -> None:
        nodes = [node(1), node(31), node(54)]
        result = elo.phase_metrics_for_scope(nodes, "black", "black", "full_game", None)
        self.assertFalse(result["completeFourPhase"])
        self.assertIsNone(result["equalPhaseGameGe4Rate"])

    def test_opponent_nodes_do_not_enter_target_metrics(self) -> None:
        nodes = [node(1), node(31), node(48), node(54)]
        nodes.extend([node(2, "white", "white", 4), node(32, "white", "white", 4)])
        result = elo.phase_metrics_for_scope(nodes, "black", "black", "full_game", None)
        self.assertEqual(result["validLossNodeCount"], 4)
        self.assertEqual(result["equalPhaseGameGe4Rate"], 0.0)

    def test_directed_record_contains_new_r_and_both_scopes(self) -> None:
        game = {"gameId": "g", "nodes": [node(1), node(31), node(48), node(54)]}
        detail = {
            "id": "g",
            "created": "2026-08-01T00:00:00Z",
            "players": [
                {"id": "black", "oldR": 1700, "newR": 1710},
                {"id": "white", "oldR": 1800, "newR": 1790},
            ],
        }
        record = elo.make_elo_directed_record(
            game, detail, "black", {"algorithmLabel": "offbook", "offBookPly": 1},
            Path("engine/game.json"), "a" * 64,
            in_main_matrix=True, partition_scope="test",
        )
        self.assertEqual(record["targetNewR"], 1710.0)
        self.assertEqual(set(record["metrics"]), {"full_game", "post_offbook_inclusive"})
        self.assertTrue(record["metrics"]["post_offbook_inclusive"]["completeFourPhase"])

    def test_neighbor_count_distance_weight_and_stable_tie_key(self) -> None:
        rows = [
            reference("b", 0.0, target_elo=1600, opponent_elo=1700),
            reference("a", 0.5, target_elo=1600, opponent_elo=1700),
            reference("c", 1.0, target_elo=1601, opponent_elo=1700),
            reference("d", 0.25, target_elo=1610, opponent_elo=1700),
        ]
        self.assertEqual(elo.neighbor_count(4), 3)
        result = elo.nearest_weighted_neighbors(rows, 1600, 1700)
        self.assertTrue(result["ok"])
        self.assertEqual([row["record"]["gameId"] for row in result["weighted"]], ["a", "b", "c"])
        self.assertEqual(result["boundaryDistance"], 10.0)
        self.assertAlmostEqual(result["weighted"][0]["referenceWeight"], 1.0)

    def test_target_display_elo_is_not_used_in_distance(self) -> None:
        rows = [reference(f"r{i}", i / 10, target_elo=1600 + i * 5) for i in range(8)]
        target_a = reference("target-a", 0.3, target_elo=1600, opponent_elo=1750, target_id="account")
        target_b = {**target_a, "targetOldR": 2495}
        result_a = elo.score_game_at_elo(target_a, rows, 1700)
        result_b = elo.score_game_at_elo(target_b, rows, 1700)
        self.assertTrue(result_a["ok"])
        self.assertEqual(result_a["gameZ"], result_b["gameZ"])

    def test_scope_color_and_complete_filters_and_account_leakage(self) -> None:
        target = reference("target", 0.2, scope="full_game", target_id="Account")
        safe = reference("safe", 0.3, scope="full_game", target_id="other")
        leaked_as_opponent = reference("leaked", 0.4, scope="full_game", opponent_id=" account ")
        wrong_color = reference("white", 0.5, scope="full_game", color="white")
        excluded = elo.excluded_reference_game_ids([target, safe, leaked_as_opponent, wrong_color], "ACCOUNT", ["target"])
        eligible = elo.eligible_reference_records(
            [target, safe, leaked_as_opponent, wrong_color], target,
            excluded_game_ids=excluded,
        )
        self.assertEqual([row["gameId"] for row in eligible], ["safe"])

    def test_score_curve_states_and_intervals(self) -> None:
        valid = elo.classify_curve({"points": [
            {"elo": 1600, "candidateZ": 1.0, "score": 1.0},
            {"elo": 1601, "candidateZ": -1.0, "score": 1.0},
        ]})
        self.assertEqual(valid["status"], "valid")
        above = elo.classify_curve({"points": [
            {"elo": 1600, "candidateZ": -2.0, "score": 2.0},
            {"elo": 1601, "candidateZ": -1.0, "score": 1.0},
        ]})
        self.assertEqual(above["status"], "above_reference_range")
        below = elo.classify_curve({"points": [
            {"elo": 1600, "candidateZ": 2.0, "score": 2.0},
            {"elo": 1601, "candidateZ": 1.0, "score": 1.0},
        ]})
        self.assertEqual(below["status"], "below_reference_range")
        multiple = elo.classify_curve({"points": [
            {"elo": 1600, "candidateZ": 1.0, "score": 1.0},
            {"elo": 1601, "candidateZ": -1.0, "score": 1.0},
            {"elo": 1602, "candidateZ": 1.0, "score": 1.0},
            {"elo": 1603, "candidateZ": -1.0, "score": 1.0},
        ]})
        self.assertEqual(multiple["status"], "multiple_crossings")
        intervals = elo.intervals_for_score_threshold([
            {"elo": 1600, "score": 2}, {"elo": 1601, "score": 0.5},
            {"elo": 1602, "score": 0.5}, {"elo": 1603, "score": 2},
        ], 1)
        self.assertEqual(intervals[0]["lower"], 1601)
        self.assertEqual(intervals[0]["upper"], 1602)

    def test_target_selection_records_fixed_exclusion_reasons(self) -> None:
        incomplete = reference("incomplete", 0.2)
        incomplete["metrics"]["full_game"]["completeFourPhase"] = False
        out_of_range = reference("out", 0.2, opponent_elo=2500)
        selected = elo.select_target_records([incomplete, out_of_range])
        self.assertEqual(selected["status"], "insufficient_target_games")
        self.assertEqual(
            {row["reason"] for row in selected["excluded"]},
            {"incomplete_phase_data", "opponent_out_of_reference_range"},
        )

    def test_latest_known_elo_uses_new_r_from_latest_detail(self) -> None:
        bundle = {
            "details": [
                {"id": "old", "created": "2026-08-01", "players": [{"id": "u", "newR": 1700}, {"id": "x"}]},
                {"id": "new", "created": "2026-08-02", "players": [{"id": "U", "newR": 1812}, {"id": "y"}]},
            ]
        }
        self.assertEqual(elo._latest_known_elos(bundle)["u"], 1812.0)

    def test_calibration_artifact_assembly_keeps_formal_bounds(self) -> None:
        artifact, cases = elo.calibrate_global_interval([], {"details": []})
        self.assertEqual(cases, [])
        self.assertEqual(artifact["formalEloMinimum"], 1600)
        self.assertEqual(artifact["formalEloMaximum"], 2495)
        self.assertEqual(artifact["status"], "calibration_unavailable")


if __name__ == "__main__":
    unittest.main()
