from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "sentinel_unified_analysis.py"
SPEC = importlib.util.spec_from_file_location("sentinel_unified_analysis_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UnifiedSentinelAnalysisTests(unittest.TestCase):
    @staticmethod
    def wld_summary() -> dict:
        return {
            "startPly": 39,
            "boundaryInclusive": True,
            "aggregationDenominator": "target_game_player_totals",
            "validGames": 4,
            "gamesWithValidNodes": 3,
            "validNodes": 12,
            "engine_wld_loss_total_from_ply39": 2.5,
            "meanPerValidGame": 0.625,
            "sourceArtifact": {"path": "wld.json", "sha256": "wld-sha"},
        }

    @staticmethod
    def nodes_for_ranges(ranges: list[tuple[int, int, float]]) -> list[dict]:
        return [
            {"gameId": f"game-{ply % 5}", "ply": ply, "discLoss": loss}
            for start, end, loss in ranges
            for ply in range(start, end + 1)
        ]

    def test_help_is_available_and_compatibility_commands_are_documented(self) -> None:
        self.assertEqual(MODULE.main(["--help"]), 0)
        self.assertIn("prepare-conditional-elo-reference", MODULE.ELO_COMMANDS)
        self.assertIn("audit-estimate-coverage", MODULE.ELO_COMMANDS)
        self.assertIn("audit-reference-z-sensitivity", MODULE.ELO_COMMANDS)

    def test_four_phase_fit_is_contiguous_complete_and_has_no_maximum_width(self) -> None:
        nodes = self.nodes_for_ranges([
            (1, 10, 0.0),
            (11, 25, 2.0),
            (26, 40, 8.0),
            (41, 60, 1.0),
        ])
        result = MODULE.analyze_player_phase_nodes(nodes, self.wld_summary())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["boundaries"], [10, 25, 40])
        self.assertEqual(
            [(phase["startPly"], phase["endPly"]) for phase in result["phases"]],
            [(1, 10), (11, 25), (26, 40), (41, 60)],
        )
        self.assertTrue(all(phase["plyWidth"] >= 6 for phase in result["phases"]))
        self.assertGreater(max(phase["plyWidth"] for phase in result["phases"]), 6)
        self.assertIsNone(result["method"]["maximumPhaseWidth"])
        self.assertEqual(result["strongestPhase"], 1)
        self.assertEqual(result["weakestPhase"], 3)

    def test_equal_cost_boundaries_use_lexicographically_smallest_split(self) -> None:
        nodes = self.nodes_for_ranges([(1, 30, 1.0)])
        result = MODULE.analyze_player_phase_nodes(nodes, self.wld_summary())
        self.assertEqual(result["boundaries"], [6, 12, 18])

    def test_probabilities_use_raw_nodes_and_ge10_is_also_ge4(self) -> None:
        losses = [0.0, 0.0, 4.0, 4.0, 10.0, 10.0]
        nodes = [
            {"gameId": f"phase1-{index}", "ply": index + 1, "discLoss": loss}
            for index, loss in enumerate(losses)
        ]
        nodes.extend(self.nodes_for_ranges([(7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]))
        result = MODULE.analyze_player_phase_nodes(nodes, self.wld_summary())
        phase = result["phases"][0]
        self.assertEqual(result["boundaries"], [6, 12, 18])
        self.assertEqual(phase["validNodes"], 6)
        self.assertEqual(phase["lossEq0Count"], 2)
        self.assertEqual(phase["lossGe4Count"], 4)
        self.assertEqual(phase["lossGe10Count"], 2)
        self.assertAlmostEqual(phase["meanDiscLoss"], sum(losses) / len(losses))
        self.assertAlmostEqual(phase["probabilityLossEq0"], 2 / 6)
        self.assertAlmostEqual(phase["probabilityLossGe4"], 4 / 6)
        self.assertAlmostEqual(phase["probabilityLossGe10"], 2 / 6)
        self.assertLessEqual(phase["probabilityLossGe10"], phase["probabilityLossGe4"])

    def test_level22_actual_disc_loss_uses_the_shared_nonnegative_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "game_g1.json"
            source.write_text(json.dumps({
                "gameId": "g1",
                "nodes": [
                    {
                        "ply": 1,
                        "playerAccount": "target",
                        "playerColor": "black",
                        "lossClipped": -2,
                        "lossPositive": -2,
                    },
                    {
                        "ply": 3,
                        "playerAccount": "target",
                        "playerColor": "black",
                        "lossClipped": 4,
                        "lossPositive": 4,
                    },
                ],
            }), encoding="utf-8", newline="\n")
            records = [{
                "gameId": "g1",
                "targetColor": "black",
                "targetPlayerId": "target",
                "sourceLevel22File": str(source),
                "sourceLevel22Sha256": MODULE.elo.sha256_file(source),
            }]
            nodes, _ = MODULE.collect_actual_disc_loss_nodes(records)
            self.assertEqual([node["discLoss"] for node in nodes], [0.0, 4.0])

    def test_wld_is_one_global_record_and_not_split_into_phases(self) -> None:
        result = MODULE.analyze_player_phase_nodes(
            self.nodes_for_ranges([(1, 6, 0.0), (7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]),
            self.wld_summary(),
        )
        self.assertEqual(result["wldFromPly39"]["startPly"], 39)
        self.assertTrue(all(not any("wld" in key.casefold() for key in phase) for phase in result["phases"]))
        csv_rows = MODULE.player_phase_csv_rows(result)
        self.assertEqual(sum(row["recordType"] == "wld_global" for row in csv_rows), 1)
        self.assertEqual(sum(row["recordType"] == "phase" for row in csv_rows), 4)

    def test_wld_summary_uses_only_nodes_at_or_after_ply39(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory)
            (engine / "engine_wld_loss_totals_from_ply39.json").write_text(json.dumps({
                "schema": "player-engine-wld-loss-totals-v1",
                "wldFromPly": 39,
                "gamePlayerTotals": [
                    {
                        "game_id": "g1", "player_id": "target", "side": "black",
                        "engine_wld_loss_total_from_ply39": 1.0,
                    },
                    {
                        "game_id": "g2", "player_id": "target", "side": "white",
                        "engine_wld_loss_total_from_ply39": 0.5,
                    },
                ],
            }), encoding="utf-8")
            records = [
                {"gameId": "g1", "targetColor": "black"},
                {"gameId": "g2", "targetColor": "white"},
            ]
            nodes = [
                {"gameId": "g1", "ply": 38, "discLoss": 10.0},
                {"gameId": "g1", "ply": 39, "discLoss": 0.0},
                {"gameId": "g2", "ply": 60, "discLoss": 4.0},
            ]
            result = MODULE.summarize_wld_from_ply39(engine, "target", records, nodes)
            self.assertEqual(result["startPly"], 39)
            self.assertEqual(result["validGames"], 2)
            self.assertEqual(result["validNodes"], 2)
            self.assertEqual(result["gamesWithValidNodes"], 2)
            self.assertEqual(result["engine_wld_loss_total_from_ply39"], 1.5)

    def test_insufficient_span_returns_explicit_status(self) -> None:
        result = MODULE.analyze_player_phase_nodes(
            self.nodes_for_ranges([(1, 23, 1.0)]), self.wld_summary()
        )
        self.assertEqual(result["status"], "insufficient_phase_data")
        self.assertEqual(result["statusReasons"], ["observed_ply_span_below_24"])
        self.assertEqual(result["phases"], [])

    def test_phase_implementation_has_no_model_or_ensemble_dependency(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8").casefold()
        self.assertNotIn("import torch", source)
        self.assertNotIn("tcn_loss_model", source)
        self.assertNotIn("ensemble_manifest", source)
        result = MODULE.analyze_player_phase_nodes(
            self.nodes_for_ranges([(1, 6, 0.0), (7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]),
            self.wld_summary(),
        )
        self.assertFalse(result["method"]["usesTcn"])
        self.assertFalse(result["method"]["estimatesPhaseElo"])

    def test_run_combines_legacy_summary_and_estimated_elo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            for name, value in {
                "per_game_reference_scores.json": {
                    "targetRecordCount": 3,
                    "calibratableGameCount": 3,
                    "excludedReferenceGameCount": 4,
                },
                "sentinel_scan_results.json": {
                    "classification": "normal",
                    "selectedK": 0,
                    "reportedGameIds": [],
                },
                "pseudo_scan_summary.json": {},
                "selection_manifest.json": {"modelReviewReady": False, "classification": "normal"},
                "model_review_groups.json": {},
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8", newline="\n")
            # The unified report now records the SHA-256 of the Level22
            # statistical offbook records.  Keep the fixture explicit rather
            # than weakening the runtime provenance contract.
            (root / "offbook.json").write_text("{}\n", encoding="utf-8", newline="\n")

            estimate = SimpleNamespace(
                payload={
                    "schema": "player-sentinel-estimated-elo-v1",
                    "status": "insufficient_target_games",
                    "estimatedElo": None,
                    "selectedGameCount": 3,
                    "databaseCalibrated95Range": None,
                    "gameDiagnostics": [],
                    "phaseDiagnostics": [],
                },
                curve={"points": []},
            )
            reference_payload = {
                "config": {},
                "configPath": root / "sentinel_elo_reference_config.json",
                "derived": reference,
                "recordsPath": reference / "directed_game_phase_records.jsonl",
                "manifestPath": reference / "reference_sha256_manifest.json",
                "calibrationPath": reference / "elo_calibration.json",
                "manifestSha256": "manifest-sha",
                "calibrationSha256": "calibration-sha",
                "calibration": {"status": "validated"},
            }
            args = SimpleNamespace(
                account="target",
                bundle=root / "bundle.json",
                engine_dir=root / "engine",
                offbook_records=root / "offbook.json",
                elo_reference_config=root / "elo.json",
                output_dir=root,
            )
            phase_analysis = MODULE.analyze_player_phase_nodes(
                self.nodes_for_ranges([(1, 6, 0.0), (7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]),
                self.wld_summary(),
            )
            with patch.object(MODULE, "load_elo_reference", return_value=reference_payload), patch.object(
                MODULE.elo, "reference_records_from_directory", return_value=[]
            ), patch.object(MODULE.elo, "target_records_from_inputs", return_value=[]), patch.object(
                MODULE, "build_player_phase_analysis", return_value=phase_analysis
            ), patch.object(
                MODULE.elo, "estimate_database_calibrated_range", return_value=estimate
            ):
                self.assertEqual(MODULE.command_run(args), 0)

            unified = json.loads((root / "sentinel_unified_analysis.json").read_text(encoding="utf-8"))
            estimated = json.loads((root / "estimated_elo" / "estimated_elo.json").read_text(encoding="utf-8"))
            self.assertEqual(unified["schema"], MODULE.SCHEMA_UNIFIED)
            self.assertEqual(unified["legacySentinel"]["classification"], "normal")
            self.assertEqual(unified["estimatedElo"]["status"], "insufficient_target_games")
            self.assertIsNone(unified["estimatedElo"]["estimatedElo"])
            self.assertIsNone(unified["estimatedElo"]["databaseCalibrated95Range"])
            self.assertEqual(unified["estimatedElo"]["selectedGameCount"], 3)
            self.assertEqual(unified["playerPhaseAnalysis"]["status"], "ok")
            self.assertTrue((root / "player_phase_analysis.json").is_file())
            self.assertTrue((root / "player_phase_analysis.csv").is_file())
            self.assertEqual(estimated["status"], "insufficient_target_games")

    def test_run_dispatches_v2_to_the_shared_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            for name, value in {
                "per_game_reference_scores.json": {"targetRecordCount": 10},
                "sentinel_scan_results.json": {"classification": "normal", "reportedGameIds": []},
                "pseudo_scan_summary.json": {},
                "selection_manifest.json": {"modelReviewReady": False, "classification": "normal"},
                "model_review_groups.json": {},
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "offbook.json").write_text("{}\n", encoding="utf-8")
            estimate = SimpleNamespace(
                payload={
                    "schema": MODULE.elo.SCHEMA_ESTIMATE_V2,
                    "status": "calibration_unavailable",
                    "estimatedElo": 1800,
                    "selectedGameCount": 10,
                    "gameDiagnostics": [],
                    "phaseDiagnostics": [],
                },
                curve={"points": [{
                    "elo": 1800, "meanNegativeLogLikelihood": 3.0,
                    "score": 3.0, "meanConditionalZ": 0.0,
                    "validTargetGameCount": 10, "failureReasons": [],
                }]},
            )
            reference_payload = {
                "config": {"schema": MODULE.elo.SCHEMA_CONFIG_V2},
                "derived": reference,
                "manifestSha256": "reference-sha",
                "conditionalRecords": [],
                "conditionalManifest": {},
                "conditionalManifestSha256": "conditional-sha",
                "calibrationSha256": "calibration-sha",
                "calibration": {"status": "calibration_unavailable"},
            }
            args = SimpleNamespace(
                account="target", bundle=root / "bundle.json", engine_dir=root / "engine",
                offbook_records=root / "offbook.json", elo_reference_config=root / "elo.json",
                output_dir=root,
            )
            phase_analysis = MODULE.analyze_player_phase_nodes(
                self.nodes_for_ranges([(1, 6, 0.0), (7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]),
                self.wld_summary(),
            )
            with patch.object(MODULE, "load_elo_reference", return_value=reference_payload), patch.object(
                MODULE.elo, "target_records_from_inputs", return_value=[]
            ), patch.object(
                MODULE, "build_player_phase_analysis", return_value=phase_analysis
            ), patch.object(
                MODULE.elo, "estimate_database_calibrated_range_v2", return_value=estimate
            ) as shared_v2:
                self.assertEqual(MODULE.command_run(args), 0)
            shared_v2.assert_called_once()
            unified = json.loads((root / "sentinel_unified_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(unified["estimatedElo"]["schema"], MODULE.elo.SCHEMA_ESTIMATE_V2)
            curve_header = (root / "estimated_elo" / "estimated_elo_curve.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("meanNegativeLogLikelihood", curve_header)

    def test_run_dispatches_v4_and_keeps_reference_contract_without_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            for name, value in {
                "per_game_reference_scores.json": {"targetRecordCount": 10},
                "sentinel_scan_results.json": {"classification": "normal", "reportedGameIds": []},
                "pseudo_scan_summary.json": {},
                "selection_manifest.json": {"modelReviewReady": False, "classification": "normal"},
                "model_review_groups.json": {},
                "bundle.json": {"details": []},
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8", newline="\n")
            (root / "offbook.json").write_text("{}\n", encoding="utf-8")
            config = MODULE.elo.default_v4_config()
            estimate = SimpleNamespace(
                payload={
                    "schema": MODULE.elo.SCHEMA_ESTIMATE_V4,
                    "algorithmVersion": MODULE.elo.ALGORITHM_VERSION_V4,
                    "status": "calibration_unavailable",
                    "estimatedElo": 1800,
                    "selectedGameCount": 10,
                    "gameDiagnostics": [],
                    "phaseDiagnostics": [],
                },
                curve={"points": [{"elo": 1800, "score": 3.0, "meanTargetZ": 0.0}]},
            )
            reference_payload = {
                "config": config,
                "derived": reference,
                "manifestSha256": "source-sha",
                "sourceReferenceManifestSha256": "source-sha",
                "referenceModelManifestSha256": "conditional-sha",
                "referenceModelContractSha256": "model-contract-sha",
                "conditionalRecords": [],
                "conditionalManifest": {},
                "conditionalManifestSha256": "conditional-sha",
                "calibrationSha256": None,
                "calibration": None,
            }
            args = SimpleNamespace(
                account="target", bundle=root / "bundle.json", engine_dir=root / "engine",
                offbook_records=root / "offbook.json", elo_reference_config=root / "elo.json",
                output_dir=root,
            )
            phase_analysis = MODULE.analyze_player_phase_nodes(
                self.nodes_for_ranges([(1, 6, 0.0), (7, 12, 1.0), (13, 18, 2.0), (19, 24, 3.0)]),
                self.wld_summary(),
            )
            with patch.object(MODULE, "load_elo_reference", return_value=reference_payload), patch.object(
                MODULE.elo, "target_records_from_inputs", return_value=[]
            ), patch.object(
                MODULE, "build_player_phase_analysis", return_value=phase_analysis
            ), patch.object(
                MODULE.elo, "estimate_database_calibrated_range_v4", return_value=estimate
            ) as shared_v4:
                self.assertEqual(MODULE.command_run(args), 0)
            shared_v4.assert_called_once()
            unified = json.loads((root / "sentinel_unified_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(unified["schema"], MODULE.SCHEMA_UNIFIED_V4)
            self.assertEqual(unified["estimatedElo"]["algorithmVersion"], MODULE.elo.ALGORITHM_VERSION_V4)
            self.assertEqual(unified["reference"]["referenceModelContractSha256"], "model-contract-sha")
            curve_header = (root / "estimated_elo" / "estimated_elo_curve.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("meanNegativeLogPredictiveDensity", curve_header)


if __name__ == "__main__":
    unittest.main()
