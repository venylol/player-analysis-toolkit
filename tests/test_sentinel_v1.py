from __future__ import annotations

import importlib.util
import json
import subprocess
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

from player_analysis_toolkit import sentinel


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


detect = load_script("sentinel_test_detect", ROOT / "scripts" / "analysis" / "detect_offbook.py")
orchestrator = load_script(
    "sentinel_test_orchestrator", ROOT / "scripts" / "analysis" / "run_player_investigation.py"
)
sentinel_cli = load_script(
    "sentinel_test_cli", ROOT / "scripts" / "analysis" / "sentinel_analysis.py"
)


def engine_node(ply: int, account: str, color: str, loss: float = 0, time: float = 100, score: float = 0) -> dict:
    return {
        "ply": ply, "move": "d3", "sourceMoveIndex": ply - 1,
        "playerAccount": account, "playerColor": color,
        "lossClipped": loss, "thinkingTimeMs": time,
        "bestEval": score, "actualEval": score,
    }


def bilateral_game() -> tuple[dict, dict]:
    nodes = []
    for ply in range(1, 7):
        black = ply % 2 == 1
        nodes.append(engine_node(
            ply, "black-id" if black else "white-id", "black" if black else "white",
            loss=float(ply), time=5501 if ply == 5 else 100,
        ))
    game = {
        "gameId": "双向局", "black": {"account": "black-id"},
        "white": {"account": "white-id"}, "nodes": nodes,
    }
    detail = {
        "id": "双向局", "created": "2026-08-14T00:00:00Z",
        "tcb": 300000,
        "players": [{"id": "black-id", "oldR": 1650}, {"id": "white-id", "oldR": 1750}],
    }
    return game, detail


def reference_record(
    game_id: str, rate: float, *, color: str = "black",
    scope: str = sentinel.SCOPES[0], target_lower: int = 1600,
    opponent_lower: int = 1600, formal: bool = True, wld: float = 0.0,
) -> dict:
    return {
        "gameId": game_id, "targetPlayerId": f"p-{game_id}-{color}",
        "opponentPlayerId": f"o-{game_id}", "targetColor": color,
        "targetOldR": sentinel.BAND_BY_LOWER.get(target_lower, {"center": target_lower + 50})["center"],
        "opponentOldR": sentinel.BAND_BY_LOWER.get(opponent_lower, {"center": opponent_lower + 50})["center"],
        "targetEloBand": sentinel.BAND_BY_LOWER.get(target_lower, {"lower": target_lower}),
        "opponentEloBand": sentinel.BAND_BY_LOWER.get(opponent_lower, {"lower": opponent_lower}),
        "formalReferenceEligible": formal,
        "algorithmLabel": "offbook" if scope == sentinel.SCOPES[0] else "no_offbook",
        "analysisScope": scope, "loss_ge4_rate": rate,
        "engine_wld_loss_total_from_ply39": wld,
    }


def slot(game_id: str, residual: float, *, rate: float = 0.25) -> dict:
    return {
        **reference_record(game_id, rate),
        "created": f"2026-08-{int(game_id[1:]) + 1:02d}T00:00:00Z" if game_id[1:].isdigit() else None,
        "calibratable": True,
        "externalStrengthResidual": residual,
        "externalWldStrengthResidual": residual,
    }


def normal_reference(count: int = 40) -> list[dict]:
    rates = (0.0, 0.25, 0.5, 0.75)
    return [reference_record(f"r{i:03d}", rates[i % len(rates)], wld=(i % 3) * 0.5) for i in range(count)]


def score_payload(residuals: list[float]) -> dict:
    return {
        "scores": [slot(f"g{i}", value) for i, value in enumerate(residuals)],
        "excludedReferenceGameIds": [],
    }


class SentinelV1Tests(unittest.TestCase):
    def test_00_network_acquisition_replaces_fetcher_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"

            def fake_fetch(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
                bundle_path = Path(command[command.index("--output") + 1])
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                bundle_path.write_text(json.dumps({
                    "account": "target",
                    "index": [{"id": "game-1"}],
                    "details": [{
                        "id": "game-1",
                        "created": "2026-08-21T00:00:00Z",
                        "players": [
                            {"id": "target", "oldR": 1700},
                            {"id": "opponent", "oldR": 1700},
                        ],
                        "position": {"moves": [{"m": "d3", "t": 100}]},
                    }],
                    "acquisition": {
                        "listedGameCount": 1,
                        "detailFetchedGameCount": 1,
                        "detailFailureGameIds": [],
                    },
                }), encoding="utf-8")
                (bundle_path.parent / "acquisition_report.json").write_text(
                    json.dumps({"schema": "oq-account-bundle-acquisition-report-v1"}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sentinel_cli.subprocess, "run", side_effect=fake_fetch):
                sentinel_cli.command_acquire(SimpleNamespace(
                    output_dir=output, bundle=None, account="target", mode="5min"
                ))

            report = json.loads(
                (output / "acquisition_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["schema"], "player-sentinel-acquisition-report-v1")

    def test_00_acquisition_excludes_zero_placement_games_before_recent_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = []
            index = []
            for number in range(30):
                game_id = f"valid-{number:02d}"
                details.append({
                    "id": game_id,
                    "created": f"2026-08-{number % 28 + 1:02d}T00:00:{number:02d}Z",
                    "players": [{"id": "target", "oldR": 1700}, {"id": "opponent", "oldR": 1700}],
                    "position": {"moves": [{"m": "d3", "t": 100}]},
                })
                index.append({"id": game_id})
            details.append({
                "id": "terminal-only",
                "created": "2026-09-01T00:00:00Z",
                "players": [{"id": "target", "oldR": 1700}, {"id": "opponent", "oldR": 1700}],
                "position": {"moves": [{"t": 690, "s": "LOSE:DISCONNECT"}]},
            })
            index.append({"id": "terminal-only"})
            source = root / "source.json"
            source.write_text(json.dumps({"details": details, "index": index}), encoding="utf-8")
            output = root / "output"

            sentinel_cli.command_acquire(SimpleNamespace(
                output_dir=output, bundle=source, account="target", mode="5min"
            ))

            selected = json.loads(
                (output / "selected_account_bundle.json").read_text(encoding="utf-8")
            )
            selected_ids = {row["id"] for row in selected["details"]}
            self.assertEqual(len(selected_ids), 30)
            self.assertNotIn("terminal-only", selected_ids)
            self.assertEqual(selected["selection"]["excludedZeroPlacementGameCount"], 1)
            self.assertEqual(
                selected["selection"]["excludedZeroPlacementGameIds"], ["terminal-only"]
            )

    def test_01_same_reference_game_runs_algorithm_for_both_sides(self) -> None:
        game, _ = bilateral_game()
        black = detect.detect_game(game, "black-id", 300000)
        white = detect.detect_game(game, "white-id", 300000)
        self.assertEqual({black["targetColor"], white["targetColor"]}, {"black", "white"})
        self.assertEqual(black["gameId"], white["gameId"])

    def test_02_one_side_can_be_offbook_and_other_no_offbook(self) -> None:
        game, _ = bilateral_game()
        black = detect.detect_game(game, "black-id", 300000)
        white = detect.detect_game(game, "white-id", 300000)
        self.assertEqual(black["algorithmLabel"], "offbook")
        self.assertEqual(white["algorithmLabel"], "no_offbook")

    def test_03_no_offbook_uses_all_target_nodes(self) -> None:
        game, detail = bilateral_game()
        mark = detect.detect_game(game, "white-id", 300000)
        record = sentinel.make_directed_record(
            game, detail, "white", mark, Path("engine/game.json"), "a" * 64,
            in_main_matrix=True, partition_scope="test",
        )
        self.assertEqual(record["analysisScope"], sentinel.SCOPES[1])
        self.assertEqual(record["eligibleTargetNodeCount"], 3)
        self.assertEqual(record["validLossNodeCount"], 3)

    def test_04_offbook_scope_includes_anchor(self) -> None:
        game, detail = bilateral_game()
        mark = detect.detect_game(game, "black-id", 300000)
        record = sentinel.make_directed_record(
            game, detail, "black", mark, Path("engine/game.json"), "a" * 64,
            in_main_matrix=True, partition_scope="test",
        )
        self.assertEqual(record["offBookPly"], 5)
        self.assertEqual(record["analysisStartPly"], 5)
        self.assertEqual(record["eligibleTargetNodeCount"], 1)
        self.assertEqual(record["loss_ge4_count"], 1)

    def test_05_pass_event_does_not_create_loss_node(self) -> None:
        game, detail = bilateral_game()
        game["events"] = [{"sourceMoveIndex": 2, "eventType": "pass", "thinkingTimeMs": 999}]
        mark = detect.detect_game(game, "white-id", 300000)
        record = sentinel.make_directed_record(
            game, detail, "white", mark, Path("engine/game.json"), "b" * 64,
            in_main_matrix=True, partition_scope="test",
        )
        self.assertEqual(record["eligibleTargetNodeCount"], 3)

    def test_06_wld_includes_global_placement_ply_39(self) -> None:
        game = {"nodes": [
            engine_node(38, "p", "black", score=1),
            {**engine_node(39, "p", "black", score=1), "actualEval": -1},
        ]}
        self.assertEqual(sentinel.engine_wld_loss_total([game], 39), 1.0)

    def test_07_target_game_excludes_both_reference_directions(self) -> None:
        target = slot("leak", 0)
        references = [
            reference_record("leak", 0.0, color="black"),
            reference_record("leak", 1.0, color="white"),
            reference_record("safe", 0.5, color="black"),
        ]
        result = sentinel.score_target_records([target], references)
        self.assertEqual(result["excludedReferenceGameIds"], ["leak"])
        self.assertEqual(result["excludedDirectedReferenceRecordCount"], 2)

    def test_08_two_dimensional_elo_and_color_matching(self) -> None:
        references = []
        values = {(1600, 1700): 0.0, (1600, 1800): 0.2, (1700, 1700): 0.6, (1700, 1800): 1.0}
        for (target_lower, opponent_lower), value in values.items():
            references.append(reference_record(
                f"b-{target_lower}-{opponent_lower}", value,
                target_lower=target_lower, opponent_lower=opponent_lower,
            ))
            references.append(reference_record(
                f"w-{target_lower}-{opponent_lower}", 1.0, color="white",
                target_lower=target_lower, opponent_lower=opponent_lower,
            ))
        match = sentinel.match_reference(
            {**slot("target", 0), "targetOldR": 1700, "opponentOldR": 1800},
            sentinel.reference_index(references),
        )
        self.assertAlmostEqual(match["expected"], 0.45)
        self.assertTrue(all(row["targetColor"] == "black" for row, _ in match["records"]))

    def test_09_missing_cell_falls_back_to_nearest_opponent_band_with_flag(self) -> None:
        references = [reference_record("near", 0.3, target_lower=1600, opponent_lower=1800)]
        match = sentinel.match_reference(
            {**slot("target", 0), "targetOldR": 1650, "opponentOldR": 1750},
            sentinel.reference_index(references),
        )
        self.assertTrue(match["fallbackApplied"])
        self.assertEqual(match["cellResolutions"][0]["fallback"], "nearest_opponent_band")
        self.assertEqual(match["usedCells"][0]["opponentBandLower"], 1800)

    def test_10_low_elo_extension_is_not_in_formal_denominator(self) -> None:
        low = reference_record("low", 0.0, target_lower=1500, formal=False)
        formal = reference_record("formal", 0.5)
        index = sentinel.reference_index([low, formal])
        self.assertEqual({row["gameId"] for rows in index.values() for row in rows}, {"formal"})

    def test_10b_single_no_offbook_reference_is_explicitly_not_calibratable(self) -> None:
        target = reference_record("target-no", 0.2, scope=sentinel.SCOPES[1])
        target["created"] = "2026-08-14T00:00:00Z"
        only_reference = reference_record("only-no", 0.3, scope=sentinel.SCOPES[1])
        result = sentinel.score_target_records([target], [only_reference])
        score = result["scores"][0]
        self.assertFalse(score["calibratable"])
        self.assertEqual(score["matchedDistinctReferenceGameCount"], 1)
        self.assertIn("leave_one", score["notCalibratableReason"])

    def test_11_selection_bias_alone_does_not_make_normal_player_anomalous(self) -> None:
        scan, _, _ = sentinel.run_pseudo_scan(
            score_payload([0.0] * 8), normal_reference(), replicates=99, bootstrap=100, seed=11
        )
        self.assertEqual(scan["classification"], "no_clear_signal")
        self.assertGreater(scan["scanCorrectedWilson95Interval"][1], 0.05)

    def test_12_interleaved_aabb_pattern_is_found_by_strength_sort(self) -> None:
        rows = [slot(f"g{i}", value) for i, value in enumerate([0.8, -0.2, 0.8, -0.2, 0.8, -0.2, 0.8, -0.2])]
        effect = sentinel.scan_effects(rows)
        self.assertEqual(effect["perK"][2]["k"], 4)
        self.assertEqual(effect["perK"][2]["gameIds"], ["g0", "g2", "g4", "g6"])

    def test_13_only_sorted_prefixes_are_scanned(self) -> None:
        rows = [slot(f"g{i}", value) for i, value in enumerate([0.4, 0.1, 0.3, 0.2, 0, -0.1, -0.2, -0.3])]
        effect = sentinel.scan_effects(rows)
        self.assertEqual([row["k"] for row in effect["perK"]], [2, 3, 4])
        for row in effect["perK"]:
            self.assertEqual(row["gameIds"], effect["orderedGameIds"][: row["k"]])

    def test_14_all_k_are_statistical_and_model_pipeline_is_called_at_most_once(self) -> None:
        fake_run = SimpleNamespace(config={})
        frozen = {"reportedGameIds": ["a"], "modelReviewReady": True}
        with patch.object(orchestrator, "run_sentinel_pre_scan_stages"), patch.object(
            orchestrator, "run_sentinel_scan_stages", return_value=frozen
        ), patch.object(orchestrator, "validate_paths"), patch.object(
            orchestrator, "run_safe_hints"
        ), patch.object(
            orchestrator, "run_sentinel_unified_analysis"
        ), patch.object(orchestrator, "run_model_and_report_stages") as model:
            orchestrator.run_sentinel(fake_run)
        model.assert_called_once_with(fake_run)

    def test_15_external_reference_failure_creates_no_reported_group(self) -> None:
        scan, _, _ = sentinel.run_pseudo_scan(
            score_payload([0.0] * 8), normal_reference(), replicates=99, bootstrap=50, seed=15
        )
        self.assertEqual(scan["reportedGameIds"], [])
        self.assertFalse(scan["modelReviewReady"])

    def test_16_concentrated_freeze_uses_selected_k_game_ids(self) -> None:
        scan, _, _ = sentinel.run_pseudo_scan(
            score_payload([0.8, 0.8, 0.8, -0.2, -0.2, -0.2, -0.2, -0.2]),
            normal_reference(), replicates=499, bootstrap=500, seed=16,
        )
        self.assertEqual(scan["classification"], "concentrated_external_internal_anomaly")
        self.assertEqual(len(scan["reportedGameIds"]), scan["selectedK"])

    def test_17_isolated_calibration_uses_each_pseudo_players_maximum(self) -> None:
        scan, replicates, _ = sentinel.run_pseudo_scan(
            score_payload([0.8, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]),
            normal_reference(), replicates=199, bootstrap=100, seed=17,
        )
        self.assertEqual(len(replicates), 199)
        self.assertTrue(all("bestSingleScore" in row for row in replicates))
        self.assertLessEqual(scan["bestSingle"]["scanCorrectedNormalExceedanceRate"], 0.05)

    def test_18_uniform_anomaly_has_no_forced_internal_control_group(self) -> None:
        scan, _, _ = sentinel.run_pseudo_scan(
            score_payload([0.8] * 8), normal_reference(), replicates=499, bootstrap=100, seed=18
        )
        self.assertEqual(scan["classification"], "external_uniform_anomaly")
        self.assertEqual(scan["reportedGameIds"], [])
        self.assertEqual(scan["statisticalControlGameIds"], [])
        self.assertEqual(
            set(scan["secondaryWld"]["candidateGameIds"]),
            {f"g{index}" for index in range(8)},
        )

    def test_19_fixed_seed_is_exactly_reproducible(self) -> None:
        args = (score_payload([0.1, 0, -0.1, 0.2, 0, -0.2, 0.1, -0.1]), normal_reference())
        first = sentinel.run_pseudo_scan(*args, replicates=30, bootstrap=30, seed=19)
        second = sentinel.run_pseudo_scan(*args, replicates=30, bootstrap=30, seed=19)
        self.assertEqual(first, second)

    def test_20_sampled_reference_record_uses_leave_one_game(self) -> None:
        original = sentinel.match_reference
        exclusions: list[set[str]] = []

        def recording(*args, **kwargs):
            excluded = set(kwargs.get("excluded_game_ids") or set())
            if excluded:
                exclusions.append(excluded)
            return original(*args, **kwargs)

        with patch.object(sentinel, "match_reference", side_effect=recording):
            sentinel.run_pseudo_scan(
                score_payload([0, 0, 0, 0]), normal_reference(), replicates=5, bootstrap=5, seed=20
            )
        self.assertTrue(exclusions)
        self.assertTrue(all(len(value) == 1 for value in exclusions))

    def test_21_plus_one_and_wilson_boundaries(self) -> None:
        p, successes, trials = sentinel.empirical_upper_p(2.0, [0.0] * 99)
        self.assertEqual((p, successes, trials), (0.01, 1, 100))
        interval = sentinel.wilson_interval(successes, trials)
        self.assertLess(interval[0], p)
        self.assertGreater(interval[1], p)
        self.assertEqual(sentinel.wilson_interval(0, 10)[0], 0.0)

    def test_22_model_result_cannot_enter_or_change_selection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name in ("config.json", "manifest.json", "bundle.json", "audit.json", "offbook.json"):
                path = root / name
                path.write_text("{}\n", encoding="utf-8", newline="\n")
                files.append(path)
            scan = {
                "testedK": [2], "perK": [], "selectedK": 2,
                "reportedGameIds": ["g0", "g1"], "statisticalControlGameIds": ["g2"],
                "modelControlGameIds": ["g2"], "classification": "concentrated_external_internal_anomaly",
                "modelReviewReady": False, "selectionPolicy": "prefixes",
            }
            scores = {"scores": [slot("g0", 1)], "excludedReferenceGameIds": []}
            manifest = sentinel.selection_manifest(
                scan, scores, reference_config_path=files[0], reference_manifest_path=files[1],
                target_bundle_path=files[2], level22_audit_path=files[3], offbook_records_path=files[4],
                seed=1, pseudo_replicates=10, bootstrap_replicates=10,
            )
            frozen_hash = manifest["payloadSha256"]
            _model_result = {"relationshipToPrimarySelection": "conflicting"}
            self.assertEqual(frozen_hash, sentinel.canonical_sha256({k: v for k, v in manifest.items() if k != "payloadSha256"}))
            self.assertNotIn("model", manifest)

    def test_23_utf8_chinese_path_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文目录" / "哨兵.json"
            sentinel.write_json(path, {"玩家": "测试棋手"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["玩家"], "测试棋手")

    def test_24_resume_skips_completed_hash_matching_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"schema": orchestrator.SCHEMA_CONFIG, "paths": {"python": sys.executable}}
            orchestrator.atomic_write_json(root / "run_config.json", config)
            orchestrator.atomic_write_json(root / "progress.json", orchestrator.initial_progress("目标", root, ["one"]))
            run = orchestrator.Run(root)
            output = root / "结果.txt"
            command = [sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('完成', encoding='utf-8')"]
            run.run_stage("one", command, [output])
            with patch.object(orchestrator.subprocess, "run", side_effect=AssertionError("must not rerun")):
                run.run_stage("one", command, [output])
            self.assertEqual(output.read_text(encoding="utf-8"), "完成")


if __name__ == "__main__":
    unittest.main()
