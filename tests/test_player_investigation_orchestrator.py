from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "run_player_investigation.py"
SPEC = importlib.util.spec_from_file_location("run_player_investigation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def game(game_id: str, created: str) -> dict:
    return {
        "gameId": game_id,
        "created": created,
        "timeLimitMs": 300000,
        "targetColor": "black",
        "opponentAccount": f"opp-{game_id}",
        "blackAccount": "target",
        "whiteAccount": f"opp-{game_id}",
        "sourceMoveCount": 60,
    }


class TimeRangeSelectionTests(unittest.TestCase):
    def make_run(self, catalog: list[dict]) -> tuple[tempfile.TemporaryDirectory, object]:
        temporary = tempfile.TemporaryDirectory()
        run_dir = Path(temporary.name)
        config = {
            "schema": MODULE.SCHEMA_CONFIG,
            "account": "target",
            "reportedGameIds": [],
            "controlGameIds": [],
            "excludedGameIds": [],
            "paths": {},
        }
        MODULE.atomic_write_json(run_dir / "run_config.json", config)
        MODULE.atomic_write_json(run_dir / "progress.json", MODULE.initial_progress("target", run_dir))
        MODULE.atomic_write_json(
            run_dir / "game_catalog.json",
            {"schema": "player-investigation-game-catalog-v1", "games": catalog},
        )
        return temporary, MODULE.Run(run_dir)

    def test_inclusive_boundaries_and_offset_conversion(self) -> None:
        temporary, run = self.make_run([
            game("before", "2026-08-08T12:59:59Z"),
            game("lower", "2026-08-08T13:00:00Z"),
            game("inside", "2026-08-08T13:30:00Z"),
            game("upper", "2026-08-08T14:00:00Z"),
            game("after", "2026-08-08T14:00:01Z"),
        ])
        try:
            reported, control, selection = MODULE.groups_from_time_range(
                run, "2026-08-08T21:00:00+08:00", "2026-08-08T22:00:00+08:00"
            )
            self.assertEqual(reported, ["lower", "inside", "upper"])
            self.assertEqual(control, ["before", "after"])
            self.assertEqual(selection["reportedFrom"], "2026-08-08T13:00:00Z")
            self.assertEqual(selection["reportedTo"], "2026-08-08T14:00:00Z")
        finally:
            temporary.cleanup()

    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include Z"):
            MODULE.parse_aware_datetime("2026-08-08T21:00:00", "bound")

    def test_reversed_range_is_rejected(self) -> None:
        temporary, run = self.make_run([game("one", "2026-08-08T13:00:00Z")])
        try:
            with self.assertRaisesRegex(ValueError, "must not be later"):
                MODULE.groups_from_time_range(
                    run, "2026-08-08T14:00:00Z", "2026-08-08T13:00:00Z"
                )
        finally:
            temporary.cleanup()


class SelectionMaterializationTests(unittest.TestCase):
    def test_selected_bundle_keeps_only_selected_games(self) -> None:
        source = {
            "schema": "source",
            "index": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "details": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        }
        selected = MODULE.selected_bundle(source, {"a", "c"})
        self.assertEqual([row["id"] for row in selected["details"]], ["a", "c"])
        self.assertEqual([row["id"] for row in selected["index"]], ["a", "c"])
        self.assertEqual(selected["selection"]["gameIds"], ["a", "c"])

    def test_atomic_json_is_utf8_and_preserves_non_ascii(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "状态.json"
            MODULE.atomic_write_json(path, {"选手": "测试"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"选手": "测试"})


class DefaultPathValidationTests(unittest.TestCase):
    def test_server_style_parallel_defaults(self) -> None:
        args = MODULE.build_parser().parse_args([
            "start", "--account", "target", "--output-dir", "run"
        ])
        self.assertEqual(args.level22_workers, 12)
        self.assertEqual(args.level22_threads, 16)
        self.assertEqual(args.level22_hash, 25)
        self.assertEqual(args.hint6_workers, 12)

    def test_offbook_detection_runs_immediately_after_level22(self) -> None:
        level_index = MODULE.STAGE_ORDER.index("level22")
        self.assertEqual(MODULE.STAGE_ORDER[level_index + 1], "offbook_detection")
        self.assertNotIn("offbook_review", MODULE.STAGE_ORDER)

    def test_default_human_opening_book_path_contract(self) -> None:
        path = Path(MODULE.default_paths()["humanOpeningBook"])
        self.assertEqual(path.name, "othelloquest_human_frequency_nodes_ply1_30_min5.runtime.json")
        self.assertIn("source_snapshot", path.parts)

    def test_human_opening_book_rejects_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human_opening_frequency_lookup.csv"
            path.write_text("ply,frequency\n1,10\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "UTF-8 JSON runtime book"):
                MODULE.validate_human_opening_book(path)

    def test_full_sentinel_flow_registers_phase_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class FakeRun:
                run_dir = root
                config = {"account": "target", "eloReferenceConfig": str(root / "elo.json")}

                @staticmethod
                def python() -> str:
                    return "python"

                @staticmethod
                def path(value: str) -> Path:
                    return root / value

                def run_stage(self, name: str, command: list[str], outputs: list[Path]) -> None:
                    self.name = name
                    self.command = command
                    self.outputs = outputs

            run = FakeRun()
            MODULE.run_sentinel_unified_analysis(run)
            self.assertEqual(run.name, "sentinel_unified_analysis")
            self.assertEqual(run.command[2], "run")
            self.assertIn(root / "player_phase_analysis.json", run.outputs)
            self.assertIn(root / "player_phase_analysis.csv", run.outputs)

    def test_sentinel_report_references_phase_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "estimated_elo").mkdir()
            for name, value in {
                "selection_manifest.json": {
                    "classification": "no_formal_reported_games",
                    "modelReviewReady": False,
                },
                "acquisition_report.json": {},
                "per_game_reference_scores.json": {},
                "sentinel_scan_results.json": {},
                "pseudo_scan_summary.json": {},
                "sentinel_unified_analysis.json": {"playerPhaseAnalysis": {"status": "ok"}},
                "player_phase_analysis.json": {"status": "ok"},
            }.items():
                MODULE.atomic_write_json(root / name, value)
            MODULE.atomic_write_json(root / "estimated_elo" / "estimated_elo.json", {})
            (root / "player_phase_analysis.csv").write_text(
                "recordType,status\nphase,ok\n", encoding="utf-8", newline="\n"
            )

            class FakeRun:
                config = {"account": "target"}

                @staticmethod
                def path(value: str) -> Path:
                    return root / value

                def set_stage(self, *args: object, **kwargs: object) -> None:
                    self.stage = (args, kwargs)

                def set_overall(self, *args: object) -> None:
                    self.overall = args

            run = FakeRun()
            MODULE.build_sentinel_report_without_model(run)
            report = MODULE.read_json(root / "report.json")
            self.assertEqual(report["playerPhaseAnalysis"]["status"], "ok")
            self.assertEqual(
                report["playerPhaseAnalysisArtifacts"]["json"]["sha256"],
                MODULE.sha256_file(root / "player_phase_analysis.json"),
            )
            self.assertEqual(
                report["playerPhaseAnalysisArtifacts"]["csv"]["sha256"],
                MODULE.sha256_file(root / "player_phase_analysis.csv"),
            )


if __name__ == "__main__":
    unittest.main()
