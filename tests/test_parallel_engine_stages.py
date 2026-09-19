from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LEVEL_WRAPPER = load_module(
    "run_egaroucid_bundle", ROOT / "scripts" / "data" / "run_egaroucid_bundle.py"
)
HINT_WRAPPER = load_module(
    "run_safe_hint_stage", ROOT / "scripts" / "data" / "run_safe_hint_stage.py"
)


class Level22AuditTests(unittest.TestCase):
    def test_full_audit_accepts_atomic_parallel_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine.exe"
            engine.write_bytes(b"engine")
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({
                "details": [{
                    "id": "g1",
                    "position": {"moves": [{"m": "d3", "t": 100}, {"t": 0}, {"m": "c3", "t": 200}]},
                }]
            }), encoding="utf-8")
            output = root / "out"
            output.mkdir()
            contract = {
                "path": str(engine.resolve()),
                "sha256": LEVEL_WRAPPER.sha256_file(engine),
                "level": 22,
                "threads": 16,
                "hash": 25,
                "book": "enabled-default",
            }
            game = {
                "schema": "ega-game-analysis-v1",
                "gameId": "g1",
                "moveCount": 2,
                "bundleWorkerId": 0,
                "engine": contract,
                "nodes": [
                    {"ply": 1, "move": "d3", "lossClipped": 0.0},
                    {"ply": 2, "move": "c3", "lossClipped": 1.0},
                ],
                "events": [
                    {"sourceMoveIndex": 0, "thinkingTimeMs": 100},
                    {"sourceMoveIndex": 1, "thinkingTimeMs": 0},
                    {"sourceMoveIndex": 2, "thinkingTimeMs": 200},
                ],
            }
            (output / "game_0_1_g1.json").write_text(json.dumps(game), encoding="utf-8")
            (output / "summary.json").write_text(json.dumps({
                "schema": "ega-account-bundle-summary-v1",
                "gameCount": 1,
                "workerCount": 12,
                "threadsPerConsole": 16,
            }), encoding="utf-8")

            audit = LEVEL_WRAPPER.audit_level22_outputs(
                bundle, output, engine.resolve(), level=22, threads=16,
                hash_level=25, workers=12, book="",
            )

            self.assertTrue(audit["ok"])
            self.assertEqual(audit["nodeCount"], 2)
            self.assertTrue((output / "audit.json").is_file())

    def test_full_audit_accepts_terminal_only_game_with_zero_moves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine.exe"
            engine.write_bytes(b"engine")
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({
                "details": [{
                    "id": "disconnect",
                    "position": {"moves": [{"t": 690, "s": "LOSE:DISCONNECT"}]},
                }]
            }), encoding="utf-8")
            output = root / "out"
            output.mkdir()
            game = {
                "schema": "ega-game-analysis-v1",
                "gameId": "disconnect",
                "moveCount": 0,
                "bundleWorkerId": 0,
                "engine": {
                    "path": str(engine.resolve()),
                    "sha256": LEVEL_WRAPPER.sha256_file(engine),
                    "level": 22,
                    "threads": 16,
                    "hash": 25,
                    "book": "enabled-default",
                },
                "nodes": [],
                "events": [{"sourceMoveIndex": 0, "thinkingTimeMs": 690}],
            }
            (output / "game_0_1_disconnect.json").write_text(
                json.dumps(game), encoding="utf-8"
            )
            (output / "summary.json").write_text(json.dumps({
                "schema": "ega-account-bundle-summary-v1",
                "gameCount": 1,
                "workerCount": 12,
                "threadsPerConsole": 16,
            }), encoding="utf-8")

            audit = LEVEL_WRAPPER.audit_level22_outputs(
                bundle, output, engine.resolve(), level=22, threads=16,
                hash_level=25, workers=12, book="",
            )

            self.assertTrue(audit["ok"])
            self.assertEqual(audit["nodeCount"], 0)
            self.assertEqual(audit["eventCount"], 1)


class SafeHintStageTests(unittest.TestCase):
    def test_server_contract_and_full_audit_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "run_manifest.json").write_text(json.dumps({
                "stage": "hint6",
                "workerCount": 12,
                "threads": 16,
                "hashLevel": 25,
                "level": 18,
                "count": 6,
                "use_book": True,
            }), encoding="utf-8")
            (output / "audit.json").write_text(json.dumps({
                "ok": True,
                "rows": 100,
                "boardMismatches": 0,
                "legalityOrCompletenessErrors": 0,
            }), encoding="utf-8")

            result = HINT_WRAPPER.validate_audit(output, "hint6", 100, 12, 25)

            self.assertEqual(result["workerCount"], 12)
            self.assertEqual(result["threadsPerConsole"], 16)


class TournamentRunnerDefaultsTests(unittest.TestCase):
    def test_direct_runner_defaults_to_two_workers(self) -> None:
        runner_dir = ROOT.parent / "wechat-decrypt"
        python = runner_dir / ".venv" / "Scripts" / "python.exe"
        code = (
            "import json,sys; "
            f"sys.path.insert(0, {str(runner_dir)!r}); "
            "import agent_egaroucid_analysis as runner; "
            "a=runner.build_parser().parse_args(['analyze-bundle','--bundle','bundle.json',"
            "'--cache-dir','cache']); "
            "print(json.dumps([a.workers,a.threads,a.hash]))"
        )
        completed = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True,
            encoding="utf-8", check=True,
        )
        self.assertEqual(json.loads(completed.stdout), [2, 16, 25])


if __name__ == "__main__":
    unittest.main()
