from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "tcn_loss_model" / "scripts" / "data" / "fetch_oq_player_profiles.py"
SPEC = importlib.util.spec_from_file_location("fetch_oq_player_profiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProfileFailurePolicyTests(unittest.TestCase):
    def run_main(self, *extra: str) -> int:
        argv = ["fetch", "--account", "missing", "--output-dir", "unused", *extra]
        with mock.patch.object(MODULE, "fetch", return_value={"failed": 1}):
            with contextlib.redirect_stdout(io.StringIO()):
                return MODULE.main(argv)

    def test_failures_remain_fatal_by_default(self) -> None:
        self.assertEqual(self.run_main(), 2)

    def test_explicit_allow_failures_returns_success(self) -> None:
        self.assertEqual(self.run_main("--allow-failures"), 0)


if __name__ == "__main__":
    unittest.main()
