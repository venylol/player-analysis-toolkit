from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.progress import read_progress, write_progress


class ProgressTests(unittest.TestCase):
  def test_atomic_utf8_progress(self):
    output = Path(__file__).resolve().parents[1] / "outputs" / "test-runtime-progress"
    write_progress(output, status="waiting-for-data", stage="setup", run_id="测试")
    payload = read_progress(output)
    self.assertEqual(payload["run_id"], "测试")
    raw = (output / "progress.json").read_bytes()
    self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
    self.assertEqual(json.loads(raw.decode("utf-8"))["status"], "waiting-for-data")
