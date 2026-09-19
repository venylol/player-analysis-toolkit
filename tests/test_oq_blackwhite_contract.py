from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFBOOK = ROOT / "research" / "offbook_detection"
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from oq_blackwhite_contract import (  # noqa: E402
    capacity_target,
    dynamic_formal_maximum,
    formal_cells,
    load_expansion_config,
    stable_summary_sha,
    unordered_cells,
)


class OQBlackWhiteContractTests(unittest.TestCase):
    def test_matchup550_config_has_directed_9x9_and_distinct_seeds(self) -> None:
        config, _path, _sha = load_expansion_config(
            ROOT / "oq_reference_blackwhite_expansion_config_v2_matchup550_20260829.json"
        )
        self.assertEqual(len(formal_cells(config)), 81)
        self.assertEqual(len(unordered_cells(config)), 45)
        self.assertEqual(config["topBinLower"], 2400)
        self.assertNotEqual(config["selectionSeed"], config["calibrationSplitSeed"])

    def test_dynamic_upper_bound_never_creates_tenth_bucket(self) -> None:
        config, _path, _sha = load_expansion_config(
            ROOT / "oq_reference_blackwhite_expansion_config_v2_matchup550_20260829.json"
        )
        self.assertEqual(dynamic_formal_maximum(config, 2498, 2497), 2498)
        self.assertEqual(dynamic_formal_maximum(config, 2495, 2500), 2500)
        self.assertEqual(len(formal_cells(config)), 81)

    def test_capacity_formula_and_stable_sha_are_deterministic(self) -> None:
        self.assertEqual(capacity_target(500, 612, 550), 550)
        self.assertEqual(capacity_target(512, 512, 550), 512)
        summary = {"id": "g1", "players": [{"oldR": 1600}, {"oldR": 1700}]}
        first = stable_summary_sha((1600, 1700), "g1", "uniqueSnapshotExpansion", summary, 20260829551)
        second = stable_summary_sha((1600, 1700), "g1", "uniqueSnapshotExpansion", summary, 20260829551)
        other_seed = stable_summary_sha((1600, 1700), "g1", "uniqueSnapshotExpansion", summary, 20260829552)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)

    def test_config_is_utf8_json(self) -> None:
        path = ROOT / "oq_reference_blackwhite_expansion_config_v2_matchup550_20260829.json"
        self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)


if __name__ == "__main__":
    unittest.main()
