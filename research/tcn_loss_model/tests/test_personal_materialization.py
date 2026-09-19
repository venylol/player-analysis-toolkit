from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "materialize_personal_oq_tcn_model_ready.py"
SPEC = importlib.util.spec_from_file_location("materialize_personal_oq_tcn_model_ready", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PersonalAuditArrayTests(unittest.TestCase):
    def test_target_mask_also_limits_wld_availability(self) -> None:
        arrays = {
            "game_id": np.asarray(["g1"]),
            "mask": np.asarray([[True, True]]),
            "wld_label_available": np.asarray([[True, True]]),
            "player_id": np.asarray([["Target", "Opponent"]]),
            "move_index": np.asarray([[0, 1]], dtype=np.int16),
            "actual_thinking_time_ms": np.asarray([[1000.0, 2000.0]], dtype=np.float32),
        }
        normalized = {
            "details": [{
                "id": "g1",
                "source_time_limit_ms": 300000,
                "effective_time_limit_ms": 300000,
                "time_scale_factor": 1.0,
                "position": {"moves": [
                    {"m": "d3", "raw_thinking_time_ms": 1000},
                    {"m": "c3", "raw_thinking_time_ms": 2000},
                ]},
            }]
        }

        MODULE.add_personal_audit_arrays(arrays, normalized, "target", {"g1"})

        self.assertEqual(arrays["mask"].tolist(), [[True, False]])
        self.assertEqual(arrays["wld_label_available"].tolist(), [[True, False]])

    def test_level18_materializer_maps_directed_records_to_nodes(self) -> None:
        decisions = pd.DataFrame({
            "game_id": ["g1", "g1", "g1", "g1"],
            "move_index": [0, 1, 2, 3],
            "global_placement_ply": [5, 6, 7, 8],
            "side_to_move": ["black", "white", "black", "white"],
            "player_id": ["black-player", "white-player", "black-player", "white-player"],
            "actual_move": ["a", "b", "c", "d"],
            "actual_thinking_time_ms": [6000.0, 1000.0, 1000.0, 1000.0],
            "hint6_1_score": [0.0, 6.0, 0.0, 6.0],
        })
        arrays = {
            "X": np.zeros((1, 5, 1), dtype=np.float32),
            "game_id": np.asarray(["g1"]),
            "move_index": np.asarray([[0, 1, 2, 3, -1]], dtype=np.int16),
            "global_placement_ply": np.asarray([[5, 6, 7, 8, 0]], dtype=np.int16),
            "side_to_move": np.asarray([["black", "white", "black", "white", ""]]),
            "player_id": np.asarray([["black-player", "white-player", "black-player", "white-player", ""]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled.csv"
            checkpoint = root / "base.pt"
            source.write_text("source\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            manifest = MODULE.materialize_level18_offbook(
                decisions, arrays, source, checkpoint, root / "output"
            )
            np.testing.assert_array_equal(arrays["offbook_ply"], [[5, 0, 5, 0, 0]])
            np.testing.assert_array_equal(arrays["offbook_present"], [[True, False, True, False, False]])
            np.testing.assert_allclose(arrays["offbook_feature"], [[5 / 60, 0, 5 / 60, 0, 0]])
            self.assertEqual(manifest["directedSideRecords"], 2)
            self.assertEqual(manifest["nodeMappingRows"], 4)
            self.assertEqual(manifest["validation"]["directedSideRecords"], 2)


if __name__ == "__main__":
    unittest.main()
