from __future__ import annotations

import copy
import unittest

import numpy as np

from src.offbook import (
    ALGORITHM_LABEL,
    OFFBOOK_ENGINE_CONTRACT,
    OFFBOOK_ENGINE_LEVEL,
    OFFBOOK_LABEL_SOURCE,
    OFFBOOK_NORMALIZATION,
    OFFBOOK_RETROSPECTIVE_DISCLOSURE,
    OFFBOOK_SCHEMA,
    OFFBOOK_TIME_LIMIT_MS,
    recover_level18_hint6_scores,
    make_directed_record,
    validate_offbook_arrays,
)


def record(game_id: str, color: str, nodes: list[dict]) -> dict:
    return make_directed_record(game_id, color, f"{color}-player", nodes)


class Level18AlgorithmTests(unittest.TestCase):
    def test_black_and_white_have_independent_directed_anchors(self) -> None:
        black = record("g1", "black", [
            {"ply": 5, "move": "a", "thinkingTimeMs": 6000, "bestEval": 0},
            {"ply": 6, "move": "b", "thinkingTimeMs": 2500, "bestEval": 0},
            {"ply": 7, "move": "c", "thinkingTimeMs": 2500, "bestEval": 0},
            {"ply": 8, "move": "d", "thinkingTimeMs": 2500, "bestEval": 0},
        ])
        white = record("g1", "white", [
            {"ply": 5, "move": "e", "thinkingTimeMs": 1000, "bestEval": 0},
            {"ply": 7, "move": "f", "thinkingTimeMs": 1000, "bestEval": 0},
        ])
        self.assertEqual(black["judgment"], "offbook")
        self.assertEqual(black["offBookPly"], 5)
        self.assertEqual(white["judgment"], "no_offbook")
        self.assertIsNone(white["offBookPly"])
        self.assertEqual(black["labelSource"], OFFBOOK_LABEL_SOURCE)
        self.assertEqual(black["recordSchema"], OFFBOOK_SCHEMA)

    def test_both_sides_no_offbook_and_pass_ply_is_excluded(self) -> None:
        nodes = [
            {"ply": 4, "move": "pass", "thinkingTimeMs": 9000, "bestEval": 20},
            {"ply": 5, "move": "a", "thinkingTimeMs": 1000, "bestEval": 6},
        ]
        self.assertEqual(record("g2", "black", nodes)["judgment"], "no_offbook")
        self.assertEqual(record("g2", "white", nodes)["judgment"], "no_offbook")

    def test_strict_abs_eval_cutoff_and_post_fast_statuses(self) -> None:
        cutoff = record("g3", "black", [
            {"ply": 5, "move": "a", "thinkingTimeMs": 1000, "bestEval": 6.0},
            {"ply": 6, "move": "b", "thinkingTimeMs": 1000, "bestEval": 6.01},
        ])
        self.assertEqual(cutoff["offBookPly"], 6)
        self.assertEqual(cutoff["postFastCheck"]["status"], "insufficient")

        rejected = record("g4", "black", [
            {"ply": 5, "move": "a", "thinkingTimeMs": 6000, "bestEval": 0},
            {"ply": 6, "move": "b", "thinkingTimeMs": 1000, "bestEval": 0},
            {"ply": 7, "move": "c", "thinkingTimeMs": 1000, "bestEval": 0},
            {"ply": 8, "move": "d", "thinkingTimeMs": 1000, "bestEval": 0},
        ])
        self.assertEqual(rejected["judgment"], "no_offbook")
        self.assertEqual(rejected["algorithmEvidence"]["candidateChecks"][0]["postFastCheck"]["status"], "rejected")

        accepted_insufficient = record("g5", "black", [
            {"ply": 5, "move": "a", "thinkingTimeMs": 6000, "bestEval": 0},
            {"ply": 6, "move": "b", "thinkingTimeMs": 1000, "bestEval": 0},
        ])
        self.assertEqual(accepted_insufficient["judgment"], "offbook")
        self.assertEqual(accepted_insufficient["postFastCheck"]["status"], "insufficient")

    def test_recovery_uses_dynamic_checkpoint_scale_and_integer_contract(self) -> None:
        scores = np.asarray([[-12.0, 0.0, 17.0]], dtype=np.float64)
        encoded = np.tanh(scores / 32.0)
        recovered, audit = recover_level18_hint6_scores(
            encoded, np.ones_like(encoded, dtype=bool), 32.0
        )
        np.testing.assert_array_equal(recovered, np.asarray([[-12, 0, 17]], dtype=np.int16))
        self.assertLess(audit["maxAbsoluteDistanceToNearestInteger"], 1.0e-3)


class Level18ArrayContractTests(unittest.TestCase):
    def base_arrays(self) -> dict[str, np.ndarray]:
        shape = (1, 3)
        arrays: dict[str, np.ndarray] = {
            "X": np.zeros((1, 3, 1), dtype=np.float32),
            "global_placement_ply": np.asarray([[5, 6, 0]], dtype=np.int16),
            "side_to_move": np.asarray([["black", "white", ""]]),
            "player_id": np.asarray([["b", "w", ""]]),
            "game_id": np.asarray(["g1"]),
            "offbook_ply": np.asarray([[5, 0, 0]], dtype=np.int16),
            "offbook_present": np.asarray([[True, False, False]], dtype=bool),
            "offbook_feature": np.asarray([[5 / 60.0, 0.0, 0.0]], dtype=np.float32),
            "offbook_schema": np.asarray(OFFBOOK_SCHEMA),
            "offbook_label_source": np.asarray(OFFBOOK_LABEL_SOURCE),
            "offbook_algorithm_version": np.asarray(ALGORITHM_LABEL),
            "offbook_source_engine_level": np.asarray(OFFBOOK_ENGINE_LEVEL, dtype=np.int16),
            "offbook_engine_contract": np.asarray(OFFBOOK_ENGINE_CONTRACT),
            "offbook_normalization": np.asarray(OFFBOOK_NORMALIZATION),
            "offbook_time_limit_ms": np.asarray(OFFBOOK_TIME_LIMIT_MS, dtype=np.int32),
            "offbook_source_data_sha256": np.asarray("0" * 64),
            "offbook_source_checkpoint_sha256": np.asarray("1" * 64),
            "offbook_records_sha256": np.asarray("2" * 64),
            "offbook_materialization_sha256": np.asarray("3" * 64),
            "offbook_retrospective_disclosure": np.asarray(OFFBOOK_RETROSPECTIVE_DISCLOSURE),
        }
        self.assertEqual(shape, arrays["offbook_ply"].shape)
        return arrays

    def test_side_mapping_repetition_padding_and_present_mask(self) -> None:
        report = validate_offbook_arrays(self.base_arrays())
        self.assertEqual(report["directedSideRecords"], 2)
        self.assertEqual(report["offbookNodes"], 1)
        self.assertEqual(report["noOffbookNodes"], 1)
        self.assertEqual(report["paddingNodes"], 1)

    def test_invalid_present_or_padding_values_fail_loudly(self) -> None:
        invalid = self.base_arrays()
        invalid["offbook_ply"][0, 1] = 7
        with self.assertRaisesRegex(ValueError, "present=false"):
            validate_offbook_arrays(invalid)
        invalid = self.base_arrays()
        invalid["offbook_ply"][0, 2] = 5
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_offbook_arrays(invalid)


if __name__ == "__main__":
    unittest.main()
