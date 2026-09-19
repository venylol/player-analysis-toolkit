from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_single_change_point.core import (  # noqa: E402
    CandidateRule, DiagonalSegmentModel, candidate_decisions, posterior_summary, state_log_scores,
)


class CandidateTest(unittest.TestCase):
    def test_frozen_candidate_boundaries_and_full_post_segment(self) -> None:
        ply = np.arange(1, 31, dtype=np.int16) * 2
        self.assertEqual(candidate_decisions(ply).tolist(), list(range(3, 20)))
        ply[18] = 40
        self.assertEqual(candidate_decisions(ply).tolist(), list(range(3, 19)))

    def test_short_game_requires_three_post_decisions(self) -> None:
        self.assertEqual(candidate_decisions(np.array([1, 3, 5, 7], dtype=np.int16)).tolist(), [])
        self.assertEqual(candidate_decisions(np.array([1, 3, 5, 7, 9], dtype=np.int16)).tolist(), [3])


class LikelihoodTest(unittest.TestCase):
    def test_state_scores_match_manual_segmentation(self) -> None:
        model = DiagonalSegmentModel(1, "gaussian")
        with torch.no_grad():
            model.location[:] = torch.tensor([[0.0], [3.0]])
            model.raw_scale[:] = torch.log(torch.expm1(torch.tensor(1.0 - model.scale_floor)))
        values = torch.tensor([[[0.0], [0.0], [3.0], [3.0], [3.0]]])
        lengths = torch.tensor([5])
        mask = torch.tensor([[False, False, True, False, False]])
        scores, valid = state_log_scores(model, values, lengths, mask)
        self.assertEqual(valid.tolist(), [[True, False, False, True, False, False]])
        self.assertGreater(float(scores[0, 3].detach()), float(scores[0, 0].detach()))

    def test_no_change_wins_summary_when_largest(self) -> None:
        summary = posterior_summary(np.array([0.0, -2.0]), np.array([3]), np.arange(1, 6))
        self.assertEqual(summary["map_state"], "no_change")
        self.assertIsNone(summary["map_decision"])

    def test_entropy_remains_finite_for_underflowed_candidates(self) -> None:
        summary = posterior_summary(np.array([0.0, -1000.0]), np.array([3]), np.arange(1, 6))
        self.assertTrue(np.isfinite(summary["posterior_entropy"]))


if __name__ == "__main__":
    unittest.main()
