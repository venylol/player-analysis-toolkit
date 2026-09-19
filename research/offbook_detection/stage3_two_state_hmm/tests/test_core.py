from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_two_state_hmm.core import (  # noqa: E402
    DiagonalHMM,
    forward_log_likelihood_from_emissions,
    infer_batch,
    thirds,
)
from research.offbook_detection.stage3_two_state_hmm.data import selected_positions  # noqa: E402


def brute_force(emission: np.ndarray, initial: np.ndarray, transition: np.ndarray):
    paths = list(itertools.product(range(2), repeat=len(emission)))
    probabilities = []
    for path in paths:
        value = initial[path[0]] * math.exp(emission[0, path[0]])
        for time_index in range(1, len(path)):
            value *= transition[path[time_index - 1], path[time_index]] * math.exp(emission[time_index, path[time_index]])
        probabilities.append(value)
    total = sum(probabilities)
    gamma = np.zeros((len(emission), 2), dtype=np.float64)
    xi = np.full((len(emission), 2, 2), np.nan, dtype=np.float64)
    xi[1:] = 0.0
    for path, probability in zip(paths, probabilities):
        weight = probability / total
        for time_index, state in enumerate(path):
            gamma[time_index, state] += weight
        for time_index in range(1, len(path)):
            xi[time_index, path[time_index - 1], path[time_index]] += weight
    best = np.asarray(paths[int(np.argmax(probabilities))], dtype=np.int64)
    return math.log(total), gamma, xi, best


class DataRuleTest(unittest.TestCase):
    def test_decisions_three_through_nineteen_and_ply_cutoff(self) -> None:
        config = {
            "target_decision_min_inclusive": 3,
            "target_decision_max_inclusive": 19,
            "strict_ply_max_inclusive": 38,
        }
        plies = np.arange(1, 22, dtype=np.int16) * 2
        self.assertEqual(selected_positions(len(plies), plies, config).tolist(), list(range(2, 19)))
        plies[17] = 40
        self.assertEqual(selected_positions(len(plies), plies, config).tolist(), list(range(2, 17)) + [18])

    def test_short_sequences_are_excluded_from_naming_only(self) -> None:
        self.assertIsNone(thirds(2))
        split = thirds(5)
        assert split is not None
        early, late = split
        self.assertEqual(early.tolist(), [0, 1])
        self.assertEqual(late.tolist(), [4])


class ExactInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = np.array([0.6, 0.4], dtype=np.float64)
        self.transition = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        self.emission = np.array([
            [-0.2, -1.1],
            [-0.5, -0.3],
            [-1.4, -0.1],
        ], dtype=np.float64)

    def test_forward_matches_path_enumeration_with_padding(self) -> None:
        padded = torch.tensor(np.stack((
            self.emission,
            np.vstack((self.emission[:2], [99.0, 99.0])),
        )), dtype=torch.float64)
        lengths = torch.tensor([3, 2])
        actual, _ = forward_log_likelihood_from_emissions(
            padded,
            lengths,
            torch.tensor(np.log(self.initial)),
            torch.tensor(np.log(self.transition)),
        )
        expected0 = brute_force(self.emission, self.initial, self.transition)[0]
        expected1 = brute_force(self.emission[:2], self.initial, self.transition)[0]
        np.testing.assert_allclose(actual.numpy(), [expected0, expected1], rtol=0, atol=1e-12)

    def test_posteriors_transition_posteriors_and_viterbi_match_enumeration(self) -> None:
        model = DiagonalHMM(1, "gaussian").double()
        values = torch.tensor([[[0.1], [0.7], [1.4]]], dtype=torch.float64)
        with torch.no_grad():
            model.location[:] = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
            desired_scale = torch.tensor([[0.8], [1.2]], dtype=torch.float64)
            model.raw_scale[:] = torch.log(torch.expm1(desired_scale - model.scale_floor))
            model.initial_logits[:] = torch.tensor(np.log(self.initial))
            model.transition_logits[:] = torch.tensor(np.log(self.transition))
        emission = model.emission_log_prob(values)[0].detach().numpy()
        expected_ll, expected_gamma, expected_xi, expected_path = brute_force(emission, self.initial, self.transition)
        actual = infer_batch(model, values, torch.tensor([3]))
        self.assertAlmostEqual(float(actual.log_likelihood[0]), expected_ll, places=12)
        np.testing.assert_allclose(actual.posterior[0], expected_gamma, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.transition_posterior[0, 1:], expected_xi[1:], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual.viterbi[0], expected_path)
        np.testing.assert_allclose(actual.transition_posterior[0, 1:].sum(axis=(1, 2)), 1.0, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
