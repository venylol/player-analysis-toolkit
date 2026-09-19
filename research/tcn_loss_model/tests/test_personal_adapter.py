import unittest

import torch
from torch.nn import functional as F

from src.personal_adapter import (
    AdapterConfig, adapter_objective, validate_personal_game_sets, wld_adapter_objective,
)


class PersonalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = AdapterConfig(
            optimizer="LBFGS", max_iter=200, line_search_fn="strong_wolfe",
            tolerance_grad=1e-7, tolerance_change=1e-9,
            kl_weight=0.25, delta_w_l2_weight=0.01, delta_b_l2_weight=0.01,
        )

    def test_zero_initialization_is_exact_identity_and_game_equal(self):
        hidden = torch.arange(4 * 64, dtype=torch.float64).reshape(4, 64) / 100
        logits = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                               [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, 3])
        game_index = torch.tensor([0, 0, 0, 1])
        delta_w = torch.zeros((64, 4), dtype=torch.float64)
        delta_b = torch.zeros(4, dtype=torch.float64)
        personal = torch.softmax(logits + hidden @ delta_w + delta_b, dim=-1)
        self.assertTrue(torch.equal(personal, torch.softmax(logits, dim=-1)))
        losses = adapter_objective(hidden, logits, targets, game_index, 2, delta_w, delta_b, self.cfg)
        node_ce = F.cross_entropy(logits, targets, reduction="none")
        expected = (node_ce[:3].mean() + node_ce[3:].mean()) / 2
        self.assertAlmostEqual(float(losses["gameEqualCrossEntropy"]), float(expected), places=12)
        self.assertEqual(float(losses["gameEqualKlBaseToPersonal"]), 0.0)

    def test_zero_target_record_stays_in_fixed_denominator(self):
        hidden = torch.zeros((2, 64), dtype=torch.float64)
        logits = torch.zeros((2, 4), dtype=torch.float64)
        targets = torch.tensor([0, 1])
        game_index = torch.tensor([0, 0])
        losses = adapter_objective(
            hidden, logits, targets, game_index, 2,
            torch.zeros((64, 4), dtype=torch.float64), torch.zeros(4, dtype=torch.float64), self.cfg,
        )
        self.assertAlmostEqual(float(losses["gameEqualCrossEntropy"]), float(torch.log(torch.tensor(4.0, dtype=torch.float64)) / 2), places=12)

    def test_wld_adapter_is_identity_and_game_equal_on_applicable_nodes(self):
        hidden = torch.arange(3 * 64, dtype=torch.float64).reshape(3, 64) / 100
        logits = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        targets = torch.tensor([0, 1, 2])
        game_index = torch.tensor([0, 0, 1])
        delta_w = torch.zeros((64, 3), dtype=torch.float64)
        delta_b = torch.zeros(3, dtype=torch.float64)
        personal = torch.softmax(logits + hidden @ delta_w + delta_b, dim=-1)
        self.assertTrue(torch.equal(personal, torch.softmax(logits, dim=-1)))
        losses = wld_adapter_objective(
            hidden, logits, targets, game_index, 3, delta_w, delta_b, self.cfg
        )
        node_ce = F.cross_entropy(logits, targets, reduction="none")
        expected = (node_ce[:2].mean() + node_ce[2:].mean() + 0.0) / 3
        self.assertAlmostEqual(float(losses["gameEqualCrossEntropy"]), float(expected), places=12)
        self.assertEqual(float(losses["gameEqualKlBaseToPersonal"]), 0.0)

    def test_personal_split_accepts_reusable_group_sizes(self):
        validate_personal_game_sets([f"c{i}" for i in range(24)], [f"r{i}" for i in range(6)])
        with self.assertRaisesRegex(ValueError, "non-empty disjoint"):
            validate_personal_game_sets(["g"], ["g"])


if __name__ == "__main__":
    unittest.main()
