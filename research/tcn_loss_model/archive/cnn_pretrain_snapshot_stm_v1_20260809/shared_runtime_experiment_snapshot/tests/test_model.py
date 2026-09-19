from __future__ import annotations

import torch
import unittest

from src.backbone import ModelConfig
from src.model import ProfileConditionedLossModel, TimeConditionedLossModel, multitask_loss


def inputs(batch=2, steps=3):
    return (
        torch.zeros(batch, steps, 362),
        torch.ones(batch, steps, 3, 64, dtype=torch.long),
        torch.zeros(batch, steps, 3, dtype=torch.long),
        torch.zeros(batch, steps, 6, dtype=torch.long),
        torch.zeros(batch, steps, 4),
        torch.zeros(batch, steps, 2),
    )


class ModelTests(unittest.TestCase):
  def test_actual_current_time_cannot_change_time_head(self):
    torch.manual_seed(1)
    model = TimeConditionedLossModel(ModelConfig())
    model.eval()
    with torch.no_grad():
        one = model(*inputs(), torch.full((2, 3), 1000.0))
        two = model(*inputs(), torch.full((2, 3), 9000.0))
    self.assertTrue(torch.equal(one.pred_time_log_seconds, two.pred_time_log_seconds))
    self.assertTrue(
        not torch.equal(one.probability_loss_ge4, two.probability_loss_ge4)
        or not torch.equal(one.probability_loss_ge10, two.probability_loss_ge10)
    )
    self.assertTrue(bool(torch.all((one.probability_loss_ge4 >= 0) & (one.probability_loss_ge4 <= 1))))
    self.assertTrue(bool(torch.all(one.probability_loss_ge10 <= one.probability_loss_ge4)))
    self.assertTrue(bool(torch.all(one.probability_loss_ge4 <= one.probability_loss_positive)))
    self.assertTrue(torch.allclose(one.severity_class_probabilities.sum(dim=-1), torch.ones(2, 3)))
    self.assertTrue(torch.allclose(one.wld_probabilities.sum(dim=-1), torch.ones(2, 3)))
    expected = 0.5 * one.wld_probabilities[..., 1] + one.wld_probabilities[..., 2]
    self.assertTrue(torch.allclose(one.expected_wld_loss, expected))
    self.assertTrue(bool(torch.all((expected >= 0) & (expected <= 1))))


  def test_multitask_loss_is_finite(self):
    model = TimeConditionedLossModel(ModelConfig())
    actual_time = torch.full((2, 3), 1000.0)
    output = model(*inputs(), actual_time)
    severity = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 2.0]])
    losses = multitask_loss(output, actual_time, severity, torch.ones(2, 3, dtype=torch.bool))
    self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
    self.assertEqual(float(losses["wld_classification"]), 0.0)

  def test_wld_cross_entropy_and_empty_batch_are_finite(self):
    model = TimeConditionedLossModel(ModelConfig())
    actual_time = torch.full((2, 3), 1000.0)
    output = model(*inputs(), actual_time)
    severity = torch.zeros((2, 3))
    wld_class = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    available = torch.tensor([[True, True, True], [False, False, False]])
    ply = torch.tensor([[39, 40, 41], [1, 2, 3]])
    losses = multitask_loss(
        output, actual_time, severity, torch.ones(2, 3, dtype=torch.bool),
        wld_class=wld_class, wld_label_available=available,
        global_placement_ply=ply, wld_weight=1.0,
    )
    self.assertGreater(float(losses["wld_classification"]), 0.0)
    empty = multitask_loss(
        output, actual_time, severity, torch.ones(2, 3, dtype=torch.bool),
        wld_class=wld_class, wld_label_available=torch.zeros_like(available),
        global_placement_ply=ply, wld_weight=1.0,
    )
    self.assertEqual(float(empty["wld_classification"]), 0.0)

  def test_profile_branch_zero_initialization_is_exact_baseline_identity(self):
    torch.manual_seed(17)
    baseline = TimeConditionedLossModel(ModelConfig()).eval()
    torch.manual_seed(17)
    profile_model = ProfileConditionedLossModel(ModelConfig(), "full-31").eval()
    actual_time = torch.full((2, 3), 1500.0)
    profile_values = torch.randn(2, 3, 31)
    profile_missing = torch.rand(2, 3, 31) > 0.7
    with torch.no_grad():
        base_output = baseline(*inputs(), actual_time)
        profile_output = profile_model(*inputs(), actual_time, profile_values, profile_missing)
    self.assertTrue(torch.equal(base_output.pred_time_log_seconds, profile_output.pred_time_log_seconds))
    self.assertTrue(torch.equal(base_output.severity_hidden, profile_output.severity_hidden))
    self.assertTrue(torch.equal(base_output.severity_logits, profile_output.severity_logits))
    self.assertTrue(torch.equal(
        base_output.severity_class_probabilities,
        profile_output.severity_class_probabilities,
    ))

  def test_profile_branch_cannot_change_thinking_time_head(self):
    torch.manual_seed(19)
    model = ProfileConditionedLossModel(ModelConfig(), "overall-both-10-plus-five-differences").eval()
    with torch.no_grad():
        model.profile_severity_film.weight.fill_(0.01)
        first = model(*inputs(), torch.ones(2, 3), torch.zeros(2, 3, 31), torch.zeros(2, 3, 31, dtype=torch.bool))
        second = model(*inputs(), torch.ones(2, 3), torch.ones(2, 3, 31), torch.zeros(2, 3, 31, dtype=torch.bool))
    self.assertTrue(torch.equal(first.pred_time_log_seconds, second.pred_time_log_seconds))
    self.assertEqual(first.severity_logits.shape, (2, 3, 4))
    self.assertFalse(torch.equal(first.severity_logits, second.severity_logits))

  def test_profile_forward_requires_exact_31_feature_contract(self):
    model = ProfileConditionedLossModel(ModelConfig()).eval()
    with self.assertRaisesRegex(ValueError, "expected OQ profile"):
        model(*inputs(), torch.ones(2, 3), torch.zeros(2, 3, 30), torch.zeros(2, 3, 30))

  def test_profile_state_dict_round_trip_is_strict(self):
    source = ProfileConditionedLossModel(ModelConfig(), "with-color")
    target = ProfileConditionedLossModel(ModelConfig(), "with-color")
    target.load_state_dict(source.state_dict(), strict=True)
    self.assertEqual(target.oq_profile_feature_indices.tolist(), list(range(23)))
