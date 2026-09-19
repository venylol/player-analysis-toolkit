import unittest

from train import parser as train_parser
from scripts.modeling.select_seed42_epoch_schedule import choose_schedule
from src.training import TrainingConfig


class Seed42ScheduleTests(unittest.TestCase):
    def test_probe_flag_is_explicit_and_defaults_off(self):
        required = [
            "train", "--data", "data.npz", "--output-dir", "out", "--base-checkpoint", "base.pt",
            "--run-name", "probe", "--confirm-new-data-ready",
        ]
        self.assertFalse(train_parser().parse_args(required).skip_test_evaluation)
        self.assertTrue(train_parser().parse_args([*required, "--skip-test-evaluation"]).skip_test_evaluation)

    def test_validation_minimum_freezes_exact_total_epoch(self):
        config = TrainingConfig(head_epochs=2, fine_tune_epochs=6, seed=42)
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.61, 0.62, 0.63]
        rows = []
        for epoch, loss in enumerate(losses, start=1):
            rows.append({
                "epoch": epoch,
                "stage": "training-heads" if epoch <= 2 else "fine-tuning",
                "validation_total_loss": loss,
                "validation_thinking_time_loss": 0.2,
                "validation_severity_classification_loss": loss - 0.05,
                "zero_log_loss": 0.4,
                "ge4_log_loss": 0.3,
                "ge10_log_loss": 0.1,
            })
        formal, decision = choose_schedule(rows, config, minimum_confirmation_epochs=3)
        self.assertEqual(formal["training"]["head_epochs"], 2)
        self.assertEqual(formal["training"]["fine_tune_epochs"], 3)
        self.assertEqual(decision["fineTuneEpochDecision"]["selectedBestEpoch"], 5)
        self.assertFalse(decision["testEvaluatedDuringProbe"])

    def test_insufficient_post_best_curve_is_rejected(self):
        config = TrainingConfig(head_epochs=2, fine_tune_epochs=3, seed=42)
        rows = []
        for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7, 0.6], start=1):
            rows.append({
                "epoch": epoch,
                "stage": "training-heads" if epoch <= 2 else "fine-tuning",
                "validation_total_loss": loss,
                "validation_thinking_time_loss": 0.2,
                "validation_severity_classification_loss": loss,
                "zero_log_loss": 0.4,
                "ge4_log_loss": 0.3,
                "ge10_log_loss": 0.1,
            })
        with self.assertRaisesRegex(ValueError, "probe is too short"):
            choose_schedule(rows, config, minimum_confirmation_epochs=2)


if __name__ == "__main__":
    unittest.main()
