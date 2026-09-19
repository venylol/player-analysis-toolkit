from __future__ import annotations

from pathlib import Path
import unittest
import torch

from src.backbone import ModelConfig
from src.checkpoint import (
    load_trained_state_with_offbook_migration,
    load_trained_state_with_wld_migration,
    verify_checkpoint,
)
from src.model import OffbookProfileConditionedLossModel, ProfileConditionedLossModel, TimeConditionedLossModel

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "base" / "tcn_board_cnn_time_model_best.pt"


class CheckpointTests(unittest.TestCase):
  def test_legacy_migration_allows_only_wld_head(self):
    source = TimeConditionedLossModel(ModelConfig())
    legacy = {key: value for key, value in source.state_dict().items() if not key.startswith("wld_head.")}
    target = TimeConditionedLossModel(ModelConfig())
    report = load_trained_state_with_wld_migration(target, legacy)
    self.assertTrue(report["migratedLegacyCheckpoint"])
    self.assertEqual(report["missingKeys"], ["wld_head.bias", "wld_head.weight"])
    bad = dict(legacy)
    bad["unexpected.parameter"] = torch.zeros(1)
    with self.assertRaisesRegex(RuntimeError, "unexpected"):
      load_trained_state_with_wld_migration(target, bad)

  def test_offbook_migration_allows_only_the_two_new_branch_keys(self):
    legacy = ProfileConditionedLossModel(ModelConfig()).state_dict()
    target = OffbookProfileConditionedLossModel(ModelConfig())
    report = load_trained_state_with_offbook_migration(target, legacy)
    self.assertTrue(report["migratedOffbookCheckpoint"])
    self.assertEqual(set(report["missingKeys"]), {"offbook_severity_film.bias", "offbook_severity_film.weight"})
    bad_missing = dict(legacy)
    bad_missing.pop("severity_head.bias")
    with self.assertRaisesRegex(RuntimeError, "missing"):
      load_trained_state_with_offbook_migration(target, bad_missing)
    bad_unexpected = dict(legacy)
    bad_unexpected["unexpected.parameter"] = torch.zeros(1)
    with self.assertRaisesRegex(RuntimeError, "unexpected"):
      load_trained_state_with_offbook_migration(target, bad_unexpected)

  @unittest.skipUnless(CHECKPOINT.is_file(), "download the tcn-base-checkpoint release asset")
  def test_official_checkpoint_strict_load(self):
    report = verify_checkpoint(
        CHECKPOINT,
        ROOT / "provenance" / "source_snapshot" / "preprocessing.json",
    )
    self.assertIs(report["compatible"], True)
    self.assertEqual(report["inputFeatures"], 362)
    self.assertEqual(report["boardCnnChannels"], 23)
    self.assertIs(report["strictBackboneLoad"], True)
    self.assertEqual(report["inputPolicy"], "uniform-no-current-player-loss-history-v1")
    self.assertEqual(report["lossHistoryInputFeatures"], 0)
    self.assertGreater(report["sourceMaxSequenceLength"], 60)
