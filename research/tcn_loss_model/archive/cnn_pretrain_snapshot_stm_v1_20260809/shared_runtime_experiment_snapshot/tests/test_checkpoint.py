from __future__ import annotations

from pathlib import Path
import unittest
import torch

from src.backbone import ModelConfig
from src.checkpoint import (
    load_trained_state_excluding_board_encoder,
    load_trained_state_with_wld_migration,
    verify_checkpoint,
)
from src.model import TimeConditionedLossModel

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "base" / "tcn_board_cnn_time_model_best.pt"


class CheckpointTests(unittest.TestCase):
  def test_strict_non_cnn_load_excludes_only_replaced_board_encoder(self):
    legacy = TimeConditionedLossModel(ModelConfig())
    residual_config = ModelConfig(
        board_channels=64, residual_blocks=6, board_embedding_dim=96,
        embedding_projection_kernel=1, board_cnn_architecture="residual-v2",
    )
    target = TimeConditionedLossModel(residual_config)
    before_cnn = {
        key: value.clone() for key, value in target.state_dict().items()
        if key.startswith("backbone.board_encoder.")
    }
    report = load_trained_state_excluding_board_encoder(target, legacy.state_dict())
    self.assertTrue(report["strictNonCnnLoad"])
    self.assertTrue(report["excludedBoardEncoderKeys"])
    for key, value in target.state_dict().items():
      if key.startswith("backbone.board_encoder."):
        self.assertTrue(torch.equal(value, before_cnn[key]))
      else:
        self.assertTrue(torch.equal(value, legacy.state_dict()[key]))

    bad = dict(legacy.state_dict())
    bad.pop("severity_head.bias")
    with self.assertRaisesRegex(RuntimeError, "non-CNN.*missing"):
      load_trained_state_excluding_board_encoder(target, bad)

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
