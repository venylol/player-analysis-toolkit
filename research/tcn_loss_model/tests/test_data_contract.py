from __future__ import annotations

import pandas as pd
import unittest
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from src.data_contract import validate_model_ready_npz, validate_raw_data

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "base" / "tcn_board_cnn_time_model_best.pt"


class DataContractTests(unittest.TestCase):
  def test_raw_validator_reindexes_pass_ply(self):
    rows = []
    for index, move in enumerate(["d3", "-", "c4"]):
        rows.append({
            "game_id": "g", "move_index": index, "ply": index + 1,
            "player_id": "p", "side_to_move": "black" if index != 1 else "white",
            "actual_move": move, "actual_thinking_time_ms": 1000,
            "board": "-" * 64, "hint6_1_score": [3, -1, 2][index], "tcb": 300000,
        })
    output = Path(__file__).resolve().parents[1] / "outputs" / "test-runtime-data"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "nodes.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    nodes, report = validate_raw_data(path)
    self.assertEqual(report["passRows"], 1)
    self.assertEqual(report["maxGlobalPlacementPly"], 2)
    self.assertEqual(nodes["ply"].tolist(), [1, 2])

  @unittest.skipUnless(CHECKPOINT.is_file(), "download the tcn-base-checkpoint release asset")
  def test_model_ready_bundle_carries_order_and_preprocessing_identity(self):
    root = ROOT
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    pre_body = json.dumps(checkpoint["preprocessing"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pre_hash = hashlib.sha256(pre_body).hexdigest()
    shape = (3, 2)
    path = root / "outputs" / "test-runtime-data" / "model_ready.npz"
    np.savez(
        path,
        X=np.zeros((*shape, 362), np.float32),
        board_tokens=np.ones((*shape, 3, 64), np.int8),
        board_move_tokens=np.zeros((*shape, 3), np.int8),
        current_hint_tokens=np.zeros((*shape, 6), np.int8),
        current_hint_values=np.zeros((*shape, 4), np.float32),
        prev_own_hint_values=np.zeros((*shape, 2), np.float32),
        actual_thinking_time_ms=np.ones(shape, np.float32),
        disc_loss=np.zeros(shape, np.float32),
        raw_loss=np.zeros(shape, np.float32),
        severity_class=np.zeros(shape, np.int8),
        label_zero=np.ones(shape, np.int8),
        label_ge4=np.zeros(shape, np.int8),
        label_ge10=np.zeros(shape, np.int8),
        move_index=np.tile(np.array([0, 1]), (3, 1)),
        source_ply_including_pass=np.tile(np.array([1, 2]), (3, 1)),
        label_available=np.ones(shape, bool),
        has_consecutive_child=np.ones(shape, bool),
        child_continuity_ok=np.ones(shape, bool),
        same_side_after_move=np.zeros(shape, bool),
        current_score=np.zeros(shape, np.float32),
        actual_move_score=np.zeros(shape, np.float32),
        wld_class=np.zeros(shape, np.int8),
        wld_loss=np.zeros(shape, np.float32),
        wld_label_available=np.zeros(shape, bool),
        mask=np.ones(shape, bool),
        game_id=np.array(["g1", "g2", "g3"]),
        player_id=np.full(shape, "p"),
        global_placement_ply=np.tile(np.array([1, 2]), (3, 1)),
        side_to_move=np.full(shape, "black"),
        split=np.array(["train", "validation", "test"]),
        input_features=np.array(checkpoint["input_features"]),
        board_cnn_channels=np.array(checkpoint["board_encoding"]["cnn_channels"]),
        preprocessing_sha256=np.array(pre_hash),
        input_policy=np.array("uniform-no-current-player-loss-history-v1"),
    )
    report = validate_model_ready_npz(
        path,
        expected_input_features=checkpoint["input_features"],
        expected_board_channels=checkpoint["board_encoding"]["cnn_channels"],
        expected_preprocessing_sha256=pre_hash,
    )
    self.assertTrue(report["ok"])
    self.assertEqual(report["splits"], {"train": 1, "validation": 1, "test": 1})
    self.assertEqual(report["inputPolicy"], "uniform-no-current-player-loss-history-v1")

  @unittest.skipUnless(CHECKPOINT.is_file(), "download the tcn-base-checkpoint release asset")
  def test_model_ready_bundle_rejects_loss_history_feature_even_if_masked(self):
    root = ROOT
    source = root / "outputs" / "test-runtime-data" / "model_ready.npz"
    target = root / "outputs" / "test-runtime-data" / "model_ready_forbidden_feature.npz"
    with np.load(source, allow_pickle=False) as data:
      payload = {name: data[name].copy() for name in data.files}
    payload["input_features"][0] = "own_previous_disc_loss__missing"
    np.savez(target, **payload)
    with self.assertRaisesRegex(ValueError, "must be omitted"):
      validate_model_ready_npz(target)
