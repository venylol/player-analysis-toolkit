import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.backbone import ModelConfig
from src.board_cnn import BoardCNNAuxiliaryHeads, legal_move_targets_from_board_tokens
from src.board_cnn_pretrain import legal_moves_bitboard
from src.checkpoint import (
    load_checkpoint_payload, load_trained_state_excluding_board_encoder,
    load_transferred_profile_model, sha256_file,
)
from src.model import ProfileConditionedLossModel, multitask_loss
from src.training import (
    TrainingConfig,
    _strict_load_auxiliary_heads,
    _strict_load_stage_a_checkpoint,
    _strict_load_stage_b_checkpoint,
    configure_cnn_stage_b,
    configure_cnn_stage_c,
    configure_cnn_direct_full,
    stage_b_auxiliary_loss,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_A = ROOT / "outputs/cnn_tcn_stage_a_snapshot_stm_v1_seed42_20260809/best.pt"
TRANSFERRED = ROOT / "checkpoints/transferred/tcn_board_cnn_residual64x6_pretrain_epoch6_seed42_v2.pt"
PRETRAIN = ROOT / "outputs/board_cnn_pretrain_64x6_seed42_20260809T143514/checkpoints/epoch-0005-step-00140628.pt"
STAGE_B = ROOT / "outputs/cnn_tcn_stage_b_snapshot_stm_v1_seed42_20260809/best.pt"
PROFILE_WARM = ROOT / "outputs/oq_profile_full31_wld_ply39_ensemble12_11200_latest10_wldhead6_total16_baselineguard_20260809/members/member_01_seed_42/best.pt"


def stage_b_config() -> TrainingConfig:
    return TrainingConfig.load(ROOT / "config/cnn_tcn_stage_b_seed42.json")


def stage_c_config() -> TrainingConfig:
    return TrainingConfig.load(ROOT / "config/cnn_tcn_stage_c_seed42.json")


class CnnStageBTests(unittest.TestCase):
    def test_legal_targets_are_derived_from_current_state_only(self):
        opening = "---------------------------OX------XO---------------------------"
        tokens = torch.tensor([[1 if cell == "-" else 2 if cell == "X" else 3 for cell in opening]])
        target = legal_move_targets_from_board_tokens(tokens)[0]
        expected = legal_moves_bitboard(opening)
        expected_indices = [index for index in range(64) if expected & (1 << index)]
        self.assertEqual(target.nonzero().flatten().tolist(), expected_indices)

    def test_freeze_scope_optimizer_ratio_and_embedding_shape(self):
        cfg = stage_b_config()
        model = ProfileConditionedLossModel(ModelConfig())
        auxiliary = BoardCNNAuxiliaryHeads()
        optimizer, report = configure_cnn_stage_b(model, auxiliary, cfg)
        shared = model.backbone.board_encoder.shared
        self.assertFalse(any(parameter.requires_grad for parameter in shared.stem.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in shared.stem_norm.parameters()))
        self.assertEqual(report["residualBlockRequiresGrad"], [False, False, False, False, True, True])
        self.assertTrue(all(parameter.requires_grad for parameter in shared.projection.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in shared.embedding_norm.parameters()))
        self.assertEqual(optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] * 0.1)
        embedding = model.backbone.board_encoder(
            torch.ones(1, 1, 3, 64, dtype=torch.long),
            torch.zeros(1, 1, 3, dtype=torch.long),
            torch.zeros(1, 1, 6, dtype=torch.long),
            torch.zeros(1, 1, 4), torch.zeros(1, 1, 2),
        )
        self.assertEqual(tuple(embedding.shape), (1, 1, 96))

    @unittest.skipUnless(STAGE_A.is_file() and TRANSFERRED.is_file() and PRETRAIN.is_file(), "handoff artifacts absent")
    def test_real_stage_a_and_auxiliary_checkpoints_load_strictly(self):
        cfg = stage_b_config()
        model, _ = load_transferred_profile_model(TRANSFERRED, "full-31")
        source = torch.load(STAGE_A, map_location="cpu", weights_only=False)
        identity = {key: source["manifest"][key] for key in (
            "modelSchema", "modelVariant", "boardPerspective", "inputPolicy", "dataSha256",
            "baseCheckpointSha256", "featureOrderSha256", "boardChannelOrderSha256",
            "preprocessingSha256", "oqProfileAblation", "oqProfileAblationSha256",
            "oqProfileFeatureOrderSha256", "oqProfilePreprocessingSha256", "oqProfilePolicy",
            "oqProfileTemporalLeakageAuthorized", "testEvaluationPlanned",
        )}
        loaded, digest = _strict_load_stage_a_checkpoint(
            model, STAGE_A, cfg.initial_stage_checkpoint_sha256, identity
        )
        self.assertEqual(digest, sha256_file(STAGE_A))
        self.assertEqual(set(model.state_dict()), set(loaded["modelStateDict"]))
        auxiliary = BoardCNNAuxiliaryHeads()
        _strict_load_auxiliary_heads(
            auxiliary, PRETRAIN, cfg.auxiliary_checkpoint_sha256,
            load_checkpoint_payload(TRANSFERRED),
        )
        pretrain = torch.load(PRETRAIN, map_location="cpu", weights_only=False)
        self.assertTrue(torch.equal(auxiliary.legal_head.weight, pretrain["legal_move_head_state_dict"]["weight"]))

    @unittest.skipUnless(STAGE_A.is_file() and TRANSFERRED.is_file(), "handoff artifacts absent")
    def test_legacy_stage_or_perspective_is_rejected(self):
        cfg = stage_b_config()
        model, _ = load_transferred_profile_model(TRANSFERRED, "full-31")
        source = torch.load(STAGE_A, map_location="cpu", weights_only=False)
        source["board_perspective"] = "legacy_fixed_color"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(source, path)
            with self.assertRaisesRegex(ValueError, "legacy or missing"):
                _strict_load_stage_a_checkpoint(model, path, sha256_file(path), {})

    def test_one_step_updates_only_authorized_parameters_and_restores_contract(self):
        torch.manual_seed(7)
        cfg = stage_b_config()
        model = ProfileConditionedLossModel(ModelConfig(dropout=0.0))
        auxiliary = BoardCNNAuxiliaryHeads()
        optimizer, report = configure_cnn_stage_b(model, auxiliary, cfg)
        model.train()
        opening = "---------------------------OX------XO---------------------------"
        current = torch.tensor([1 if cell == "-" else 2 if cell == "X" else 3 for cell in opening])
        board_tokens = current.reshape(1, 1, 1, 64).expand(1, 1, 3, 64).clone()
        batch = {
            "X": torch.randn(1, 1, 362), "board_tokens": board_tokens,
            "board_move_tokens": torch.zeros(1, 1, 3, dtype=torch.long),
            "current_hint_tokens": torch.zeros(1, 1, 6, dtype=torch.long),
            "current_hint_values": torch.zeros(1, 1, 4),
            "prev_own_hint_values": torch.zeros(1, 1, 2),
            "actual_thinking_time_ms": torch.full((1, 1), 1000.0),
            "severity_class": torch.zeros(1, 1), "mask": torch.ones(1, 1, dtype=torch.bool),
            "wld_class": torch.zeros(1, 1), "wld_label_available": torch.ones(1, 1, dtype=torch.bool),
            "global_placement_ply": torch.full((1, 1), 39),
            "oq_profile_features": torch.zeros(1, 1, 31),
            "oq_profile_missing": torch.zeros(1, 1, 31, dtype=torch.bool),
            "current_score": torch.zeros(1, 1),
            "legal_move_target": legal_move_targets_from_board_tokens(current.reshape(1, 1, 64)),
        }
        frozen_before = copy.deepcopy(model.backbone.board_encoder.shared.blocks[0].state_dict())
        trainable_before = copy.deepcopy(model.backbone.board_encoder.shared.blocks[5].state_dict())
        non_cnn_before = model.severity_head.weight.detach().clone()
        output = model(
            batch["X"], batch["board_tokens"], batch["board_move_tokens"],
            batch["current_hint_tokens"], batch["current_hint_values"], batch["prev_own_hint_values"],
            batch["actual_thinking_time_ms"], batch["oq_profile_features"], batch["oq_profile_missing"],
        )
        formal = multitask_loss(
            output, batch["actual_thinking_time_ms"], batch["severity_class"], batch["mask"],
            wld_class=batch["wld_class"], wld_label_available=batch["wld_label_available"],
            global_placement_ply=batch["global_placement_ply"],
        )["total"]
        auxiliary_losses = stage_b_auxiliary_loss(model, auxiliary, batch, cfg)
        (formal + auxiliary_losses["weighted_total"]).backward()
        self.assertIsNotNone(model.backbone.board_encoder.shared.blocks[5].conv1.weight.grad)
        self.assertEqual(int(auxiliary_losses["state_only_nonzero_forbidden_channels"]), 0)
        optimizer.step()
        self.assertTrue(all(torch.equal(value, frozen_before[key]) for key, value in model.backbone.board_encoder.shared.blocks[0].state_dict().items()))
        self.assertTrue(any(not torch.equal(value, trainable_before[key]) for key, value in model.backbone.board_encoder.shared.blocks[5].state_dict().items()))
        self.assertFalse(torch.equal(model.severity_head.weight, non_cnn_before))

        payload = {
            "schema": "tcn-loss-profile-wld-stage-b-checkpoint-v1", "stage": "cnn-stage-b",
            "board_perspective": "snapshot_side_to_move_v1", "modelStateDict": model.state_dict(),
            "auxiliaryHeadsStateDict": auxiliary.state_dict(), "optimizerStateDict": optimizer.state_dict(),
            "optimizerParameterGroups": report["parameterGroups"],
        }
        restored_model = ProfileConditionedLossModel(ModelConfig(dropout=0.0))
        restored_auxiliary = BoardCNNAuxiliaryHeads()
        restored_optimizer, restored_report = configure_cnn_stage_b(restored_model, restored_auxiliary, cfg)
        restored_model.load_state_dict(payload["modelStateDict"], strict=True)
        restored_auxiliary.load_state_dict(payload["auxiliaryHeadsStateDict"], strict=True)
        self.assertEqual(payload["optimizerParameterGroups"], restored_report["parameterGroups"])
        restored_optimizer.load_state_dict(payload["optimizerStateDict"])
        self.assertEqual(payload["stage"], "cnn-stage-b")
        self.assertEqual(payload["board_perspective"], "snapshot_side_to_move_v1")


class CnnStageCTests(unittest.TestCase):
    def test_patience_six_stage_c_config(self):
        cfg = TrainingConfig.load(ROOT / "config/cnn_tcn_stage_c_seed42_patience6.json")
        self.assertEqual(cfg.early_stopping_patience, 6)
        model = ProfileConditionedLossModel(ModelConfig())
        auxiliary = BoardCNNAuxiliaryHeads()
        _optimizer, report = configure_cnn_stage_c(model, auxiliary, cfg)
        self.assertEqual(report["earlyStoppingPatience"], 6)

    def test_full_cnn_unfreeze_and_reduced_non_cnn_learning_rate(self):
        cfg = stage_c_config()
        model = ProfileConditionedLossModel(ModelConfig())
        auxiliary = BoardCNNAuxiliaryHeads()
        optimizer, report = configure_cnn_stage_c(model, auxiliary, cfg)
        self.assertTrue(report["allCnnTrainable"])
        self.assertEqual(report["frozenCnnParameterCount"], 0)
        self.assertEqual(report["residualBlockRequiresGrad"], [True] * 6)
        self.assertTrue(report["stemRequiresGrad"])
        self.assertTrue(report["stemNormRequiresGrad"])
        self.assertTrue(report["projectionRequiresGrad"])
        self.assertTrue(report["embeddingNormRequiresGrad"])
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[2]["lr"], 5e-5)
        self.assertEqual(report["earlyStoppingPatience"], 2)

    @unittest.skipUnless(STAGE_B.is_file() and TRANSFERRED.is_file(), "Stage B handoff artifacts absent")
    def test_real_stage_b_best_strictly_restores_full_model_and_auxiliary_heads(self):
        cfg = stage_c_config()
        model, _ = load_transferred_profile_model(TRANSFERRED, "full-31")
        auxiliary = BoardCNNAuxiliaryHeads()
        source = torch.load(STAGE_B, map_location="cpu", weights_only=False)
        identity = {key: source["manifest"][key] for key in (
            "modelSchema", "modelVariant", "boardPerspective", "inputPolicy", "dataSha256",
            "baseCheckpointSha256", "featureOrderSha256", "boardChannelOrderSha256",
            "preprocessingSha256", "oqProfileAblation", "oqProfileAblationSha256",
            "oqProfileFeatureOrderSha256", "oqProfilePreprocessingSha256", "oqProfilePolicy",
            "oqProfileTemporalLeakageAuthorized", "testEvaluationPlanned",
        )}
        loaded, digest = _strict_load_stage_b_checkpoint(
            model, auxiliary, STAGE_B, cfg.initial_stage_checkpoint_sha256, identity
        )
        self.assertEqual(digest, sha256_file(STAGE_B))
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(set(model.state_dict()), set(loaded["modelStateDict"]))
        self.assertEqual(set(auxiliary.state_dict()), set(loaded["auxiliaryHeadsStateDict"]))
        self.assertTrue(all(
            torch.equal(value, loaded["auxiliaryHeadsStateDict"][key])
            for key, value in auxiliary.state_dict().items()
        ))

    def test_stage_c_config_rejects_unconfirmed_non_cnn_learning_rate(self):
        cfg = stage_c_config()
        document = {
            "input_policy": "uniform-no-current-player-loss-history-v1",
            "board_perspective": "snapshot_side_to_move_v1",
            "training": {**cfg.__dict__, "non_cnn_learning_rate": 1e-4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            import json
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be 5e-5"):
                TrainingConfig.load(path)


class CnnDirectFullTests(unittest.TestCase):
    def test_direct_full_config_and_parameter_groups(self):
        cfg = TrainingConfig.load(ROOT / "config/cnn_tcn_direct_full_38_seed42.json")
        self.assertEqual(cfg.epochs, 38)
        self.assertEqual(cfg.early_stopping_patience, 6)
        model = ProfileConditionedLossModel(ModelConfig())
        auxiliary = BoardCNNAuxiliaryHeads()
        optimizer, report = configure_cnn_direct_full(model, auxiliary, cfg)
        self.assertEqual(report["strategy"], "full-joint-from-epoch-one")
        self.assertTrue(report["allCnnTrainable"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-4, 1e-4])

    @unittest.skipUnless(
        PROFILE_WARM.is_file() and TRANSFERRED.is_file() and PRETRAIN.is_file(),
        "direct-full initialization artifacts absent",
    )
    def test_real_direct_full_initialization_preserves_cnn_and_strictly_loads_non_cnn_and_auxiliary(self):
        cfg = TrainingConfig.load(ROOT / "config/cnn_tcn_direct_full_38_seed42.json")
        model, _ = load_transferred_profile_model(TRANSFERRED, "full-31")
        cnn_before = {
            key: value.clone() for key, value in model.state_dict().items()
            if key.startswith("backbone.board_encoder.")
        }
        warm = torch.load(PROFILE_WARM, map_location="cpu", weights_only=False)
        migration = load_trained_state_excluding_board_encoder(model, warm["modelStateDict"])
        self.assertTrue(migration["strictNonCnnLoad"])
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in cnn_before.items()))
        self.assertTrue(all(
            torch.equal(model.state_dict()[key], value)
            for key, value in warm["modelStateDict"].items()
            if not key.startswith("backbone.board_encoder.")
        ))
        auxiliary = BoardCNNAuxiliaryHeads()
        _strict_load_auxiliary_heads(
            auxiliary, PRETRAIN, cfg.auxiliary_checkpoint_sha256,
            load_checkpoint_payload(TRANSFERRED),
        )
        optimizer, report = configure_cnn_direct_full(model, auxiliary, cfg)
        self.assertTrue(report["allCnnTrainable"])
        self.assertEqual(sum(group["parameterCount"] for group in report["parameterGroups"]), 1172670)
        self.assertEqual(len(optimizer.param_groups), 3)


if __name__ == "__main__":
    unittest.main()
