import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.backbone import BoardCNNEncoder, BoardConditionedBackbone, ModelConfig
from src.board_cnn import (
    BoardCNNAuxiliaryHeads, SharedBoardCNN, parameter_count, sample_auxiliary_node_indices,
)
from src.board_cnn_pretrain import (
    BoardCNNPretrainModel,
    BoardShardDataset,
    DATA_CONTRACT,
    checkpoint_payload,
    collate_records,
    legal_moves_bitboard,
    load_pretrain_checkpoint,
    materialize_source_file,
    pack_board,
    save_checkpoint,
    unpack_board,
    BoardCNNTrainingConfig,
)
from src.board_cnn_transfer import transfer_board_cnn


OPENING = "---------------------------OX------XO---------------------------"
ROOT = Path(__file__).resolve().parents[1]


def training_config(root: Path) -> BoardCNNTrainingConfig:
    return BoardCNNTrainingConfig(
        data_dir=root,
        shard_dir=root / "shards",
        splits={"train": ["0000000.txt"], "validation": ["0000000.txt"], "test": ["0000000.txt"]},
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        legal_loss_weight=1.0,
        value_loss_weight=1.0,
        num_workers=0,
        seed=42,
        checkpoint_interval_epochs=1,
        validation_interval_epochs=1,
        mixed_precision=False,
        resume_strategy="latest",
        smoke_max_samples=2,
        board_channels=64,
        residual_blocks=6,
        board_embedding_dim=96,
        embedding_projection_kernel=1,
        input_channels=3,
        gradient_accumulation_steps=1,
    )


class LegalMoveRulesTest(unittest.TestCase):
    def test_opening(self):
        expected = sum(1 << square for square in (19, 26, 37, 44))
        self.assertEqual(legal_moves_bitboard(OPENING), expected)

    def test_ordinary_position(self):
        self.assertEqual(legal_moves_bitboard("XO-" + "-" * 61), 1 << 2)

    def test_pass_position(self):
        self.assertEqual(legal_moves_bitboard("X" + "-" * 63), 0)

    def test_terminal_position(self):
        self.assertEqual(legal_moves_bitboard("X" * 64), 0)

    def test_two_bit_round_trip(self):
        codes = unpack_board(pack_board(OPENING))
        reconstructed = "".join("-XO"[int(code)] for code in codes)
        self.assertEqual(reconstructed, OPENING)


class ShardAndTrainingContractTest(unittest.TestCase):
    def test_materialize_reuse_decode_forward_backward_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "0000000.txt"
            source.write_text(f"{OPENING} 3\n{'XO-' + '-' * 61} -4\n", encoding="utf-8")
            first = materialize_source_file(source, root / "shards")
            second = materialize_source_file(source, root / "shards")
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["records"], 2)
            dataset = BoardShardDataset([Path(first["shard"])])
            boards, legal, values = collate_records([dataset[0], dataset[1]])
            self.assertEqual(tuple(boards.shape), (2, 3, 8, 8))
            self.assertEqual(tuple(legal.shape), (2, 64))
            self.assertTrue(torch.equal(legal[0].nonzero().flatten(), torch.tensor([19, 26, 37, 44])))
            self.assertTrue(torch.allclose(values, torch.tensor([3 / 64, -4 / 64])))
            dataset.close()

            model = BoardCNNPretrainModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            logits, prediction = model(boards)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, legal)
            loss = loss + torch.nn.functional.smooth_l1_loss(prediction, values)
            loss.backward()
            optimizer.step()
            cfg = training_config(root)
            data_manifest = {"shards": [first]}
            checkpoint = root / "checkpoint.pt"
            save_checkpoint(checkpoint, checkpoint_payload(model, optimizer, 0, 1, cfg, data_manifest, {"selectionMetric": 1.0}))
            restored = BoardCNNPretrainModel()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            loaded = load_pretrain_checkpoint(checkpoint, restored, restored_optimizer)
            self.assertEqual(loaded["data_contract"], DATA_CONTRACT)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_failed_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bad.txt"
            source.write_text("not-a-board 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                materialize_source_file(source, root / "shards")
            manifests = list((root / "shards").glob("*.attempt-*.part.manifest.json"))
            parts = list((root / "shards").glob("*.attempt-*.part"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(parts), 1)
            self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8"))["status"], "failed")


class TransferTest(unittest.TestCase):
    def test_strict_three_to_twenty_three_channel_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(7)
            pretrained_model = BoardCNNPretrainModel()
            optimizer = torch.optim.AdamW(pretrained_model.parameters())
            cfg = training_config(root)
            pretrained = root / "pretrained.pt"
            payload = checkpoint_payload(pretrained_model, optimizer, 0, 0, cfg, {"shards": []}, {})
            torch.save(payload, pretrained)
            target = root / "target.pt"
            backbone = BoardConditionedBackbone(ModelConfig(board_cnn_architecture="legacy-three-conv-v1"))
            torch.save({"model_state_dict": backbone.state_dict(), "sentinel": "preserved"}, target)
            output = root / "transferred.pt"
            manifest = transfer_board_cnn(pretrained, target, output)
            self.assertTrue(all(manifest["checks"].values()))
            transferred = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(transferred["board_perspective"], "snapshot_side_to_move_v1")
            self.assertEqual(manifest["boardPerspective"], "snapshot_side_to_move_v1")
            first = transferred["model_state_dict"]["board_encoder.shared.stem.weight"]
            self.assertTrue(torch.equal(first[:, :3], pretrained_model.shared.stem.weight))
            self.assertEqual(int(torch.count_nonzero(first[:, 3:])), 0)
            self.assertEqual(transferred["sentinel"], "preserved")
            self.assertNotIn("legal_move_head_state_dict", transferred["model_state_dict"])
            self.assertTrue(manifest["checks"]["allNonCnnKeysStrictlyLoaded"])
            self.assertTrue(manifest["excludedOldBoardEncoderKeys"])
            formal = BoardConditionedBackbone(ModelConfig())
            formal.load_state_dict(transferred["model_state_dict"], strict=True)
            with torch.no_grad():
                embedding = formal.board_encoder(
                    torch.ones(2, 4, 3, 64, dtype=torch.long),
                    torch.zeros(2, 4, 3, dtype=torch.long),
                    torch.zeros(2, 4, 6, dtype=torch.long),
                    torch.zeros(2, 4, 4),
                    torch.zeros(2, 4, 2),
                )
            self.assertEqual(tuple(embedding.shape), (2, 4, 96))


class FinalArchitectureContractTest(unittest.TestCase):
    def test_default_pretraining_config_contract(self):
        cfg = BoardCNNTrainingConfig.load(ROOT / "config" / "board_cnn_pretrain.json")
        self.assertEqual(cfg.input_channels, 3)
        self.assertEqual(cfg.board_channels, 64)
        self.assertEqual(cfg.residual_blocks, 6)
        self.assertEqual(cfg.board_embedding_dim, 96)
        self.assertEqual(cfg.embedding_projection_kernel, 1)
        self.assertGreaterEqual(cfg.gradient_accumulation_steps, 1)

    def test_default_architecture_shapes_and_exact_parameter_counts(self):
        pretrain = BoardCNNPretrainModel()
        formal = BoardCNNEncoder(ModelConfig())
        self.assertEqual(pretrain.shared.board_channels, 64)
        self.assertEqual(pretrain.shared.residual_blocks_count, 6)
        self.assertEqual(pretrain.shared.board_embedding_dim, 96)
        self.assertEqual(pretrain.shared.projection.kernel_size, (1, 1))
        self.assertEqual(len(pretrain.shared.blocks), 6)
        self.assertEqual(parameter_count(pretrain.shared), 453024)
        self.assertEqual(parameter_count(formal.shared), 464544)
        self.assertTrue(430000 <= parameter_count(pretrain.shared) <= 480000)
        self.assertTrue(440000 <= parameter_count(formal.shared) <= 500000)
        self.assertEqual(parameter_count(pretrain.legal_head), 65)
        self.assertEqual(parameter_count(pretrain.value_head), 6273)
        logits, value = pretrain(torch.zeros(3, 3, 8, 8))
        self.assertEqual(tuple(logits.shape), (3, 64))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertEqual(tuple(pretrain.shared(torch.zeros(3, 3, 8, 8)).shape), (3, 96))

    def test_state_only_auxiliary_forward_zeros_every_non_current_channel(self):
        encoder = BoardCNNEncoder(ModelConfig()).eval()
        heads = BoardCNNAuxiliaryHeads().eval()
        planes = torch.randn(2, 23, 8, 8)
        changed = planes.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        with torch.no_grad():
            spatial_one, embedding_one = encoder.state_only_features(planes)
            spatial_two, embedding_two = encoder.state_only_features(changed)
            output_one = heads(spatial_one, embedding_one)
            output_two = heads(spatial_two, embedding_two)
        self.assertTrue(torch.equal(spatial_one, spatial_two))
        self.assertTrue(torch.equal(embedding_one, embedding_two))
        self.assertTrue(torch.equal(output_one[0], output_two[0]))
        self.assertTrue(torch.equal(output_one[1], output_two[1]))

    def test_projection_kernel_and_embedding_dimension_are_fixed(self):
        with self.assertRaisesRegex(ValueError, "projection_kernel"):
            SharedBoardCNN(3, embedding_projection_kernel=3)
        with self.assertRaisesRegex(ValueError, "fixed at 96"):
            SharedBoardCNN(3, board_embedding_dim=128)

    def test_auxiliary_sampling_selects_only_valid_nodes_per_game(self):
        valid = torch.tensor([[True, False, True, True], [False, True, False, False]])
        selected = sample_auxiliary_node_indices(valid, 2, torch.Generator().manual_seed(4))
        self.assertEqual(int((selected[:, 0] == 0).sum()), 2)
        self.assertEqual(int((selected[:, 0] == 1).sum()), 1)
        self.assertTrue(all(bool(valid[game, node]) for game, node in selected.tolist()))


if __name__ == "__main__":
    unittest.main()
