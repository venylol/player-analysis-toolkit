from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_11200_pass_supplement import legal_moves_bitboard, normalize_for_side_to_move
from legal_data import PassAwareBatchSampler, collate_legal_records, load_combined_dataset
from model import SpatialLegalTransformer, SpatialTransformerConfig, parameter_count
from transforms import apply_d4, to_target_perspective, transform_flat

import torch


class RulesTest(unittest.TestCase):
    def test_opening_legal_moves(self) -> None:
        opening = "---------------------------OX------XO---------------------------"
        self.assertEqual(legal_moves_bitboard(opening), sum(1 << i for i in (19, 26, 37, 44)))

    def test_white_to_move_normalization(self) -> None:
        self.assertEqual(normalize_for_side_to_move("X-O", "white"), "O-X")


class CombinedDataTest(unittest.TestCase):
    def test_all_split_counts_and_supplement_tail(self) -> None:
        expected = {"train": 24_008_755, "validation": 1_001_121, "test": 515_219}
        config = ROOT / "data_config.json"
        for split, count in expected.items():
            with self.subTest(split=split):
                dataset = load_combined_dataset(config, split)
                self.assertEqual(len(dataset), count)
                codes, legal, is_pass = collate_legal_records([dataset[-1]])
                self.assertEqual(tuple(codes.shape), (1, 64))
                self.assertEqual(tuple(legal.shape), (1, 64))
                self.assertTrue(is_pass.item())
                self.assertEqual(legal.sum().item(), 0.0)
                dataset.base.close()

    def test_pass_aware_sampler_covers_base_and_uses_requested_pass_count(self) -> None:
        sampler = PassAwareBatchSampler(10, 3, batch_size=5, pass_fraction=0.2, seed=7)
        batches = list(sampler)
        base = [index for batch in batches for index in batch if index < 10]
        self.assertEqual(sorted(base), list(range(10)))
        self.assertTrue(all(sum(index >= 10 for index in batch) == 1 for batch in batches))


class TransformAndModelTest(unittest.TestCase):
    def test_target_perspective_swap(self) -> None:
        codes = torch.tensor([[0, 1, 2] + [0] * 61, [0, 1, 2] + [0] * 61])
        result = to_target_perspective(codes, torch.tensor([True, False]))
        self.assertEqual(result[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(result[1, :3].tolist(), [0, 2, 1])

    def test_all_d4_transforms_keep_board_and_mask_aligned(self) -> None:
        codes = torch.arange(64).reshape(1, 64)
        legal = (codes == 9).float()
        for transform_id in range(8):
            transformed_codes, transformed_legal = apply_d4(
                codes, legal, torch.tensor([transform_id])
            )
            self.assertTrue(torch.equal(transformed_legal.bool(), transformed_codes == 9))
            self.assertTrue(torch.equal(transform_flat(legal, transform_id), transformed_legal))

    def test_model_shapes_and_size(self) -> None:
        model = SpatialLegalTransformer(SpatialTransformerConfig(dropout=0.0))
        logits, embedding = model(torch.zeros(3, 64, dtype=torch.long), torch.tensor([True, False, True]))
        self.assertEqual(tuple(logits.shape), (3, 64))
        self.assertEqual(tuple(embedding.shape), (3, 96))
        self.assertLess(parameter_count(model), 500_000)


if __name__ == "__main__":
    unittest.main()
