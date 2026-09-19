import unittest
from pathlib import Path

import numpy as np

from src.ensemble import deterministic_fixed_test_split, resolve_member_checkpoint


class EnsembleSplitTests(unittest.TestCase):
    def test_fixed_test_and_deterministic_game_split(self):
        game_ids = np.asarray([f"g{i}" for i in range(100)])
        original = np.asarray(["test" if i >= 90 else "train" for i in range(100)])
        first, first_summary = deterministic_fixed_test_split(game_ids, original, 42)
        second, second_summary = deterministic_fixed_test_split(game_ids, original, 42)
        other, _ = deterministic_fixed_test_split(game_ids, original, 43)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first == "test", original == "test")
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(set(first), {"train", "validation", "test"})

    def test_member_checkpoint_is_resolved_relative_to_manifest(self):
        manifest = Path("models/primary/ensemble_manifest.json")
        expected = manifest.resolve().parent / "member_01.pt"
        self.assertEqual(resolve_member_checkpoint(manifest, "member_01.pt"), expected)

        absolute = Path.cwd().resolve() / "member_01.pt"
        self.assertEqual(resolve_member_checkpoint(manifest, absolute), absolute)


if __name__ == "__main__":
    unittest.main()
