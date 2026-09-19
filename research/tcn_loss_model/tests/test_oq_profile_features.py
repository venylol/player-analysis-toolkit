from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.data.materialize_oq_profile_context import materialize
from src.data_contract import validate_model_ready_npz
from src.oq_player_profile import PROFILE_SCHEMA, canonical_json_sha256
from src.oq_profile_features import (
    OQ_PROFILE_FEATURE_NAMES,
    RETROSPECTIVE_TRUSTED_POLICY,
    STRICT_PRE_GAME_POLICY,
    ProfileSnapshot,
    build_profile_feature_vector,
    fit_train_only_normalization,
    select_snapshot,
)


def profile(account: str, rating: float, win: int, loss: int, draw: int) -> dict:
    def record(rating_value, w, l, d):
        return {"rating": rating_value, "win": w, "loss": l, "draw": d, "played": w + l + d}
    document = {
        "schema": PROFILE_SCHEMA,
        "normalized_account": account,
        "id": account,
        "name": account,
        "gtype": "reversi",
        "rating": rating,
        "high": rating + 20,
        "win": win,
        "loss": loss,
        "draw": draw,
        "played": win + loss + draw,
        "sente": record(rating + 10, 6, 3, 1),
        "gote": record(rating - 10, 4, 5, 1),
        "strong": record(rating - 30, 2, 7, 1),
        "weak": record(rating + 30, 8, 1, 1),
        "missing_categories": [],
        "profile_fetched_at_utc": "2026-08-04T00:00:00Z",
        "raw_response_sha256": "a" * 64,
    }
    return document


class OqProfileFeatureTests(unittest.TestCase):
    def test_fixed_order_and_formulas(self):
        self.assertEqual(len(OQ_PROFILE_FEATURE_NAMES), 31)
        player = profile("p", 2000, 30, 9, 1)
        opponent = profile("o", 1800, 15, 14, 1)
        values, missing = build_profile_feature_vector(
            player, opponent, player_color="black", opponent_color="white"
        )
        self.assertFalse(missing.any())
        self.assertAlmostEqual(float(values[1]), 30 / 40)
        self.assertAlmostEqual(float(values[2]), 1 / 40)
        self.assertAlmostEqual(float(values[3]), np.log1p(40), places=6)
        self.assertEqual(float(values[4]), 1.0)
        self.assertEqual(float(values[5]), 2010.0)  # black -> sente
        self.assertEqual(float(values[14]), 1790.0)  # opponent white -> gote
        self.assertEqual(float(values[18]), 200.0)
        self.assertAlmostEqual(float(values[23]), 2 / 10)  # current player strong only
        self.assertEqual(float(values[26]), 1970.0)

    def test_player_opponent_swap_reverses_all_difference_signs(self):
        player = profile("p", 2000, 30, 9, 1)
        opponent = profile("o", 1800, 15, 14, 1)
        forward, _ = build_profile_feature_vector(player, opponent, player_color="black", opponent_color="white")
        reverse, _ = build_profile_feature_vector(opponent, player, player_color="white", opponent_color="black")
        np.testing.assert_allclose(forward[18:23], -reverse[18:23])

    def test_zero_games_and_missing_category_have_missing_masks(self):
        zero = profile("zero", 0, 0, 0, 0)
        zero["sente"] = {"rating": 0, "win": 0, "loss": 0, "draw": 0, "played": 0}
        zero["strong"] = None
        values, missing = build_profile_feature_vector(zero, None, player_color="black", opponent_color="white")
        self.assertTrue(missing[:5].all())
        self.assertTrue(missing[5:9].all())
        self.assertTrue(missing[9:18].all())
        self.assertTrue(missing[23:27].all())
        self.assertTrue(np.all(values[missing] == 0))

    def test_future_snapshot_is_rejected_and_latest_eligible_is_selected(self):
        def snapshot(day: int) -> ProfileSnapshot:
            document = profile("p", 1900 + day, 10, 5, 0)
            document["profile_fetched_at_utc"] = f"2026-08-{day:02d}T00:00:00Z"
            return ProfileSnapshot(
                "p", datetime(2026, 8, day, tzinfo=timezone.utc), document,
                Path(f"p-{day}.json"), str(day) * 64,
            )
        snapshots = [snapshot(1), snapshot(3), snapshot(5)]
        game_time = datetime(2026, 8, 4, tzinfo=timezone.utc)
        self.assertEqual(select_snapshot(snapshots, game_time, STRICT_PRE_GAME_POLICY).fetched_at.day, 3)
        self.assertEqual(select_snapshot(snapshots, game_time, RETROSPECTIVE_TRUSTED_POLICY).fetched_at.day, 5)
        self.assertIsNone(select_snapshot(snapshots, datetime(2026, 7, 1, tzinfo=timezone.utc), STRICT_PRE_GAME_POLICY))

    def test_normalization_fits_train_only(self):
        raw = np.zeros((3, 2, 31), dtype=np.float32)
        raw[0, 0, :] = 1
        raw[0, 1, :] = 3
        raw[1, :, :] = 1000  # validation must not affect train means
        raw[2, :, :] = -1000  # test must not affect train means
        missing = np.zeros_like(raw, dtype=bool)
        node_valid = np.ones((3, 2), dtype=bool)
        normalized, means, stds, _hash = fit_train_only_normalization(
            raw, missing, node_valid, np.asarray(["train", "validation", "test"])
        )
        np.testing.assert_allclose(means, 2.0)
        np.testing.assert_allclose(stds, 1.0)
        np.testing.assert_allclose(normalized[0, :, 0], [-1, 1])

    def test_toy_materializer_preserves_original_arrays_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            for account, rating in (("black", 2000), ("white", 1800)):
                document = profile(account, rating, 20, 10, 0)
                document["raw_response_sha256"] = canonical_json_sha256(document)
                (snapshots / f"{account}.json").write_text(
                    json.dumps(document, ensure_ascii=False), encoding="utf-8"
                )
            games = root / "games.csv"
            games.write_text(
                "game_id,created,black_id,white_id\ng,2026-08-01T00:00:00Z,black,white\n",
                encoding="utf-8",
            )
            source = root / "source.npz"
            x = np.arange(2 * 362, dtype=np.float32).reshape(1, 2, 362)
            shape = (1, 2)
            np.savez(
                source,
                X=x,
                board_tokens=np.ones((*shape, 3, 64), dtype=np.int8),
                board_move_tokens=np.zeros((*shape, 3), dtype=np.int8),
                current_hint_tokens=np.zeros((*shape, 6), dtype=np.int8),
                current_hint_values=np.zeros((*shape, 4), dtype=np.float32),
                prev_own_hint_values=np.zeros((*shape, 2), dtype=np.float32),
                actual_thinking_time_ms=np.asarray([[1000.0, 0.0]], dtype=np.float32),
                disc_loss=np.zeros(shape, dtype=np.float32),
                raw_loss=np.zeros(shape, dtype=np.float32),
                severity_class=np.zeros(shape, dtype=np.int8),
                label_zero=np.asarray([[1, 0]], dtype=np.int8),
                label_ge4=np.zeros(shape, dtype=np.int8),
                label_ge10=np.zeros(shape, dtype=np.int8),
                mask=np.asarray([[True, False]]),
                game_id=np.asarray(["g"]),
                split=np.asarray(["train"]),
                global_placement_ply=np.asarray([[1, 0]], dtype=np.int16),
                player_id=np.asarray([["black", ""]]),
                side_to_move=np.asarray([["black", ""]]),
                move_index=np.asarray([[0, -1]], dtype=np.int16),
                source_ply_including_pass=np.asarray([[1, 0]], dtype=np.int16),
                label_available=np.asarray([[True, False]]),
                has_consecutive_child=np.asarray([[True, False]]),
                child_continuity_ok=np.asarray([[True, False]]),
                same_side_after_move=np.zeros(shape, dtype=bool),
                current_score=np.zeros(shape, dtype=np.float32),
                actual_move_score=np.zeros(shape, dtype=np.float32),
                wld_class=np.zeros(shape, dtype=np.int8),
                wld_loss=np.zeros(shape, dtype=np.float32),
                wld_label_available=np.zeros(shape, dtype=bool),
                input_features=np.asarray([f"feature_{i}" for i in range(362)]),
                board_cnn_channels=np.asarray([f"c{i}" for i in range(23)]),
                preprocessing_sha256=np.asarray("b" * 64),
                input_policy=np.asarray("uniform-no-current-player-loss-history-v1"),
            )
            output_dir = root / "output"
            manifest = materialize(Namespace(
                input_npz=source,
                games=games,
                snapshots_dir=snapshots,
                output_dir=output_dir,
                output_name="profile.npz",
                policy=RETROSPECTIVE_TRUSTED_POLICY,
                allow_temporal_leakage=True,
            ))
            with np.load(output_dir / "profile.npz", allow_pickle=False) as result:
                np.testing.assert_array_equal(result["X"], x)
                self.assertEqual(result["oq_profile_features"].shape, (1, 2, 31))
                self.assertFalse(result["oq_profile_missing"][0, 0].any())
                self.assertTrue(result["oq_profile_missing"][0, 1].all())
                self.assertTrue(np.all(result["oq_profile_features"][0, 1] == 0))
                self.assertEqual(result["oq_profile_policy"].item(), RETROSPECTIVE_TRUSTED_POLICY)
            self.assertTrue(manifest["originalArraysPreserved"])
            self.assertEqual(manifest["coverage"]["snapshotSelectionsAfterGameCreated"], 2)
            good = output_dir / "profile.npz"
            with np.load(good, allow_pickle=False) as data:
                payload = {name: data[name].copy() for name in data.files}
            bad_order = root / "bad_order.npz"
            order_payload = {**payload, "oq_profile_feature_names": payload["oq_profile_feature_names"].copy()}
            order_payload["oq_profile_feature_names"][0] = "wrong_name"
            np.savez(bad_order, **order_payload)
            with self.assertRaisesRegex(ValueError, "feature order"):
                validate_model_ready_npz(bad_order, require_oq_profile=True)
            bad_hash = root / "bad_hash.npz"
            np.savez(bad_hash, **{**payload, "oq_profile_preprocessing_sha256": np.asarray("0" * 64)})
            with self.assertRaisesRegex(ValueError, "preprocessing hash"):
                validate_model_ready_npz(bad_hash, require_oq_profile=True)
            bad_policy = root / "bad_policy.npz"
            np.savez(bad_policy, **{**payload, "oq_profile_policy": np.asarray("unknown-policy")})
            with self.assertRaisesRegex(ValueError, "unsupported OQ profile policy"):
                validate_model_ready_npz(bad_policy, require_oq_profile=True)
            strict_future = root / "strict_future.npz"
            np.savez(strict_future, **{
                **payload,
                "oq_profile_policy": np.asarray(STRICT_PRE_GAME_POLICY),
                "oq_profile_temporal_leakage_authorized": np.asarray(False),
            })
            with self.assertRaisesRegex(ValueError, "future snapshot"):
                validate_model_ready_npz(strict_future, require_oq_profile=True)


if __name__ == "__main__":
    unittest.main()
