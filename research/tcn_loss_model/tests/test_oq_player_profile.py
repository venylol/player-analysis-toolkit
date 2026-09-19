from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.oq_player_profile import (
    PROFILE_SCHEMA,
    accounts_from_file,
    canonical_json_sha256,
    normalize_account,
    normalize_profile_response,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "oq_player_profile_response.json"


class OqPlayerProfileTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixed_fixture_is_parsed_and_unapproved_fields_are_not_normalized(self):
        profile = normalize_profile_response(
            self.raw,
            requested_account="FIXTURE_PLAYER",
            fetched_at_utc="2026-08-02T00:00:00Z",
        )
        self.assertEqual(profile["schema"], PROFILE_SCHEMA)
        self.assertEqual(profile["id"], "fixture_player")
        self.assertEqual(profile["name"], "Fixture Player 测试")
        self.assertEqual(profile["normalized_account"], "fixture_player")
        self.assertEqual(profile["played"], 40)
        self.assertEqual(profile["sente"]["played"], 20)
        self.assertEqual(profile["gote"]["played"], 20)
        self.assertEqual(profile["strong"]["played"], 0)
        self.assertEqual(profile["weak"]["played"], 26)
        self.assertEqual(profile["missing_categories"], [])
        self.assertEqual(profile["raw_response_sha256"], canonical_json_sha256(self.raw))
        self.assertNotIn("chart", profile)
        self.assertNotIn("hiddenR", profile)

    def test_case_insensitive_matching_preserves_returned_identity(self):
        self.raw["id"] = "Fixture_Player"
        profile = normalize_profile_response(
            self.raw,
            requested_account="fixture_player",
            fetched_at_utc="2026-08-02T00:00:00Z",
        )
        self.assertEqual(profile["id"], "Fixture_Player")
        self.assertEqual(profile["normalized_id"], "fixture_player")
        self.assertEqual(normalize_account("ＦＩＸＴＵＲＥ＿ＰＬＡＹＥＲ"), "fixture_player")

    def test_missing_category_is_explicit_not_a_zero_record(self):
        self.raw["srecords"] = [row for row in self.raw["srecords"] if row.get("name") != "weak"]
        profile = normalize_profile_response(
            self.raw,
            requested_account="fixture_player",
            fetched_at_utc="2026-08-02T00:00:00Z",
        )
        self.assertIsNone(profile["weak"])
        self.assertEqual(profile["missing_categories"], ["opp/weak"])

    def test_played_identity_is_enforced(self):
        self.raw["played"] = 41
        with self.assertRaisesRegex(ValueError, r"played != win \+ loss \+ draw"):
            normalize_profile_response(
                self.raw,
                requested_account="fixture_player",
                fetched_at_utc="2026-08-02T00:00:00Z",
            )

    def test_utf8_csv_json_and_text_account_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "accounts.csv"
            json_path = root / "accounts.json"
            text_path = root / "accounts.txt"
            games_path = root / "games.csv"
            csv_path.write_text("oq_account,note\nHero9,测试\nxiaojianbao,二\n", encoding="utf-8")
            json_path.write_text(
                json.dumps({"players": [{"account": "Hero9"}, {"id": "xiaojianbao"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            text_path.write_text("# UTF-8 名单\nHero9\nXIAOJIANBAO\nhero9\n", encoding="utf-8")
            games_path.write_text(
                "game_id,black_id,white_id\ng1,Hero9,xiaojianbao\ng2,hero9,Third_Player\n",
                encoding="utf-8",
            )
            self.assertEqual(accounts_from_file(csv_path), ["Hero9", "xiaojianbao"])
            self.assertEqual(accounts_from_file(json_path), ["Hero9", "xiaojianbao"])
            self.assertEqual(accounts_from_file(text_path), ["Hero9", "XIAOJIANBAO"])
            self.assertEqual(accounts_from_file(games_path), ["Hero9", "xiaojianbao", "Third_Player"])


if __name__ == "__main__":
    unittest.main()
