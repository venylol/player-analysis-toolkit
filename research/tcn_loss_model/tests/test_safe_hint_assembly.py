import unittest

from scripts.pipeline.assemble_safe_hint_recompute import (
    PROVENANCE_FIELDS,
    apply_hint_fields,
    clear_pass_engine_fields,
)


class SafeHintAssemblyTests(unittest.TestCase):
    def test_new_results_replace_every_old_hint_field(self):
        row = {"hint1_move": "old", "hint6_1_move": "old"}
        hint1 = {
            "hints": [{"move": "d3", "score": 1, "nodes": 2, "depth": "2@100%", "is_book": False}],
            "request_board": "b" * 65,
            "console_board": "b" * 65,
            "setboard_console_board": "b" * 65,
            "request_id": "h1",
            "worker_id": 1,
            "batch_id": 2,
            "contract_hash": "c1",
        }
        hint6 = {
            "hints": [
                {"move": move, "score": rank, "nodes": rank + 10, "depth": "18@74%", "is_book": False}
                for rank, move in enumerate(("d3", "c4", "f5", "e6"), start=1)
            ],
            "request_board": "b" * 65,
            "console_board": "b" * 65,
            "setboard_console_board": "b" * 65,
            "request_id": "h6",
            "worker_id": 3,
            "batch_id": 4,
            "contract_hash": "c6",
        }
        apply_hint_fields(row, hint1, hint6)
        self.assertEqual(row["hint1_move"], "d3")
        self.assertEqual(row["hint6_4_move"], "e6")
        self.assertEqual(row["hint6_5_move"], "")
        self.assertEqual(row["hint1_request_id"], "h1")
        self.assertEqual(row["hint6_contract_hash"], "c6")

    def test_pass_rows_cannot_retain_old_engine_values(self):
        row = {"hint1_move": "old", "hint6_1_score": "12"}
        clear_pass_engine_fields(row)
        self.assertEqual(row["hint1_move"], "")
        self.assertEqual(row["hint6_1_score"], "")
        self.assertTrue(all(row[field] == "" for field in PROVENANCE_FIELDS))


if __name__ == "__main__":
    unittest.main()
