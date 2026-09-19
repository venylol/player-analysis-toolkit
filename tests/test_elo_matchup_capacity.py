from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "offbook_detection"))

from build_elo_reference import OthelloBoard, bin_for_rating, bin_lowers, load_inputs
from summarize_elo_matchup_capacity import (
    bin_label,
    build_capacity,
    long_rows,
    matrix_rows,
    scan_cache_metadata,
    write_outputs,
)


def record(black_rating: float, white_rating: float, black_id: str = "black", white_id: str = "white"):
    return ({}, {"players": [{"id": black_id, "oldR": black_rating}, {"id": white_id, "oldR": white_rating}]})


def valid_jsonl_row(game_id: str, black_rating: float, white_rating: float) -> dict:
    board = OthelloBoard()
    move = board.legal_moves()[0]
    detail = {
        "gtype": "reversi",
        "finished": True,
        "tcb": 300000,
        "players": [{"id": f"b-{game_id}", "oldR": black_rating}, {"id": f"w-{game_id}", "oldR": white_rating}],
        "position": {"moves": [{"m": move, "t": 1000}]},
    }
    return {"game_id": game_id, "valid": True, "summary": {"finalStatus": "SCORE:1-0"}, "detail": detail}


class EloMatchupCapacityTests(unittest.TestCase):
    def test_duplicate_jsonl_game_id_is_not_counted_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "details.jsonl"
            row = valid_jsonl_row("g1", 1650, 1750)
            path.write_text("\n".join(json.dumps(row) for _ in range(2)) + "\n", encoding="utf-8")
            metadata = scan_cache_metadata(path)
            _, records, counts = load_inputs(path, "test", minimum=1600, maximum=2486, width=100, seed=0)
            cells, _ = build_capacity(records, minimum=1600, maximum=2486, width=100)
            self.assertEqual(metadata["duplicateCacheMarkedValidLines"], 1)
            self.assertEqual(counts["excludedDuplicateJsonlGameId"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(cells[(1600, 1700)].game_ids), 1)

    def test_cross_bin_game_enters_opposite_cells(self) -> None:
        cells, counts = build_capacity({"g": record(1650, 1750)}, minimum=1600, maximum=2486, width=100)
        self.assertEqual(cells[(1600, 1700)].game_ids, {"g"})
        self.assertEqual(cells[(1700, 1600)].game_ids, {"g"})
        self.assertEqual(counts["inRangeUniqueGames"], 1)

    def test_same_bin_game_is_unique_once_but_has_both_side_capacities(self) -> None:
        cells, _ = build_capacity({"g": record(1650, 1699)}, minimum=1600, maximum=2486, width=100)
        cell = cells[(1600, 1600)]
        self.assertEqual(cell.game_ids, {"g"})
        self.assertEqual(cell.target_black_game_ids, {"g"})
        self.assertEqual(cell.target_white_game_ids, {"g"})
        self.assertEqual(cell.target_player_ids, {"black", "white"})

    def test_boundaries_and_maximum(self) -> None:
        self.assertEqual(bin_for_rating(1600, 1600, 2486, 100), 1600)
        self.assertEqual(bin_for_rating(1699.999, 1600, 2486, 100), 1600)
        self.assertEqual(bin_for_rating(1700, 1600, 2486, 100), 1700)
        self.assertEqual(bin_for_rating(2486, 1600, 2486, 100), 2400)
        self.assertIsNone(bin_for_rating(1599.999, 1600, 2486, 100))
        self.assertIsNone(bin_for_rating(2486.001, 1600, 2486, 100))

    def test_zero_combinations_are_retained(self) -> None:
        cells, _ = build_capacity({"g": record(1650, 1750)}, minimum=1600, maximum=2486, width=100)
        lowers = bin_lowers(1600, 2486, 100)
        rows = long_rows(cells, lowers, 2486, 100)
        self.assertEqual(len(rows), 81)
        self.assertTrue(any(row["uniqueGameCount"] == 0 for row in rows))

    def test_matrix_long_csv_and_markdown_are_cellwise_consistent(self) -> None:
        cells, _ = build_capacity({"g": record(1650, 1750)}, minimum=1600, maximum=2486, width=100)
        lowers = bin_lowers(1600, 2486, 100)
        matrix = matrix_rows(cells, lowers, 2486, 100)
        long = long_rows(cells, lowers, 2486, 100)
        long_map = {(row["targetBinLower"], row["opponentBinLower"]): row["uniqueGameCount"] for row in long}
        for row, target in zip(matrix, lowers):
            for opponent in lowers:
                self.assertEqual(row[bin_label(opponent, 2486, 100)], long_map[(target, opponent)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_outputs(output, cells=cells, lowers=lowers, maximum=2486, width=100, summary={"ok": True})
            with (output / "target_vs_opponent_elo_unique_game_matrix.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            markdown = (output / "target_vs_opponent_elo_unique_game_matrix.md").read_text(encoding="utf-8")
            self.assertEqual(csv_rows[0]["[1700,1800)"], "1")
            self.assertIn("| [1600,1700) | 0 | 1 |", markdown)


if __name__ == "__main__":
    unittest.main()
