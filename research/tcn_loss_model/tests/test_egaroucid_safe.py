import unittest

from src.egaroucid_safe import (
    HintTransaction,
    legal_moves_from_setboard,
    normalize_setboard,
    parse_hint_rows,
    parse_unique_native_console_board,
    validate_hint_transaction,
)


INITIAL = "---------------------------OX------XO---------------------------X"
BOARD = """  a b c d e f g h
1 . . . . . . . .
2 . . . . . . . .   BLACK to move
3 . . . . . . . .   ply 1 60 empties
4 . . . O X . . .   BLACK: 2 WHITE: 2
5 . . . X O . . .   BLACK Remaining -s
6 . . . . . . . .   WHITE Remaining -s
7 . . . . . . . .
8 . . . . . . . .
"""
HINT = """|          Level|          Depth|           Move|          Score|           Time|          Nodes|            NPS|
|              2|         2@100%|             e6|             +0|  000:00:00.000|             17|          17000|
""" + BOARD


class SafeEgaroucidTests(unittest.TestCase):
    def test_initial_legal_moves_are_derived_independently(self):
        self.assertEqual(legal_moves_from_setboard(INITIAL), ["d3", "c4", "f5", "e6"])

    def test_native_board_and_hint_are_strictly_parsed(self):
        board, count = parse_unique_native_console_board(HINT)
        self.assertEqual(board, INITIAL)
        self.assertEqual(count, 1)
        self.assertEqual(parse_hint_rows(HINT)[0]["move"], "e6")

    def test_identical_multiple_boards_are_allowed_but_conflicts_are_not(self):
        board, count = parse_unique_native_console_board(BOARD + BOARD)
        self.assertEqual(board, INITIAL)
        self.assertEqual(count, 2)
        conflicting = BOARD + BOARD.replace("BLACK to move", "WHITE to move")
        with self.assertRaises(RuntimeError):
            parse_unique_native_console_board(conflicting)

    def test_duplicate_or_out_of_sequence_rows_are_rejected(self):
        malformed = BOARD.replace(
            "2 . . . . . . . .   BLACK to move",
            "1 . . . . . . . .   BLACK to move",
        )
        with self.assertRaises(RuntimeError):
            parse_unique_native_console_board(malformed)

    def test_transaction_requires_board_legality_and_complete_candidates(self):
        hints = parse_hint_rows(HINT)
        transaction = HintTransaction(
            request_board_setboard=INITIAL,
            setboard_response_board_setboard=INITIAL,
            hint_response_board_setboard=INITIAL,
            setboard_response_board_count=1,
            hint_response_board_count=1,
            hints=hints,
            setboard_raw_response=BOARD,
            hint_raw_response=HINT,
            elapsed_seconds=0.01,
        )
        validate_hint_transaction(transaction, ["d3", "c4", "f5", "e6"], 1)
        with self.assertRaises(RuntimeError):
            validate_hint_transaction(transaction, ["a1"], 1)

    def test_setboard_normalization_rejects_bad_shape(self):
        self.assertEqual(normalize_setboard(INITIAL.lower()), INITIAL)
        with self.assertRaises(ValueError):
            normalize_setboard(INITIAL[:-1])


if __name__ == "__main__":
    unittest.main()
