#!/usr/bin/env python3
"""Report strict-game, complete-hint, and current bilateral-ranked coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True)
    parser.add_argument("--moves", required=True)
    parser.add_argument("--hints", required=True)
    parser.add_argument("--users", required=True)
    parser.add_argument("--bilateral-target", type=int, default=4000)
    args = parser.parse_args()

    users: set[str] = set()
    with Path(args.users).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = row.get("id", "").strip().lower()
            if user_id:
                users.add(user_id)

    games: dict[str, dict[str, str]] = {}
    bilateral = 0
    with Path(args.games).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if not game_id or row.get("tcb") != "300000":
                continue
            if not row.get("finalStatus", "").startswith("SCORE:"):
                continue
            if game_id in games:
                continue
            games[game_id] = row
            if row.get("black_id", "").strip().lower() in users and row.get("white_id", "").strip().lower() in users:
                bilateral += 1

    passes: dict[str, set[int]] = defaultdict(set)
    with Path(args.moves).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id in games and row.get("move", "").strip() == "-":
                passes[game_id].add(int(row["move_index"]))

    hinted: dict[str, set[int]] = defaultdict(set)
    with Path(args.hints).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row.get("game_id", "").strip()
            if game_id not in games:
                continue
            move_index = int(row["move_index"])
            if row.get("actual_move", "").strip() == "-":
                passes[game_id].add(move_index)
            elif row.get("hint6_1_score", "").strip():
                hinted[game_id].add(move_index)

    complete = 0
    for game_id, game in games.items():
        length = int(game.get("length", 0) or 0)
        if set(range(length)).issubset(hinted.get(game_id, set()) | passes.get(game_id, set())):
            complete += 1
    additional_bilateral_needed = max(0, args.bilateral_target - bilateral)
    report = {
        "strict_games": len(games),
        "complete_bilateral_hint_games": complete,
        "remaining_incomplete_games": len(games) - complete,
        "current_both_ranked_games": bilateral,
        "bilateral_target": args.bilateral_target,
        "additional_bilateral_games_needed": additional_bilateral_needed,
        "expected_strict_total_at_bilateral_target": len(games) + additional_bilateral_needed,
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
