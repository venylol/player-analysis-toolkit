#!/usr/bin/env python3
"""Summarize directed target-vs-opponent Elo capacity from the full OQ cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_elo_reference import (
    bin_for_rating,
    bin_lowers,
    bin_upper,
    load_inputs,
    sha256_file,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "research" / "offbook_detection" / "data" / "oq_transformer_100000_20260813"
DEFAULT_DETAILS = DEFAULT_DATASET / "acquisition" / "game_details.jsonl"
DEFAULT_LEADERBOARD_JSON = DEFAULT_DATASET / "leaderboard.json"
DEFAULT_LEADERBOARD_CSV = DEFAULT_DATASET / "leaderboard.csv"
DEFAULT_OUTPUT_NAME = "elo_matchup_capacity_1600plus_20260814"
SIDES = ("black", "white")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unique_output_dir(parent: Path, preferred_name: str) -> Path:
    candidate = parent / preferred_name
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{preferred_name}_{suffix:02d}"
        suffix += 1
    return candidate


def parse_rating(value: Any) -> float | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if math.isfinite(rating) else None


def scan_cache_metadata(path: Path) -> Counter[str]:
    """Count row-level cache metadata using load_inputs' first-seen order."""
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    valid_ids_anywhere: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            counts["jsonlLines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid UTF-8 JSONL at {path}:{line_number}") from exc
            game_id = str(row.get("game_id") or "").strip()
            cache_valid = row.get("valid") is True
            if cache_valid:
                counts["cacheMarkedValidLines"] += 1
                if game_id:
                    valid_ids_anywhere.add(game_id)
            if not game_id:
                counts["missingGameIdLines"] += 1
                continue
            if game_id in seen:
                counts["duplicateGameIdLines"] += 1
                if cache_valid:
                    counts["duplicateCacheMarkedValidLines"] += 1
                continue
            seen.add(game_id)
            counts["firstOccurrenceGameIds"] += 1
            if cache_valid:
                counts["firstOccurrenceCacheMarkedValidGameIds"] += 1
    counts["distinctCacheMarkedValidGameIdsAnywhere"] = len(valid_ids_anywhere)
    return counts


@dataclass
class Cell:
    game_ids: set[str] = field(default_factory=set)
    target_black_game_ids: set[str] = field(default_factory=set)
    target_white_game_ids: set[str] = field(default_factory=set)
    target_player_ids: set[str] = field(default_factory=set)

    def add(self, game_id: str, side: str, target_player_id: str) -> None:
        self.game_ids.add(game_id)
        if side == "black":
            self.target_black_game_ids.add(game_id)
        elif side == "white":
            self.target_white_game_ids.add(game_id)
        else:
            raise ValueError(f"invalid target side: {side}")
        self.target_player_ids.add(target_player_id.casefold())


def build_capacity(
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    minimum: int,
    maximum: int,
    width: int,
) -> tuple[dict[tuple[int, int], Cell], Counter[str]]:
    lowers = bin_lowers(minimum, maximum, width)
    cells = {(target, opponent): Cell() for target in lowers for opponent in lowers}
    counts: Counter[str] = Counter()
    for game_id, (_, detail) in records.items():
        players = detail.get("players") or []
        ratings = [parse_rating(player.get("oldR")) for player in players]
        missing = any(rating is None for rating in ratings)
        below = any(rating is not None and rating < minimum for rating in ratings)
        above = any(rating is not None and rating > maximum for rating in ratings)
        if missing:
            counts["excludedAnyOldRMissingOrInvalid"] += 1
        if below:
            counts["excludedAnyOldRBelowMinimum"] += 1
        if above:
            counts["excludedAnyOldRAboveMaximum"] += 1
        if missing or below or above:
            counts["excludedOutsideMainMatrixUnion"] += 1
            continue

        assert ratings[0] is not None and ratings[1] is not None
        black_lower = bin_for_rating(ratings[0], minimum, maximum, width)
        white_lower = bin_for_rating(ratings[1], minimum, maximum, width)
        if black_lower is None or white_lower is None:
            raise AssertionError("in-range finite rating did not map to a bin")
        counts["inRangeUniqueGames"] += 1
        black_id = str(players[0]["id"])
        white_id = str(players[1]["id"])
        cells[(black_lower, white_lower)].add(game_id, "black", black_id)
        if black_lower == white_lower:
            # Unique capacity is one game, while both target-side options remain visible.
            cells[(black_lower, white_lower)].add(game_id, "white", white_id)
        else:
            cells[(white_lower, black_lower)].add(game_id, "white", white_id)
    return cells, counts


def bin_label(lower: int, maximum: int, width: int) -> str:
    upper = bin_upper(lower, maximum, width)
    return f"[{lower},{upper}]" if upper == maximum else f"[{lower},{upper})"


def long_rows(
    cells: dict[tuple[int, int], Cell],
    lowers: list[int],
    maximum: int,
    width: int,
) -> list[dict[str, Any]]:
    rows = []
    for target in lowers:
        for opponent in lowers:
            cell = cells[(target, opponent)]
            target_upper = bin_upper(target, maximum, width)
            opponent_upper = bin_upper(opponent, maximum, width)
            rows.append({
                "targetBinLower": target,
                "targetBinUpper": target_upper,
                "targetUpperInclusive": target_upper == maximum,
                "opponentBinLower": opponent,
                "opponentBinUpper": opponent_upper,
                "opponentUpperInclusive": opponent_upper == maximum,
                "uniqueGameCount": len(cell.game_ids),
                "targetBlackGameCount": len(cell.target_black_game_ids),
                "targetWhiteGameCount": len(cell.target_white_game_ids),
                "distinctTargetPlayerCount": len(cell.target_player_ids),
            })
    return rows


def matrix_rows(cells: dict[tuple[int, int], Cell], lowers: list[int], maximum: int, width: int) -> list[dict[str, Any]]:
    labels = [bin_label(lower, maximum, width) for lower in lowers]
    rows = []
    for target, target_label in zip(lowers, labels):
        row: dict[str, Any] = {"targetEloBin": target_label}
        for opponent, opponent_label in zip(lowers, labels):
            row[opponent_label] = len(cells[(target, opponent)].game_ids)
        rows.append(row)
    return rows


def markdown_matrix(matrix: list[dict[str, Any]], labels: list[str]) -> str:
    lines = [
        "# OQ 自己 Elo × 对手 Elo 可用唯一棋谱容量",
        "",
        "每格是该有方向组合可提供的唯一 `gameId` 数。跨档对局会进入两个方向相反的单元格；同档对局在同一单元格只计一次。因此矩阵单元格总和不等于范围内唯一游戏总数。",
        "",
        "| 自己 \\ 对手 | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for row in matrix:
        lines.append("| " + str(row["targetEloBin"]) + " | " + " | ".join(str(row[label]) for label in labels) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    output: Path,
    *,
    cells: dict[tuple[int, int], Cell],
    lowers: list[int],
    maximum: int,
    width: int,
    summary: dict[str, Any],
) -> None:
    matrix = matrix_rows(cells, lowers, maximum, width)
    labels = [bin_label(lower, maximum, width) for lower in lowers]
    long = long_rows(cells, lowers, maximum, width)
    write_csv(output / "target_vs_opponent_elo_unique_game_matrix.csv", matrix, ["targetEloBin", *labels])
    (output / "target_vs_opponent_elo_unique_game_matrix.md").write_text(
        markdown_matrix(matrix, labels), encoding="utf-8", newline="\n"
    )
    long_fields = list(long[0])
    write_csv(output / "target_vs_opponent_elo_capacity_long.csv", long, long_fields)
    write_json(output / "target_vs_opponent_elo_capacity_long.json", {
        "schema": "oq-target-vs-opponent-elo-capacity-long-v1",
        "rows": long,
    })
    write_json(output / "target_vs_opponent_elo_capacity_summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--leaderboard-json", type=Path, default=DEFAULT_LEADERBOARD_JSON)
    parser.add_argument("--leaderboard-csv", type=Path, default=DEFAULT_LEADERBOARD_CSV)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--minimum-elo", type=int, default=1600)
    parser.add_argument("--bin-width", type=int, default=100)
    args = parser.parse_args()

    details = args.game_details.resolve()
    leaderboard_json = args.leaderboard_json.resolve()
    leaderboard_csv = args.leaderboard_csv.resolve()
    snapshot = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    with leaderboard_csv.open("r", encoding="utf-8", newline="") as handle:
        leaderboard_rows = list(csv.DictReader(handle))
    ratings = [int(row["rating"]) for row in leaderboard_rows]
    minimum = args.minimum_elo
    maximum = max(ratings)
    if int(snapshot.get("cutoff_rating") or -1) != minimum:
        raise ValueError("leaderboard cutoff does not match --minimum-elo")
    if int(snapshot.get("count") or -1) != len(leaderboard_rows):
        raise ValueError("leaderboard JSON/CSV count mismatch")

    cache_counts = scan_cache_metadata(details)
    _, records, validation_counts = load_inputs(
        details,
        str(details.parent.parent),
        minimum=minimum,
        maximum=maximum,
        width=args.bin_width,
        seed=0,
    )
    if cache_counts["jsonlLines"] != validation_counts["jsonlLines"]:
        raise AssertionError("independent line counts disagree")
    cells, elo_counts = build_capacity(
        records, minimum=minimum, maximum=maximum, width=args.bin_width
    )
    lowers = bin_lowers(minimum, maximum, args.bin_width)
    long = long_rows(cells, lowers, maximum, args.bin_width)
    capacities = [row["uniqueGameCount"] for row in long]
    nonzero = [count for count in capacities if count]
    directed_total = sum(capacities)
    same_bin_games = sum(len(cells[(lower, lower)].game_ids) for lower in lowers)
    expected_directed = 2 * elo_counts["inRangeUniqueGames"] - same_bin_games
    if directed_total != expected_directed:
        raise AssertionError("directed total does not reconcile to cross-bin/same-bin semantics")

    script_path = Path(__file__).resolve()
    builder_path = script_path.with_name("build_elo_reference.py")
    puller_path = script_path.with_name("pull_oq_transformer_dataset.py")
    threshold_lists = {
        str(threshold): [
            {
                "targetBin": bin_label(row["targetBinLower"], maximum, args.bin_width),
                "opponentBin": bin_label(row["opponentBinLower"], maximum, args.bin_width),
                "uniqueGameCount": row["uniqueGameCount"],
            }
            for row in long if 0 < row["uniqueGameCount"] < threshold
        ]
        for threshold in (5, 10, 30, 40)
    }
    summary: dict[str, Any] = {
        "schema": "oq-target-vs-opponent-elo-capacity-summary-v1",
        "generatedAtUtc": utc_now(),
        "sources": {
            "gameDetailsJsonl": {"path": str(details), "sha256": sha256_file(details)},
            "leaderboardJson": {"path": str(leaderboard_json), "sha256": sha256_file(leaderboard_json)},
            "leaderboardCsv": {"path": str(leaderboard_csv), "sha256": sha256_file(leaderboard_csv)},
        },
        "leaderboard": {
            "fetchedAtUtc": snapshot.get("fetched_at"),
            "rowCount": len(leaderboard_rows),
            "minimumRating": min(ratings),
            "maximumRating": maximum,
        },
        "eloBins": {
            "minimumInclusive": minimum,
            "maximumInclusive": maximum,
            "width": args.bin_width,
            "labels": [bin_label(lower, maximum, args.bin_width) for lower in lowers],
        },
        "inputCounts": {
            **dict(sorted(cache_counts.items())),
            "deduplicatedValidUniqueGameIdCount": cache_counts["firstOccurrenceCacheMarkedValidGameIds"],
            "revalidatedLegalUniqueGameCount": len(records),
        },
        "qualityValidationCounts": dict(sorted(validation_counts.items())),
        "eloEligibilityCounts": {
            "excludedAnyOldRMissingOrInvalid": elo_counts["excludedAnyOldRMissingOrInvalid"],
            "excludedAnyOldRBelowMinimum": elo_counts["excludedAnyOldRBelowMinimum"],
            "excludedAnyOldRAboveMaximum": elo_counts["excludedAnyOldRAboveMaximum"],
            "excludedOutsideMainMatrixUnion": elo_counts["excludedOutsideMainMatrixUnion"],
            "inRangeUniqueGames": elo_counts["inRangeUniqueGames"],
        },
        "matrix": {
            "inRangeUniqueGameCount": elo_counts["inRangeUniqueGames"],
            "sameBinUniqueGameCount": same_bin_games,
            "crossBinUniqueGameCount": elo_counts["inRangeUniqueGames"] - same_bin_games,
            "directedCellCountTotal": directed_total,
            "directedTotalReconciliation": "2 * crossBinUniqueGameCount + sameBinUniqueGameCount",
            "combinationCount": len(capacities),
            "emptyCombinationCount": sum(count == 0 for count in capacities),
            "nonEmptyBelow5Count": len(threshold_lists["5"]),
            "nonEmptyBelow10Count": len(threshold_lists["10"]),
            "nonEmptyBelow30Count": len(threshold_lists["30"]),
            "nonEmptyBelow40Count": len(threshold_lists["40"]),
            "maximumUniqueGameCount": max(capacities),
            "minimumNonzeroUniqueGameCount": min(nonzero) if nonzero else 0,
            "medianAllCombinationUniqueGameCount": statistics.median(capacities),
            "medianNonzeroCombinationUniqueGameCount": statistics.median(nonzero) if nonzero else 0,
            "combinationsBelowThreshold": threshold_lists,
        },
        "rules": {
            "sourceScope": "full acquisition/game_details.jsonl; the 1,222-game Reference is not an input",
            "deduplication": "exactly load_inputs order: require non-empty game_id, keep first occurrence, reject later occurrences before cache/revalidation checks",
            "quality": "build_elo_reference.load_inputs using valid_detail and replay_is_legal, plus reversi, finished=true, distinct non-empty player IDs",
            "ratingField": "detail.players[].oldR only; both values must parse to finite numbers in the inclusive global range",
            "bins": "build_elo_reference bin_lowers/bin_for_rating/bin_upper; left-closed/right-open except final maximum inclusive",
            "direction": "cross-bin games enter two opposite directed cells; same-bin games enter one cell once",
            "sideCounts": "for same-bin games, black-target and white-target availability are both counted while uniqueGameCount remains one",
            "eloExclusionPredicates": "missing/invalid, below-minimum, and above-maximum counts are independent game-level predicates and may overlap",
        },
        "codeVersion": {
            "scriptPath": str(script_path),
            "scriptSha256": sha256_file(script_path),
            "referenceBuilderPath": str(builder_path),
            "referenceBuilderSha256": sha256_file(builder_path),
            "acquisitionScriptPath": str(puller_path),
            "acquisitionScriptSha256": sha256_file(puller_path),
            "repositoryRevision": None,
            "repositoryRevisionNote": "workspace root is not a Git worktree",
        },
    }

    output = unique_output_dir(args.output_parent.resolve(), args.output_name)
    output.mkdir(parents=True, exist_ok=False)
    write_outputs(
        output,
        cells=cells,
        lowers=lowers,
        maximum=maximum,
        width=args.bin_width,
        summary=summary,
    )
    print(json.dumps({"ok": True, "outputDir": str(output), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
