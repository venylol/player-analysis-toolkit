#!/usr/bin/env python3
"""Keep a complete Level22 baseline and fill sparse unordered Elo matchups toward a cap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
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
DATA = ROOT / "research" / "offbook_detection" / "data"
SOURCE = DATA / "oq_transformer_100000_20260813"
BASE = DATA / "oq_elo100_reference_level22_1600plus_1260_20260814"
DEFAULT_OUTPUT_NAME = "oq_elo_matchup30_reference_level22_1600plus_20260814"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_rating(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def label(lower: int, maximum: int, width: int) -> str:
    upper = bin_upper(lower, maximum, width)
    return f"[{lower},{upper}]" if upper == maximum else f"[{lower},{upper})"


def archive_bin_for_rating(rating: float | None, minimum: int, maximum: int, width: int) -> int | None:
    """Use official bins in range and natural 100-point bins below the cutoff."""
    if rating is None or not math.isfinite(rating) or rating > maximum:
        return None
    if rating < minimum:
        return math.floor(rating / width) * width
    return bin_for_rating(rating, minimum, maximum, width)


def archive_label(lower: int, minimum: int, maximum: int, width: int) -> str:
    return f"[{lower},{lower + width})" if lower < minimum else label(lower, maximum, width)


def pair_for_detail(
    detail: dict[str, Any], minimum: int, maximum: int, width: int
) -> tuple[int, int] | None:
    players = detail.get("players") or []
    if len(players) != 2:
        return None
    ratings = [parse_rating(player.get("oldR")) for player in players]
    if any(rating is None for rating in ratings):
        return None
    black = bin_for_rating(ratings[0], minimum, maximum, width)  # type: ignore[arg-type]
    white = bin_for_rating(ratings[1], minimum, maximum, width)  # type: ignore[arg-type]
    return tuple(sorted((black, white))) if black is not None and white is not None else None


def archive_pair_for_detail(
    detail: dict[str, Any], minimum: int, maximum: int, width: int
) -> tuple[int, int] | None:
    players = detail.get("players") or []
    if len(players) != 2:
        return None
    ratings = [parse_rating(player.get("oldR")) for player in players]
    bins = [archive_bin_for_rating(rating, minimum, maximum, width) for rating in ratings]
    return tuple(sorted((bins[0], bins[1]))) if bins[0] is not None and bins[1] is not None else None


def stable_game_key(seed: int, pair: tuple[int, int], game_id: str) -> str:
    return hashlib.sha256(f"{seed}|{pair[0]}|{pair[1]}|{game_id}".encode("utf-8")).hexdigest()


def select_expansion(
    available_by_pair: dict[tuple[int, int], set[str]],
    baseline_by_pair: dict[tuple[int, int], set[str]],
    pairs: list[tuple[int, int]],
    *,
    target: int,
    seed: int,
) -> tuple[dict[tuple[int, int], list[str]], list[dict[str, Any]]]:
    selected: dict[tuple[int, int], list[str]] = {}
    coverage = []
    for pair in pairs:
        available = set(available_by_pair.get(pair, set()))
        baseline = set(baseline_by_pair.get(pair, set()))
        if not baseline <= available:
            raise ValueError(f"baseline contains games outside validated acquisition pair {pair}")
        floor = min(target, len(available))
        deficit = max(0, floor - len(baseline))
        candidates = sorted(
            available - baseline,
            key=lambda game_id: (stable_game_key(seed, pair, game_id), game_id),
        )
        if len(candidates) < deficit:
            raise ValueError(f"insufficient candidates for pair {pair}")
        added = candidates[:deficit]
        selected[pair] = added
        coverage.append({
            "pairLowerA": pair[0],
            "pairLowerB": pair[1],
            "availableUniqueGameCount": len(available),
            "fillTarget": floor,
            "baselineGameCount": len(baseline),
            "addedGameCount": len(added),
            "mergedGameCount": len(baseline) + len(added),
            "baselineAlreadyAtOrAboveTarget": len(baseline) >= floor,
            "capacityExhaustedBelowRequestedTarget": len(available) < target,
            "shortfallAfterExpansion": max(0, floor - len(baseline) - len(added)),
        })
    return selected, coverage


def selected_game_row(
    detail: dict[str, Any], source_kind: str, minimum: int, maximum: int, width: int
) -> dict[str, Any]:
    players = detail.get("players") or []
    ratings = [parse_rating(player.get("oldR")) for player in players]
    main_bins = [
        bin_for_rating(rating, minimum, maximum, width) if rating is not None else None
        for rating in ratings
    ]
    bins = [archive_bin_for_rating(rating, minimum, maximum, width) for rating in ratings]
    in_main = main_bins[0] is not None and main_bins[1] is not None
    in_low_extension = (
        not in_main and bins[0] is not None and bins[1] is not None
        and any(bin_lower < minimum for bin_lower in bins if bin_lower is not None)
    )
    unordered = tuple(sorted((bins[0], bins[1]))) if in_main else None
    if in_low_extension:
        unordered = tuple(sorted((bins[0], bins[1])))
    partition_scope = (
        "main_bilateral" if in_main else
        "baseline_low_elo_extension" if in_low_extension else
        "outside_unpartitioned"
    )
    return {
        "gameId": str(detail.get("id") or ""),
        "created": str(detail.get("created") or ""),
        "sourceKind": source_kind,
        "blackPlayerId": str(players[0].get("id") or "") if len(players) == 2 else "",
        "blackOldR": ratings[0] if len(ratings) == 2 else None,
        "blackBinLower": bins[0],
        "blackBinUpper": bin_upper(bins[0], maximum, width) if bins[0] is not None else None,
        "blackBinLabel": archive_label(bins[0], minimum, maximum, width) if bins[0] is not None else "outside_partition_range",
        "whitePlayerId": str(players[1].get("id") or "") if len(players) == 2 else "",
        "whiteOldR": ratings[1] if len(ratings) == 2 else None,
        "whiteBinLower": bins[1],
        "whiteBinUpper": bin_upper(bins[1], maximum, width) if bins[1] is not None else None,
        "whiteBinLabel": archive_label(bins[1], minimum, maximum, width) if bins[1] is not None else "outside_partition_range",
        "inMainMatrix": in_main,
        "partitionScope": partition_scope,
        "unorderedPartitionKey": (
            f"{archive_label(unordered[0], minimum, maximum, width)}__{archive_label(unordered[1], minimum, maximum, width)}"
            if unordered is not None else "outside_unpartitioned"
        ),
        "blackTargetDirectedPartition": (
            f"{archive_label(bins[0], minimum, maximum, width)}__vs__{archive_label(bins[1], minimum, maximum, width)}"
            if unordered is not None else "outside_unpartitioned"
        ),
        "whiteTargetDirectedPartition": (
            f"{archive_label(bins[1], minimum, maximum, width)}__vs__{archive_label(bins[0], minimum, maximum, width)}"
            if unordered is not None else "outside_unpartitioned"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-details", type=Path, default=SOURCE / "acquisition" / "game_details.jsonl")
    parser.add_argument("--leaderboard-json", type=Path, default=SOURCE / "leaderboard.json")
    parser.add_argument("--leaderboard-csv", type=Path, default=SOURCE / "leaderboard.csv")
    parser.add_argument("--base-bundle", type=Path, default=BASE / "selected_account_bundle.json")
    parser.add_argument("--base-engine-dir", type=Path, default=BASE / "engine_level22_attempt_2")
    parser.add_argument("--output-parent", type=Path, default=DATA)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--minimum-elo", type=int, default=1600)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--target-per-pair", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    output = args.output_parent.resolve() / args.output_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")

    details_path = args.game_details.resolve()
    leaderboard_json = args.leaderboard_json.resolve()
    leaderboard_csv = args.leaderboard_csv.resolve()
    base_bundle_path = args.base_bundle.resolve()
    base_engine_dir = args.base_engine_dir.resolve()
    snapshot = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    with leaderboard_csv.open("r", encoding="utf-8", newline="") as handle:
        leaderboard_rows = list(csv.DictReader(handle))
    maximum = max(int(row["rating"]) for row in leaderboard_rows)
    minimum = args.minimum_elo
    if int(snapshot.get("cutoff_rating") or -1) != minimum:
        raise ValueError("leaderboard cutoff mismatch")

    _, records, validation_counts = load_inputs(
        details_path,
        str(details_path.parent.parent),
        minimum=minimum,
        maximum=maximum,
        width=args.width,
        seed=args.seed,
    )
    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    base_details = base_bundle.get("details") if isinstance(base_bundle.get("details"), list) else []
    base_index = base_bundle.get("index") if isinstance(base_bundle.get("index"), list) else []
    if len(base_details) != 1222 or len(base_index) != len(base_details):
        raise ValueError("expected complete 1,222-game baseline bundle")
    base_ids = [str(detail.get("id") or "") for detail in base_details]
    if len(set(base_ids)) != len(base_ids) or any(not game_id for game_id in base_ids):
        raise ValueError("baseline game IDs are empty or duplicated")
    if any(game_id not in records for game_id in base_ids):
        raise ValueError("baseline includes a game absent from the validated acquisition cache")

    lowers = bin_lowers(minimum, maximum, args.width)
    pairs = [(first, second) for index, first in enumerate(lowers) for second in lowers[index:]]
    available_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    for game_id, (_, detail) in records.items():
        pair = pair_for_detail(detail, minimum, maximum, args.width)
        if pair is not None:
            available_by_pair[pair].add(game_id)
    baseline_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    for detail in base_details:
        pair = pair_for_detail(detail, minimum, maximum, args.width)
        if pair is not None:
            baseline_by_pair[pair].add(str(detail["id"]))

    added_by_pair, coverage = select_expansion(
        available_by_pair,
        baseline_by_pair,
        pairs,
        target=args.target_per_pair,
        seed=args.seed,
    )
    added_ids = [game_id for pair in pairs for game_id in added_by_pair[pair]]
    if len(added_ids) != 273 or len(set(added_ids)) != len(added_ids):
        raise ValueError(f"expected exactly 273 unique additions, got {len(added_ids)}")
    if set(added_ids) & set(base_ids):
        raise ValueError("expansion overlaps the baseline")

    added_index = [records[game_id][0] for game_id in added_ids]
    added_details = [records[game_id][1] for game_id in added_ids]
    merged_details = [*base_details, *added_details]
    merged_index = [*base_index, *added_index]
    merged_ids = [str(detail.get("id") or "") for detail in merged_details]
    if len(merged_ids) != 1495 or len(set(merged_ids)) != 1495:
        raise ValueError("merged bundle must contain 1,495 unique games")

    table_by_id = {
        str(detail.get("id") or ""): ordinal
        for ordinal, detail in enumerate(
            sorted(merged_details, key=lambda item: str(item.get("created") or "")), start=1
        )
    }
    rows = []
    for detail in merged_details:
        game_id = str(detail.get("id") or "")
        row = selected_game_row(
            detail,
            "baseline1222" if game_id in set(base_ids) else "matchup30Expansion",
            minimum,
            maximum,
            args.width,
        )
        row["bundleTable"] = table_by_id[game_id]
        row["expectedEngineFile"] = f"engine_level22/game_0_{table_by_id[game_id]}_{game_id}.json"
        rows.append(row)
    rows.sort(key=lambda row: int(row["bundleTable"]))
    row_by_id = {str(row["gameId"]): row for row in rows}

    for item in coverage:
        pair = (int(item["pairLowerA"]), int(item["pairLowerB"]))
        base_games = sorted(baseline_by_pair.get(pair, set()))
        added_games = sorted(added_by_pair.get(pair, []))
        item["pairLabelA"] = label(pair[0], maximum, args.width)
        item["pairLabelB"] = label(pair[1], maximum, args.width)
        item["partitionKey"] = f"{item['pairLabelA']}__{item['pairLabelB']}"
        item["partitionScope"] = "main_bilateral"
        item["targetPolicy"] = "fill_toward_30_or_full_library_capacity"
        item["baselineGameIds"] = base_games
        item["addedGameIds"] = added_games
        item["mergedGameIds"] = sorted([*base_games, *added_games])

    full_low_extension: dict[tuple[int, int], set[str]] = defaultdict(set)
    for game_id, (_, detail) in records.items():
        pair = archive_pair_for_detail(detail, minimum, maximum, args.width)
        if pair is not None and pair[0] < minimum:
            full_low_extension[pair].add(game_id)
    merged_low_extension: dict[tuple[int, int], set[str]] = defaultdict(set)
    for detail in merged_details:
        pair = archive_pair_for_detail(detail, minimum, maximum, args.width)
        if pair is not None and pair[0] < minimum:
            merged_low_extension[pair].add(str(detail["id"]))
    for pair in sorted(merged_low_extension):
        base_games = sorted(merged_low_extension[pair])
        coverage.append({
            "pairLowerA": pair[0],
            "pairLowerB": pair[1],
            "availableUniqueGameCount": len(full_low_extension.get(pair, set())),
            "fillTarget": len(base_games),
            "baselineGameCount": len(base_games),
            "addedGameCount": 0,
            "mergedGameCount": len(base_games),
            "baselineAlreadyAtOrAboveTarget": True,
            "capacityExhaustedBelowRequestedTarget": None,
            "shortfallAfterExpansion": 0,
            "pairLabelA": archive_label(pair[0], minimum, maximum, args.width),
            "pairLabelB": archive_label(pair[1], minimum, maximum, args.width),
            "partitionKey": (
                f"{archive_label(pair[0], minimum, maximum, args.width)}__"
                f"{archive_label(pair[1], minimum, maximum, args.width)}"
            ),
            "partitionScope": "baseline_low_elo_extension",
            "targetPolicy": "retain_baseline_only_no_acquisition_or_engine_fill",
            "baselineGameIds": base_games,
            "addedGameIds": [],
            "mergedGameIds": base_games,
        })

    directed = []
    directed_pairs = {(target, opponent) for target in lowers for opponent in lowers}
    for row in rows:
        if row["partitionScope"] == "baseline_low_elo_extension":
            directed_pairs.add((int(row["blackBinLower"]), int(row["whiteBinLower"])))
            directed_pairs.add((int(row["whiteBinLower"]), int(row["blackBinLower"])))
    for target, opponent in sorted(directed_pairs):
            partition_game_ids = sorted({
                str(row["gameId"])
                for row in rows
                if row["partitionScope"] != "outside_unpartitioned" and (
                    (row["blackBinLower"] == target and row["whiteBinLower"] == opponent)
                    or (row["whiteBinLower"] == target and row["blackBinLower"] == opponent)
                )
            })
            baseline_games = [gid for gid in partition_game_ids if row_by_id[gid]["sourceKind"] == "baseline1222"]
            added_games = [gid for gid in partition_game_ids if row_by_id[gid]["sourceKind"] == "matchup30Expansion"]
            directed.append({
                "targetBinLower": target,
                "targetBinUpper": target + args.width if target < minimum else bin_upper(target, maximum, args.width),
                "targetBinLabel": archive_label(target, minimum, maximum, args.width),
                "opponentBinLower": opponent,
                "opponentBinUpper": opponent + args.width if opponent < minimum else bin_upper(opponent, maximum, args.width),
                "opponentBinLabel": archive_label(opponent, minimum, maximum, args.width),
                "partitionKey": f"{archive_label(target, minimum, maximum, args.width)}__vs__{archive_label(opponent, minimum, maximum, args.width)}",
                "partitionScope": (
                    "baseline_low_elo_extension" if target < minimum or opponent < minimum
                    else "main_bilateral"
                ),
                "baselineGameCount": len(baseline_games),
                "addedGameCount": len(added_games),
                "mergedGameCount": len(partition_game_ids),
                "baselineGameIds": baseline_games,
                "addedGameIds": added_games,
                "mergedGameIds": partition_game_ids,
            })

    output.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema": "oq-account-bundle-elo-matchup30-expansion-v1",
        "account": "elo_matchup30_reference_multi_target",
        "fetchedAt": snapshot.get("fetched_at"),
        "selection": {
            "baselineBundle": str(base_bundle_path),
            "baselineGameCount": len(base_ids),
            "addedGameCount": len(added_ids),
            "mergedGameCount": len(merged_ids),
            "targetPerUnorderedPair": args.target_per_pair,
            "gameIds": merged_ids,
            "addedGameIds": added_ids,
        },
        "index": merged_index,
        "details": merged_details,
    }
    write_json(output / "selected_account_bundle.json", bundle)
    csv_fields = list(rows[0])
    write_csv(output / "selected_games_with_partitions.csv", rows, csv_fields)
    write_json(output / "selected_games_with_partitions.json", {
        "schema": "oq-matchup30-selected-games-partitions-v1", "games": rows,
    })
    added_rows = [row for row in rows if row["sourceKind"] == "matchup30Expansion"]
    write_csv(output / "added_games.csv", added_rows, csv_fields)
    write_json(output / "added_games.json", {
        "schema": "oq-matchup30-added-games-v1", "games": added_rows,
    })
    coverage_csv = [{key: value for key, value in item.items() if not key.endswith("GameIds")} for item in coverage]
    write_csv(output / "partitions_unordered.csv", coverage_csv, list(coverage_csv[0]))
    write_json(output / "partitions_unordered.json", {
        "schema": "oq-matchup30-unordered-partitions-v1", "partitions": coverage,
    })
    directed_csv = [{key: value for key, value in item.items() if not key.endswith("GameIds")} for item in directed]
    write_csv(output / "partitions_directed.csv", directed_csv, list(directed_csv[0]))
    write_json(output / "partitions_directed.json", {
        "schema": "oq-matchup30-directed-partitions-v1", "partitions": directed,
    })

    manifest = {
        "schema": "oq-matchup30-level22-expansion-manifest-v1",
        "createdAtUtc": utc_now(),
        "sources": {
            "gameDetailsJsonl": {"path": str(details_path), "sha256": sha256_file(details_path)},
            "leaderboardJson": {"path": str(leaderboard_json), "sha256": sha256_file(leaderboard_json)},
            "leaderboardCsv": {"path": str(leaderboard_csv), "sha256": sha256_file(leaderboard_csv)},
            "baselineBundle": {"path": str(base_bundle_path), "sha256": sha256_file(base_bundle_path)},
            "baselineEngineAudit": {
                "path": str(base_engine_dir / "audit.json"),
                "sha256": sha256_file(base_engine_dir / "audit.json"),
            },
        },
        "leaderboard": {
            "fetchedAtUtc": snapshot.get("fetched_at"),
            "minimumElo": minimum,
            "maximumEloInclusive": maximum,
            "binWidth": args.width,
        },
        "selection": {
            "seed": args.seed,
            "targetPerUnorderedPair": args.target_per_pair,
            "baselinePolicy": "retain all 1,222 baseline games without clipping any partition",
            "fillPolicy": "for each unordered in-range Elo pair, add max(0, min(30, full capacity) - baseline count) games",
            "tieOrder": "SHA-256(seed|lowerPairA|lowerPairB|gameId), then gameId",
            "crossBinDirectionPolicy": "one engine-analyzed game covers both opposite directed target/opponent partitions",
            "outsideRangePolicy": "retain baseline games outside the bilateral main matrix; do not add new outside-range games",
        },
        "counts": {
            "validatedAcquisitionUniqueGames": validation_counts["validUniqueGames"],
            "baselineGames": len(base_ids),
            "baselineInMainMatrix": sum(len(ids) for ids in baseline_by_pair.values()),
            "baselineOutsideMainMatrix": len(base_ids) - sum(len(ids) for ids in baseline_by_pair.values()),
            "addedGames": len(added_ids),
            "mergedGames": len(merged_ids),
            "mergedInMainMatrix": sum(int(row["inMainMatrix"]) for row in rows),
            "mergedOutsideMainMatrix": sum(not bool(row["inMainMatrix"]) for row in rows),
            "mergedLowEloExtensionGames": sum(row["partitionScope"] == "baseline_low_elo_extension" for row in rows),
            "mergedOutsideUnpartitioned": sum(row["partitionScope"] == "outside_unpartitioned" for row in rows),
            "unorderedPartitions": len(coverage),
            "directedPartitions": len(directed),
            "mainUnorderedPartitions": sum(item["partitionScope"] == "main_bilateral" for item in coverage),
            "lowEloExtensionUnorderedPartitions": sum(item["partitionScope"] == "baseline_low_elo_extension" for item in coverage),
            "emptyFullLibraryMainPartitions": sum(
                item["partitionScope"] == "main_bilateral" and item["availableUniqueGameCount"] == 0
                for item in coverage
            ),
            "mainPartitionsReceivingAdditions": sum(item["addedGameCount"] > 0 for item in coverage),
            "postExpansionMainShortfall": sum(
                item["shortfallAfterExpansion"] for item in coverage if item["partitionScope"] == "main_bilateral"
            ),
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    write_json(output / "expansion_manifest.json", manifest)
    print(json.dumps({"ok": True, "outputDir": str(output), "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
