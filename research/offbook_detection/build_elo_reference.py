#!/usr/bin/env python3
"""Build a reproducible, audited Elo-stratified OQ reference bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from pull_oq_transformer_dataset import OthelloBoard, valid_detail


MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)
SIDES = ("black", "white")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bin_lowers(minimum: int, maximum: int, width: int) -> list[int]:
    return list(range(minimum, maximum + 1, width))


def bin_for_rating(rating: float, minimum: int, maximum: int, width: int) -> int | None:
    if not math.isfinite(rating) or rating < minimum or rating > maximum:
        return None
    return minimum + int((rating - minimum) // width) * width


def bin_upper(lower: int, maximum: int, width: int) -> int:
    return maximum if lower + width > maximum else lower + width


def stable_key(seed: int, lower: int, game_id: str, target_id: str, side: str) -> str:
    raw = f"{seed}|{lower}|{game_id}|{target_id.casefold()}|{side}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Candidate:
    game_id: str
    created: str
    lower: int
    upper: int
    target_id: str
    target_side: str
    target_old_r: float
    opponent_id: str
    opponent_old_r: float
    source_dataset: str
    tie_key: str


def replay_is_legal(detail: dict[str, Any]) -> bool:
    board = OthelloBoard()
    try:
        for event in (detail.get("position") or {}).get("moves") or []:
            board.apply(str(event.get("m") or "").lower())
    except (TypeError, ValueError):
        return False
    return True


def choose_candidates(
    candidates_by_bin: dict[int, list[Candidate]],
    lowers: list[int],
    *,
    target_per_bin: int,
    initial_used_games: set[str] | None = None,
    initial_global_players: Counter[str] | None = None,
) -> tuple[list[tuple[Candidate, str, bool]], dict[int, Counter[str]]]:
    """Select scarce bins first, preserving game uniqueness and side balance."""
    selected: list[tuple[Candidate, str, bool]] = []
    used_games: set[str] = set(initial_used_games or ())
    global_players: Counter[str] = Counter(initial_global_players or {})
    counts_by_bin: dict[int, Counter[str]] = {lower: Counter() for lower in lowers}
    selected_players_by_bin: dict[int, set[str]] = {lower: set() for lower in lowers}
    side_cap = target_per_bin // 2

    for lower in sorted(lowers, reverse=True):
        pool = sorted(candidates_by_bin.get(lower, []), key=lambda item: (item.tie_key, item.game_id))
        side_counts: Counter[str] = Counter()

        def available(phase: str, side: str) -> list[Candidate]:
            result = []
            for candidate in pool:
                player = candidate.target_id.casefold()
                if candidate.target_side != side or candidate.game_id in used_games:
                    continue
                if phase == "global_unique" and global_players[player]:
                    continue
                if phase in {"global_unique", "bin_unique"} and player in selected_players_by_bin[lower]:
                    continue
                result.append(candidate)
            return result

        for phase in ("global_unique", "bin_unique", "round_robin_repeat"):
            while sum(side_counts.values()) < target_per_bin:
                eligible_sides = [side for side in SIDES if side_counts[side] < side_cap]
                eligible_sides.sort(key=lambda side: (side_counts[side], SIDES.index(side)))
                picked: Candidate | None = None
                for side in eligible_sides:
                    options = available(phase, side)
                    if not options:
                        continue
                    if phase == "round_robin_repeat":
                        options.sort(key=lambda item: (
                            counts_by_bin[lower][item.target_id.casefold()],
                            global_players[item.target_id.casefold()],
                            item.tie_key,
                            item.game_id,
                        ))
                    picked = options[0]
                    break
                if picked is None:
                    break
                player = picked.target_id.casefold()
                repeated = bool(global_players[player])
                selected.append((picked, phase, repeated))
                used_games.add(picked.game_id)
                global_players[player] += 1
                counts_by_bin[lower][player] += 1
                selected_players_by_bin[lower].add(player)
                side_counts[picked.target_side] += 1
    return selected, counts_by_bin


def load_inputs(
    detail_path: Path,
    source_dataset: str,
    *,
    minimum: int,
    maximum: int,
    width: int,
    seed: int,
) -> tuple[dict[int, list[Candidate]], dict[str, tuple[dict[str, Any], dict[str, Any]]], Counter[str]]:
    candidates: dict[int, list[Candidate]] = defaultdict(list)
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with detail_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            counts["jsonlLines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid UTF-8 JSONL at {detail_path}:{line_number}") from exc
            game_id = str(row.get("game_id") or "").strip()
            if not game_id:
                counts["excludedMissingGameId"] += 1
                continue
            if game_id in seen:
                counts["excludedDuplicateJsonlGameId"] += 1
                continue
            seen.add(game_id)
            if not row.get("valid"):
                counts[f"excludedCacheMarkedInvalid:{row.get('reason') or 'unknown'}"] += 1
                continue
            summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
            detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
            ok, reason = valid_detail(summary, detail)
            if not ok:
                counts[f"excludedRevalidation:{reason}"] += 1
                continue
            if str(detail.get("gtype") or "") != "reversi":
                counts["excludedWrongGameType"] += 1
                continue
            if detail.get("finished") is not True:
                counts["excludedNotFinished"] += 1
                continue
            players = detail.get("players") or []
            player_ids = [str(player.get("id") or "").strip() for player in players]
            if len(players) != 2 or not all(player_ids) or player_ids[0].casefold() == player_ids[1].casefold():
                counts["excludedInvalidPlayerIds"] += 1
                continue
            if not replay_is_legal(detail):
                counts["excludedIllegalReplay"] += 1
                continue
            counts["validUniqueGames"] += 1
            records[game_id] = (summary, detail)
            for side_index, side in enumerate(SIDES):
                target, opponent = players[side_index], players[1 - side_index]
                try:
                    target_rating = float(target["oldR"])
                    opponent_rating = float(opponent["oldR"])
                except (KeyError, TypeError, ValueError):
                    counts["excludedTargetSideMissingOldR"] += 1
                    continue
                lower = bin_for_rating(target_rating, minimum, maximum, width)
                if lower is None:
                    counts["excludedTargetSideOutsideEloRange"] += 1
                    continue
                target_id = str(target["id"])
                candidates[lower].append(Candidate(
                    game_id=game_id,
                    created=str(detail.get("created") or summary.get("created") or ""),
                    lower=lower,
                    upper=bin_upper(lower, maximum, width),
                    target_id=target_id,
                    target_side=side,
                    target_old_r=target_rating,
                    opponent_id=str(opponent["id"]),
                    opponent_old_r=opponent_rating,
                    source_dataset=source_dataset,
                    tie_key=stable_key(seed, lower, game_id, target_id, side),
                ))
                counts["validTargetSideCandidates"] += 1
    return candidates, records, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-csv", type=Path, required=True)
    parser.add_argument("--leaderboard-json", type=Path, required=True)
    parser.add_argument("--game-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-elo", type=int, default=1600)
    parser.add_argument("--bin-width", type=int, default=100)
    parser.add_argument("--per-bin", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--base-reference-dir",
        type=Path,
        help="preserve an existing reference and add --per-bin new games to each Elo bin",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    leaderboard_csv = args.leaderboard_csv.resolve()
    leaderboard_json = args.leaderboard_json.resolve()
    detail_path = args.game_details.resolve()
    snapshot = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    with leaderboard_csv.open("r", encoding="utf-8", newline="") as handle:
        leaderboard_rows = list(csv.DictReader(handle))
    ratings = [int(row["rating"]) for row in leaderboard_rows]
    maximum = max(ratings)
    minimum = args.minimum_elo
    if int(snapshot.get("cutoff_rating") or -1) != minimum:
        raise ValueError("leaderboard cutoff does not match --minimum-elo")
    if int(snapshot.get("count") or -1) != len(leaderboard_rows):
        raise ValueError("leaderboard JSON/CSV row count mismatch")
    lowers = bin_lowers(minimum, maximum, args.bin_width)
    source_dataset = str(detail_path.parent.parent)
    candidates, records, exclusion_counts = load_inputs(
        detail_path, source_dataset, minimum=minimum, maximum=maximum,
        width=args.bin_width, seed=args.seed,
    )
    base_rows: list[dict[str, Any]] = []
    base_source: dict[str, Any] | None = None
    if args.base_reference_dir is not None:
        base_dir = args.base_reference_dir.resolve()
        base_selection_path = base_dir / "selected_games.json"
        base_payload = json.loads(base_selection_path.read_text(encoding="utf-8"))
        base_rows = base_payload.get("games") if isinstance(base_payload.get("games"), list) else []
        if not base_rows:
            raise ValueError("base reference has no selected games")
        if any(str(row.get("gameId") or "") not in records for row in base_rows):
            raise ValueError("base reference contains games absent from the detail cache")
        base_source = {
            "directory": str(base_dir),
            "selectedGames": str(base_selection_path),
            "selectedGamesSha256": sha256_file(base_selection_path),
        }
    base_game_ids = {str(row["gameId"]) for row in base_rows}
    base_global_players = Counter(str(row["targetPlayerId"]).casefold() for row in base_rows)
    selected, _ = choose_candidates(
        candidates, lowers, target_per_bin=args.per_bin,
        initial_used_games=base_game_ids,
        initial_global_players=base_global_players,
    )
    selected.sort(key=lambda value: (value[0].lower, value[0].target_side, value[0].tie_key))

    added_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set(base_game_ids)
    for candidate, phase, repeated in selected:
        if candidate.game_id in selected_ids:
            raise AssertionError(f"duplicate selected game: {candidate.game_id}")
        selected_ids.add(candidate.game_id)
        added_rows.append({
            "gameId": candidate.game_id,
            "created": candidate.created,
            "binLower": candidate.lower,
            "binUpper": candidate.upper,
            "targetPlayerId": candidate.target_id,
            "targetSide": candidate.target_side,
            "targetOldR": candidate.target_old_r,
            "opponentPlayerId": candidate.opponent_id,
            "opponentOldR": candidate.opponent_old_r,
            "sourceDataset": candidate.source_dataset,
            "targetAccountRepeatRelaxed": repeated,
            "selectionPhase": phase,
        })

    selected_rows = [dict(row) for row in base_rows] + added_rows
    selected_rows.sort(key=lambda row: (
        int(row["binLower"]), str(row["targetSide"]), str(row["gameId"])
    ))
    coverage: list[dict[str, Any]] = []
    for lower in lowers:
        rows = [row for row in selected_rows if row["binLower"] == lower]
        account_counts = Counter(str(row["targetPlayerId"]).casefold() for row in rows)
        base_bin_count = sum(int(row["binLower"]) == lower for row in base_rows)
        target_count = base_bin_count + args.per_bin
        repeated_accounts = sorted(
            ({"targetPlayerId": player, "gameCount": count} for player, count in account_counts.items() if count > 1),
            key=lambda item: (-item["gameCount"], item["targetPlayerId"]),
        )
        coverage.append({
            "binLower": lower,
            "binUpper": bin_upper(lower, maximum, args.bin_width),
            "targetGameCount": target_count,
            "actualGameCount": len(rows),
            "shortfall": target_count - len(rows),
            "blackGameCount": sum(row["targetSide"] == "black" for row in rows),
            "whiteGameCount": sum(row["targetSide"] == "white" for row in rows),
            "distinctTargetPlayerCount": len(account_counts),
            "repeatedTargetAccountCount": len(repeated_accounts),
            "maxGamesFromSingleAccount": max(account_counts.values(), default=0),
            "maxAccountShare": max(account_counts.values(), default=0) / len(rows) if rows else 0.0,
            "validCandidateCount": sum(
                candidate.game_id not in base_game_ids for candidate in candidates.get(lower, [])
            ),
            "repeatedAccounts": repeated_accounts,
        })

    fields = list(selected_rows[0]) if selected_rows else [
        "gameId", "created", "binLower", "binUpper", "targetPlayerId", "targetSide",
        "targetOldR", "opponentPlayerId", "opponentOldR", "sourceDataset",
        "targetAccountRepeatRelaxed", "selectionPhase",
    ]
    write_csv(output / "selected_games.csv", selected_rows, fields)
    write_json(output / "selected_games.json", {
        "schema": "oq-elo100-reference-selected-games-v1", "games": selected_rows,
    })
    coverage_csv = [{key: value for key, value in row.items() if key != "repeatedAccounts"} for row in coverage]
    write_csv(output / "coverage_by_elo.csv", coverage_csv, list(coverage_csv[0]))
    write_json(output / "coverage_by_elo.json", {
        "schema": "oq-elo100-reference-coverage-v1", "bins": coverage,
    })

    bundle_details = []
    bundle_index = []
    for row in selected_rows:
        summary, detail = records[row["gameId"]]
        bundle_index.append(summary)
        bundle_details.append(detail)
    bundle = {
        "schema": "oq-account-bundle-elo-reference-v1",
        "account": "elo100_reference_multi_target",
        "fetchedAt": snapshot.get("fetched_at"),
        "selection": {
            "gameIds": [row["gameId"] for row in selected_rows],
            "targetByGameId": {row["gameId"]: {
                "playerId": row["targetPlayerId"], "side": row["targetSide"],
                "oldR": row["targetOldR"], "binLower": row["binLower"], "binUpper": row["binUpper"],
            } for row in selected_rows},
        },
        "index": bundle_index,
        "details": bundle_details,
    }
    write_json(output / "selected_account_bundle.json", bundle)

    global_counts = Counter(row["targetPlayerId"].casefold() for row in selected_rows)
    manifest = {
        "schema": "oq-elo100-reference-source-manifest-v1",
        "createdAtUtc": utc_now(),
        "sources": {
            "leaderboardCsv": {"path": str(leaderboard_csv), "sha256": sha256_file(leaderboard_csv)},
            "leaderboardJson": {"path": str(leaderboard_json), "sha256": sha256_file(leaderboard_json)},
            "gameDetailsJsonl": {"path": str(detail_path), "sha256": sha256_file(detail_path)},
        },
        "leaderboard": {
            "fetchedAtUtc": snapshot.get("fetched_at"), "rowCount": len(leaderboard_rows),
            "uniquePlayerIdCount": len({row["id"].casefold() for row in leaderboard_rows}),
            "minimumRating": min(ratings), "maximumRating": maximum,
        },
        "sampling": {
            "minimumElo": minimum, "maximumEloInclusive": maximum, "binWidth": args.bin_width,
            "additionalGamesPerBin": args.per_bin if base_rows else None,
            "perBinMaximum": None if base_rows else args.per_bin, "seed": args.seed,
            "targetEligibility": "either side with finite oldR in [minimumElo, maximumEloInclusive]",
            "binIntervals": "left-closed/right-open except the final truncated bin includes maximumEloInclusive",
            "gameDeduplication": "first valid JSONL occurrence per gameId; selected gameId globally unique",
            "ordering": "process Elo bins from highest/scarcest to lowest; within ties use SHA-256(seed|binLower|gameId|targetPlayerId.casefold()|side), then gameId",
            "selectionPriority": [
                "fill each bin to at most 40 with exactly 20 black and 20 white when full",
                "global_unique: target accounts not previously selected anywhere",
                "bin_unique: allow global reuse but still at most one game for that account in this bin",
                "round_robin_repeat: minimize per-bin account contribution, then global contribution, then seeded stable tie order",
            ],
            "quality": "existing valid_detail contract plus reversi, finished=true, non-empty distinct player IDs, finite oldR, and full legal replay",
        },
        "inputAndExclusionCounts": dict(sorted(exclusion_counts.items())),
        "baseReference": base_source,
        "result": {
            "gameCount": len(selected_rows), "addedGameCount": len(added_rows),
            "distinctTargetPlayerCount": len(global_counts),
            "repeatedTargetAccounts": [
                {"targetPlayerId": player, "gameCount": count}
                for player, count in sorted(global_counts.items(), key=lambda item: (-item[1], item[0])) if count > 1
            ],
            "maxGamesFromSingleAccount": max(global_counts.values(), default=0),
            "maxAccountShare": max(global_counts.values(), default=0) / len(selected_rows) if selected_rows else 0.0,
        },
        "outputFiles": {},
    }
    for name in ("selected_games.csv", "selected_games.json", "coverage_by_elo.csv", "coverage_by_elo.json", "selected_account_bundle.json"):
        path = output / name
        manifest["outputFiles"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output / "reference_source_manifest.json", manifest)
    print(json.dumps({"ok": True, "output": str(output), **manifest["result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
