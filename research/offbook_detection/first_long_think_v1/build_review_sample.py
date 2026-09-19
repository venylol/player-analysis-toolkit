#!/usr/bin/env python3
"""Detect first-long-think anchors and build a stratified replay review sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.first_long_think_v1.core import (  # noqa: E402
    RuleConfig,
    detect_game_views,
    stratified_review_sample,
)
from research.offbook_detection.pull_oq_transformer_dataset import OthelloBoard  # noqa: E402


DEFAULT_DATA = ROOT / "research/offbook_detection/data/oq_transformer_61145_20260813"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_games(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = row["game_id"]
            if game_id in result:
                raise ValueError(f"duplicate game_id in games.csv: {game_id}")
            result[game_id] = row
    return result


def detect_all(
    nodes_path: Path,
    games: Mapping[str, Mapping[str, str]],
    config: RuleConfig,
    output_path: Path,
) -> list[dict[str, Any]]:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records: list[dict[str, Any]] = []
    seen_games: set[str] = set()

    with nodes_path.open("r", encoding="utf-8", newline="") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        current_game_id: str | None = None
        current_rows: list[dict[str, str]] = []

        def finish_game() -> None:
            if current_game_id is None:
                return
            if current_game_id in seen_games:
                raise ValueError(f"nodes.csv is not grouped by game_id: {current_game_id}")
            if current_game_id not in games:
                raise ValueError(f"nodes.csv game missing from games.csv: {current_game_id}")
            seen_games.add(current_game_id)
            for record in detect_game_views(games[current_game_id], current_rows, config):
                records.append(record)
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

        for row in csv.DictReader(source):
            game_id = row["game_id"]
            if current_game_id is not None and game_id != current_game_id:
                finish_game()
                current_rows = []
            current_game_id = game_id
            current_rows.append(row)
        finish_game()
        target.flush()
        os.fsync(target.fileno())

    if seen_games != set(games):
        missing = sorted(set(games) - seen_games)
        raise ValueError(f"games.csv entries missing from nodes.csv: {missing[:5]}")
    os.replace(temporary, output_path)
    return records


def load_selected_nodes(
    path: Path, selected_game_ids: set[str]
) -> dict[str, list[dict[str, str]]]:
    result = {game_id: [] for game_id in selected_game_ids}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["game_id"] in result:
                result[row["game_id"]].append(row)
    for game_id, rows in result.items():
        if not rows:
            raise ValueError(f"selected game missing from nodes.csv: {game_id}")
        rows.sort(key=lambda row: int(row["node_index"]))
    return result


def replay_game(
    evaluation: Mapping[str, Any],
    metadata: Mapping[str, str],
    nodes: Sequence[Mapping[str, str]],
    config: RuleConfig,
) -> dict[str, Any]:
    target_color = str(evaluation["target_color"])
    target_id = str(evaluation["target_id"])
    board = OthelloBoard()
    plies: list[dict[str, Any]] = []
    side_times = {"black": 0.0, "white": 0.0}
    side_moves = {"black": 0, "white": 0}
    target_decision = 0

    for expected_node, row in enumerate(nodes):
        node_index = int(row["node_index"])
        if node_index != expected_node:
            raise ValueError(f"non-contiguous source nodes in {metadata['game_id']}")
        if row["board_before"] != board.text():
            raise ValueError(f"board_before mismatch in {metadata['game_id']} node {node_index}")
        move = row["move"].strip().lower()
        color = row["actor_color"].strip().lower()
        expected_color = "black" if board.current == "X" else "white"
        if color != expected_color:
            raise ValueError(f"actor/board color mismatch in {metadata['game_id']} node {node_index}")
        is_pass = row["is_pass"] == "1"
        if is_pass != (move == "-"):
            raise ValueError(f"pass flag mismatch in {metadata['game_id']} node {node_index}")
        if not is_pass:
            strict_ply = int(row["strict_ply"])
            if strict_ply != len(plies) + 1:
                raise ValueError(f"strict ply mismatch in {metadata['game_id']} node {node_index}")
            is_target = color == target_color
            if is_target:
                target_decision += 1
            thinking_time = float(row["thinking_time_ms"])
            plies.append({
                "ply": strict_ply,
                "sourceMoveIndex": node_index,
                "move": move,
                "playerColor": color,
                "playerAccount": row["actor_id"],
                "isTargetMove": is_target,
                "targetDecisionNumber": target_decision if is_target else None,
                "thinkingTimeMs": thinking_time,
            })
            side_times[color] += thinking_time
            side_moves[color] += 1
        board.apply(move)

    anchor = evaluation["anchor"]
    has_anchor = evaluation["result"] == "anchor"
    if has_anchor:
        matching = [
            ply for ply in plies
            if ply["isTargetMove"] and ply["targetDecisionNumber"] == anchor["anchor_target_decision"]
        ]
        if (
            len(matching) != 1
            or matching[0]["ply"] != anchor["anchor_strict_ply"]
            or matching[0]["sourceMoveIndex"] != anchor["anchor_original_node_index"]
        ):
            raise ValueError(f"anchor/source mapping mismatch in {metadata['game_id']}/{target_color}")
        confidence = (
            f"time ratio={anchor['time_ratio']:.4f}; current={anchor['thinking_time_ms']:.0f} ms; "
            f"prior median={anchor['prior_time_median_ms']:.0f} ms"
            if anchor["time_ratio"] is not None
            else "prior median=0 ms and current thinking time > 0 ms"
        )
    else:
        confidence = (
            f"no target node exceeded {config.multiplier:g}x its prior median "
            f"within strict ply {config.min_ply}-{config.max_ply}"
        )

    initial_time = float(metadata["tcb"])
    return {
        "account": target_id,
        "gameId": metadata["game_id"],
        "created": metadata.get("created") or None,
        "targetColor": target_color,
        "opponentAccount": evaluation["opponent_id"],
        "actualMoveCount": len(plies),
        "targetMoveCount": target_decision,
        "sampleGroup": evaluation["sample_group"],
        "sideTimeSummary": {
            color: {
                "initialTimeLimitMs": initial_time,
                "summedThinkingTimeMs": side_times[color],
                "moveCount": side_moves[color],
                "timedMoveCount": side_moves[color],
            }
            for color in ("black", "white")
        },
        "manualReview": None,
        "algorithm": {
            "hasCandidate": has_anchor,
            "candidatePly": anchor["anchor_strict_ply"] if has_anchor else None,
            "candidateTargetDecision": anchor["anchor_target_decision"] if has_anchor else None,
            "confidence": confidence,
            "supportCount": None,
            "supportingScales": [],
            "supportingDecisions": None,
            "convergenceScore": anchor["time_ratio"] if has_anchor else None,
            "foldThreshold": config.multiplier,
            "scaleCandidateCount": None,
            "convergenceClusterCount": None,
        },
        "firstLongThink": {
            "result": evaluation["result"],
            "anchor": anchor,
            "rule": evaluation["rule"],
        },
        "reviewerMark": None,
        "plies": plies,
    }


def build_bundle(
    selected: Sequence[Mapping[str, Any]],
    bin_summary: Sequence[Mapping[str, int]],
    games: Mapping[str, Mapping[str, str]],
    selected_nodes: Mapping[str, Sequence[Mapping[str, str]]],
    config: RuleConfig,
    sources: Mapping[str, Mapping[str, str]],
    seed: int,
) -> dict[str, Any]:
    replay_games = [
        replay_game(row, games[row["game_id"]], selected_nodes[row["game_id"]], config)
        for row in selected
    ]
    replay_games.sort(key=lambda game: (
        game["algorithm"]["candidatePly"] is None,
        game["algorithm"]["candidatePly"] or 10_000,
        game["gameId"],
    ))
    return {
        "schema": "offbook-replay-review-bundle-v1",
        "generatedAt": utc_now(),
        "purpose": "Blind review of deterministic first-long-think median-ratio v1 anchors",
        "sampling": {
            "method": "10 contiguous strict-ply bins across 5-38, 3 deterministic pseudo-random distinct games per bin, plus 2 no-anchor games",
            "seed": seed,
            "manual_labels_used": False,
            "model_labels_used": False,
            "anchor_examples": 30,
            "no_anchor_examples": 2,
            "distinct_original_games": len({game["gameId"] for game in replay_games}),
            "ply_bins": list(bin_summary),
        },
        "counts": {
            "players": len({game["account"].casefold() for game in replay_games}),
            "playerGameEvaluations": len(replay_games),
            "algorithmCandidates": sum(game["algorithm"]["hasCandidate"] for game in replay_games),
            "noAnchor": sum(not game["algorithm"]["hasCandidate"] for game in replay_games),
        },
        "sources": dict(sources),
        "games": replay_games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--games-csv", type=Path, default=DEFAULT_DATA / "games.csv")
    parser.add_argument("--nodes-csv", type=Path, default=DEFAULT_DATA / "nodes.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    games_path = args.games_csv.resolve(strict=True)
    nodes_path = args.nodes_csv.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    with config_path.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    config = RuleConfig.from_mapping(config_payload)
    sampling = config_payload["sampling"]
    games = load_games(games_path)

    anchors_path = output_dir / "anchor_records.jsonl"
    records = detect_all(nodes_path, games, config, anchors_path)
    selected, bin_summary = stratified_review_sample(
        records,
        min_ply=config.min_ply,
        max_ply=config.max_ply,
        bin_count=int(sampling["anchor_ply_bins"]),
        anchors_per_bin=int(sampling["anchors_per_bin"]),
        no_anchor_count=int(sampling["no_anchor_examples"]),
        seed=int(sampling["seed"]),
    )
    selected_ids = {row["game_id"] for row in selected}
    selected_nodes = load_selected_nodes(nodes_path, selected_ids)

    sources = {
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "gamesCsv": {"path": str(games_path), "sha256": sha256_file(games_path)},
        "nodesCsv": {"path": str(nodes_path), "sha256": sha256_file(nodes_path)},
        "anchorRecords": {"path": str(anchors_path), "sha256": sha256_file(anchors_path)},
    }
    bundle = build_bundle(
        selected, bin_summary, games, selected_nodes, config, sources, int(sampling["seed"])
    )
    bundle_path = output_dir / "review_bundle_stratified30_plus2_no_anchor.json"
    atomic_json(bundle_path, bundle)

    result_counts = Counter(record["result"] for record in records)
    anchor_ply_counts = Counter(
        int(record["anchor"]["anchor_strict_ply"])
        for record in records if record["result"] == "anchor"
    )
    split_counts: dict[str, Counter[str]] = {}
    for record in records:
        split = str(record["split"])
        split_counts.setdefault(split, Counter())[record["result"]] += 1
    summary = {
        "schema": "first-long-think-median-ratio-summary-v1",
        "generatedAt": utc_now(),
        "rule": config_payload,
        "sources": sources,
        "counts": {
            "originalGames": len(games),
            "playerViews": len(records),
            "anchor": result_counts["anchor"],
            "noAnchor": result_counts["no_anchor"],
            "anchorRate": result_counts["anchor"] / len(records),
        },
        "anchorPlyCounts": {
            str(ply): anchor_ply_counts[ply] for ply in range(config.min_ply, config.max_ply + 1)
        },
        "splitCounts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "reviewSample": {
            "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
            "anchorExamples": 30,
            "noAnchorExamples": 2,
            "distinctOriginalGames": len(selected_ids),
            "plyBins": bin_summary,
            "selected": [
                {
                    "game_id": row["game_id"],
                    "target_color": row["target_color"],
                    "result": row["result"],
                    "anchor_strict_ply": row["anchor"]["anchor_strict_ply"] if row["anchor"] else None,
                    "sample_group": row["sample_group"],
                }
                for row in selected
            ],
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "review_bundle": str(bundle_path),
        "player_views": len(records),
        "anchor": result_counts["anchor"],
        "no_anchor": result_counts["no_anchor"],
        "review_examples": len(bundle["games"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
