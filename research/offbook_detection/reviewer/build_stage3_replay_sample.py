#!/usr/bin/env python3
"""Build a stratified 30-example replay bundle from frozen stage-3 posteriors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.pull_oq_transformer_dataset import OthelloBoard  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_records(path: Path) -> list[dict[str, Any]]:
    by_decision: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["result"] != "change_point":
                continue
            decision = int(record["map_target_decision"])
            if 5 <= decision <= 19:
                by_decision[decision].append(record)
    selected: list[dict[str, Any]] = []
    used_games: set[str] = set()
    target_counts = {decision: 2 for decision in range(5, 20)}
    target_counts[5] = 1
    target_counts[6] = 3
    for decision in range(5, 20):
        ranked = sorted(by_decision[decision], key=lambda row: (-float(row["map_probability"]), row["game_id"], row["target_view"]))
        choices = [row for row in ranked if row["game_id"] not in used_games][:target_counts[decision]]
        if len(choices) != target_counts[decision]:
            raise ValueError(f"not enough distinct games for MAP decision {decision}")
        selected.extend(choices)
        used_games.update(row["game_id"] for row in choices)
    if len(selected) != 30 or len(used_games) != 30:
        raise AssertionError("stratified selection must contain 30 distinct games")
    return selected


def load_games(path: Path, selected_ids: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["game_id"] in selected_ids:
                result[row["game_id"]] = row
    if set(result) != selected_ids:
        raise ValueError(f"games.csv lacks selected ids: {sorted(selected_ids - set(result))}")
    return result


def load_nodes(path: Path, selected_ids: set[str]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {game_id: [] for game_id in selected_ids}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["game_id"] in result:
                result[row["game_id"]].append(row)
    for game_id, rows in result.items():
        rows.sort(key=lambda row: int(row["node_index"]))
        if not rows:
            raise ValueError(f"nodes.csv lacks selected game: {game_id}")
    return result


def replay_game(
    posterior: dict[str, Any], metadata: dict[str, str], nodes: list[dict[str, str]]
) -> dict[str, Any]:
    target_view = int(posterior["target_view"])
    target_color = "black" if target_view == 0 else "white"
    target_id = metadata["black_id"] if target_view == 0 else metadata["white_id"]
    opponent_id = metadata["white_id"] if target_view == 0 else metadata["black_id"]
    board = OthelloBoard()
    plies: list[dict[str, Any]] = []
    target_decision = 0
    side_times = {"black": 0.0, "white": 0.0}
    side_moves = {"black": 0, "white": 0}
    for expected_node, row in enumerate(nodes):
        if int(row["node_index"]) != expected_node:
            raise ValueError(f"non-contiguous source nodes in {metadata['game_id']}")
        move = row["move"].strip().lower()
        side = row["actor_color"]
        is_pass = row["is_pass"] == "1"
        if is_pass != (move == "-"):
            raise ValueError(f"pass flag mismatch in {metadata['game_id']} node {expected_node}")
        if not is_pass:
            is_target = row["actor_id"].casefold() == target_id.casefold()
            if is_target:
                target_decision += 1
            strict_ply = int(row["strict_ply"])
            if strict_ply != len(plies) + 1:
                raise ValueError(f"strict ply mismatch in {metadata['game_id']}")
            thinking = float(row["thinking_time_ms"])
            plies.append({
                "ply": strict_ply, "sourceMoveIndex": expected_node, "move": move,
                "playerColor": side, "playerAccount": row["actor_id"], "isTargetMove": is_target,
                "targetDecisionNumber": target_decision if is_target else None, "thinkingTimeMs": thinking,
            })
            side_times[side] += thinking
            side_moves[side] += 1
        board.apply(move)
    map_ply = int(posterior["map_strict_ply"])
    map_decision = int(posterior["map_target_decision"])
    matches = [ply for ply in plies if ply["isTargetMove"] and ply["targetDecisionNumber"] == map_decision]
    if len(matches) != 1 or matches[0]["ply"] != map_ply or matches[0]["sourceMoveIndex"] != posterior["map_original_node_index"]:
        raise ValueError(f"posterior/source mapping mismatch for {metadata['game_id']}/{target_id}")
    candidate_text = "; ".join(
        f"d{item['target_decision']}=ply{item['strict_ply']}:P{float(item['probability']):.4f}"
        for item in posterior["candidates"]
    )
    return {
        "account": target_id, "gameId": metadata["game_id"], "created": metadata.get("created") or None,
        "targetColor": target_color, "opponentAccount": opponent_id,
        "actualMoveCount": len(plies), "targetMoveCount": target_decision,
        "sideTimeSummary": {
            color: {
                "initialTimeLimitMs": 300000.0, "summedThinkingTimeMs": side_times[color],
                "moveCount": side_moves[color], "timedMoveCount": side_moves[color],
            }
            for color in ("black", "white")
        },
        "manualReview": None,
        "algorithm": {
            "hasCandidate": True, "candidatePly": map_ply,
            "candidateTargetDecision": map_decision,
            "confidence": f"MAP P={float(posterior['map_probability']):.4f}",
            "supportCount": len(posterior["candidates"]), "supportingScales": [],
            "supportingDecisions": candidate_text,
            "convergenceScore": float(posterior["map_probability"]), "foldThreshold": None,
            "scaleCandidateCount": len(posterior["candidates"]), "convergenceClusterCount": None,
        },
        "stage3Posterior": {
            "pNoChange": float(posterior["p_no_change"]),
            "mapProbability": float(posterior["map_probability"]),
            "posteriorEntropy": float(posterior["posterior_entropy"]),
            "candidates": posterior["candidates"],
        },
        "reviewerMark": None, "plies": plies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posteriors", type=Path, required=True)
    parser.add_argument("--games-csv", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    posterior_path = args.posteriors.resolve(strict=True)
    selected = choose_records(posterior_path)
    selected_ids = {row["game_id"] for row in selected}
    games = load_games(args.games_csv.resolve(strict=True), selected_ids)
    nodes = load_nodes(args.nodes_csv.resolve(strict=True), selected_ids)
    replay_games = [replay_game(row, games[row["game_id"]], nodes[row["game_id"]]) for row in selected]
    replay_games.sort(key=lambda row: (row["algorithm"]["candidateTargetDecision"], -row["algorithm"]["convergenceScore"]))
    selection_path = args.selection.resolve(strict=True)
    payload = {
        "schema": "offbook-replay-review-bundle-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Blind replay review of 30 frozen stage-3 change-point examples",
        "sampling": {
            "method": "highest-MAP-posterior distinct games stratified across decisions 5-19; counts are d5=1, d6=3, all others=2",
            "manual_labels_used": False, "examples": 30, "distinct_original_games": 30,
        },
        "counts": {"players": len({game["account"].casefold() for game in replay_games}), "playerGameEvaluations": 30, "algorithmCandidates": 30},
        "sources": {
            "stage3Posteriors": {"path": str(posterior_path), "sha256": sha256_file(posterior_path)},
            "modelSelection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "gamesCsv": {"path": str(args.games_csv.resolve()), "sha256": sha256_file(args.games_csv.resolve())},
            "nodesCsv": {"path": str(args.nodes_csv.resolve()), "sha256": sha256_file(args.nodes_csv.resolve())},
        },
        "games": replay_games,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": "complete", "output": str(output), "sha256": sha256_file(output),
        "examples": len(replay_games),
        "MAP_decision_counts": {str(k): sum(game["algorithm"]["candidateTargetDecision"] == k for game in replay_games) for k in range(5, 20)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
