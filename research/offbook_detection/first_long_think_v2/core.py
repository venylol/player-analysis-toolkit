from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RuleConfig:
    min_ply: int
    max_ply: int
    time_multiplier: float
    absolute_evaluation_threshold: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuleConfig":
        config = cls(
            min_ply=int(value["clip_min_strict_ply_inclusive"]),
            max_ply=int(value["cap_max_strict_ply_inclusive"]),
            time_multiplier=float(value["time_multiplier"]),
            absolute_evaluation_threshold=float(value["absolute_evaluation_threshold"]),
        )
        if config.min_ply < 1 or config.max_ply < config.min_ply:
            raise ValueError("invalid inclusive strict-ply range")
        if not math.isfinite(config.time_multiplier) or config.time_multiplier <= 1:
            raise ValueError("time multiplier must be finite and greater than one")
        if not math.isfinite(config.absolute_evaluation_threshold) or config.absolute_evaluation_threshold < 0:
            raise ValueError("absolute evaluation threshold must be finite and nonnegative")
        return config


def _recorded_colors(metadata: Mapping[str, str]) -> tuple[str, ...]:
    colors = tuple(part.strip().lower() for part in metadata["recorded_sides"].split("|") if part.strip())
    if not colors or any(color not in {"black", "white"} for color in colors) or len(set(colors)) != len(colors):
        raise ValueError(f"invalid recorded_sides in {metadata['game_id']}: {metadata['recorded_sides']!r}")
    return colors


def detect_game_views(
    metadata: Mapping[str, str],
    node_rows: Sequence[Mapping[str, str]],
    current_scores_by_ply: Mapping[int, float],
    config: RuleConfig,
) -> list[dict[str, Any]]:
    game_id = metadata["game_id"]
    player_ids = {"black": metadata["black_id"], "white": metadata["white_id"]}
    moves_by_color: dict[str, list[dict[str, Any]]] = {"black": [], "white": []}
    decisions = {"black": 0, "white": 0}
    actual_ply = 0

    for expected_move_index, row in enumerate(node_rows):
        if row["game_id"] != game_id:
            raise ValueError(f"mixed game rows while processing {game_id}")
        move_index = int(row["move_index"])
        if move_index != expected_move_index:
            raise ValueError(f"non-contiguous move_index in {game_id}: {move_index}")
        color = row["side_to_move"].strip().lower()
        if color not in moves_by_color:
            raise ValueError(f"invalid side_to_move in {game_id}: {color}")
        is_pass = row["is_pass_record"] == "1"
        if is_pass:
            if row["actual_move"].strip() != "-":
                raise ValueError(f"pass/move mismatch in {game_id} node {move_index}")
            continue

        # The retained source assigns pass-row player_id by source-list parity,
        # while side_to_move follows the board. Only actual placements carry an
        # authoritative player identity in this dataset.
        if row["player_id"].casefold() != player_ids[color].casefold():
            raise ValueError(f"player identity/color mismatch in {game_id} node {move_index}")

        actual_ply += 1
        strict_ply = int(row["global_placement_ply"])
        if strict_ply != actual_ply:
            raise ValueError(f"strict ply mismatch in {game_id} node {move_index}")
        thinking_time = float(row["actual_thinking_time_ms"])
        current_score = float(current_scores_by_ply[strict_ply])
        if not math.isfinite(thinking_time) or thinking_time < 0:
            raise ValueError(f"invalid thinking time in {game_id} node {move_index}")
        if not math.isfinite(current_score):
            raise ValueError(f"invalid current score in {game_id} ply {strict_ply}")
        decisions[color] += 1
        moves_by_color[color].append({
            "strict_ply": strict_ply,
            "target_decision": decisions[color],
            "original_node_index": move_index,
            "thinking_time_ms": thinking_time,
            "current_score": current_score,
        })

    if set(current_scores_by_ply) != set(range(1, actual_ply + 1)):
        raise ValueError(f"evaluation ply coverage mismatch in {game_id}")

    records: list[dict[str, Any]] = []
    for color in _recorded_colors(metadata):
        target_view = 0 if color == "black" else 1
        opponent_color = "white" if color == "black" else "black"
        target_moves = moves_by_color[color]
        cutoff_move = next((
            move for move in target_moves
            if config.min_ply <= move["strict_ply"] <= config.max_ply
            and abs(move["current_score"]) > config.absolute_evaluation_threshold
        ), None)
        evaluation_cutoff = None if cutoff_move is None else {
            "strict_ply": cutoff_move["strict_ply"],
            "target_decision": cutoff_move["target_decision"],
            "original_node_index": cutoff_move["original_node_index"],
            "current_score": cutoff_move["current_score"],
            "absolute_current_score": abs(cutoff_move["current_score"]),
            "absolute_threshold": config.absolute_evaluation_threshold,
            "comparison": ">",
        }

        history: list[float] = []
        time_anchor: dict[str, Any] | None = None
        cutoff_ply = cutoff_move["strict_ply"] if cutoff_move is not None else None
        for move in target_moves:
            ply = move["strict_ply"]
            eligible = (
                config.min_ply <= ply <= config.max_ply
                and (cutoff_ply is None or ply < cutoff_ply)
                and bool(history)
            )
            if eligible:
                prior_median = float(statistics.median(history))
                threshold = config.time_multiplier * prior_median
                if move["thinking_time_ms"] > threshold:
                    time_anchor = {
                        "anchor_source": "time_rule_before_evaluation_cutoff" if cutoff_ply is not None else "time_rule_without_evaluation_cutoff",
                        "anchor_strict_ply": ply,
                        "anchor_target_decision": move["target_decision"],
                        "anchor_original_node_index": move["original_node_index"],
                        "thinking_time_ms": move["thinking_time_ms"],
                        "prior_time_median_ms": prior_median,
                        "time_threshold_ms": threshold,
                        "time_ratio": None if prior_median == 0 else move["thinking_time_ms"] / prior_median,
                        "prior_observation_count": len(history),
                    }
                    break
            history.append(move["thinking_time_ms"])
            if cutoff_ply is not None and ply >= cutoff_ply:
                break

        if time_anchor is not None:
            anchor = time_anchor
        elif cutoff_move is not None:
            anchor = {
                "anchor_source": "absolute_evaluation_cutoff",
                "anchor_strict_ply": cutoff_move["strict_ply"],
                "anchor_target_decision": cutoff_move["target_decision"],
                "anchor_original_node_index": cutoff_move["original_node_index"],
                "current_score": cutoff_move["current_score"],
                "absolute_current_score": abs(cutoff_move["current_score"]),
                "absolute_evaluation_threshold": config.absolute_evaluation_threshold,
                "evaluation_comparison": ">",
            }
        else:
            anchor = None

        records.append({
            "schema": "first-long-think-absolute-evaluation-cutoff-record-v2",
            "game_id": game_id,
            "split": metadata.get("split") or None,
            "target_view": target_view,
            "target_color": color,
            "target_id": player_ids[color],
            "opponent_id": player_ids[opponent_color],
            "target_move_count": decisions[color],
            "result": "anchor" if anchor is not None else "no_anchor",
            "anchor": anchor,
            "evaluation_cutoff": evaluation_cutoff,
            "rule": {
                "clip_min_strict_ply_inclusive": config.min_ply,
                "cap_max_strict_ply_inclusive": config.max_ply,
                "time_multiplier": config.time_multiplier,
                "absolute_evaluation_threshold": config.absolute_evaluation_threshold,
                "evaluation_comparison": "abs(current_score) > threshold",
                "evaluation_uses_loss": False,
                "time_search_end": "strictly before evaluation cutoff, or through cap when cutoff is absent",
            },
        })
    return records
