from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RuleConfig:
    min_ply: int
    max_ply: int
    multiplier: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuleConfig":
        config = cls(
            min_ply=int(value["clip_min_strict_ply_inclusive"]),
            max_ply=int(value["cap_max_strict_ply_inclusive"]),
            multiplier=float(value["time_multiplier"]),
        )
        if config.min_ply < 1 or config.max_ply < config.min_ply:
            raise ValueError("invalid inclusive strict-ply range")
        if not math.isfinite(config.multiplier) or config.multiplier <= 1:
            raise ValueError("time multiplier must be finite and greater than one")
        return config


def detect_game_views(
    metadata: Mapping[str, str],
    node_rows: Sequence[Mapping[str, str]],
    config: RuleConfig,
) -> list[dict[str, Any]]:
    game_id = metadata["game_id"]
    player_ids = {"black": metadata["black_id"], "white": metadata["white_id"]}
    histories: dict[str, list[float]] = {"black": [], "white": []}
    target_decisions = {"black": 0, "white": 0}
    anchors: dict[str, dict[str, Any] | None] = {"black": None, "white": None}
    actual_ply = 0

    for expected_node_index, row in enumerate(node_rows):
        if row["game_id"] != game_id:
            raise ValueError(f"mixed game rows while processing {game_id}")
        node_index = int(row["node_index"])
        if node_index != expected_node_index:
            raise ValueError(f"non-contiguous node_index in {game_id}: {node_index}")
        color = row["actor_color"].strip().lower()
        if color not in histories:
            raise ValueError(f"invalid actor color in {game_id}: {color}")
        if row["actor_id"].casefold() != player_ids[color].casefold():
            raise ValueError(f"actor identity/color mismatch in {game_id} node {node_index}")
        is_pass = row["is_pass"] == "1"
        if is_pass:
            if row["move"].strip() != "-":
                raise ValueError(f"pass/move mismatch in {game_id} node {node_index}")
            continue

        actual_ply += 1
        strict_ply = int(row["strict_ply"])
        if strict_ply != actual_ply:
            raise ValueError(f"strict ply mismatch in {game_id} node {node_index}")
        thinking_time = float(row["thinking_time_ms"])
        if not math.isfinite(thinking_time) or thinking_time < 0:
            raise ValueError(f"invalid thinking time in {game_id} node {node_index}")

        target_decisions[color] += 1
        history = histories[color]
        if (
            anchors[color] is None
            and config.min_ply <= strict_ply <= config.max_ply
            and history
        ):
            prior_median = float(statistics.median(history))
            threshold = config.multiplier * prior_median
            if thinking_time > threshold:
                anchors[color] = {
                    "anchor_strict_ply": strict_ply,
                    "anchor_target_decision": target_decisions[color],
                    "anchor_original_node_index": node_index,
                    "thinking_time_ms": thinking_time,
                    "prior_time_median_ms": prior_median,
                    "threshold_ms": threshold,
                    "time_ratio": None if prior_median == 0 else thinking_time / prior_median,
                    "prior_observation_count": len(history),
                }
        history.append(thinking_time)

    records: list[dict[str, Any]] = []
    for target_view, color in enumerate(("black", "white")):
        opponent_color = "white" if color == "black" else "black"
        anchor = anchors[color]
        records.append({
            "schema": "first-long-think-median-ratio-record-v1",
            "game_id": game_id,
            "split": metadata.get("split") or None,
            "target_view": target_view,
            "target_color": color,
            "target_id": player_ids[color],
            "opponent_id": player_ids[opponent_color],
            "target_move_count": target_decisions[color],
            "result": "anchor" if anchor is not None else "no_anchor",
            "anchor": anchor,
            "rule": {
                "clip_min_strict_ply_inclusive": config.min_ply,
                "cap_max_strict_ply_inclusive": config.max_ply,
                "time_multiplier": config.multiplier,
                "comparison": ">",
                "history": "all prior same-player non-pass thinking times, including pre-clip nodes",
            },
        })
    return records


def contiguous_ply_bins(min_ply: int, max_ply: int, count: int) -> list[tuple[int, int]]:
    width = max_ply - min_ply + 1
    if count < 1 or count > width:
        raise ValueError("ply-bin count must be between one and the inclusive range width")
    base, remainder = divmod(width, count)
    bins: list[tuple[int, int]] = []
    start = min_ply
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        stop = start + size - 1
        bins.append((start, stop))
        start = stop + 1
    return bins


def _stable_sample_key(record: Mapping[str, Any], seed: int) -> str:
    source = f"{seed}\0{record['game_id']}\0{record['target_color']}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def stratified_review_sample(
    records: Iterable[Mapping[str, Any]],
    *,
    min_ply: int,
    max_ply: int,
    bin_count: int,
    anchors_per_bin: int,
    no_anchor_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    rows = [dict(record) for record in records]
    bins = contiguous_ply_bins(min_ply, max_ply, bin_count)
    used_games: set[str] = set()
    selected: list[dict[str, Any]] = []
    bin_summary: list[dict[str, int]] = []

    for bin_index, (start, stop) in enumerate(bins, start=1):
        eligible = [
            row for row in rows
            if row["result"] == "anchor"
            and start <= int(row["anchor"]["anchor_strict_ply"]) <= stop
        ]
        eligible.sort(key=lambda row: _stable_sample_key(row, seed))
        choices: list[dict[str, Any]] = []
        for row in eligible:
            if row["game_id"] in used_games:
                continue
            choice = dict(row)
            choice["sample_group"] = f"anchor_ply_{start}_{stop}"
            choices.append(choice)
            used_games.add(row["game_id"])
            if len(choices) == anchors_per_bin:
                break
        if len(choices) != anchors_per_bin:
            raise ValueError(f"not enough distinct games in anchor-ply bin {start}-{stop}")
        selected.extend(choices)
        bin_summary.append({
            "bin": bin_index,
            "min_ply_inclusive": start,
            "max_ply_inclusive": stop,
            "selected": len(choices),
        })

    no_anchor_rows = [row for row in rows if row["result"] == "no_anchor"]
    no_anchor_rows.sort(key=lambda row: _stable_sample_key(row, seed))
    no_anchor_selected = 0
    for row in no_anchor_rows:
        if row["game_id"] in used_games:
            continue
        choice = dict(row)
        choice["sample_group"] = "no_anchor"
        selected.append(choice)
        used_games.add(row["game_id"])
        no_anchor_selected += 1
        if no_anchor_selected == no_anchor_count:
            break
    if no_anchor_selected != no_anchor_count:
        raise ValueError("not enough distinct no-anchor games")
    return selected, bin_summary
