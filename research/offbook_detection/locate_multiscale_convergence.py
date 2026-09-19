#!/usr/bin/env python3
"""Locate candidate off-book anchors by multiscale convergence of v and a.

The input is a completed ``monotonic-offbook-evidence-fit-v1`` directory.  For
each outer leave-one-player-out fold, this script rebuilds the fold's training
curves, derives scale-specific empirical references from training ``no_offbook``
games, calibrates a convergence threshold on training games, and applies the
frozen rule to the held-out player's existing cross-validated curve.

This remains a research candidate detector.  It never edits manual anchors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import fit_monotonic_evidence as base


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SCALES = (1, 2, 4, 8)
DEFAULT_CLUSTER_RADIUS = 2
DEFAULT_MIN_SUPPORT = 2
LOGGER = logging.getLogger("multiscale_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-file",
        type=Path,
        help="UTF-8 progress log (default: an adjacent <output-dir>.log file).",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Target-player decision window sizes (default: 1 2 4 8).",
    )
    parser.add_argument(
        "--cluster-radius",
        type=int,
        default=DEFAULT_CLUSTER_RADIUS,
        help="Maximum pairwise distance in target-player decisions within a cluster.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help="Minimum number of distinct scales required for convergence.",
    )
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_parameters(scales: Iterable[int], cluster_radius: int, min_support: int) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(scale) for scale in scales)))
    if not normalized or normalized[0] <= 0:
        raise ValueError("scales must be distinct positive integers")
    if cluster_radius < 0:
        raise ValueError("cluster radius must be non-negative")
    if not 2 <= min_support <= len(normalized):
        raise ValueError("min support must lie between 2 and the number of scales")
    return normalized


@dataclass(frozen=True)
class ScaleReference:
    scale: int
    v_values: np.ndarray
    a_values: np.ndarray

    def percentile_v(self, value: float) -> float:
        return empirical_percentile(self.v_values, value)

    def percentile_a(self, value: float) -> float:
        return empirical_percentile(self.a_values, value)

    def as_dict(self) -> dict[str, Any]:
        quantiles = (0.5, 0.75, 0.9, 0.95, 0.99)
        return {
            "scale": self.scale,
            "referencePositions": int(len(self.v_values)),
            "vWindowMeanQuantiles": {
                str(quantile): float(np.quantile(self.v_values, quantile))
                for quantile in quantiles
            },
            "aWindowChangeQuantiles": {
                str(quantile): float(np.quantile(self.a_values, quantile))
                for quantile in quantiles
            },
        }


def empirical_percentile(sorted_values: np.ndarray, value: float) -> float:
    if len(sorted_values) == 0:
        raise ValueError("empirical reference is empty")
    rank = int(np.searchsorted(sorted_values, value, side="right"))
    return float(rank / len(sorted_values))


def ordered_game(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("target_decision_number", kind="stable").copy()
    decisions = ordered["target_decision_number"].to_numpy(dtype=int)
    if not np.array_equal(decisions, np.arange(1, len(ordered) + 1)):
        raise ValueError("target decision numbers must be consecutive and start at 1")
    h = ordered["h_increment"].to_numpy(dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("h increments must be finite and non-negative")
    return ordered


def window_statistics(game: pd.DataFrame, scales: tuple[int, ...]) -> pd.DataFrame:
    """Return forward-window v and pre/post a at each possible transition onset.

    Candidate decision 1 is excluded because there is no earlier target decision
    from which a change can be observed.  The forward window must be complete;
    the preceding window uses all available history up to the requested scale.
    """
    ordered = ordered_game(game)
    h = ordered["h_increment"].to_numpy(dtype=float)
    strict_ply = ordered["strict_ply"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for scale in scales:
        for start in range(1, len(h)):
            stop = start + scale
            if stop > len(h):
                break
            pre_start = max(0, start - scale)
            pre_v = float(np.mean(h[pre_start:start]))
            post_v = float(np.mean(h[start:stop]))
            rows.append({
                "scale": scale,
                "candidate_target_decision": start + 1,
                "candidate_strict_ply": int(strict_ply[start]),
                "pre_v_window_mean": pre_v,
                "v_window_mean": post_v,
                "a_window_change": post_v - pre_v,
                "pre_window_observations": start - pre_start,
                "post_window_observations": scale,
            })
    return pd.DataFrame(rows)


def build_scale_references(
    curves: pd.DataFrame,
    scales: tuple[int, ...],
) -> dict[int, ScaleReference]:
    no_offbook = curves.loc[curves["judgment"].eq("no_offbook")]
    if no_offbook.empty:
        raise ValueError("training fold contains no no_offbook nodes")
    frames = [
        window_statistics(game, scales)
        for _, game in no_offbook.groupby(["account", "game_id"], sort=False)
        if len(game) > 1
    ]
    if not frames:
        raise ValueError("training no_offbook games have no usable transition positions")
    stats = pd.concat(frames, ignore_index=True)
    references: dict[int, ScaleReference] = {}
    for scale in scales:
        selected = stats.loc[stats["scale"].eq(scale)]
        if selected.empty:
            raise ValueError(f"training no_offbook games have no positions at scale {scale}")
        references[scale] = ScaleReference(
            scale=scale,
            v_values=np.sort(selected["v_window_mean"].to_numpy(dtype=float)),
            a_values=np.sort(selected["a_window_change"].to_numpy(dtype=float)),
        )
    return references


def scale_candidates_for_game(
    game: pd.DataFrame,
    scales: tuple[int, ...],
    references: dict[int, ScaleReference],
) -> pd.DataFrame:
    stats = window_statistics(game, scales)
    if stats.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scale, scale_stats in stats.groupby("scale", sort=True):
        reference = references[int(scale)]
        ordered = scale_stats.sort_values("candidate_target_decision", kind="stable").copy()
        ordered["v_reference_percentile"] = [
            reference.percentile_v(value)
            for value in ordered["v_window_mean"].to_numpy(dtype=float)
        ]
        ordered["a_reference_percentile"] = [
            reference.percentile_a(value)
            for value in ordered["a_window_change"].to_numpy(dtype=float)
        ]
        ordered["scale_strength"] = np.minimum(
            ordered["v_reference_percentile"],
            ordered["a_reference_percentile"],
        )
        eligible_strength = np.where(
            ordered["a_window_change"].to_numpy(dtype=float) > 0,
            ordered["scale_strength"].to_numpy(dtype=float),
            0.0,
        )
        for index, strength in enumerate(eligible_strength):
            previous = eligible_strength[index - 1] if index > 0 else 0.0
            following = eligible_strength[index + 1] if index + 1 < len(eligible_strength) else 0.0
            if strength <= 0 or not (strength > previous and strength >= following):
                continue
            row = ordered.iloc[index].to_dict()
            row["scale_strength"] = float(strength)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["candidate_target_decision", "scale"], kind="stable"
    ).reset_index(drop=True)


def confidence_from_support(support_count: int) -> str:
    if support_count >= 4:
        return "high"
    if support_count == 3:
        return "medium"
    if support_count == 2:
        return "low"
    raise ValueError(f"unsupported convergence support count: {support_count}")


def convergence_clusters_for_game(
    candidates: pd.DataFrame,
    game: pd.DataFrame,
    cluster_radius: int,
    min_support: int,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    proposals: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    minimum = int(candidates["candidate_target_decision"].min())
    maximum = int(candidates["candidate_target_decision"].max())
    for left in range(minimum, maximum + 1):
        window = candidates.loc[
            candidates["candidate_target_decision"].between(left, left + cluster_radius)
        ]
        if window.empty:
            continue
        selected_rows = []
        for _, same_scale in window.groupby("scale", sort=True):
            selected_rows.append(
                same_scale.sort_values(
                    ["scale_strength", "candidate_target_decision"],
                    ascending=[False, True],
                    kind="stable",
                ).iloc[0]
            )
        if len(selected_rows) < min_support:
            continue
        selected = pd.DataFrame(selected_rows).sort_values("scale", kind="stable")
        positions = selected["candidate_target_decision"].to_numpy(dtype=int)
        if int(positions.max() - positions.min()) > cluster_radius:
            raise AssertionError("convergence cluster exceeds the pairwise radius")
        key = tuple(
            (int(row.scale), int(row.candidate_target_decision))
            for row in selected.itertuples(index=False)
        )
        support_count = len(selected)
        mean_strength = float(selected["scale_strength"].mean())
        anchor_decision = int(positions.min())
        proposal = {
            "anchor_target_decision": anchor_decision,
            "support_count": support_count,
            "confidence": confidence_from_support(support_count),
            "supporting_scales": ";".join(str(int(value)) for value in selected["scale"]),
            "supporting_candidate_decisions": ";".join(
                f"{int(row.scale)}:{int(row.candidate_target_decision)}"
                for row in selected.itertuples(index=False)
            ),
            "supporting_scale_strengths": ";".join(
                f"{int(row.scale)}:{float(row.scale_strength):.9f}"
                for row in selected.itertuples(index=False)
            ),
            "mean_scale_strength": mean_strength,
            "minimum_scale_strength": float(selected["scale_strength"].min()),
            "convergence_score": float(support_count + mean_strength),
        }
        existing = proposals.get(key)
        if existing is None or proposal["convergence_score"] > existing["convergence_score"]:
            proposals[key] = proposal

    if not proposals:
        return pd.DataFrame()

    ranked = sorted(
        proposals.values(),
        key=lambda row: (-row["convergence_score"], row["anchor_target_decision"]),
    )
    kept: list[dict[str, Any]] = []
    for proposal in ranked:
        if any(
            abs(proposal["anchor_target_decision"] - other["anchor_target_decision"])
            <= cluster_radius
            for other in kept
        ):
            continue
        kept.append(proposal)

    ordered = ordered_game(game)
    ply_by_decision = dict(zip(
        ordered["target_decision_number"].to_numpy(dtype=int),
        ordered["strict_ply"].to_numpy(dtype=int),
        strict=True,
    ))
    kept.sort(key=lambda row: row["anchor_target_decision"])
    for cluster_number, proposal in enumerate(kept, start=1):
        proposal["cluster_number"] = cluster_number
        proposal["anchor_strict_ply"] = int(ply_by_decision[proposal["anchor_target_decision"]])
    return pd.DataFrame(kept)


def derive_candidates_and_clusters(
    curves: pd.DataFrame,
    scales: tuple[int, ...],
    references: dict[int, ScaleReference],
    cluster_radius: int,
    min_support: int,
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    identity_columns = ("account", "game_id", "judgment", "manual_offbook_ply")
    grouped = list(curves.groupby(["account", "game_id"], sort=True))
    for game_number, ((account, game_id), game) in enumerate(grouped, start=1):
        candidates = scale_candidates_for_game(game, scales, references)
        clusters = convergence_clusters_for_game(
            candidates,
            game,
            cluster_radius,
            min_support,
        )
        identity = {
            "account": str(account),
            "game_id": str(game_id),
            "judgment": str(game["judgment"].iloc[0]),
            "manual_offbook_ply": game["manual_offbook_ply"].iloc[0],
        }
        if not candidates.empty:
            for column in reversed(identity_columns):
                candidates.insert(0, column, identity[column])
            candidate_frames.append(candidates)
        if not clusters.empty:
            for column in reversed(identity_columns):
                clusters.insert(0, column, identity[column])
            cluster_frames.append(clusters)
        if progress_label and (game_number == 1 or game_number % 25 == 0 or game_number == len(grouped)):
            LOGGER.info(
                "%s game progress %d/%d",
                progress_label,
                game_number,
                len(grouped),
            )
    candidates_out = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    clusters_out = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    return candidates_out, clusters_out


def manual_anchor_decisions(curves: pd.DataFrame) -> dict[tuple[str, str], int]:
    anchors = curves.loc[curves["is_manual_anchor"].astype(bool)]
    if anchors.duplicated(["account", "game_id"]).any():
        raise ValueError("a player/game evaluation has multiple manual anchors")
    return {
        (str(row.account), str(row.game_id)): int(row.target_decision_number)
        for row in anchors.itertuples(index=False)
    }


def earliest_qualified_cluster(game_clusters: pd.DataFrame, threshold: float) -> pd.Series | None:
    if game_clusters.empty:
        return None
    qualified = game_clusters.loc[game_clusters["convergence_score"].ge(threshold)]
    if qualified.empty:
        return None
    return qualified.sort_values(
        ["anchor_target_decision", "convergence_score"],
        ascending=[True, False],
        kind="stable",
    ).iloc[0]


def evaluate_threshold(
    evaluation_games: list[tuple[tuple[str, str], str, int | None]],
    cluster_options: dict[tuple[str, str], list[tuple[int, float]]],
    threshold: float,
) -> dict[str, Any]:
    anchored_total = 0
    no_offbook_total = 0
    exact = 0
    within_one = 0
    within_two = 0
    anchored_with_candidate = 0
    no_offbook_without_candidate = 0
    for key, judgment, manual in evaluation_games:
        selected_decision = next(
            (
                anchor_decision
                for anchor_decision, score in cluster_options.get(key, [])
                if score >= threshold
            ),
            None,
        )
        if judgment == "offbook":
            anchored_total += 1
            if manual is None:
                raise ValueError(f"anchored game lacks a mapped target decision: {key}")
            if selected_decision is not None:
                anchored_with_candidate += 1
                distance = abs(selected_decision - manual)
                exact += int(distance == 0)
                within_one += int(distance <= 1)
                within_two += int(distance <= 2)
        elif judgment == "no_offbook":
            no_offbook_total += 1
            no_offbook_without_candidate += int(selected_decision is None)
        else:
            raise ValueError(f"unexpected judgment: {judgment!r}")
    if anchored_total == 0 or no_offbook_total == 0:
        raise ValueError("threshold calibration requires anchored and no_offbook games")
    within_two_rate = within_two / anchored_total
    no_offbook_nonreport_rate = no_offbook_without_candidate / no_offbook_total
    return {
        "threshold": float(threshold),
        "anchoredGames": anchored_total,
        "noOffbookGames": no_offbook_total,
        "anchoredCandidateRate": anchored_with_candidate / anchored_total,
        "exactRateAllAnchored": exact / anchored_total,
        "withinOneRateAllAnchored": within_one / anchored_total,
        "withinTwoRateAllAnchored": within_two_rate,
        "noOffbookNonreportRate": no_offbook_nonreport_rate,
        "balancedLocalizationObjective": 0.5 * (within_two_rate + no_offbook_nonreport_rate),
    }


def calibrate_threshold(
    marked_games: pd.DataFrame,
    clusters: pd.DataFrame,
    anchor_decisions: dict[tuple[str, str], int],
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    if clusters.empty:
        raise ValueError("training fold produced no convergence clusters")
    scores = np.sort(clusters["convergence_score"].unique())
    thresholds = list(float(value) for value in scores)
    thresholds.append(float(np.nextafter(scores[-1], math.inf)))
    cluster_options = {
        (str(account), str(game_id)): [
            (int(row.anchor_target_decision), float(row.convergence_score))
            for row in frame.sort_values(
                ["anchor_target_decision", "convergence_score"],
                ascending=[True, False],
                kind="stable",
            ).itertuples(index=False)
        ]
        for (account, game_id), frame in clusters.groupby(["account", "game_id"], sort=False)
    }
    evaluation_games = [
        (
            (str(row.account), str(row.game_id)),
            str(row.judgment),
            anchor_decisions.get((str(row.account), str(row.game_id))),
        )
        for row in marked_games.itertuples(index=False)
    ]
    LOGGER.info("threshold sweep started: %d finite thresholds", len(thresholds))
    rows = []
    for threshold_number, threshold in enumerate(thresholds, start=1):
        rows.append(evaluate_threshold(evaluation_games, cluster_options, threshold))
        if threshold_number % 100 == 0 or threshold_number == len(thresholds):
            LOGGER.info("threshold sweep progress %d/%d", threshold_number, len(thresholds))
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(
        [
            "balancedLocalizationObjective",
            "noOffbookNonreportRate",
            "withinTwoRateAllAnchored",
            "threshold",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    ).iloc[0].to_dict()
    return float(best["threshold"]), sweep, best


def game_summary(
    marked_games: pd.DataFrame,
    curves: pd.DataFrame,
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    anchor_decisions: dict[tuple[str, str], int],
    thresholds_by_account: dict[str, float],
) -> pd.DataFrame:
    nodes_by_game = {
        (str(account), str(game_id)): frame
        for (account, game_id), frame in curves.groupby(["account", "game_id"], sort=False)
    }
    candidates_by_game = {
        (str(account), str(game_id)): frame
        for (account, game_id), frame in candidates.groupby(["account", "game_id"], sort=False)
    } if not candidates.empty else {}
    clusters_by_game = {
        (str(account), str(game_id)): frame
        for (account, game_id), frame in clusters.groupby(["account", "game_id"], sort=False)
    } if not clusters.empty else {}
    rows: list[dict[str, Any]] = []
    for mark in marked_games.sort_values(["account", "game_id"], kind="stable").itertuples(index=False):
        account = str(mark.account)
        game_id = str(mark.game_id)
        key = (account, game_id)
        threshold = float(thresholds_by_account[account])
        game_nodes = nodes_by_game.get(key, pd.DataFrame())
        game_candidates = candidates_by_game.get(key, pd.DataFrame())
        game_clusters = clusters_by_game.get(key, pd.DataFrame())
        selected = earliest_qualified_cluster(game_clusters, threshold)
        manual_decision = anchor_decisions.get(key)
        predicted_decision = int(selected["anchor_target_decision"]) if selected is not None else None
        signed_distance = (
            predicted_decision - manual_decision
            if predicted_decision is not None and manual_decision is not None
            else None
        )
        rows.append({
            "account": account,
            "game_id": game_id,
            "judgment": str(mark.judgment),
            "manual_offbook_ply": mark.manual_offbook_ply,
            "manual_offbook_target_decision": manual_decision,
            "target_nodes": int(len(game_nodes)),
            "scale_candidate_count": int(len(game_candidates)),
            "convergence_cluster_count": int(len(game_clusters)),
            "fold_convergence_threshold": threshold,
            "has_converged_candidate": selected is not None,
            "candidate_strict_ply": int(selected["anchor_strict_ply"]) if selected is not None else None,
            "candidate_target_decision": predicted_decision,
            "candidate_signed_distance_from_anchor": signed_distance,
            "candidate_absolute_distance_from_anchor": (
                abs(signed_distance) if signed_distance is not None else None
            ),
            "candidate_support_count": int(selected["support_count"]) if selected is not None else None,
            "candidate_confidence": str(selected["confidence"]) if selected is not None else None,
            "candidate_supporting_scales": (
                str(selected["supporting_scales"]) if selected is not None else None
            ),
            "candidate_supporting_decisions": (
                str(selected["supporting_candidate_decisions"]) if selected is not None else None
            ),
            "candidate_convergence_score": (
                float(selected["convergence_score"]) if selected is not None else None
            ),
        })
    return pd.DataFrame(rows)


def finite_rate(condition: pd.Series) -> float | None:
    return float(condition.mean()) if len(condition) else None


def build_summary(
    input_summary: dict[str, Any],
    nodes: pd.DataFrame,
    games: pd.DataFrame,
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    folds: list[dict[str, Any]],
    scales: tuple[int, ...],
    cluster_radius: int,
    min_support: int,
    input_dir: Path,
) -> dict[str, Any]:
    anchored = games.loc[games["judgment"].eq("offbook")]
    no_offbook = games.loc[games["judgment"].eq("no_offbook")]
    selected_anchored = anchored.loc[anchored["has_converged_candidate"]]
    selected_all = games.loc[games["has_converged_candidate"]]
    confidence_counts = {
        str(key): int(value)
        for key, value in selected_all["candidate_confidence"].value_counts().sort_index().items()
    }
    return {
        "schema": "monotonic-offbook-multiscale-convergence-v1",
        "status": "completed",
        "source": {
            "inputDirectory": str(input_dir),
            "inputSchema": input_summary["schema"],
            "inputNodeCurveSha256": base.sha256_file(input_dir / "cross_validated_node_curves.csv"),
            "inputSummarySha256": base.sha256_file(input_dir / "summary.json"),
        },
        "methodPolicy": {
            "sequenceUnit": "target_decision_number; final anchors are mapped back to strict_ply",
            "scales": list(scales),
            "vAtScale": "mean h from candidate onset through the complete forward scale window",
            "aAtScale": "post-window mean h minus up-to-scale available pre-window mean h",
            "firstDecision": "not eligible because no prior target decision exists",
            "shortPreWindow": "use all available prior target decisions, at least one",
            "shortPostWindow": "scale does not participate when its full forward window is unavailable",
            "scaleCandidate": (
                "positive a local maximum; strength is the minimum of training-no_offbook "
                "empirical percentiles for v and a"
            ),
            "clusterRadiusTargetDecisions": cluster_radius,
            "minimumSupportingScales": min_support,
            "clusterScore": "distinct supporting scale count plus mean scale strength",
            "unifiedAnchor": "earliest supporting candidate position in the earliest qualified cluster",
            "confidence": "low/medium/high for 2/3/4 supporting scales",
            "thresholdCalibration": (
                "within each outer training fold, maximize the equal-weight mean of anchored-game "
                "within-two-target-decision localization and no_offbook non-report rate; ties prefer "
                "higher non-report rate, then higher localization, then higher threshold"
            ),
            "crossValidation": "leave one player out; held-out labels never select that fold's threshold",
        },
        "counts": {
            "players": int(nodes["account"].nunique()),
            "playerGameEvaluations": int(len(games)),
            "offbookGames": int(len(anchored)),
            "noOffbookGames": int(len(no_offbook)),
            "targetNodes": int(len(nodes)),
            "scaleCandidates": int(len(candidates)),
            "convergenceClusters": int(len(clusters)),
            "selectedCandidates": int(games["has_converged_candidate"].sum()),
            "selectedConfidenceCounts": confidence_counts,
        },
        "crossValidatedMetrics": {
            "anchoredGameCandidateRate": finite_rate(anchored["has_converged_candidate"]),
            "exactAnchorRateAllAnchored": finite_rate(
                anchored["candidate_absolute_distance_from_anchor"].eq(0)
            ),
            "withinOneRateAllAnchored": finite_rate(
                anchored["candidate_absolute_distance_from_anchor"].le(1)
            ),
            "withinTwoRateAllAnchored": finite_rate(
                anchored["candidate_absolute_distance_from_anchor"].le(2)
            ),
            "exactAnchorRateAmongCandidates": finite_rate(
                selected_anchored["candidate_absolute_distance_from_anchor"].eq(0)
            ),
            "withinOneRateAmongCandidates": finite_rate(
                selected_anchored["candidate_absolute_distance_from_anchor"].le(1)
            ),
            "withinTwoRateAmongCandidates": finite_rate(
                selected_anchored["candidate_absolute_distance_from_anchor"].le(2)
            ),
            "noOffbookNonreportRate": finite_rate(~no_offbook["has_converged_candidate"]),
        },
        "baselineMetrics": input_summary.get("crossValidatedMetrics", {}),
        "folds": folds,
        "limitations": [
            "This is a first multiscale research detector and does not replace manual anchors.",
            "Scale references and convergence thresholds are estimated from only 35 no_offbook games in total.",
            "Training-fold curve scores used for calibration are in-sample predictions from the fold hazard model.",
            "Confidence is an auditable scale-support grade, not a calibrated probability.",
        ],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    metrics = summary["crossValidatedMetrics"]
    baseline = summary["baselineMetrics"]
    confidence = counts["selectedConfidenceCounts"]
    lines = [
        "# 多尺度差分收敛与候选脱谱锚点（第一版）",
        "",
        "## 方法",
        "",
        "- 使用目标选手决策序号上的 1/2/4/8 尺度。",
        "- 每个尺度计算后窗平均 `v=h`，以及后窗平均 `v` 减前窗平均 `v` 的粗尺度 `a`。",
        "- `a>0` 的局部峰形成尺度候选；候选强度同时要求 `v` 和 `a` 相对训练折 `no_offbook` 参照较高。",
        "- 不同尺度候选相差不超过 2 个目标决策时形成收敛簇；至少两个尺度支持。",
        "- 每个外层按棋手留一训练折单独校准收敛阈值，留出棋手不参与阈值选择。",
        "",
        "## 留一验证结果",
        "",
        f"- {counts['players']} 位选手、{counts['playerGameEvaluations']} 个选手－对局；其中 {counts['offbookGames']} 局有人工锚点，{counts['noOffbookGames']} 局为 `no_offbook`。",
        f"- 生成 {counts['scaleCandidates']} 个逐尺度候选、{counts['convergenceClusters']} 个收敛簇，最终 {counts['selectedCandidates']} 局输出候选锚点。",
        f"- 锚点局候选覆盖率：{metrics['anchoredGameCandidateRate']:.1%}。",
        f"- 所有锚点局精确 / 前后1决策 / 前后2决策定位率：{metrics['exactAnchorRateAllAnchored']:.1%} / {metrics['withinOneRateAllAnchored']:.1%} / {metrics['withinTwoRateAllAnchored']:.1%}。",
        f"- 已输出候选的锚点局中，精确 / 前后1决策 / 前后2决策定位率：{metrics['exactAnchorRateAmongCandidates']:.1%} / {metrics['withinOneRateAmongCandidates']:.1%} / {metrics['withinTwoRateAmongCandidates']:.1%}。",
        f"- `no_offbook` 不误报率：{metrics['noOffbookNonreportRate']:.1%}。",
        f"- 置信等级数量：低 {confidence.get('low', 0)}，中 {confidence.get('medium', 0)}，高 {confidence.get('high', 0)}。",
        "",
        "## 与单点最大 h 诊断的对照",
        "",
        f"- 原单点最大 `h` 精确 / 前后1决策 / 前后2决策：{baseline['topHExactAnchorRate']:.1%} / {baseline['topHWithinOneTargetDecisionRate']:.1%} / {baseline['topHWithinTwoTargetDecisionsRate']:.1%}。",
        f"- 原诊断阈值的锚点局覆盖率 / `no_offbook` 不误报率：{baseline['anchoredGameCandidateRate']:.1%} / {baseline['noOffbookSpecificity']:.1%}。",
        "",
        "## 边界",
        "",
        "本结果是第一版候选定位器，不改写人工锚点。置信等级只表示支持尺度数量，不是脱谱概率。阈值由每折训练数据上的定位与不误报等权目标选择，尚未经过独立外部样本冻结验证。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    scales = validate_parameters(args.scales, args.cluster_radius, args.min_support)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    log_path = (
        args.log_file.resolve()
        if args.log_file is not None
        else output_dir.parent / f"{output_dir.name}.log"
    )
    configure_logging(log_path)
    run_started = perf_counter()
    LOGGER.info("run started")
    LOGGER.info("input directory: %s", input_dir)
    LOGGER.info("output directory: %s", output_dir)
    LOGGER.info("scales=%s cluster_radius=%d min_support=%d", scales, args.cluster_radius, args.min_support)

    input_summary_path = input_dir / "summary.json"
    input_nodes_path = input_dir / "cross_validated_node_curves.csv"
    if not input_summary_path.is_file() or not input_nodes_path.is_file():
        raise FileNotFoundError("input directory lacks summary.json or cross_validated_node_curves.csv")
    input_summary = read_json(input_summary_path)
    if input_summary.get("schema") != "monotonic-offbook-evidence-fit-v1":
        raise ValueError(f"unexpected input schema: {input_summary.get('schema')!r}")

    cohort_manifest_path = Path(input_summary["inputs"]["cohortManifest"]).resolve()
    frequency_book_path = Path(input_summary["inputs"]["frequencyBook"]).resolve()
    if base.sha256_file(cohort_manifest_path) != input_summary["inputs"]["cohortManifestSha256"]:
        raise ValueError("cohort manifest no longer matches the input model summary")
    if base.sha256_file(frequency_book_path) != input_summary["inputs"]["frequencyBookSha256"]:
        raise ValueError("frequency book no longer matches the input model summary")
    LOGGER.info("input summary and frozen source hashes validated")

    phase_started = perf_counter()
    source_cv = pd.read_csv(input_nodes_path, encoding="utf-8")
    book = base.GlobalFrequencyBook.load(frequency_book_path)
    nodes, marked_games, _ = base.load_all_nodes(cohort_manifest_path, book)
    LOGGER.info(
        "inputs materialized: nodes=%d games=%d elapsed=%.2fs",
        len(nodes),
        len(marked_games),
        perf_counter() - phase_started,
    )
    if len(source_cv) != len(nodes):
        raise ValueError("input cross-validated node count differs from current frozen inputs")
    anchor_decisions = manual_anchor_decisions(source_cv)
    regularization_c = float(input_summary["modelPolicy"]["regularizationC"])

    all_candidates: list[pd.DataFrame] = []
    all_clusters: list[pd.DataFrame] = []
    all_sweeps: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    thresholds_by_account: dict[str, float] = {}

    for held_out in sorted(nodes["account"].unique(), key=str.casefold):
        fold_started = perf_counter()
        LOGGER.info("fold %s started", held_out)
        train_nodes = nodes.loc[nodes["account"].ne(held_out)].copy()
        test_nodes = nodes.loc[nodes["account"].eq(held_out)].copy()
        train_games = marked_games.loc[marked_games["account"].ne(held_out)].copy()
        model = base.fit_hazard_model(train_nodes, regularization_c)
        train_curves = base.add_curve_columns(train_nodes, model.predict_probability(train_nodes))
        regenerated_test = base.add_curve_columns(test_nodes, model.predict_probability(test_nodes))
        LOGGER.info("fold %s hazard curves ready elapsed=%.2fs", held_out, perf_counter() - fold_started)

        source_test = source_cv.loc[source_cv["account"].eq(held_out)].sort_values(
            ["account", "game_id", "target_decision_number"], kind="stable"
        )
        regenerated_test = regenerated_test.sort_values(
            ["account", "game_id", "target_decision_number"], kind="stable"
        )
        source_keys = source_test[["account", "game_id", "target_decision_number"]].reset_index(drop=True)
        regenerated_keys = regenerated_test[["account", "game_id", "target_decision_number"]].reset_index(drop=True)
        if not source_keys.equals(regenerated_keys):
            raise ValueError(f"held-out node keys changed for {held_out}")
        maximum_h_difference = float(np.max(np.abs(
            source_test["h_increment"].to_numpy(dtype=float)
            - regenerated_test["h_increment"].to_numpy(dtype=float)
        )))
        if maximum_h_difference > 1e-10:
            raise ValueError(
                f"reproduced held-out h differs from frozen input for {held_out}: {maximum_h_difference}"
            )
        LOGGER.info("fold %s held-out h reproduced max_abs_diff=%.3g", held_out, maximum_h_difference)

        references = build_scale_references(train_curves, scales)
        LOGGER.info(
            "fold %s scale references ready counts=%s",
            held_out,
            {scale: len(references[scale].v_values) for scale in scales},
        )
        train_candidates, train_clusters = derive_candidates_and_clusters(
            train_curves,
            scales,
            references,
            args.cluster_radius,
            args.min_support,
            progress_label=f"fold {held_out} training candidate generation",
        )
        LOGGER.info(
            "fold %s training candidates ready scale_candidates=%d clusters=%d",
            held_out,
            len(train_candidates),
            len(train_clusters),
        )
        threshold, sweep, best_training = calibrate_threshold(
            train_games,
            train_clusters,
            manual_anchor_decisions(train_curves),
        )
        thresholds_by_account[str(held_out)] = threshold
        sweep.insert(0, "held_out_account", str(held_out))
        all_sweeps.append(sweep)

        test_candidates, test_clusters = derive_candidates_and_clusters(
            source_test,
            scales,
            references,
            args.cluster_radius,
            args.min_support,
            progress_label=f"fold {held_out} held-out candidate generation",
        )
        if not test_candidates.empty:
            test_candidates["held_out_account"] = str(held_out)
            all_candidates.append(test_candidates)
        if not test_clusters.empty:
            test_clusters["held_out_account"] = str(held_out)
            test_clusters["fold_convergence_threshold"] = threshold
            test_clusters["passes_fold_threshold"] = test_clusters["convergence_score"].ge(threshold)
            all_clusters.append(test_clusters)
        folds.append({
            "heldOutAccount": str(held_out),
            "trainingPlayers": int(train_nodes["account"].nunique()),
            "trainingGames": int(len(train_games)),
            "trainingOffbookGames": int(train_games["judgment"].eq("offbook").sum()),
            "trainingNoOffbookGames": int(train_games["judgment"].eq("no_offbook").sum()),
            "trainingScaleCandidates": int(len(train_candidates)),
            "trainingConvergenceClusters": int(len(train_clusters)),
            "selectedConvergenceThreshold": threshold,
            "selectedTrainingMetrics": best_training,
            "scaleReferences": {
                str(scale): references[scale].as_dict() for scale in scales
            },
            "maximumReproducedHeldOutHDifference": maximum_h_difference,
        })
        LOGGER.info(
            "fold %s completed threshold=%.9f test_scale_candidates=%d test_clusters=%d elapsed=%.2fs",
            held_out,
            threshold,
            len(test_candidates),
            len(test_clusters),
            perf_counter() - fold_started,
        )

    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    clusters = pd.concat(all_clusters, ignore_index=True) if all_clusters else pd.DataFrame()
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    games = game_summary(
        marked_games,
        source_cv,
        candidates,
        clusters,
        anchor_decisions,
        thresholds_by_account,
    )
    summary = build_summary(
        input_summary,
        source_cv,
        games,
        candidates,
        clusters,
        folds,
        scales,
        args.cluster_radius,
        args.min_support,
        input_dir,
    )
    report = markdown_report(summary)
    LOGGER.info("all folds complete; writing output artifacts")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates.to_csv(output_dir / "cross_validated_scale_candidates.csv", index=False, encoding="utf-8")
    clusters.to_csv(output_dir / "cross_validated_convergence_clusters.csv", index=False, encoding="utf-8")
    games.to_csv(output_dir / "cross_validated_game_summary.csv", index=False, encoding="utf-8")
    sweeps.to_csv(output_dir / "fold_threshold_sweep.csv", index=False, encoding="utf-8")
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    LOGGER.info("run completed elapsed=%.2fs", perf_counter() - run_started)
    print(json.dumps({
        "status": "completed",
        "outputDir": str(output_dir),
        "counts": summary["counts"],
        "crossValidatedMetrics": summary["crossValidatedMetrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
