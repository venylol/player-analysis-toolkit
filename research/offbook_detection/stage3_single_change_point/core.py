"""Data contracts, candidate construction, likelihoods, and posterior summaries."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CandidateRule:
    minimum_decision: int = 3
    maximum_decision: int = 19
    maximum_strict_ply: int = 38
    minimum_post_decisions: int = 3


def candidate_decisions(strict_ply: np.ndarray, rule: CandidateRule = CandidateRule()) -> np.ndarray:
    """Return legal one-based decisions k where the post segment begins."""
    m = int(len(strict_ply))
    upper = min(rule.maximum_decision, m - rule.minimum_post_decisions + 1)
    if upper < rule.minimum_decision:
        return np.empty(0, dtype=np.int16)
    values = np.arange(rule.minimum_decision, upper + 1, dtype=np.int16)
    return values[strict_ply[values - 1] <= rule.maximum_strict_ply]


def robust_standardize(values_ms: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (np.log1p(values_ms.astype(np.float32)) - center) / scale


def raw_stage3_features(
    archive: np.lib.npyio.NpzFile,
    target_view: int,
    offset: int,
    stop: int,
    time_stats: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    actor = archive["actor_is_target"][target_view, offset:stop].astype(np.bool_, copy=False)
    non_pass = ~archive["is_pass"][offset:stop].astype(np.bool_, copy=False)
    selected = np.flatnonzero(actor & non_pass)
    black = archive["black_remaining_time_ms_after"][offset:stop][selected]
    white = archive["white_remaining_time_ms_after"][offset:stop][selected]
    if target_view == 0:
        target_remaining, opponent_remaining = black, white
    else:
        target_remaining, opponent_remaining = white, black
    thinking = robust_standardize(
        archive["thinking_time_ms"][offset:stop][selected],
        time_stats["thinking_time"]["center"], time_stats["thinking_time"]["scale"],
    )
    target = robust_standardize(
        target_remaining, time_stats["target_remaining_time"]["center"],
        time_stats["target_remaining_time"]["scale"],
    )
    opponent = robust_standardize(
        opponent_remaining, time_stats["opponent_remaining_time"]["center"],
        time_stats["opponent_remaining_time"]["scale"],
    )
    hidden = archive["hidden_states"][target_view, offset:stop][selected].astype(np.float32)
    return np.concatenate((hidden, thinking[:, None], target[:, None], opponent[:, None]), axis=1), selected


class DiagonalSegmentModel(nn.Module):
    def __init__(self, dimension: int, family: str, degrees_of_freedom: float = 5.0, scale_floor: float = 1e-4):
        super().__init__()
        if family not in ("student_t", "gaussian"):
            raise ValueError(f"unsupported family: {family}")
        self.family = family
        self.degrees_of_freedom = degrees_of_freedom
        self.scale_floor = scale_floor
        self.location = nn.Parameter(torch.zeros(2, dimension))
        self.raw_scale = nn.Parameter(torch.zeros(2, dimension))

    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + self.scale_floor

    def node_log_prob(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, scale = self.location, self.scale()
        centered = (values[:, :, None, :] - location[None, None, :, :]) / scale[None, None, :, :]
        if self.family == "gaussian":
            terms = -0.5 * centered.square() - torch.log(scale)[None, None, :, :] - 0.5 * math.log(2 * math.pi)
        else:
            nu = self.degrees_of_freedom
            constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            terms = constant - torch.log(scale)[None, None, :, :] - ((nu + 1) / 2) * torch.log1p(centered.square() / nu)
        summed = terms.sum(-1)
        return summed[:, :, 0], summed[:, :, 1]


def state_log_scores(
    model: DiagonalSegmentModel,
    values: torch.Tensor,
    lengths: torch.Tensor,
    candidate_mask: torch.Tensor,
    no_change_prior: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded state scores [no-change, k=1..K] and their valid mask."""
    before, after = model.node_log_prob(values)
    node_mask = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
    before = before.masked_fill(~node_mask, 0.0)
    after = after.masked_fill(~node_mask, 0.0)
    prefix = torch.cat((torch.zeros((values.shape[0], 1), device=values.device), before.cumsum(1)), dim=1)
    suffix = torch.flip(torch.flip(after, dims=(1,)).cumsum(1), dims=(1,))
    no_change = before.sum(1) + math.log(no_change_prior)
    finite_count = candidate_mask.sum(1)
    finite_prior = math.log(1.0 - no_change_prior) - finite_count.clamp_min(1).float().log()
    # Column j represents one-based decision k=j+1; prefix uses nodes before k and suffix begins at k.
    finite = prefix[:, :-1] + suffix + finite_prior[:, None]
    finite = finite.masked_fill(~candidate_mask, -torch.inf)
    scores = torch.cat((no_change[:, None], finite), dim=1)
    valid = torch.cat((torch.ones_like(no_change[:, None], dtype=torch.bool), candidate_mask), dim=1)
    return scores, valid


def marginal_nll(
    model: DiagonalSegmentModel,
    values: torch.Tensor,
    lengths: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, _ = state_log_scores(model, values, lengths, candidate_mask)
    eligible = candidate_mask.any(1)
    if not eligible.any():
        raise ValueError("batch has no sequences with finite candidates")
    total_nll = -torch.logsumexp(scores[eligible], dim=1).sum()
    total_nodes = lengths[eligible].sum()
    return total_nll, total_nodes


def posterior_summary(scores: np.ndarray, candidate_decisions_one_based: np.ndarray, strict_ply: np.ndarray) -> dict[str, object]:
    scores = scores.astype(np.float64, copy=False)
    shifted = scores - np.max(scores)
    posterior = np.exp(shifted) / np.exp(shifted).sum()
    map_state = int(np.argmax(posterior))
    positive = posterior > 0
    entropy = float(-(posterior[positive] * np.log(posterior[positive])).sum())
    finite = posterior[1:]
    finite_total = float(finite.sum())
    candidate_plies = strict_ply[candidate_decisions_one_based - 1].astype(np.float64)
    if finite_total:
        conditional = finite / finite_total
        mean = float(np.dot(conditional, candidate_plies))
        std = float(np.sqrt(np.dot(conditional, (candidate_plies - mean) ** 2)))
    else:
        mean = std = float("nan")
    return {
        "p_no_change": float(posterior[0]),
        "map_state": "no_change" if map_state == 0 else "change",
        "map_probability": float(posterior[map_state]),
        "map_decision": None if map_state == 0 else int(candidate_decisions_one_based[map_state - 1]),
        "posterior_entropy": entropy,
        "finite_strict_ply_mean": mean,
        "finite_strict_ply_std": std,
        "candidate_probabilities": [float(value) for value in finite],
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
