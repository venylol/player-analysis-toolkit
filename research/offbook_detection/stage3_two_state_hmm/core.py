"""Exact two-state HMM likelihood, posterior, Viterbi, and semantic summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DiagonalHMM(nn.Module):
    def __init__(
        self,
        dimension: int,
        family: str,
        degrees_of_freedom: float = 5.0,
        scale_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if family not in ("student_t", "gaussian"):
            raise ValueError(f"unsupported family: {family}")
        self.dimension = dimension
        self.family = family
        self.degrees_of_freedom = degrees_of_freedom
        self.scale_floor = scale_floor
        self.location = nn.Parameter(torch.zeros(2, dimension))
        self.raw_scale = nn.Parameter(torch.zeros(2, dimension))
        self.initial_logits = nn.Parameter(torch.zeros(2))
        self.transition_logits = nn.Parameter(torch.zeros(2, 2))

    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + self.scale_floor

    def log_initial(self) -> torch.Tensor:
        return F.log_softmax(self.initial_logits, dim=-1)

    def log_transition(self) -> torch.Tensor:
        return F.log_softmax(self.transition_logits, dim=-1)

    def emission_log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Return log p(x_t | z_t) with shape [batch, time, 2]."""
        scale = self.scale()
        centered = (values[:, :, None, :] - self.location[None, None, :, :]) / scale[None, None, :, :]
        if self.family == "gaussian":
            terms = -0.5 * centered.square() - torch.log(scale)[None, None, :, :] - 0.5 * math.log(2 * math.pi)
        else:
            nu = self.degrees_of_freedom
            constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            terms = constant - torch.log(scale)[None, None, :, :] - ((nu + 1) / 2) * torch.log1p(centered.square() / nu)
        return terms.sum(-1)


def initialize_model(
    model: DiagonalHMM,
    locations: torch.Tensor,
    scales: torch.Tensor,
    initial_probabilities: torch.Tensor,
    transition_matrix: torch.Tensor,
) -> None:
    if locations.shape != model.location.shape or scales.shape != model.raw_scale.shape:
        raise ValueError("emission initialization shape mismatch")
    with torch.no_grad():
        model.location.copy_(locations)
        target = (scales - model.scale_floor).clamp_min(1e-6)
        model.raw_scale.copy_(torch.log(torch.expm1(target)))
        model.initial_logits.copy_(initial_probabilities.log())
        model.transition_logits.copy_(transition_matrix.log())


def forward_log_likelihood_from_emissions(
    emission: torch.Tensor,
    lengths: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    return_alpha: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if emission.ndim != 3 or emission.shape[-1] != 2:
        raise ValueError("emission must have shape [batch, time, 2]")
    if torch.any(lengths < 1) or torch.any(lengths > emission.shape[1]):
        raise ValueError("invalid sequence lengths")
    alpha = log_initial[None, :] + emission[:, 0, :]
    history = [alpha]
    for time_index in range(1, emission.shape[1]):
        proposed = torch.logsumexp(alpha[:, :, None] + log_transition[None, :, :], dim=1) + emission[:, time_index, :]
        active = time_index < lengths
        alpha = torch.where(active[:, None], proposed, alpha)
        history.append(alpha)
    log_likelihood = torch.logsumexp(alpha, dim=-1)
    return log_likelihood, torch.stack(history, dim=1) if return_alpha else None


def marginal_nll(model: DiagonalHMM, values: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    emission = model.emission_log_prob(values)
    log_likelihood, _ = forward_log_likelihood_from_emissions(
        emission, lengths, model.log_initial(), model.log_transition()
    )
    return -log_likelihood.sum(), lengths.sum()


@dataclass(frozen=True)
class InferenceResult:
    log_likelihood: np.ndarray
    posterior: np.ndarray
    transition_posterior: np.ndarray
    viterbi: np.ndarray


@torch.inference_mode()
def infer_batch(model: DiagonalHMM, values: torch.Tensor, lengths: torch.Tensor) -> InferenceResult:
    """Run exact inference; padded posterior entries are NaN and Viterbi entries are -1."""
    emission = model.emission_log_prob(values)
    log_initial = model.log_initial()
    log_transition = model.log_transition()
    log_likelihood, alpha = forward_log_likelihood_from_emissions(
        emission, lengths, log_initial, log_transition, return_alpha=True
    )
    assert alpha is not None
    batch, steps, _ = emission.shape
    beta = torch.zeros_like(emission)
    for time_index in range(steps - 2, -1, -1):
        proposed = torch.logsumexp(
            log_transition[None, :, :] + emission[:, time_index + 1, None, :] + beta[:, time_index + 1, None, :],
            dim=2,
        )
        has_next = (time_index + 1) < lengths
        beta[:, time_index, :] = torch.where(has_next[:, None], proposed, torch.zeros_like(proposed))
    log_gamma = alpha + beta - log_likelihood[:, None, None]
    posterior = log_gamma.exp()
    valid_nodes = torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]
    posterior = posterior.masked_fill(~valid_nodes[:, :, None], torch.nan)
    xi = torch.full((batch, steps, 2, 2), torch.nan, dtype=emission.dtype, device=emission.device)
    for time_index in range(1, steps):
        log_xi = (
            alpha[:, time_index - 1, :, None]
            + log_transition[None, :, :]
            + emission[:, time_index, None, :]
            + beta[:, time_index, None, :]
            - log_likelihood[:, None, None]
        )
        active = time_index < lengths
        xi[:, time_index, :, :] = torch.where(active[:, None, None], log_xi.exp(), xi[:, time_index, :, :])

    delta = log_initial[None, :] + emission[:, 0, :]
    backpointers = torch.zeros((batch, steps, 2), dtype=torch.int64, device=values.device)
    delta_history = [delta]
    for time_index in range(1, steps):
        scores = delta[:, :, None] + log_transition[None, :, :]
        best_score, best_state = scores.max(dim=1)
        proposed = best_score + emission[:, time_index, :]
        active = time_index < lengths
        delta = torch.where(active[:, None], proposed, delta)
        backpointers[:, time_index, :] = best_state
        delta_history.append(delta)
    delta_all = torch.stack(delta_history, dim=1)
    paths = torch.full((batch, steps), -1, dtype=torch.int64, device=values.device)
    for row in range(batch):
        length = int(lengths[row])
        state = int(delta_all[row, length - 1].argmax())
        paths[row, length - 1] = state
        for time_index in range(length - 1, 0, -1):
            state = int(backpointers[row, time_index, state])
            paths[row, time_index - 1] = state
    posterior_numpy = posterior.cpu().double().numpy()
    transition_numpy = xi.cpu().double().numpy()
    valid_numpy = valid_nodes.cpu().numpy()
    posterior_numpy[valid_numpy] /= posterior_numpy[valid_numpy].sum(axis=1, keepdims=True)
    boundary_numpy = valid_numpy.copy()
    boundary_numpy[:, 0] = False
    transition_numpy[boundary_numpy] /= transition_numpy[boundary_numpy].sum(axis=(1, 2), keepdims=True)
    return InferenceResult(
        log_likelihood=log_likelihood.cpu().double().numpy(),
        posterior=posterior_numpy,
        transition_posterior=transition_numpy,
        viterbi=paths.cpu().numpy(),
    )


def thirds(length: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return early and late indices using numpy.array_split-equivalent sizes."""
    if length < 3:
        return None
    parts = np.array_split(np.arange(length, dtype=np.int64), 3)
    return parts[0], parts[2]


def semantic_indices(raw_to_semantic: list[int] | tuple[int, int]) -> tuple[int, int]:
    if sorted(raw_to_semantic) != [0, 1]:
        raise ValueError("raw_to_semantic must be a permutation of [0, 1]")
    return raw_to_semantic.index(0), raw_to_semantic.index(1)
