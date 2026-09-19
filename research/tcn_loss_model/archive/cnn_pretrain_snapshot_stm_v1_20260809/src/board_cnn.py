"""Shared residual board-CNN components for pretraining and formal inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_BOARD_CHANNELS = 64
DEFAULT_RESIDUAL_BLOCKS = 6
BOARD_EMBEDDING_DIM = 96
DEFAULT_PROJECTION_KERNEL = 1


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise AssertionError("unreachable")


class BoardResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.gelu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return F.gelu(value + residual)


class SharedBoardCNN(nn.Module):
    """Stem, residual spatial trunk, and 96-dimensional pooled projection."""

    def __init__(
        self,
        input_channels: int,
        board_channels: int = DEFAULT_BOARD_CHANNELS,
        residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
        board_embedding_dim: int = BOARD_EMBEDDING_DIM,
        embedding_projection_kernel: int = DEFAULT_PROJECTION_KERNEL,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or board_channels <= 0 or residual_blocks <= 0:
            raise ValueError("input_channels, board_channels, and residual_blocks must be positive")
        if board_embedding_dim != BOARD_EMBEDDING_DIM:
            raise ValueError(f"board_embedding_dim is fixed at {BOARD_EMBEDDING_DIM}")
        if embedding_projection_kernel != 1:
            raise ValueError("embedding_projection_kernel must be 1 for the current architecture")
        self.input_channels = input_channels
        self.board_channels = board_channels
        self.residual_blocks_count = residual_blocks
        self.board_embedding_dim = board_embedding_dim
        self.embedding_projection_kernel = embedding_projection_kernel
        self.stem = nn.Conv2d(input_channels, board_channels, 3, padding=1)
        self.stem_norm = nn.GroupNorm(_group_count(board_channels), board_channels)
        self.blocks = nn.ModuleList(BoardResidualBlock(board_channels) for _ in range(residual_blocks))
        self.projection = nn.Conv2d(board_channels, board_embedding_dim, 1)
        self.embedding_norm = nn.LayerNorm(board_embedding_dim)

    def spatial_features(self, board_planes: torch.Tensor) -> torch.Tensor:
        expected = (self.input_channels, 8, 8)
        if board_planes.ndim != 4 or tuple(board_planes.shape[1:]) != expected:
            raise ValueError(f"expected Bx{self.input_channels}x8x8 input, got {tuple(board_planes.shape)}")
        value = F.gelu(self.stem_norm(self.stem(board_planes)))
        for block in self.blocks:
            value = block(value)
        return value

    def embedding_from_spatial(self, spatial: torch.Tensor) -> torch.Tensor:
        projected = F.gelu(self.projection(spatial))
        return self.embedding_norm(F.adaptive_avg_pool2d(projected, 1).flatten(1))

    def forward(self, board_planes: torch.Tensor) -> torch.Tensor:
        return self.embedding_from_spatial(self.spatial_features(board_planes))


class BoardCNNAuxiliaryHeads(nn.Module):
    """Training-only legal-move and normalized-value heads."""

    def __init__(self, board_channels: int = DEFAULT_BOARD_CHANNELS) -> None:
        super().__init__()
        self.legal_head = nn.Conv2d(board_channels, 1, 1)
        self.value_head = nn.Sequential(
            nn.Linear(BOARD_EMBEDDING_DIM, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, spatial: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.legal_head(spatial).flatten(1), self.value_head(embedding).squeeze(1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def sample_auxiliary_node_indices(
    valid_nodes: torch.Tensor,
    nodes_per_game: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample at most N valid sequence nodes independently for every game."""
    if valid_nodes.ndim != 2:
        raise ValueError("valid_nodes must have shape BxT")
    if nodes_per_game <= 0:
        raise ValueError("nodes_per_game must be positive")
    selected: list[torch.Tensor] = []
    for game_index in range(valid_nodes.shape[0]):
        node_indices = torch.nonzero(valid_nodes[game_index].bool(), as_tuple=False).flatten()
        if node_indices.numel() == 0:
            continue
        order = torch.randperm(node_indices.numel(), generator=generator, device=node_indices.device)
        node_indices = node_indices[order[:nodes_per_game]]
        game_indices = torch.full_like(node_indices, game_index)
        selected.append(torch.stack((game_indices, node_indices), dim=1))
    if not selected:
        return torch.empty((0, 2), dtype=torch.long, device=valid_nodes.device)
    return torch.cat(selected, dim=0)


def legal_move_targets_from_board_tokens(current_board_tokens: torch.Tensor) -> torch.Tensor:
    """Generate legal-X targets from snapshot-side-to-move board tokens only.

    Token 1 is empty, token 2 is the local side-to-move (X), and token 3 is
    the opponent (O). Leading dimensions are preserved and the final dimension
    must be the 64 squares ordered a1..h8.
    """
    if current_board_tokens.ndim < 2 or current_board_tokens.shape[-1] != 64:
        raise ValueError("current_board_tokens must have shape ...x64")
    if bool(((current_board_tokens < 0) | (current_board_tokens > 3)).any()):
        raise ValueError("current-board tokens must be in 0..3; zero is padding")
    shape = current_board_tokens.shape
    board = current_board_tokens.reshape(-1, 8, 8)
    empty, own, opponent = board == 1, board == 2, board == 3

    def shifted(value: torch.Tensor, dr: int, dc: int, distance: int) -> torch.Tensor:
        result = torch.zeros_like(value)
        row_source_start = max(0, dr * distance)
        row_source_end = min(8, 8 + dr * distance)
        col_source_start = max(0, dc * distance)
        col_source_end = min(8, 8 + dc * distance)
        row_target_start = max(0, -dr * distance)
        row_target_end = min(8, 8 - dr * distance)
        col_target_start = max(0, -dc * distance)
        col_target_end = min(8, 8 - dc * distance)
        result[:, row_target_start:row_target_end, col_target_start:col_target_end] = value[
            :, row_source_start:row_source_end, col_source_start:col_source_end
        ]
        return result

    legal = torch.zeros_like(empty)
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        run = shifted(opponent, dr, dc, 1)
        for distance in range(2, 8):
            legal |= run & shifted(own, dr, dc, distance)
            run &= shifted(opponent, dr, dc, distance)
    return (legal & empty).reshape(*shape[:-1], 64)
