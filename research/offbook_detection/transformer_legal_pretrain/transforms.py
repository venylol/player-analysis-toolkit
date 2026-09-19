"""Target-perspective and D4 transforms for 8x8 Othello positions."""

from __future__ import annotations

import torch


def to_target_perspective(codes: torch.Tensor, actor_is_target: torch.Tensor) -> torch.Tensor:
    """Convert empty/current/opponent codes to empty/target/opponent codes."""
    if codes.ndim != 2 or codes.shape[1] != 64:
        raise ValueError(f"expected codes with shape Bx64, got {tuple(codes.shape)}")
    if actor_is_target.shape != (codes.shape[0],):
        raise ValueError("actor_is_target must have shape B")
    if torch.any((codes < 0) | (codes > 2)):
        raise ValueError("board codes must be in 0..2")
    result = codes.clone()
    swap = ~actor_is_target.bool()
    if swap.any():
        selected = result[swap]
        result[swap] = torch.where(selected == 1, 2, torch.where(selected == 2, 1, selected))
    return result


def _transform_grid(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
    if not 0 <= transform_id < 8:
        raise ValueError(f"D4 transform id must be in 0..7, got {transform_id}")
    result = grid
    if transform_id >= 4:
        result = torch.flip(result, dims=(-1,))
    return torch.rot90(result, transform_id % 4, dims=(-2, -1))


def apply_d4(
    codes: torch.Tensor,
    legal: torch.Tensor,
    transform_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one independently selected D4 transform to each record."""
    if codes.ndim != 2 or codes.shape[1] != 64:
        raise ValueError("codes must have shape Bx64")
    if legal.shape != codes.shape:
        raise ValueError("legal targets must match codes shape")
    if transform_ids.shape != (codes.shape[0],):
        raise ValueError("transform_ids must have shape B")
    board_grid = codes.reshape(-1, 8, 8)
    legal_grid = legal.reshape(-1, 8, 8)
    transformed_boards = torch.empty_like(board_grid)
    transformed_legal = torch.empty_like(legal_grid)
    for transform_id in range(8):
        selected = transform_ids == transform_id
        if selected.any():
            transformed_boards[selected] = _transform_grid(board_grid[selected], transform_id)
            transformed_legal[selected] = _transform_grid(legal_grid[selected], transform_id)
    return transformed_boards.reshape(-1, 64), transformed_legal.reshape(-1, 64)


def transform_flat(values: torch.Tensor, transform_id: int) -> torch.Tensor:
    if values.shape[-1] != 64:
        raise ValueError("last dimension must contain 64 squares")
    leading = values.shape[:-1]
    return _transform_grid(values.reshape(-1, 8, 8), transform_id).reshape(*leading, 64)
