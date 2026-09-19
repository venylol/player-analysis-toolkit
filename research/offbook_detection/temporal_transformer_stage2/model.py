"""Bidirectional temporal Transformer and its three single-node recovery heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalTransformerConfig:
    board_embedding_dim: int = 96
    hidden_size: int = 128
    layers: int = 3
    attention_heads: int = 4
    feedforward_size: int = 512
    dropout: float = 0.1
    max_nodes: int = 128
    time_controls: int = 1

    def __post_init__(self) -> None:
        if self.board_embedding_dim != 96:
            raise ValueError("frozen stage-2 contract requires 96-dimensional board embeddings")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.max_nodes <= 0 or self.time_controls <= 0:
            raise ValueError("max_nodes and time_controls must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TemporalTransformer(nn.Module):
    def __init__(self, config: TemporalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.board_projection = nn.Linear(config.board_embedding_dim, hidden)
        self.time_projection = nn.Linear(3, hidden)
        self.structure_projection = nn.Linear(3, hidden)
        self.position_embedding = nn.Embedding(config.max_nodes, hidden)
        self.actor_target_embedding = nn.Embedding(2, hidden)
        self.actor_color_embedding = nn.Embedding(2, hidden)
        self.pass_embedding = nn.Embedding(2, hidden)
        self.time_control_embedding = nn.Embedding(config.time_controls, hidden)
        self.mask_embedding = nn.Embedding(4, hidden, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False
        )
        self.thinking_head = nn.Linear(hidden, 1)
        self.remaining_head = nn.Linear(hidden, 2)
        self.board_head = nn.Linear(hidden, config.board_embedding_dim)

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        masked_node: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        board = batch["board_embedding"]
        thinking = batch["thinking_time"]
        target_remaining = batch["target_remaining_time"]
        opponent_remaining = batch["opponent_remaining_time"]
        batch_size, length, _ = board.shape
        if length > self.config.max_nodes:
            raise ValueError(f"sequence length {length} exceeds max_nodes={self.config.max_nodes}")
        mask_codes = torch.zeros((batch_size, length), dtype=torch.long, device=board.device)
        if (masked_node is None) != (task_id is None):
            raise ValueError("masked_node and task_id must be provided together")
        if masked_node is not None and task_id is not None:
            rows = torch.arange(batch_size, device=board.device)
            if torch.any((task_id < 1) | (task_id > 3)):
                raise ValueError("task ids must be 1=thinking, 2=remaining, or 3=board")
            board = board.clone()
            thinking = thinking.clone()
            target_remaining = target_remaining.clone()
            opponent_remaining = opponent_remaining.clone()
            thinking_rows = rows[task_id == 1]
            thinking_nodes = masked_node[task_id == 1]
            thinking[thinking_rows, thinking_nodes] = 0
            remaining_rows = rows[task_id == 2]
            remaining_nodes = masked_node[task_id == 2]
            target_remaining[remaining_rows, remaining_nodes] = 0
            opponent_remaining[remaining_rows, remaining_nodes] = 0
            board_rows = rows[task_id == 3]
            board_nodes = masked_node[task_id == 3]
            board[board_rows, board_nodes] = 0
            mask_codes[rows, masked_node] = task_id
        node_index = batch["node_index"].long()
        lengths = batch["lengths"].float().clamp_min(1).unsqueeze(1)
        structure = torch.stack(
            (
                batch["strict_ply"].float() / 60.0,
                node_index.float() / 64.0,
                node_index.float() / (lengths - 1).clamp_min(1),
            ),
            dim=-1,
        )
        times = torch.stack((thinking, target_remaining, opponent_remaining), dim=-1)
        time_control = self.time_control_embedding(batch["time_control_index"].long()).unsqueeze(1)
        inputs = (
            self.board_projection(board)
            + self.time_projection(times)
            + self.structure_projection(structure)
            + self.position_embedding(node_index)
            + self.actor_target_embedding(batch["actor_is_target"].long())
            + self.actor_color_embedding(batch["actor_is_black"].long())
            + self.pass_embedding(batch["is_pass"].long())
            + time_control
            + self.mask_embedding(mask_codes)
        )
        return self.encoder(inputs, src_key_padding_mask=batch["padding_mask"])

    def recover(
        self, batch: dict[str, torch.Tensor], masked_node: torch.Tensor, task_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch, masked_node, task_id)
        rows = torch.arange(encoded.shape[0], device=encoded.device)
        selected = encoded[rows, masked_node]
        return encoded, self.thinking_head(selected).squeeze(-1), self.remaining_head(selected), self.board_head(selected)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

