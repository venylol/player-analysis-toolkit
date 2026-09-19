"""Small spatial Transformer used for legal-move board pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SpatialTransformerConfig:
    hidden_size: int = 96
    layers: int = 2
    attention_heads: int = 4
    feedforward_size: int = 384
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.layers <= 0 or self.attention_heads <= 0:
            raise ValueError("hidden size, layers, and attention heads must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.feedforward_size <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid feedforward size or dropout")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SpatialLegalTransformer(nn.Module):
    """Encode 64 board cells and predict a legal-move logit for each cell."""

    def __init__(self, config: SpatialTransformerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.occupancy_embedding = nn.Embedding(3, hidden)
        self.row_embedding = nn.Embedding(8, hidden)
        self.column_embedding = nn.Embedding(8, hidden)
        self.actor_embedding = nn.Embedding(2, hidden)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden))
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
        self.legal_head = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.occupancy_embedding.weight, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.column_embedding.weight, std=0.02)
        nn.init.normal_(self.actor_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.legal_head.weight)
        nn.init.zeros_(self.legal_head.bias)

    def forward(
        self,
        target_perspective_codes: torch.Tensor,
        actor_is_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_perspective_codes.ndim != 2 or target_perspective_codes.shape[1] != 64:
            raise ValueError("target_perspective_codes must have shape Bx64")
        batch = target_perspective_codes.shape[0]
        if actor_is_target.shape != (batch,):
            raise ValueError("actor_is_target must have shape B")
        device = target_perspective_codes.device
        rows = torch.arange(8, device=device).repeat_interleave(8)
        columns = torch.arange(8, device=device).repeat(8)
        actor = self.actor_embedding(actor_is_target.long()).unsqueeze(1)
        cells = (
            self.occupancy_embedding(target_perspective_codes.long())
            + self.row_embedding(rows).unsqueeze(0)
            + self.column_embedding(columns).unsqueeze(0)
            + actor
        )
        cls = self.cls_token.expand(batch, -1, -1) + actor
        encoded = self.encoder(torch.cat((cls, cells), dim=1))
        board_embedding = encoded[:, 0]
        legal_logits = self.legal_head(encoded[:, 1:]).squeeze(-1)
        return legal_logits, board_embedding


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
