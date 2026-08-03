from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import RouterConfig


class IndexTower(nn.Module):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(config.residual_dim)]
        input_dim = config.residual_dim
        for _ in range(config.depth):
            layers.extend(
                [
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, config.projection_dim))
        self.network = nn.Sequential(*layers)
        self.normalize = config.normalize_embeddings

    def forward(self, residual_summary: torch.Tensor) -> torch.Tensor:
        projected = self.network(residual_summary)
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected


@dataclass(frozen=True)
class RouterOutput:
    scores: torch.Tensor
    demand_logits: torch.Tensor
    query_embeddings: torch.Tensor
    key_embeddings: torch.Tensor


class LearnableBlockIndex(nn.Module):
    """Two-tower prompt-free block router with an explicit demand head."""

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_network = IndexTower(config)
        self.key_network = IndexTower(config)
        self.demand_head = nn.Linear(config.projection_dim, 1)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature), dtype=torch.float32)
        )

    def forward(
        self,
        query_summaries: torch.Tensor,
        block_summaries: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> RouterOutput:
        if query_summaries.ndim != 2:
            raise ValueError("query_summaries must have shape [batch, residual_dim]")
        if block_summaries.ndim != 3:
            raise ValueError("block_summaries must have shape [batch, blocks, residual_dim]")
        if candidate_mask.shape != block_summaries.shape[:2]:
            raise ValueError("candidate_mask must align with batch and block dimensions")
        if query_summaries.shape[0] != block_summaries.shape[0]:
            raise ValueError("query and block batch dimensions do not match")
        if query_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("query residual dimension does not match router config")
        if block_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("block residual dimension does not match router config")
        if torch.any(candidate_mask.sum(dim=-1) == 0):
            raise ValueError("every sample must expose at least one candidate block")

        query_embeddings = self.query_network(query_summaries)
        key_embeddings = self.key_network(block_summaries)
        scale = self.logit_scale.exp().clamp(max=100.0)
        scores = torch.einsum("bd,bnd->bn", query_embeddings, key_embeddings) * scale
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        demand_logits = self.demand_head(query_embeddings).squeeze(-1)
        return RouterOutput(
            scores=scores,
            demand_logits=demand_logits,
            query_embeddings=query_embeddings,
            key_embeddings=key_embeddings,
        )

