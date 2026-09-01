from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import RegionRouterConfig


class ResidualTower(nn.Module):
    def __init__(self, config: RegionRouterConfig, output_dim: int) -> None:
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
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass(frozen=True)
class GaussianRegionRouterOutput:
    query_mean: torch.Tensor
    query_scale: torch.Tensor
    key_positions: torch.Tensor
    squared_distances: torch.Tensor
    gates: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def hard_mask(self) -> torch.Tensor:
        return self.candidate_mask & (self.squared_distances <= 1.0)


class GaussianRegionRouter(nn.Module):
    """Map a query to one diagonal-Gaussian region in learned key space."""

    def __init__(self, config: RegionRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_network = ResidualTower(config, 2 * config.feature_dim)
        self.key_network = ResidualTower(config, config.feature_dim)

    def encode_query(self, query_summaries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query_summaries.ndim != 2:
            raise ValueError("query_summaries must have shape [batch, residual_dim]")
        if query_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("query residual dimension does not match router config")
        parameters = self.query_network(query_summaries)
        mean, raw_scale = parameters.chunk(2, dim=-1)
        scale = F.softplus(raw_scale.float()) + self.config.minimum_scale
        return mean.float(), scale

    def encode_keys(self, block_summaries: torch.Tensor) -> torch.Tensor:
        if block_summaries.ndim not in {2, 3}:
            raise ValueError(
                "block_summaries must have shape [blocks, residual_dim] or "
                "[batch, blocks, residual_dim]"
            )
        if block_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("block residual dimension does not match router config")
        return self.key_network(block_summaries).float()

    def score_features(
        self,
        query_mean: torch.Tensor,
        query_scale: torch.Tensor,
        key_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        gate_temperature: float | None = None,
    ) -> GaussianRegionRouterOutput:
        if query_mean.ndim != 2 or query_scale.shape != query_mean.shape:
            raise ValueError("query mean and scale must have shape [batch, feature_dim]")
        if key_positions.ndim != 3:
            raise ValueError("key_positions must have shape [batch, blocks, feature_dim]")
        if query_mean.shape[0] != key_positions.shape[0]:
            raise ValueError("query and key batch dimensions do not match")
        if query_mean.shape[-1] != key_positions.shape[-1]:
            raise ValueError("query and key feature dimensions do not match")
        if candidate_mask.shape != key_positions.shape[:2]:
            raise ValueError("candidate_mask must align with key positions")
        if torch.any(candidate_mask.sum(dim=-1) == 0):
            raise ValueError("every sample must expose at least one candidate block")
        if torch.any(query_scale <= 0):
            raise ValueError("query scales must be strictly positive")
        temperature = (
            self.config.gate_temperature
            if gate_temperature is None
            else float(gate_temperature)
        )
        if temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        standardized = (
            key_positions.float() - query_mean.float().unsqueeze(1)
        ) / query_scale.float().unsqueeze(1)
        # Use the mean per-axis Mahalanobis contribution so the initial gate
        # calibration does not change merely because feature_dim changes.  The
        # level set remains one diagonal ellipsoid; ``radius`` is its RMS
        # standardized radius rather than a raw chi-square radius.
        squared_distances = standardized.square().mean(dim=-1) / (
            self.config.radius**2
        )
        gates = torch.sigmoid((1.0 - squared_distances) / temperature)
        gates = gates.masked_fill(~candidate_mask, 0.0)
        squared_distances = squared_distances.masked_fill(
            ~candidate_mask, float("inf")
        )
        return GaussianRegionRouterOutput(
            query_mean=query_mean.float(),
            query_scale=query_scale.float(),
            key_positions=key_positions.float(),
            squared_distances=squared_distances,
            gates=gates,
            candidate_mask=candidate_mask,
        )

    def forward(
        self,
        query_summaries: torch.Tensor,
        block_summaries: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        gate_temperature: float | None = None,
    ) -> GaussianRegionRouterOutput:
        if block_summaries.ndim != 3:
            raise ValueError("block_summaries must have shape [batch, blocks, residual_dim]")
        mean, scale = self.encode_query(query_summaries)
        return self.score_features(
            mean,
            scale,
            self.encode_keys(block_summaries),
            candidate_mask,
            gate_temperature=gate_temperature,
        )
