from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ProbabilityRouterConfig


class PositiveFeatureTower(nn.Module):
    """Map one frozen-model residual summary to a strictly positive feature vector."""

    def __init__(self, config: ProbabilityRouterConfig) -> None:
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
        layers.append(nn.Linear(input_dim, config.feature_dim))
        self.network = nn.Sequential(*layers)
        self.positive_floor = float(config.positive_floor)

    def forward(self, residual_summary: torch.Tensor) -> torch.Tensor:
        # Softplus is strictly positive and avoids the overflow failure mode of
        # an unconstrained exp feature map.  The floor also keeps log(w) finite.
        return F.softplus(self.network(residual_summary)) + self.positive_floor


@dataclass(frozen=True)
class ProbabilityRouterOutput:
    weights: torch.Tensor
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    normalizer: torch.Tensor
    query_features: torch.Tensor
    key_features: torch.Tensor
    key_sum: torch.Tensor


class BlockProbabilityRouter(nn.Module):
    """Two-tower positive router with an explicit memory normalizer.

    For candidate memory blocks ``M`` this implements

    ``w_b = phi_q(q)^T phi_k(k_b)``
    ``Z_M = phi_q(q)^T sum_{b in M} phi_k(k_b)``
    ``p_b = w_b / Z_M``.

    The current live block must not be included in ``M``.  That causal boundary
    is encoded by the shared collection contract, not repaired inside this
    model.
    """

    def __init__(self, config: ProbabilityRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_network = PositiveFeatureTower(config)
        self.key_network = PositiveFeatureTower(config)

    def encode_queries(self, query_summaries: torch.Tensor) -> torch.Tensor:
        if query_summaries.ndim != 2:
            raise ValueError("query_summaries must have shape [batch, residual_dim]")
        if query_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("query residual dimension does not match router config")
        return self.query_network(query_summaries)

    def encode_keys(self, block_summaries: torch.Tensor) -> torch.Tensor:
        if block_summaries.ndim not in {2, 3}:
            raise ValueError(
                "block_summaries must have shape [blocks, residual_dim] or "
                "[batch, blocks, residual_dim]"
            )
        if block_summaries.shape[-1] != self.config.residual_dim:
            raise ValueError("block residual dimension does not match router config")
        return self.key_network(block_summaries)

    def score_features(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> ProbabilityRouterOutput:
        if query_features.ndim != 2 or key_features.ndim != 3:
            raise ValueError("encoded queries/keys must have shapes [B,F] and [B,N,F]")
        if query_features.shape[0] != key_features.shape[0]:
            raise ValueError("query and key batch dimensions do not match")
        if query_features.shape[-1] != key_features.shape[-1]:
            raise ValueError("query and key feature dimensions do not match")
        if candidate_mask.shape != key_features.shape[:2]:
            raise ValueError("candidate_mask must align with batch and block dimensions")
        if torch.any(candidate_mask.sum(dim=-1) == 0):
            raise ValueError("every sample must expose at least one historical memory block")

        # Accumulate and score in fp32 even when the towers run in bf16.  This
        # is the same sum used by online retrieval, so training cannot depend on
        # a hidden softmax path that will be absent at inference.
        query_fp32 = query_features.float()
        keys_fp32 = key_features.float()
        mask_fp32 = candidate_mask.unsqueeze(-1).to(keys_fp32.dtype)
        key_sum = (keys_fp32 * mask_fp32).sum(dim=1)
        weights = torch.einsum("bf,bnf->bn", query_fp32, keys_fp32)
        weights = weights.masked_fill(~candidate_mask, 0.0)
        normalizer = torch.einsum("bf,bf->b", query_fp32, key_sum)
        denominator = normalizer.clamp_min(self.config.normalization_epsilon)
        probabilities = weights / denominator.unsqueeze(-1)
        log_probabilities = (
            weights.clamp_min(self.config.normalization_epsilon).log()
            - denominator.log().unsqueeze(-1)
        ).masked_fill(~candidate_mask, float("-inf"))
        return ProbabilityRouterOutput(
            weights=weights,
            probabilities=probabilities,
            log_probabilities=log_probabilities,
            normalizer=normalizer,
            query_features=query_fp32,
            key_features=keys_fp32,
            key_sum=key_sum,
        )

    def forward(
        self,
        query_summaries: torch.Tensor,
        block_summaries: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> ProbabilityRouterOutput:
        if block_summaries.ndim != 3:
            raise ValueError("block_summaries must have shape [batch, blocks, residual_dim]")
        if query_summaries.shape[0] != block_summaries.shape[0]:
            raise ValueError("query and block batch dimensions do not match")
        return self.score_features(
            self.encode_queries(query_summaries),
            self.encode_keys(block_summaries),
            candidate_mask,
        )

    @staticmethod
    def threshold_mask(
        output: ProbabilityRouterOutput,
        candidate_mask: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if candidate_mask.shape != output.weights.shape:
            raise ValueError("candidate_mask must align with router output")
        # p_b > threshold iff w_b > threshold * Z_M.  Keeping this in weight
        # space is the range-search interface used by a MIPS implementation.
        return candidate_mask & (
            output.weights > threshold * output.normalizer.unsqueeze(-1)
        )
