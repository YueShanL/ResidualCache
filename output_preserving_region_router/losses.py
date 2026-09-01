from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import OutputPreservationLossConfig


@dataclass(frozen=True)
class OutputPreservationLoss:
    total: torch.Tensor
    output_kl: torch.Tensor
    all_history_output_kl: torch.Tensor
    excess_output_kl: torch.Tensor
    output_kl_violation: torch.Tensor
    expected_selected_blocks: torch.Tensor
    expected_selected_fraction: torch.Tensor
    gate_entropy: torch.Tensor


def output_preservation_loss(
    full_logits: torch.Tensor,
    gated_logits: torch.Tensor,
    all_history_logits: torch.Tensor,
    gates: torch.Tensor,
    candidate_mask: torch.Tensor,
    config: OutputPreservationLossConfig,
) -> OutputPreservationLoss:
    if (
        full_logits.shape != gated_logits.shape
        or full_logits.shape != all_history_logits.shape
        or full_logits.ndim != 3
    ):
        raise ValueError(
            "full, gated, and all-history logits must share shape [batch, tokens, vocab]"
        )
    if gates.shape != candidate_mask.shape or gates.ndim != 2:
        raise ValueError("gates and candidate_mask must share shape [batch, blocks]")
    if torch.any(candidate_mask.sum(dim=-1) == 0):
        raise ValueError("every sample must expose at least one candidate block")
    if torch.any(gates[candidate_mask] < 0) or torch.any(gates[candidate_mask] > 1):
        raise ValueError("candidate gates must lie in [0, 1]")

    temperature = config.output_temperature
    teacher = F.softmax(full_logits.detach().float() / temperature, dim=-1)
    student_log = F.log_softmax(gated_logits.float() / temperature, dim=-1)
    teacher_log = teacher.clamp_min(torch.finfo(torch.float32).tiny).log()
    per_token_kl = (teacher * (teacher_log - student_log)).sum(dim=-1)
    output_kl = per_token_kl.mean() * (temperature**2)
    reference_log = F.log_softmax(all_history_logits.detach().float() / temperature, dim=-1)
    all_history_kl = (
        (teacher * (teacher_log - reference_log)).sum(dim=-1).mean()
        * (temperature**2)
    )
    excess_output_kl = output_kl - all_history_kl
    violation = (
        excess_output_kl - config.maximum_excess_output_kl
    ).clamp_min(0.0)

    candidate_gates = gates.masked_fill(~candidate_mask, 0.0)
    expected_blocks = candidate_gates.sum(dim=-1).mean()
    candidate_counts = candidate_mask.sum(dim=-1).clamp_min(1)
    expected_fraction = (
        candidate_gates.sum(dim=-1) / candidate_counts.to(candidate_gates.dtype)
    ).mean()
    safe = candidate_gates[candidate_mask].clamp(1e-6, 1.0 - 1e-6)
    gate_entropy = (-(safe * safe.log() + (1.0 - safe) * (1.0 - safe).log())).mean()
    total = (
        config.preservation_weight * violation
        + config.sparsity_weight * expected_blocks
        + config.gate_entropy_weight * gate_entropy
    )
    return OutputPreservationLoss(
        total=total,
        output_kl=output_kl,
        all_history_output_kl=all_history_kl,
        excess_output_kl=excess_output_kl,
        output_kl_violation=violation,
        expected_selected_blocks=expected_blocks,
        expected_selected_fraction=expected_fraction,
        gate_entropy=gate_entropy,
    )
