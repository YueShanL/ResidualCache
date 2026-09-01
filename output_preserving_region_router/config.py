from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionRouterConfig:
    residual_dim: int
    feature_dim: int = 128
    hidden_dim: int = 256
    depth: int = 2
    dropout: float = 0.1
    minimum_scale: float = 0.05
    radius: float = 1.0
    gate_temperature: float = 0.25
    gate_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("residual_dim", "feature_dim", "hidden_dim", "depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if not 0 < self.gate_epsilon < 1:
            raise ValueError("gate_epsilon must be in (0, 1)")


@dataclass(frozen=True)
class OutputPreservationLossConfig:
    output_temperature: float = 1.0
    maximum_excess_output_kl: float = 0.02
    preservation_weight: float = 100.0
    sparsity_weight: float = 1.0
    gate_entropy_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.output_temperature <= 0:
            raise ValueError("output_temperature must be positive")
        if self.maximum_excess_output_kl < 0:
            raise ValueError("maximum_excess_output_kl must be non-negative")
        if self.preservation_weight <= 0:
            raise ValueError("preservation_weight must be positive")
        if self.sparsity_weight <= 0:
            raise ValueError("sparsity_weight must be positive")
        if self.gate_entropy_weight < 0:
            raise ValueError("gate_entropy_weight must be non-negative")


@dataclass(frozen=True)
class RegionTrainConfig:
    epochs: int = 3
    early_stopping_patience: int | None = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 13
    device: str = "cuda"
    prefill_chunk_size: int = 256
    final_gate_temperature: float | None = 0.05
    maximum_train_samples: int | None = None
    maximum_validation_samples: int | None = None
    progress_every: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when set")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if self.final_gate_temperature is not None and self.final_gate_temperature <= 0:
            raise ValueError("final_gate_temperature must be positive when set")
        for name in ("maximum_train_samples", "maximum_validation_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
