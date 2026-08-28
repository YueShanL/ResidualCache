from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityRouterConfig:
    residual_dim: int
    feature_dim: int = 128
    hidden_dim: int = 256
    depth: int = 2
    dropout: float = 0.0
    positive_floor: float = 1e-6
    normalization_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("residual_dim", "feature_dim", "hidden_dim", "depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.positive_floor <= 0:
            raise ValueError("positive_floor must be strictly positive")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be strictly positive")


@dataclass(frozen=True)
class ProbabilityLossConfig:
    minimum_historical_mass: float = 1e-8

    def __post_init__(self) -> None:
        if self.minimum_historical_mass < 0:
            raise ValueError("minimum_historical_mass must be non-negative")


@dataclass(frozen=True)
class ProbabilityTrainConfig:
    epochs: int = 10
    early_stopping_patience: int | None = 2
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    seed: int = 13
    gradient_clip_norm: float = 1.0
    top_n: int = 4
    probability_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when set")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not self.probability_thresholds:
            raise ValueError("probability_thresholds cannot be empty")
        if any(not 0 < threshold < 1 for threshold in self.probability_thresholds):
            raise ValueError("every probability threshold must be in (0, 1)")
        if tuple(sorted(set(self.probability_thresholds))) != self.probability_thresholds:
            raise ValueError("probability_thresholds must be unique and sorted")
