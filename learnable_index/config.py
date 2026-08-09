from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AttentionAggregationConfig:
    """Explicit reductions used to create a teacher target.

    ``future_reduction="mean"`` keeps total historical mass in ``[0, 1]``
    when the input contains normalized attention probabilities.  The raw
    absolute block masses are always retained even when block-length
    normalization is used for the conditional distribution.
    """

    teacher_layers: tuple[int, ...] | None = None
    teacher_heads: tuple[int, ...] | None = None
    future_reduction: Literal["mean", "sum"] = "mean"
    future_weights: tuple[float, ...] | None = None
    length_normalize_blocks: bool = False
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.future_reduction not in {"mean", "sum"}:
            raise ValueError("future_reduction must be 'mean' or 'sum'")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.future_weights is not None:
            if not self.future_weights or any(weight < 0 for weight in self.future_weights):
                raise ValueError("future_weights must be non-empty and non-negative")
            if sum(self.future_weights) <= 0:
                raise ValueError("future_weights must contain positive mass")


@dataclass(frozen=True)
class RouterConfig:
    residual_dim: int
    projection_dim: int = 128
    hidden_dim: int = 256
    depth: int = 2
    dropout: float = 0.0
    initial_temperature: float = 0.07
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in ("residual_dim", "projection_dim", "hidden_dim", "depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")


@dataclass(frozen=True)
class LossConfig:
    minimum_historical_mass: float = 1e-8

    def __post_init__(self) -> None:
        if self.minimum_historical_mass < 0:
            raise ValueError("minimum_historical_mass must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 13
    gradient_clip_norm: float = 1.0
    top_n: int = 4
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
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


def dataclass_dict(value: object) -> dict[str, Any]:
    """Return a JSON-compatible dictionary for one of this module's configs."""

    return asdict(value)  # tuples are accepted by json.dump and become arrays
