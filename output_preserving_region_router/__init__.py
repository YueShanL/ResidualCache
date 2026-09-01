"""Output-preserving Gaussian region router.

This package deliberately has its own model, loss, checkpoints, and command
entry.  It reuses the block-aligned streaming and Gemma replay contracts, but
does not add another operating mode to :mod:`block_probability_router`.
"""

from .config import (
    OutputPreservationLossConfig,
    RegionRouterConfig,
    RegionTrainConfig,
)
from .model import GaussianRegionRouter, GaussianRegionRouterOutput

__all__ = [
    "GaussianRegionRouter",
    "GaussianRegionRouterOutput",
    "OutputPreservationLossConfig",
    "RegionRouterConfig",
    "RegionTrainConfig",
]
