"""Positive block-probability router trained from frozen-model attention labels.

The package deliberately depends only on the data contract produced by
``learnable_index``.  Its model, loss, checkpoints, and command-line entry are
separate so the probability factorization cannot silently fall back to the
legacy softmax-logit router.
"""

from .config import ProbabilityLossConfig, ProbabilityRouterConfig, ProbabilityTrainConfig
from .model import BlockProbabilityRouter, ProbabilityRouterOutput

__all__ = [
    "BlockProbabilityRouter",
    "ProbabilityLossConfig",
    "ProbabilityRouterConfig",
    "ProbabilityRouterOutput",
    "ProbabilityTrainConfig",
]
