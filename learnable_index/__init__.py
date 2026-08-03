"""Standalone learned block-attention index research pipeline.

This package intentionally does not import :mod:`residual_cache`.  Integration
with model-specific cache adapters belongs at the boundary of the pipeline,
after the tensor contracts implemented here have been validated.
"""

from .config import AttentionAggregationConfig, LossConfig, RouterConfig, TrainConfig
from .contracts import BlockRange, RetrievalSample
from .model import LearnableBlockIndex
from .planning import PlanConfig, RetrievalPlan, SequenceRecord
from .replay import ReplayConfig
from .retrieval import RetrievalPolicyConfig

__all__ = [
    "AttentionAggregationConfig",
    "BlockRange",
    "LearnableBlockIndex",
    "LossConfig",
    "PlanConfig",
    "ReplayConfig",
    "RetrievalPlan",
    "RetrievalPolicyConfig",
    "RetrievalSample",
    "RouterConfig",
    "SequenceRecord",
    "TrainConfig",
]
