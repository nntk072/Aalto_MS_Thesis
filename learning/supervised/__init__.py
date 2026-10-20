"""Supervised-learning backends behind the generic learner/provider contracts."""

from learning.supervised.reference import (
    ReferenceTorchLearner,
    ReferenceTorchProvider,
    build_reference_torch_learner,
    load_reference_torch_provider,
)

__all__ = [
    "ReferenceTorchLearner",
    "ReferenceTorchProvider",
    "build_reference_torch_learner",
    "load_reference_torch_provider",
]
