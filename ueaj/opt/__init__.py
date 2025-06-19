"""Optimization utilities: loss functions and optimizer management."""

from .next_token_loss import (
    ntp_loss_mask,
    ntp_args,
    chunked_softmax_cross_entropy,
)

from .optimizer_state import (
    OptimizerManager,
    BatchMarker,
    BatchAccessor,
)

__all__ = [
    # Loss functions
    "ntp_loss_mask",
    "ntp_args",
    "chunked_softmax_cross_entropy",
    # Optimizer management
    "OptimizerManager",
    "BatchMarker",
    "BatchAccessor",
]