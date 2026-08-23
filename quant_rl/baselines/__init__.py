"""Baseline strategies for fair comparison against RL agents."""

from .base import BaseStrategy
from .breakout import MultiLevelBreakoutStrategy
from .classical import BuyAndHoldStrategy, EMAMACDRSIStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "EMAMACDRSIStrategy",
    "MultiLevelBreakoutStrategy",
]
