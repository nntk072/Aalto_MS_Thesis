"""Baseline strategies for fair comparison against RL agents."""

from .base import BaseStrategy
from .breakout import MultiLevelBreakoutStrategy
from .classical import BuyAndHoldStrategy, EMAMACDRSIStrategy
from .lstm_classifier import LSTMStrategy, LSTMSweepClassifier, build_sweep_dataset

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "EMAMACDRSIStrategy",
    "LSTMSweepClassifier",
    "LSTMStrategy",
    "MultiLevelBreakoutStrategy",
    "build_sweep_dataset",
]
