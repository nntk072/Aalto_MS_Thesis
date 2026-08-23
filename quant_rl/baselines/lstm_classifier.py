"""LSTM sweep-direction classifier (supervised baseline, thesis Week 8).

Predicts the direction of the next liquidity sweep from a window of
market features: class 1 = upward sweep of a high level within the
horizon, class -1 = downward sweep of a low level, class 0 = no sweep.
A fixed-position-size wrapper converts predictions into continuous
actions compatible with ``TradingEnv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from ..features.structure import detect_session_levels
from .base import BaseStrategy


def build_sweep_dataset(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    window: int = 60,
    horizon: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build labelled (features-window, direction) pairs for training.

    A bar is labelled +1 (-1) when any future close within ``horizon``
    bars rises above the current asian/london high (falls below the
    corresponding low). Bars with no sweep in either direction within the
    horizon are labelled 0.

    Args:
        bars: OHLCV DataFrame with DatetimeIndex.
        features: Feature matrix aligned row-for-row with ``bars``.
        window: Number of past feature rows per sample.
        horizon: Look-ahead bars for labelling.

    Returns:
        Tuple ``(X, y)`` where ``X`` is ``(n_samples, window, n_features)``
        float32 and ``y`` is ``(n_samples,)`` int64 with values in {-1, 0, 1}.
    """
    levels = detect_session_levels(bars)
    close = bars["close"].astype(float).to_numpy()
    highs = np.nanmax(
        np.column_stack(
            [
                levels.get("asian_high", pd.Series(np.nan, index=bars.index)),
                levels.get("london_high", pd.Series(np.nan, index=bars.index)),
            ]
        ).astype(float),
        axis=1,
    )
    lows = np.nanmin(
        np.column_stack(
            [
                levels.get("asian_low", pd.Series(np.nan, index=bars.index)),
                levels.get("london_low", pd.Series(np.nan, index=bars.index)),
            ]
        ).astype(float),
        axis=1,
    )

    feat_values = features.to_numpy(dtype=np.float32)
    n = len(close)
    samples: list[NDArray[np.float32]] = []
    labels: list[int] = []

    for i in range(window, n - horizon):
        future = close[i : i + horizon]
        valid_high = np.isfinite(highs[i])
        valid_low = np.isfinite(lows[i])
        label = 0
        if valid_high and float((future > highs[i]).any()):
            label = 1
        elif valid_low and float((future < lows[i]).any()):
            label = -1
        if label == 0:
            continue

        seq = feat_values[i - window : i]
        seq = np.nan_to_num(seq, nan=0.0)
        samples.append(seq)
        labels.append(label)

    if not samples:
        raise ValueError("no labelled sweep samples found; check data/levels")

    x = torch.from_numpy(np.stack(samples))
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


class LSTMSweepClassifier(nn.Module):
    """Small LSTM over feature windows producing a 3-class direction logit."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the classifier.

        Args:
            n_features: Number of input features per timestep.
            hidden_size: LSTM hidden dimension.
            num_layers: Stacked LSTM layers.
            dropout: Dropout between LSTM layers (ignored when 1 layer).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, 3)`` for classes {-1, 0, 1}."""
        output, _ = self.lstm(x)
        logits: torch.Tensor = self.head(output[:, -1, :])
        return logits


class LSTMStrategy(BaseStrategy):
    """Fixed-position-size strategy driven by a trained classifier.

    Pre-computes predictions over the full feature matrix at construction;
    ``act()`` then just replays them bar by bar.
    """

    def __init__(
        self,
        model: LSTMSweepClassifier,
        features: pd.DataFrame,
        window: int = 60,
        long_size: float = 1.0,
        short_size: float = -1.0,
        threshold: float = 0.5,
    ) -> None:
        """Run inference once and store per-bar actions.

        Args:
            model: Trained classifier (switched to eval mode internally).
            features: Feature matrix covering the evaluation bars.
            window: Feature window used at prediction time.
            long_size: Positive action magnitude for upward sweeps.
            short_size: Negative action magnitude for downward sweeps.
            threshold: Minimum softmax probability to act.
        """
        model.eval()
        values = np.nan_to_num(features.to_numpy(dtype=np.float32), nan=0.0)
        n = len(values)
        actions = np.zeros(n, dtype=float)
        with torch.no_grad():
            for i in range(window, n):
                seq = torch.from_numpy(values[i - window : i]).unsqueeze(0)
                probs = torch.softmax(model(seq), dim=-1)[0]
                if probs[2] >= threshold:  # class index 2 == label +1
                    actions[i] = long_size
                elif probs[0] >= threshold:  # class index 0 == label -1
                    actions[i] = short_size
        self._signals = actions
        super().__init__(len(actions))

    def _signal(self, idx: int) -> float:
        return float(self._signals[idx])
