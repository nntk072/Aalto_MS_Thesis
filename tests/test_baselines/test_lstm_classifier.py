"""Tests for the LSTM sweep classifier, dataset builder and strategy wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quant_rl.baselines import (
    LSTMStrategy,
    LSTMSweepClassifier,
    build_sweep_dataset,
)

# Real torch training inside these tests: minutes on CPU. Tagged slow so the
# `-m 'not slow'` inner loop stays fast; CI runs the full suite.
pytestmark = pytest.mark.slow


def _make_bars_with_sweep(n: int = 200) -> pd.DataFrame:
    """Range-bound bars that break out upward near the end."""
    rng = np.random.default_rng(7)
    close = np.concatenate(
        [
            20_000.0 + rng.normal(0, 2.0, n - 10),
            20_000.0 + 50.0 + rng.normal(0, 1.0, 10),  # breakout above range
        ]
    )
    index = pd.date_range("2025-01-02 01:05", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.uniform(900, 1_100, n),
        },
        index=index,
    )


@pytest.mark.unit
class TestLSTMSweepClassifier:
    def test_forward_output_shape(self) -> None:
        # Arrange
        model = LSTMSweepClassifier(n_features=8)
        batch = torch.randn(4, 60, 8)

        # Act
        logits = model(batch)

        # Assert
        assert logits.shape == (4, 3)

    def test_overfits_single_batch(self) -> None:
        # Arrange
        torch.manual_seed(0)
        model = LSTMSweepClassifier(n_features=4)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()
        x, y = torch.randn(16, 30, 4), torch.randint(0, 3, (16,))
        first_loss = float(loss_fn(model(x), y))

        # Act
        for _ in range(200):
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()

        # Assert
        assert float(loss) < first_loss
        assert float(loss) < 0.1


@pytest.mark.unit
class TestBuildSweepDataset:
    def test_labels_are_in_valid_set(self) -> None:
        # Arrange
        bars = _make_bars_with_sweep()
        features = pd.DataFrame(
            {"ret_1": np.zeros(len(bars)), "atr_5": np.ones(len(bars))}, index=bars.index
        )

        # Act
        x, y = build_sweep_dataset(bars, features, window=20, horizon=12)

        # Assert
        assert x.dtype == torch.float32 and y.dtype == torch.int64
        assert set(y.unique().tolist()).issubset({-1, 0, 1})  # type: ignore[no-untyped-call]
        assert len(x) == len(y) > 0

    def test_no_samples_raises(self) -> None:
        # Arrange — flat bars never sweep any level
        index = pd.date_range("2025-01-02 01:05", periods=30, freq="15min")
        flat = np.full(30, 20_000.0)
        bars = pd.DataFrame(
            {
                "open": flat,
                "high": flat + 1.0,
                "low": flat - 1.0,
                "close": flat,
                "volume": np.full(30, 1_000.0),
            },
            index=index,
        )
        features = pd.DataFrame({"a": np.zeros(30)}, index=index)

        # Act / Assert
        with pytest.raises(ValueError, match="no labelled sweep samples"):
            build_sweep_dataset(bars, features, window=10, horizon=2)


@pytest.mark.integration
class TestLSTMStrategy:
    def test_replays_predictions_as_actions(self) -> None:
        # Arrange
        torch.manual_seed(0)
        bars = _make_bars_with_sweep()
        features = pd.DataFrame(
            np.random.default_rng(1).normal(size=(len(bars), 4)).astype(np.float32),
            index=bars.index,
        )
        model = LSTMSweepClassifier(n_features=4)
        strategy = LSTMStrategy(model, features, window=30)

        # Act
        actions = [strategy.act({}) for _ in range(40)]

        # Assert
        assert len(actions) == 40
        assert all(a in (-1.0, 0.0, 1.0) for a in actions)
