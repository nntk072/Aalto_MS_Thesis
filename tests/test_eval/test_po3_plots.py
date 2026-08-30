"""Smoke tests for PO3/IFVG signal charts.

Uses the matplotlib Agg backend so tests run headless in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from quant_rl.eval.plots_interactive import plot_fvg_signals_interactive
from quant_rl.eval.po3_plots import plot_fvg_signals
from quant_rl.features.po3_config import detect_fvg, detect_po3_entries


@pytest.fixture()
def signal_bars() -> pd.DataFrame:
    """Synthetic M1 bars with forced imbalances so zones are non-empty."""
    idx = pd.date_range("2025-01-01", periods=200, freq="1min")
    close = 20000.0 + np.cumsum(np.random.default_rng(7).normal(0, 1.5, 200))
    close[20] = close[19] + 12  # create FVG
    close[25] = close[24] - 14  # create bearish move
    return pd.DataFrame(
        {
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        },
        index=idx,
    )


@pytest.fixture()
def po3_signals(signal_bars: pd.DataFrame) -> pd.DataFrame:
    return detect_po3_entries(signal_bars, htf="M15", primary_tf="M5")


def test_static_plot_writes_png(
    signal_bars: pd.DataFrame, po3_signals: pd.DataFrame, tmp_path
) -> None:
    out = tmp_path / "po3.png"
    fig = plot_fvg_signals(signal_bars, po3_signals, candle_tf="1min", out_path=out, dpi=80)
    assert out.exists()
    assert out.stat().st_size > 0
    assert len(fig.axes) == 1


def test_static_plot_with_window(
    signal_bars: pd.DataFrame, po3_signals: pd.DataFrame, tmp_path
) -> None:
    start = signal_bars.index[10]
    end = signal_bars.index[40]
    fig = plot_fvg_signals(
        signal_bars, po3_signals, window=(start, end), candle_tf="1min", out_path=tmp_path / "w.png"
    )
    # Window plotted: 31 bars -> candlestick patches fit within that range.
    assert len(fig.axes) == 1


def test_static_plot_no_entries(signal_bars: pd.DataFrame, po3_signals: pd.DataFrame) -> None:
    fig = plot_fvg_signals(signal_bars, po3_signals, candle_tf="5min", show_entries=False)
    assert len(fig.axes) == 1


def test_static_plot_empty_data_raises() -> None:
    bars = pd.DataFrame(columns=["open", "high", "low", "close"])
    with pytest.raises(ValueError):
        plot_fvg_signals(bars, pd.DataFrame(), candle_tf="1min")


def test_interactive_plot_writes_html(
    signal_bars: pd.DataFrame, po3_signals: pd.DataFrame, tmp_path
) -> None:
    out = tmp_path / "po3.html"
    fig = plot_fvg_signals_interactive(signal_bars, po3_signals, candle_tf="1min", out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert len(fig.data) >= 1


def test_interactive_plot_has_zone_shapes(
    signal_bars: pd.DataFrame, po3_signals: pd.DataFrame
) -> None:
    fig = plot_fvg_signals_interactive(signal_bars, po3_signals, candle_tf="1min")
    assert len(fig.layout.shapes) > 0


def test_interactive_plot_window(signal_bars: pd.DataFrame, po3_signals: pd.DataFrame) -> None:
    start = signal_bars.index[0]
    end = signal_bars.index[50]
    fig = plot_fvg_signals_interactive(
        signal_bars, po3_signals, window=(start, end), candle_tf="1min"
    )
    assert len(fig.data) >= 1


def test_static_plot_with_plain_fvg(signal_bars: pd.DataFrame, tmp_path) -> None:
    """Works with just FVG signal columns (no HTF/IFVG columns)."""
    fvg = detect_fvg(signal_bars)
    fig = plot_fvg_signals(signal_bars, fvg, candle_tf="5min", out_path=tmp_path / "fvg.png")
    assert len(fig.axes) == 1
