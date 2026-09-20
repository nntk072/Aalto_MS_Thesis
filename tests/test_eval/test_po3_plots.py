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


def test_candle_body_width_matches_bar_spacing(signal_bars: pd.DataFrame) -> None:
    """Candle bodies must be on the order of bar spacing, not ~0.7 date-days."""
    from matplotlib.patches import Rectangle

    from quant_rl.eval.po3_plots import _draw_candles, _resample_ohlc

    ohlcv = _resample_ohlc(signal_bars, "1min")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_candles(ax, ohlcv)
    widths = [p.get_width() for p in ax.patches if isinstance(p, Rectangle)]
    plt.close(fig)
    assert widths, "expected at least one candle body patch"
    # 1-minute bars are ~1/1440 days; body width should be ≪ 0.1 days.
    assert max(widths) < 0.1
    assert max(widths) < 0.01  # ~0.0004 for 1-min spacing * 0.6


def test_entry_markers_at_candle_close(signal_bars: pd.DataFrame) -> None:
    """Long/short markers sit on the candle close, not wick ± pad or price×1.004."""
    import matplotlib.pyplot as plt

    from quant_rl.eval.po3_plots import _draw_entries, _resample_ohlc

    ohlcv = _resample_ohlc(signal_bars, "1min")
    signals = pd.DataFrame(0, index=signal_bars.index, columns=["entry_long", "entry_short"])
    hit_ts = signal_bars.index[50]
    signals.loc[hit_ts, "entry_long"] = 1
    fig, ax = plt.subplots()
    _draw_entries(ax, signals, ohlcv)
    collections = ax.collections
    plt.close(fig)
    assert collections, "expected entry scatter"
    offsets = np.asarray(collections[0].get_offsets(), dtype=float)
    ys = offsets[:, 1]
    loc = int(ohlcv.index.searchsorted(hit_ts, side="left"))
    assert float(ys[0]) == pytest.approx(float(ohlcv["close"].iloc[loc]))
    # Old bug: high * 1.004 is far from close at ~20k.
    assert abs(float(ys[0]) - float(ohlcv["high"].iloc[loc]) * 1.004) > 1.0


def test_plot_window_keeps_lookback_for_zones(signal_bars: pd.DataFrame, tmp_path) -> None:
    """Display window may be short; zone builder still sees full ``bars`` spine."""
    start = signal_bars.index[100]
    end = signal_bars.index[150]
    signals = detect_po3_entries(signal_bars, htf="M15", primary_tf="M5")
    fig = plot_fvg_signals(
        signal_bars,
        signals,
        window=(start, end),
        candle_tf="1min",
        out_path=tmp_path / "win.png",
        dpi=80,
    )
    assert len(fig.axes) == 1
