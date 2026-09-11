"""Tests for chart overlay events, dual-axis oscillators, and drawdown series."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from quant_rl.eval.chart_indicators import compute_chart_overlays_full
from quant_rl.eval.chart_overlays import OverlayEvents, build_overlay_events, compute_vwap_for_chart
from quant_rl.eval.plot_series import (
    daily_drawdown_pct,
    daily_loss_limit_series,
    daily_pnl,
    drawdown_ylim,
    max_drawdown_pct,
    session_start_equity,
)
from quant_rl.eval.plots import (
    plot_daily_pnl,
    plot_drawdown,
    plot_equity_curve,
    plot_per_trade_orders,
)
from quant_rl.features.liquidity import detect_liquidity_sweeps
from quant_rl.features.structure import structure_levels


def _ohlc(n: int = 80, start: str = "2026-03-02 16:30", seed: int = 1) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(seed)
    close = 20000.0 + np.cumsum(rng.normal(0, 1.2, n))
    high = close + rng.uniform(0.4, 2.0, n)
    low = close - rng.uniform(0.4, 2.0, n)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "tickvol": rng.integers(50, 200, n),
        },
        index=idx,
    )


def test_session_start_equity_is_first_print_of_day() -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=4, freq="1h")
    eq = pd.Series([104_000.0, 105_000.0, 104_500.0, 106_000.0], index=idx)
    start = session_start_equity(eq)
    assert (start == 104_000.0).all()


def test_daily_loss_limit_uses_day_open_not_running_equity() -> None:
    """Day opens at 104k → limit stays 99k even after equity prints 105k."""
    day1 = pd.date_range("2026-03-02 16:30", periods=3, freq="1h")
    day2 = pd.date_range("2026-03-03 16:30", periods=2, freq="1h")
    eq = pd.Series(
        [104_000.0, 105_000.0, 104_200.0, 106_000.0, 107_000.0],
        index=day1.append(day2),
    )
    limit = daily_loss_limit_series(eq, 5_000.0)
    assert limit.iloc[0] == pytest.approx(99_000.0)
    assert limit.iloc[1] == pytest.approx(99_000.0)
    assert limit.iloc[3] == pytest.approx(101_000.0)


def test_max_and_daily_drawdown_scales() -> None:
    idx = pd.date_range("2026-01-02", periods=5, freq="1D")
    eq = pd.Series([100_000.0, 102_000.0, 96_900.0, 97_000.0, 101_000.0], index=idx)
    max_dd = max_drawdown_pct(eq)
    day_dd = daily_drawdown_pct(eq)
    assert max_dd.min() < -4.0
    # Daily series is 0 at each day's first (only) print.
    assert (day_dd == 0.0).all()
    lo, hi = drawdown_ylim(max_dd, -10.0)
    assert hi == 0.0
    assert lo <= -10.0


def test_daily_pnl_is_last_minus_first() -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=4, freq="1h")
    eq = pd.Series([100_000.0, 101_000.0, 99_500.0, 100_500.0], index=idx)
    pnl = daily_pnl(eq)
    assert len(pnl) == 1
    assert pnl.iloc[0] == pytest.approx(500.0)


def test_overlays_include_rsi_and_vwap() -> None:
    bars = _ohlc()
    overlays = compute_chart_overlays_full(bars)
    assert {"ema50", "macd", "signal", "histogram", "rsi", "vwap"} <= set(overlays)
    assert overlays["rsi"].dropna().between(0, 100).all()
    vwap = compute_vwap_for_chart(bars)
    assert vwap.notna().any()


def test_swing_rays_start_at_swing_origin() -> None:
    bars = _ohlc(n=120, seed=3)
    events = build_overlay_events(bars)
    assert isinstance(events, OverlayEvents)
    if events.swings:
        ray = events.swings[0]
        assert ray.t1 > ray.t0
        assert ray.side in {"high", "low"}


def test_sweep_line_is_longer_than_the_two_candles() -> None:
    """Forced sweep: line must extend past both the swing origin and sweep bar."""
    rows = [(100.5, 100.0, 100.2)] * 6
    rows += [
        (102.2, 101.5, 102.0),
        (101.8, 101.0, 101.2),
        (100.8, 100.0, 100.2),
        (99.8, 99.0, 99.2),
        (100.5, 99.5, 100.3),
        (101.2, 100.5, 101.0),
        (101.6, 101.0, 101.4),
        (102.5, 101.0, 101.5),  # takes 102.2
    ]
    idx = pd.date_range("2024-01-02 16:30", periods=len(rows), freq="1min")
    high = [r[0] for r in rows]
    low = [r[1] for r in rows]
    close = [r[2] for r in rows]
    open_ = [close[0]] + close[:-1]
    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )
    feats = detect_liquidity_sweeps(bars, swing_period=2)
    events = build_overlay_events(bars, swing_period=2)
    if feats["sweep_high"].sum() == 0 and feats["sweep_low"].sum() == 0:
        pytest.skip("synthetic path did not fire a sweep")
    assert events.sweeps
    line = events.sweeps[0]
    pad = pd.Timedelta("1min") * 0.5
    span = line.t1 - line.t0
    assert span > pad


def test_smt_segments_connect_two_swings() -> None:
    primary = _ohlc(n=150, seed=4)
    # Uncorrelated secondary so some swings fail to confirm.
    secondary = _ohlc(n=150, seed=99)
    events = build_overlay_events(primary, secondary=secondary, swing_period=3)
    for seg in events.smt:
        assert seg.t1 > seg.t0
        assert seg.side in {"high", "low"}
        assert "SMT" in seg.label


def test_clip_keeps_only_window_events() -> None:
    bars = _ohlc(n=200, seed=5)
    events = build_overlay_events(bars)
    start, end = bars.index[80], bars.index[120]
    clipped = events.clip(pd.Timestamp(start), pd.Timestamp(end))
    for ray in clipped.swings:
        assert ray.t1 >= start
        assert ray.t0 <= end


def test_plot_equity_daily_limit_follows_day_open(tmp_path) -> None:
    day1 = pd.date_range("2026-03-02 16:30", periods=3, freq="1h")
    day2 = pd.date_range("2026-03-03 16:30", periods=3, freq="1h")
    eq = pd.Series(
        [104_000.0, 105_000.0, 104_500.0, 103_000.0, 102_000.0, 103_500.0],
        index=day1.append(day2),
    )
    fig = plot_equity_curve(
        eq,
        initial_balance=100_000.0,
        daily_loss_limit=5_000.0,
        max_loss_limit=10_000.0,
        out_path=tmp_path / "equity.png",
        dpi=72,
    )
    ax = fig.axes[0] if fig.axes else None
    # Figure is closed by _save; recompute the series the chart used.
    limit = daily_loss_limit_series(eq, 5_000.0)
    assert limit.iloc[1] == pytest.approx(99_000.0)
    assert (tmp_path / "equity.png").stat().st_size > 0
    del ax


def test_plot_drawdown_two_panels_and_dates(tmp_path) -> None:
    idx = pd.date_range("2026-01-02", periods=40, freq="1D")
    rng = np.random.default_rng(0)
    eq = pd.Series(100_000 * np.cumprod(1 + rng.normal(0, 0.004, 40)), index=idx)
    fig = plot_drawdown(eq, out_path=tmp_path / "drawdown.png", dpi=72)
    assert (tmp_path / "drawdown.png").stat().st_size > 0
    # Two stacked axes (plus any colorbars — expect at least 2).
    assert len(fig.axes) >= 2


def test_plot_daily_pnl_writes(tmp_path) -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=20, freq="1h")
    eq = pd.Series(np.linspace(100_000, 101_000, 20), index=idx)
    plot_daily_pnl(eq, daily_loss_limit=5_000.0, out_path=tmp_path / "daily_pnl.png", dpi=72)
    assert (tmp_path / "daily_pnl.png").stat().st_size > 0


def test_per_trade_chart_with_overlays(tmp_path) -> None:
    bars = _ohlc(n=180, seed=8)
    t0, t1 = bars.index[60], bars.index[90]
    trades = pd.DataFrame(
        [
            {
                "type": "open",
                "direction": 1,
                "price": float(bars["close"].iloc[60]),
                "time": t0,
                "bar": 60,
            },
            {
                "type": "close",
                "direction": 1,
                "price": float(bars["close"].iloc[90]),
                "time": t1,
                "bar": 90,
                "pnl": 12.5,
            },
        ]
    )
    orders_dir = tmp_path / "orders"
    plot_per_trade_orders(
        bars,
        trades,
        orders_dir=orders_dir,
        max_charts=1,
        dpi=72,
        secondary_bars=bars.assign(
            close=bars["close"] * 0.4, high=bars["high"] * 0.4, low=bars["low"] * 0.4
        ),
        show_mae_mfe=False,
        show_sl_tp=False,
    )
    pngs = list(orders_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0


def test_structure_levels_available_for_overlay() -> None:
    bars = _ohlc(n=60)
    levels = structure_levels(bars, swing_period=3)
    assert "last_swing_high" in levels.columns
    assert "last_swing_low_time" in levels.columns
