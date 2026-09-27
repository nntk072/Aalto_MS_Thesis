"""Order-chart levels, swing pivots, and the ATR reward-ratio cap."""

from __future__ import annotations

import inspect
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quant_rl.backtest.tp_reach import max_tp_distance, rr_reaches
from quant_rl.envs.strategies.po3_ifvg import PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.eval.chart_levels import (
    at_sample_extreme,
    deviation_levels,
    isolated_spike_mask,
    ohlc_ok_mask,
)
from quant_rl.eval.chart_overlays import (
    OverlayEvents,
    SmtSegment,
    SweepLine,
    SwingRay,
    build_overlay_events,
    draw_overlays_mpl,
)
from quant_rl.eval.order_chart import order_levels
from quant_rl.eval.plots import plot_per_trade_orders
from quant_rl.eval.plots_interactive import plot_per_trade_orders as plot_per_trade_orders_html
from quant_rl.eval.trade_metrics import compute_trade_metrics
from quant_rl.eval.overlay_events import _swing_origins
from quant_rl.features.structure import structure_levels


def test_clip_cuts_smt_that_runs_months_past_the_trade() -> None:
    start = pd.Timestamp("2024-12-31 16:00")
    end = pd.Timestamp("2024-12-31 17:00")
    seg = SmtSegment(
        t0=pd.Timestamp("2024-12-31 16:20"),
        p0=21200.0,
        t1=pd.Timestamp("2025-03-08 16:00"),
        p1=22000.0,
        side="high",
        label="SMT",
    )
    clipped = OverlayEvents(smt=[seg]).clip(start, end)
    assert clipped.smt == []
    inside = SmtSegment(
        t0=pd.Timestamp("2024-12-31 16:20"),
        p0=21200.0,
        t1=pd.Timestamp("2024-12-31 16:40"),
        p1=21240.0,
        side="high",
        label="SMT",
    )
    kept = OverlayEvents(smt=[inside]).clip(start, end)
    assert len(kept.smt) == 1
    assert kept.smt[0].t0 == inside.t0
    assert kept.smt[0].t1 == inside.t1


def test_clip_drops_a_swing_whose_pivot_is_outside_the_window() -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=10, freq="1min")
    ray = SwingRay(t0=idx[0], t1=idx[8], price=10.0, side="high", origin=idx[0])
    clipped = OverlayEvents(swings=[ray]).clip(pd.Timestamp(idx[4]), pd.Timestamp(idx[8]))
    assert clipped.swings == []

    inside = SwingRay(t0=idx[5], t1=idx[9], price=10.0, side="high", origin=idx[5])
    kept = OverlayEvents(swings=[inside]).clip(pd.Timestamp(idx[4]), pd.Timestamp(idx[8]))
    assert len(kept.swings) == 1
    assert kept.swings[0].origin == idx[5]
    assert kept.swings[0].t0 == idx[5]
    assert kept.swings[0].t1 == idx[8]
    fig, ax = plt.subplots()
    draw_overlays_mpl(ax, kept)
    assert len(ax.collections) == 1
    plt.close(fig)


def test_swing_origin_is_the_pivot_candle() -> None:
    n = 40
    idx = pd.date_range("2026-03-02 16:30", periods=n, freq="1min")
    close = np.full(n, 100.0)
    close[20] = 110.0
    high = close + 0.2
    high[20] = 112.0
    low = close - 0.2
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=idx,
    )
    levels = structure_levels(bars, swing_period=3)
    events = build_overlay_events(bars, swing_period=3)
    highs = [r for r in events.swings if r.side == "high"]
    assert highs
    ray = highs[0]
    assert ray.origin is not None
    assert pd.Timestamp(ray.origin) == pd.Timestamp(levels["last_swing_high_time"].dropna().iloc[0])
    loc = int(bars.index.get_indexer(pd.Index([ray.origin]))[0])
    assert float(bars["high"].iloc[loc]) == float(ray.price)


def test_swing_off_the_wick_still_draws_on_that_candle() -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=10, freq="1min")
    high = np.full(10, 100.0)
    high[4] = 112.0
    levels = pd.DataFrame(
        {
            "last_swing_high": np.nan,
            "last_swing_high_time": pd.Series(pd.NaT, index=idx),
        },
        index=idx,
    )
    levels.loc[idx[4], "last_swing_high"] = 999.0
    levels.loc[idx[4], "last_swing_high_time"] = idx[4]
    origins = _swing_origins(levels, pd.Series(high, index=idx), idx, 3, "high")
    assert origins
    origin_i, price = origins[0]
    assert origin_i == 4
    assert price == 112.0


def test_sweep_x_sits_on_the_right_end_of_the_line() -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=8, freq="1min")
    line = SweepLine(t0=idx[1], t1=idx[5], price=101.5, side="high", label="X")
    fig, ax = plt.subplots()
    draw_overlays_mpl(ax, OverlayEvents(sweeps=[line]))
    ann = cast(Any, ax.texts[0])
    assert ann.get_text() == "X"
    assert ann.get_ha() == "left"
    assert ann.get_va() == "center"
    assert ann.xy[1] == 101.5
    plt.close(fig)


def test_order_png_legend_omits_vwap(tmp_path, monkeypatch) -> None:
    idx = pd.date_range("2026-03-02 16:30", periods=80, freq="1min")
    close = 20000.0 + np.linspace(0, 5, 80)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "tickvol": np.full(80, 10),
        },
        index=idx,
    )
    trades = pd.DataFrame(
        [
            {
                "type": "open",
                "direction": 1,
                "price": float(close[20]),
                "time": idx[20],
                "bar": 20,
                "lots": 1.0,
            },
            {
                "type": "close",
                "direction": 1,
                "price": float(close[40]),
                "time": idx[40],
                "bar": 40,
                "pnl": 1.0,
                "reason": "close",
            },
        ]
    )
    labels: list[str] = []
    orig = Figure.savefig

    def _save(self, *args, **kwargs):
        legend = self.axes[0].get_legend()
        if legend is not None:
            labels.extend(t.get_text() for t in legend.get_texts())
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _save)
    plot_per_trade_orders(bars, trades, orders_dir=tmp_path, max_charts=1, dpi=72, show_sl_tp=False)
    assert labels
    assert "VWAP" not in labels
    assert "EMA50" in labels


def test_both_plotters_call_prepare_order_chart() -> None:
    assert "prepare_order_chart" in inspect.getsource(plot_per_trade_orders)
    assert "prepare_order_chart" in inspect.getsource(plot_per_trade_orders_html)


def test_trade_like_13_keeps_mfe_on_the_wick_and_off_the_rr_target() -> None:
    idx = pd.date_range("2025-01-06 18:21", periods=42, freq="1min")
    high = np.full(42, 21690.0)
    high[8] = 21699.45
    low = np.full(42, 21650.0)
    low[30] = 21641.55
    close = np.full(42, 21670.0)
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=idx,
    )
    open_row = pd.Series(
        {
            "time": idx[0],
            "direction": 1,
            "price": 21679.55,
            "sl_price": 21451.35,
            "tp_price": 22421.44,
            "rr_ratio": 3.25,
        }
    )
    close_row = pd.Series({"time": idx[-1], "price": 21658.75, "pnl": -55.68})
    metrics = compute_trade_metrics(bars, open_row, close_row)
    assert metrics.mfe_price == 21699.45
    assert metrics.tp_time is None
    levels = order_levels(
        bars,
        metrics,
        OverlayEvents(),
        bars,
        idx[0],
        idx[-1],
        open_row,
    )
    assert levels.ylim[1] < 21800
    assert any(note.startswith("TP 22421.44") for note in levels.notes)
    assert all(seg.label != "TP" for seg in levels.segs)
    assert levels.green is not None
    assert levels.green[1] <= levels.ylim[1]
    assert levels.green[1] < 21800
    mfe = next(seg for seg in levels.segs if seg.label == "MFE")
    assert mfe.price == 21699.45
    assert mfe.mark == idx[8]


def test_isolated_spike_is_ignored_and_news_is_kept() -> None:
    idx = pd.date_range("2026-01-02 16:30", periods=12, freq="1min")
    high = np.full(12, 100.0)
    low = np.full(12, 99.0)
    high[6] = 500.0
    bars = pd.DataFrame(
        {"open": np.full(12, 99.5), "high": high, "low": low, "close": np.full(12, 99.5)},
        index=idx,
    )
    assert isolated_spike_mask(bars)[6]
    news = bars.copy()
    news.loc[news.index[6] :, "high"] = 500.0
    news.loc[news.index[6] :, "close"] = 499.0
    news.loc[news.index[6] :, "open"] = 100.0
    assert not isolated_spike_mask(news)[6]
    broken = bars.copy()
    broken.loc[broken.index[3], "low"] = 200.0
    assert not ohlc_ok_mask(broken)[3]


def test_deviation_levels_skip_far_multiples_and_need_an_extreme() -> None:
    levels = deviation_levels(100.0, 110.0, 98.0, 112.0)
    ks = [k for k, _ in levels]
    assert 0.0 in ks and 1.0 in ks
    assert 8.0 not in ks
    idx = pd.date_range("2026-01-02 16:30", periods=30, freq="1min")
    high = np.linspace(100, 110, 30)
    bars = pd.DataFrame(
        {"open": high, "high": high + 0.2, "low": high - 0.2, "close": high},
        index=idx,
    )
    # Last bar is the sample high.
    assert at_sample_extreme(bars, idx[-1], 1)
    fallen = bars.copy()
    fallen.loc[fallen.index[-1], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    assert not at_sample_extreme(fallen, idx[-1], 1)


def test_ny_trade_window_ignores_earlier_pivots() -> None:
    idx = pd.date_range("2024-12-30 07:00", "2024-12-31 23:00", freq="1min")
    close = np.linspace(21000.0, 21200.0, len(idx))
    bars = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close},
        index=idx,
    )
    t_open = pd.Timestamp("2024-12-31 16:32")
    t_close = pd.Timestamp("2024-12-31 16:33")
    open_row = pd.Series(
        {
            "time": t_open,
            "direction": 1,
            "price": float(bars.loc[t_open, "close"]),
            "type": "open",
            "lots": 1.0,
        }
    )
    close_row = pd.Series(
        {
            "time": t_close,
            "price": float(bars.loc[t_close, "close"]),
            "pnl": -100.0,
            "type": "stop_close",
        }
    )
    events = OverlayEvents(
        swings=[
            SwingRay(
                t0=pd.Timestamp("2024-12-30 07:15"),
                t1=pd.Timestamp("2024-12-31 18:00"),
                price=21100.0,
                side="high",
                origin=pd.Timestamp("2024-12-30 07:15"),
            ),
            SwingRay(
                t0=pd.Timestamp("2024-12-31 10:00"),
                t1=pd.Timestamp("2024-12-31 18:00"),
                price=21150.0,
                side="low",
                origin=pd.Timestamp("2024-12-31 10:00"),
            ),
        ],
        sweeps=[
            SweepLine(
                t0=pd.Timestamp("2024-12-31 10:00"),
                t1=pd.Timestamp("2024-12-31 16:40"),
                price=21150.0,
                side="low",
                label="SSL",
            )
        ],
        smt=[
            SmtSegment(
                t0=pd.Timestamp("2024-12-30 08:00"),
                p0=21100.0,
                t1=pd.Timestamp("2024-12-31 12:00"),
                p1=21140.0,
                side="high",
                label="SMT",
            )
        ],
    )
    flat = pd.Series(close, index=idx)
    overlays = {
        key: flat.copy()
        for key in ("ema50", "macd", "signal", "histogram", "rsi", "vwap")
    }
    from quant_rl.eval.chart_indicators import break_overlay_gaps
    from quant_rl.eval.order_window import prepare_order_chart

    prepared = prepare_order_chart(
        bars,
        open_row,
        close_row,
        overlay_events=events,
        full_overlays=overlays,
        context_bars=60,
    )
    assert prepared is not None
    assert prepared.window.index[0] == pd.Timestamp("2024-12-31 15:32")
    assert prepared.window.index[-1] <= pd.Timestamp("2024-12-31 17:33")
    assert all(ts.normalize() == pd.Timestamp("2024-12-31") for ts in prepared.window.index)
    assert t_open in prepared.window.index
    assert t_close in prepared.window.index
    assert prepared.events.smt == []
    assert prepared.events.swings == []
    assert prepared.events.sweeps == []

    gapped = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.to_datetime(
            [
                "2024-12-30 22:59",
                "2024-12-30 23:00",
                "2024-12-31 16:30",
                "2024-12-31 16:31",
            ]
        ),
    )
    broken = break_overlay_gaps({"ema50": gapped})["ema50"]
    assert broken.isna().any()
    assert list(broken.dropna()) == [1.0, 2.0, 3.0, 4.0]


def _session(
    start: str, n: int, price: float, span: float, atr: float
) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range(start, periods=n, freq="1min")
    high = np.full(n, price + span / 2)
    low = np.full(n, price - span / 2)
    bars = pd.DataFrame(
        {"open": np.full(n, price), "high": high, "low": low, "close": np.full(n, price)},
        index=idx,
    )
    return bars, pd.Series(np.full(n, atr), index=idx)


def test_rr_cap_scales_with_atr_not_a_fixed_point_distance() -> None:
    quiet_bars, quiet_atr = _session("2026-01-02 16:30", 30, 100.0, span=20.0, atr=2.0)
    quiet_bars2, quiet_atr2 = _session("2026-01-05 16:30", 30, 100.0, span=20.0, atr=2.0)
    quiet = pd.concat([quiet_bars, quiet_bars2])
    qatr = pd.concat([quiet_atr, quiet_atr2])
    qdist = max_tp_distance(quiet, qatr)
    # Prior session range 20 / atr 2 = 10 ATR, times current atr 2 → 20 points.
    assert qdist.iloc[-1] == 20.0
    assert not rr_reaches(100.0, 50.0, 4.0, float(qdist.iloc[-1]))

    big_bars, big_atr = _session("2200-01-02 16:30", 30, 1_000_000.0, span=2000.0, atr=200.0)
    big_bars2, big_atr2 = _session("2200-01-05 16:30", 30, 1_000_000.0, span=2000.0, atr=200.0)
    big = pd.concat([big_bars, big_bars2])
    batr = pd.concat([big_atr, big_atr2])
    bdist = max_tp_distance(big, batr)
    # Same 10 ATR ratio, but the point cap is 2000. A 50-point stop at 4R now fits.
    assert bdist.iloc[-1] == 2000.0
    assert rr_reaches(100.0, 50.0, 4.0, float(bdist.iloc[-1]))


def test_structural_target_past_the_cap_is_ignored() -> None:
    idx = pd.date_range("2026-01-02 16:30", periods=40, freq="1min")
    close = np.full(40, 100.0)
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "session": "ny"},
        index=idx,
    )
    n = 40
    feats = pd.DataFrame(
        {
            "atr_5": np.ones(n),
            "asian_high": np.full(n, 102.0),
            "asian_low": np.full(n, 98.0),
            "sweep_high": np.zeros(n),
            "sweep_low": np.ones(n),
            "po3_manipulation_low": np.full(n, 95.0),
            "po3_manipulation_high": np.full(n, 105.0),
            "po3_manipulation_end": np.zeros(n),
            "po3_distribution": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "ifvg_bear_low": np.full(n, 99.0),
            "ifvg_bear_high": np.full(n, 100.0),
        },
        index=idx,
    )
    env = TradingEnv(
        bars,
        feats,
        strategy_actions=True,
        strategy=PO3IFVGStrategy(enforce_gate=False),
        obs_window=5,
        rr_ratio_range=(1.5, 5.0),
    )
    row = pd.Series({"prev_day_high": 1000.0, "sweep_high_level": np.nan})
    tp = env._resolve_trader_tp(
        direction=1,
        entry_price=100.0,
        sl_price=90.0,
        rr_ratio=1.5,
        feat_row=row,
        max_tp_distance=30.0,
    )
    assert tp == 115.0
    nearer = env._resolve_trader_tp(
        direction=1,
        entry_price=100.0,
        sl_price=90.0,
        rr_ratio=1.5,
        feat_row=pd.Series({"prev_day_high": 140.0, "sweep_high_level": 130.0}),
        max_tp_distance=50.0,
    )
    assert nearer == 130.0
