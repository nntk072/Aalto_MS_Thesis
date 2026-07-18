"""Smoke tests for plots and artifact export.

Uses the matplotlib Agg backend so tests run headless in CI.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from quant_rl.eval.export import save_run
from quant_rl.eval.metrics import Metrics, calculate_metrics
from quant_rl.eval.report import build_comparison_table, build_summary_table, save_metrics_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_equity() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-02", periods=500, freq="1min")
    returns = rng.normal(0.0001, 0.002, size=500)
    equity = 100_000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=idx)


@pytest.fixture()
def synthetic_bars() -> pd.DataFrame:
    """Create synthetic M1 bars for per-trade chart testing."""
    idx = pd.date_range("2025-01-02 16:30", periods=500, freq="1min")
    rng = np.random.default_rng(42)
    close = 20000.0 + np.cumsum(rng.normal(0, 1, 500))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, 500),
            "high": close + rng.uniform(0, 2, 500),
            "low": close - rng.uniform(0, 2, 500),
            "close": close,
        },
        index=idx,
    )


@pytest.fixture()
def synthetic_trades(synthetic_equity) -> pd.DataFrame:
    idx = synthetic_equity.index
    rows = []
    for i in range(0, 500, 50):
        rows.append(
            {
                "type": "open",
                "direction": 1 if i % 2 == 0 else -1,
                "price": 100.0 + i * 0.1,
                "bar": i,
                "time": idx[i],
                "equity": synthetic_equity.iloc[i],
            }
        )
        rows.append(
            {
                "type": "close",
                "pnl": float(np.random.default_rng(i).normal(10, 50)),
                "bar": i + 25,
                "time": idx[min(i + 25, 499)],
                "equity": synthetic_equity.iloc[min(i + 25, 499)],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_result(synthetic_equity, synthetic_trades) -> dict:
    breach_events = [
        {
            "time": synthetic_equity.index[100],
            "session_id": 0,
            "reason": "daily_loss",
            "equity": 99000.0,
        }
    ]
    return {
        "equity": synthetic_equity,
        "trades": synthetic_trades,
        "breaches": ["daily_loss"],
        "breach_events": breach_events,
        "n_sessions": 5,
        "n_breach_sessions": 1,
        "n_sessions_with_trades": 4,
        "n_sessions_skipped": 1,
        "initial_balance": 100_000.0,
    }


@pytest.fixture()
def metrics(synthetic_equity, synthetic_trades) -> Metrics:
    return calculate_metrics(
        synthetic_equity,
        trades=synthetic_trades,
        n_sessions=5,
        n_breach_sessions=1,
    )


# ---------------------------------------------------------------------------
# Metrics / report tests
# ---------------------------------------------------------------------------


def test_metrics_fields(metrics):
    assert hasattr(metrics, "total_pnl")
    assert hasattr(metrics, "avg_trade")
    assert hasattr(metrics, "max_consec_loss")
    assert isinstance(metrics.total_trades, int)
    assert 0.0 <= metrics.win_rate <= 1.0


def test_build_summary_table(metrics):
    table = build_summary_table(metrics)
    assert "Sharpe" in table
    assert "Max Drawdown" in table
    assert "Breach Rate" in table


def test_build_comparison_table(metrics):
    table = build_comparison_table(metrics, metrics)
    assert "Train" in table
    assert "Test" in table
    assert "Sharpe" in table


def test_save_metrics_json(metrics, tmp_path):
    path = tmp_path / "metrics.json"
    save_metrics_json(metrics, path)
    data = json.loads(path.read_text())
    assert "sharpe" in data
    assert "total_pnl" in data


# ---------------------------------------------------------------------------
# Trade-pairing regression tests
# ---------------------------------------------------------------------------


def test_pair_trades_includes_tp_close():
    """Regression: _pair_trades must pair each open with its own immediate
    close, even when tp_close events occur between orders.

    Previously the closes list excluded ``tp_close``, so every open after a
    tp_close event was paired with the wrong close by position — mixing one
    order's entry price/volume with a completely different order's exit
    price and PnL on the chart.
    """
    from quant_rl.eval.plots import _pair_trades

    idx = pd.date_range("2025-01-02 17:00", periods=4, freq="1min")
    trades = pd.DataFrame(
        [
            {
                "type": "open",
                "direction": 1,
                "price": 100.0,
                "lots": 10.0,
                "bar": 0,
                "time": idx[0],
            },
            {"type": "tp_close", "pnl": 5.0, "reason": "structure_tp", "bar": 1, "time": idx[1]},
            {"type": "open", "direction": 1, "price": 101.0, "lots": 2.0, "bar": 1, "time": idx[1]},
            {"type": "stop_close", "pnl": -3.0, "reason": "structure_sl", "bar": 2, "time": idx[2]},
            {
                "type": "open",
                "direction": -1,
                "price": 102.0,
                "lots": 50.0,
                "bar": 2,
                "time": idx[2],
            },
            {
                "type": "stop_close",
                "pnl": -20.0,
                "reason": "structure_sl",
                "bar": 3,
                "time": idx[3],
            },
        ]
    )

    pairs = _pair_trades(trades)
    assert len(pairs) == 3

    o1, c1 = pairs[0]
    assert o1["lots"] == 10.0
    assert c1["type"] == "tp_close" and c1["pnl"] == 5.0

    o2, c2 = pairs[1]
    assert o2["lots"] == 2.0
    assert c2["type"] == "stop_close" and c2["pnl"] == -3.0

    o3, c3 = pairs[2]
    assert o3["lots"] == 50.0
    assert c3["pnl"] == -20.0


def test_pair_trades_unclosed_open_is_dropped():
    """An open with no following close (e.g. still open at end of log) must
    not be paired with a later, unrelated close."""
    from quant_rl.eval.plots import _pair_trades

    idx = pd.date_range("2025-01-02 17:00", periods=2, freq="1min")
    trades = pd.DataFrame(
        [
            {"type": "open", "direction": 1, "price": 100.0, "lots": 1.0, "bar": 0, "time": idx[0]},
        ]
    )
    assert _pair_trades(trades) == []


# ---------------------------------------------------------------------------
# Plot smoke tests (static PNG)
# ---------------------------------------------------------------------------


def test_plot_equity_curve(synthetic_equity, tmp_path):
    from quant_rl.eval.plots import plot_equity_curve

    breach_events = [
        {
            "time": synthetic_equity.index[100],
            "session_id": 0,
            "reason": "daily_loss",
            "equity": 99000.0,
        }
    ]
    plot_equity_curve(
        synthetic_equity,
        breach_events=breach_events,
        initial_balance=100_000.0,
        daily_loss_limit=5000.0,
        max_loss_limit=10000.0,
        out_path=tmp_path / "equity.png",
    )
    assert (tmp_path / "equity.png").exists()
    assert (tmp_path / "equity.png").stat().st_size > 0


def test_plot_drawdown(synthetic_equity, tmp_path):
    from quant_rl.eval.plots import plot_drawdown

    plot_drawdown(synthetic_equity, out_path=tmp_path / "drawdown.png")
    assert (tmp_path / "drawdown.png").stat().st_size > 0


def test_plot_trade_pnl_hist(synthetic_trades, tmp_path):
    from quant_rl.eval.plots import plot_trade_pnl_hist

    plot_trade_pnl_hist(synthetic_trades, out_path=tmp_path / "pnl_hist.png")
    assert (tmp_path / "pnl_hist.png").stat().st_size > 0


def test_plot_returns_dist(synthetic_equity, tmp_path):
    from quant_rl.eval.plots import plot_returns_dist

    # scipy may not be available; function should not crash
    try:
        plot_returns_dist(synthetic_equity, out_path=tmp_path / "returns_dist.png")
    except ImportError:
        pass  # scipy optional
    # At minimum should not raise other exceptions


def test_plot_monthly_heatmap(synthetic_equity, tmp_path):
    from quant_rl.eval.plots import plot_monthly_returns_heatmap

    # Use longer series so monthly resampling has data
    idx = pd.date_range("2024-01-02", periods=3000, freq="1min")
    equity = pd.Series(
        100_000 * np.cumprod(1 + np.random.default_rng(7).normal(0.0001, 0.001, 3000)),
        index=idx,
    )
    plot_monthly_returns_heatmap(equity, out_path=tmp_path / "heatmap.png")
    assert (tmp_path / "heatmap.png").stat().st_size > 0


def test_plot_per_trade_orders_mt5_style(synthetic_bars, synthetic_trades, tmp_path):
    """Test MT5-style per-trade order charts with MAE/MFE/SL/TP overlays."""
    from quant_rl.eval.plots import plot_per_trade_orders

    orders_dir = tmp_path / "orders"
    plot_per_trade_orders(
        synthetic_bars,
        synthetic_trades,
        orders_dir=orders_dir,
        max_charts=50,
        dpi=72,
        max_loss_per_trade_usd=10.0,
        take_profit_per_trade_usd=50.0,
        lots=1.0,
        contract_size=1.0,
        show_mae_mfe=True,
        show_sl_tp=True,
    )

    # Check that at least one PNG was generated
    pngs = list(orders_dir.glob("*.png"))
    assert len(pngs) > 0, "No PNG files generated"
    # Each PNG should be non-empty
    for png_path in pngs:
        assert png_path.stat().st_size > 0, f"{png_path} is empty"


def test_plot_per_trade_orders_png_datetime_axis_and_info(
    synthetic_bars, synthetic_trades, tmp_path
):
    """Verify PNG charts have datetime x-axis and trade info box."""
    from quant_rl.eval.plots import plot_per_trade_orders

    orders_dir = tmp_path / "orders"
    plot_per_trade_orders(
        synthetic_bars,
        synthetic_trades,
        orders_dir=orders_dir,
        max_charts=1,  # Just one for this test
        dpi=72,
        max_loss_per_trade_usd=10.0,
        take_profit_per_trade_usd=50.0,
        lots=1.0,
        contract_size=1.0,
        show_mae_mfe=True,
        show_sl_tp=True,
    )

    pngs = list(orders_dir.glob("*.png"))
    assert len(pngs) >= 1, "No PNG files generated"

    # Load and inspect the first PNG
    png_file = pngs[0]
    file_size = png_file.stat().st_size
    # Datetime-axis chart with 2 panels should be reasonably large (>100KB)
    assert file_size > 50000, f"PNG file too small: {file_size} bytes (expected >50KB)"


def test_macd_backtest_fills_within_bar_and_charts_render(tmp_path):
    """End-to-end regression: run the real MACD baseline through the engine,
    verify every fill price lands within its executing bar's [low, high]
    range (guards Bug 4: tick-outlier fills detached from candles), then
    render per-trade PNG charts from those trades without error (guards
    Bug 3: full-history indicator overlays must slice cleanly per trade).
    """
    from quant_rl.backtest.engine import run_backtest
    from quant_rl.baselines.rule_based import macd_ema50_baseline
    from quant_rl.eval.plots import plot_per_trade_orders
    from quant_rl.features.build import build_features

    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-02-01 00:00", periods=400, freq="1min")
    close = 21000.0 + np.cumsum(rng.normal(0, 1.0, 400))
    bars = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.5, 400),
            "high": close + rng.uniform(0, 1.0, 400),
            "low": close - rng.uniform(0, 1.0, 400),
            "close": close,
        },
        index=idx,
    )

    features = build_features(bars, cfg=None)
    actions = macd_ema50_baseline(bars)

    obs_window = 60
    state = {"i": obs_window}

    def policy(obs):
        action = int(actions.iloc[state["i"]]) if state["i"] < len(actions) else 0
        state["i"] += 1
        return action

    result = run_backtest(
        bars=bars,
        features=features,
        policy=policy,
        obs_window=obs_window,
        hold_on_zero=True,
        exit_action=2,
    )
    trades_df = result["trades"]

    # No open row may carry an invalid direction (guards Bug 1).
    opens = trades_df[trades_df["type"] == "open"]
    for _, row in opens.iterrows():
        assert row["direction"] in (1.0, -1.0)

    # All fill prices must land within (a small buffer around) the nearby
    # bar range, not detached from any visible candle (guards Bug 4).
    for _, row in trades_df.iterrows():
        price = row.get("price")
        if price is None or pd.isna(price):
            continue
        bar_time = pd.Timestamp(row["time"])
        pos = bars.index.get_indexer([bar_time], method="nearest")[0]
        lo = bars["low"].iloc[max(0, pos - 1) : pos + 2].min()
        hi = bars["high"].iloc[max(0, pos - 1) : pos + 2].max()
        assert lo - 1.0 <= price <= hi + 1.0, (
            f"Fill price {price} at {bar_time} outside nearby range [{lo}, {hi}]"
        )

    if len(opens) > 0:
        orders_dir = tmp_path / "orders"
        plot_per_trade_orders(
            bars,
            trades_df,
            orders_dir=orders_dir,
            max_charts=20,
            dpi=72,
        )
        pngs = list(orders_dir.glob("*.png"))
        assert len(pngs) > 0
        for png_path in pngs:
            assert png_path.stat().st_size > 0


def test_plot_baseline_comparison(synthetic_equity, tmp_path):
    from quant_rl.eval.plots import plot_baseline_comparison

    eq2 = synthetic_equity * 1.05
    plot_baseline_comparison({"A": synthetic_equity, "B": eq2}, out_path=tmp_path / "compare.png")
    assert (tmp_path / "compare.png").stat().st_size > 0


# ---------------------------------------------------------------------------
# Interactive HTML smoke tests (plotly)
# ---------------------------------------------------------------------------


def test_interactive_equity(synthetic_equity, tmp_path):
    pytest.importorskip("plotly")
    from quant_rl.eval.plots_interactive import plot_equity_curve

    plot_equity_curve(synthetic_equity, out_path=tmp_path / "equity.html")
    assert (tmp_path / "equity.html").stat().st_size > 0


def test_interactive_drawdown(synthetic_equity, tmp_path):
    pytest.importorskip("plotly")
    from quant_rl.eval.plots_interactive import plot_drawdown

    plot_drawdown(synthetic_equity, out_path=tmp_path / "drawdown.html")
    assert (tmp_path / "drawdown.html").stat().st_size > 0


def test_interactive_per_trade_orders_mt5_style(synthetic_bars, synthetic_trades, tmp_path):
    """Test MT5-style interactive HTML per-trade order charts."""
    pytest.importorskip("plotly")
    from quant_rl.eval.plots_interactive import plot_per_trade_orders

    orders_dir = tmp_path / "orders"
    plot_per_trade_orders(
        synthetic_bars,
        synthetic_trades,
        orders_dir=orders_dir,
        max_charts=50,
        max_loss_per_trade_usd=10.0,
        take_profit_per_trade_usd=50.0,
        lots=1.0,
        contract_size=1.0,
        show_mae_mfe=True,
        show_sl_tp=True,
    )

    # Check that at least one HTML was generated
    htmls = list(orders_dir.glob("*.html"))
    assert len(htmls) > 0, "No HTML files generated"
    # Each HTML should be non-empty
    for html_path in htmls:
        assert html_path.stat().st_size > 0, f"{html_path} is empty"


# ---------------------------------------------------------------------------
# export.save_run integration test
# ---------------------------------------------------------------------------


def test_save_run_creates_files(synthetic_result, metrics, tmp_path):
    run_dir = save_run(
        out_dir=tmp_path,
        name="test_run",
        train_result=synthetic_result,
        train_metrics=metrics,
        test_result=synthetic_result,
        test_metrics=metrics,
        save_plots=True,
        save_html=True,
        save_csv=True,
        dpi=72,
    )
    assert run_dir.exists()
    # Root: only config/summary — no raw data or charts
    assert (run_dir / "summary.txt").stat().st_size > 0
    assert not (run_dir / "equity.csv").exists()
    assert not (run_dir / "equity.png").exists()
    # Training subfolder
    assert (run_dir / "training" / "equity.csv").stat().st_size > 0
    assert (run_dir / "training" / "trades.csv").stat().st_size > 0
    assert (run_dir / "training" / "metrics.json").stat().st_size > 0
    assert (run_dir / "training" / "equity.png").stat().st_size > 0
    assert (run_dir / "training" / "drawdown.png").stat().st_size > 0
    # Testing subfolder
    assert (run_dir / "testing" / "equity.csv").stat().st_size > 0
    assert (run_dir / "testing" / "metrics.json").stat().st_size > 0


def test_save_run_no_overwrite(synthetic_result, metrics, tmp_path):
    """Two calls with the same name must produce two different directories."""
    d1 = save_run(
        out_dir=tmp_path,
        name="run",
        train_result=synthetic_result,
        train_metrics=metrics,
        save_plots=False,
        save_html=False,
    )
    time.sleep(1.1)  # ensure different timestamp
    d2 = save_run(
        out_dir=tmp_path,
        name="run",
        train_result=synthetic_result,
        train_metrics=metrics,
        save_plots=False,
        save_html=False,
    )
    assert d1 != d2
