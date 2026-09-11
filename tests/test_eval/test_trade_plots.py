"""Tests for MAE/MFE, hold-time, win-rate, and time-of-day diagnostic charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from quant_rl.eval.trade_diagnostics import (
    closed_trades_table,
    mae_mfe_table,
    rolling_win_rate,
    weekday_hour_pnl,
)
from quant_rl.eval.trade_plots import (
    plot_hold_time_pnl,
    plot_mae_mfe,
    plot_rolling_winrate,
    plot_tod_heatmap,
    write_trade_diagnostics,
)


def _trades() -> pd.DataFrame:
    idx = pd.date_range("2026-03-02 16:30", periods=40, freq="15min")
    rows: list[dict[str, object]] = []
    for i, t in enumerate(idx[::4]):
        close_i = min(i * 4 + 2, len(idx) - 1)
        rows.append(
            {
                "type": "open",
                "direction": 1 if i % 2 == 0 else -1,
                "price": 20000.0 + i,
                "time": t,
                "bar": i * 4,
                "lots": 1.0,
            }
        )
        rows.append(
            {
                "type": "tp_close" if i % 3 == 0 else "close",
                "direction": 1 if i % 2 == 0 else -1,
                "price": 20000.0 + i + (5 if i % 2 == 0 else -4),
                "time": idx[close_i],
                "bar": close_i,
                "pnl": 20.0 if i % 2 == 0 else -8.0,
            }
        )
    return pd.DataFrame(rows)


def _bars() -> pd.DataFrame:
    import numpy as np

    idx = pd.date_range("2026-03-02 16:30", periods=40, freq="15min")
    close = 20000.0 + np.linspace(0, 10, 40)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        },
        index=idx,
    )


def test_closed_trades_table_hold_time() -> None:
    closed = closed_trades_table(_trades())
    assert not closed.empty
    assert (closed["hold_min"] > 0).all()
    assert set(closed["win"].unique()).issubset({True, False})


def test_rolling_win_rate_window() -> None:
    closed = closed_trades_table(_trades())
    roll = rolling_win_rate(closed, window=5)
    assert len(roll) == len(closed)
    assert roll.dropna().between(0, 100).all()


def test_weekday_hour_heatmap_shape() -> None:
    import numpy as np

    closed = closed_trades_table(_trades())
    matrix, weekdays, hours = weekday_hour_pnl(closed)
    assert matrix.shape[0] == 7
    assert len(weekdays) == 7
    assert len(hours) == matrix.shape[1]
    assert np.isfinite(matrix).any()


def test_mae_mfe_positive_adverse() -> None:
    table = mae_mfe_table(_bars(), _trades(), lots=1.0, contract_size=1.0)
    assert not table.empty
    assert (table["mae_usd"] >= -1e-9).all()
    assert (table["mfe_usd"] >= -1e-9).all()


def test_diagnostic_pngs(tmp_path) -> None:
    write_trade_diagnostics(
        tmp_path,
        _trades(),
        _bars(),
        dpi=72,
        save_plots=True,
        save_html=False,
    )
    for name in ("hold_time_pnl.png", "rolling_winrate.png", "tod_heatmap.png", "mae_mfe.png"):
        path = tmp_path / name
        assert path.exists() and path.stat().st_size > 0


def test_empty_hold_time_still_writes(tmp_path) -> None:
    fig = plot_hold_time_pnl(pd.DataFrame(), out_path=tmp_path / "empty.png", dpi=72)
    assert (tmp_path / "empty.png").stat().st_size > 0
    del fig


def test_plot_mae_mfe_and_winrate_smoke(tmp_path) -> None:
    closed = closed_trades_table(_trades())
    mae = mae_mfe_table(_bars(), _trades())
    plot_mae_mfe(mae, out_path=tmp_path / "mae.png", dpi=72)
    plot_rolling_winrate(closed, out_path=tmp_path / "wr.png", dpi=72)
    plot_tod_heatmap(closed, out_path=tmp_path / "tod.png", dpi=72)
    assert (tmp_path / "mae.png").stat().st_size > 0


def test_interactive_hold_time(tmp_path) -> None:
    pytest.importorskip("plotly")
    from quant_rl.eval.trade_plots import plot_hold_time_pnl_html

    plot_hold_time_pnl_html(closed_trades_table(_trades()), out_path=tmp_path / "h.html")
    assert (tmp_path / "h.html").stat().st_size > 0
