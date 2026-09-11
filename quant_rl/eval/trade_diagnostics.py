"""Closed-trade tables for MAE/MFE, hold time, win rate, and session heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from quant_rl.eval.plots import _pair_trades
from quant_rl.eval.trade_metrics import compute_trade_metrics

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_SL_TYPES = {"stop_close", "forced_close"}
_TP_TYPES = {"tp_close"}


def closed_trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per completed trade: times, PnL, side, lots, close type."""
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for open_row, close_row in _pair_trades(trades):
        t_open = pd.Timestamp(open_row["time"])
        t_close = pd.Timestamp(close_row["time"])
        pnl = float(close_row["pnl"]) if pd.notna(close_row.get("pnl")) else 0.0
        lots = float(open_row["lots"]) if pd.notna(open_row.get("lots")) else 1.0
        direction = int(open_row["direction"]) if pd.notna(open_row.get("direction")) else 0
        hold_min = max((t_close - t_open).total_seconds() / 60.0, 0.0)
        rows.append(
            {
                "t_open": t_open,
                "t_close": t_close,
                "direction": direction,
                "entry_price": (
                    float(open_row["price"]) if pd.notna(open_row.get("price")) else np.nan
                ),
                "exit_price": (
                    float(close_row["price"]) if pd.notna(close_row.get("price")) else np.nan
                ),
                "pnl": pnl,
                "lots": lots,
                "hold_min": hold_min,
                "close_type": str(close_row.get("type", "")),
                "win": pnl >= 0.0,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def mae_mfe_table(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    lots: float = 1.0,
    contract_size: float = 1.0,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
) -> pd.DataFrame:
    """MAE/MFE in USD plus whether price reached the logged SL/TP."""
    if trades.empty or bars.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for open_row, close_row in _pair_trades(trades):
        trade_lots = float(open_row["lots"]) if pd.notna(open_row.get("lots")) else lots
        metrics = compute_trade_metrics(
            bars,
            open_row,
            close_row,
            max_loss_per_trade_usd=max_loss_per_trade_usd,
            take_profit_per_trade_usd=take_profit_per_trade_usd,
            lots=trade_lots,
            contract_size=contract_size,
        )
        notional = trade_lots * contract_size
        entry = metrics.entry_price
        if metrics.direction == 1:
            mae_usd = (entry - metrics.mae_price) * notional
            mfe_usd = (metrics.mfe_price - entry) * notional
        else:
            mae_usd = (metrics.mae_price - entry) * notional
            mfe_usd = (entry - metrics.mfe_price) * notional
        close_type = str(close_row.get("type", ""))
        sl_reached = close_type in _SL_TYPES
        tp_reached = close_type in _TP_TYPES
        if metrics.sl_price is not None:
            if metrics.direction == 1:
                sl_reached = sl_reached or metrics.mae_price <= metrics.sl_price
            else:
                sl_reached = sl_reached or metrics.mae_price >= metrics.sl_price
        if metrics.tp_price is not None:
            if metrics.direction == 1:
                tp_reached = tp_reached or metrics.mfe_price >= metrics.tp_price
            else:
                tp_reached = tp_reached or metrics.mfe_price <= metrics.tp_price
        pnl = float(close_row["pnl"]) if pd.notna(close_row.get("pnl")) else 0.0
        rows.append(
            {
                "mae_usd": float(max(mae_usd, 0.0)),
                "mfe_usd": float(max(mfe_usd, 0.0)),
                "pnl": pnl,
                "win": pnl >= 0.0,
                "sl_reached": bool(sl_reached),
                "tp_reached": bool(tp_reached),
                "close_type": close_type,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def rolling_win_rate(closed: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling win rate in percent, indexed by trade number (1-based)."""
    if closed.empty or "win" not in closed.columns:
        return pd.Series(dtype=float)
    wins = closed["win"].astype(float)
    min_p = min(5, len(wins))
    roll = wins.rolling(window, min_periods=min_p).mean() * 100.0
    roll.index = pd.RangeIndex(1, len(roll) + 1, name="trade")
    return roll


def weekday_hour_pnl(
    closed: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], list[str], list[int]]:
    """Sum PnL heatmap: rows = weekdays, columns = hours that appear in the log."""
    if closed.empty:
        return np.full((7, 0), np.nan), list(_WEEKDAYS), []
    t_open = pd.DatetimeIndex(closed["t_open"])
    hours = sorted({int(h) for h in t_open.hour})
    matrix = np.full((7, len(hours)), np.nan)
    hour_ix = {h: j for j, h in enumerate(hours)}
    tmp = pd.DataFrame(
        {
            "dow": np.asarray(t_open.dayofweek, dtype=int),
            "hour": np.asarray(t_open.hour, dtype=int),
            "pnl": closed["pnl"].to_numpy(dtype=float),
        }
    )
    grouped = tmp.groupby(["dow", "hour"], sort=True)["pnl"].sum().reset_index()
    dows = grouped["dow"].to_numpy(dtype=int)
    hrs = grouped["hour"].to_numpy(dtype=int)
    pnls = grouped["pnl"].to_numpy(dtype=float)
    for dow, hour, pnl in zip(dows, hrs, pnls, strict=True):
        j = hour_ix.get(int(hour))
        if j is None:
            continue
        matrix[int(dow), j] = float(pnl)
    return matrix, list(_WEEKDAYS), hours
