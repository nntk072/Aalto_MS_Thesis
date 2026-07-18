"""Compute trade-level chart metrics: MAE, MFE, SL, TP price levels and timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeChartMetrics:
    """Chart overlay metrics for a single trade.

    Attributes
    ----------
    entry_price : float
        The entry price (fill price of the open order).
    exit_price : float
        The exit price (fill price of the close order).
    direction : int
        Trade direction: +1 for long, -1 for short.
    mae_price : float
        Maximum Adverse Excursion price (worst price during trade).
    mfe_price : float
        Maximum Favorable Excursion price (best price during trade).
    mae_time : pd.Timestamp
        Timestamp of the bar where MAE occurred.
    mfe_time : pd.Timestamp
        Timestamp of the bar where MFE occurred.
    sl_price : float | None
        Stop Loss level based on max_loss_per_trade_usd. None if not configured.
    tp_price : float | None
        Take Profit level based on take_profit_per_trade_usd. None if not configured.
    """

    entry_price: float
    exit_price: float
    direction: int
    mae_price: float
    mfe_price: float
    mae_time: pd.Timestamp
    mfe_time: pd.Timestamp
    sl_price: float | None
    tp_price: float | None


def compute_trade_metrics(
    bars: pd.DataFrame,
    open_row: pd.Series,
    close_row: pd.Series,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    lots: float = 1.0,
    contract_size: float = 1.0,
) -> TradeChartMetrics:
    """Compute chart metrics for a single trade.

    Extracts the trade window from open time to close time (inclusive),
    computes MAE/MFE from in-trade bar extrema, and obtains SL/TP levels
    from trade log (if available) or computes from configuration limits.

    If open_row contains 'sl_price' or 'tp_price' (logged by structure-aware engine),
    those values take precedence over computed defaults.

    Parameters
    ----------
    bars : pd.DataFrame
        M1 price bars with index as DatetimeIndex and columns: open, high, low, close.
    open_row : pd.Series
        Trade open record with fields: time, direction, price, and optionally sl_price, tp_price.
    close_row : pd.Series
        Trade close record with fields: time, price.
    max_loss_per_trade_usd : float | None
        Maximum loss limit in USD; if set, SL level is computed (fallback).
    take_profit_per_trade_usd : float | None
        Maximum profit limit in USD; if set, TP level is computed (fallback).
    lots : float
        Position size in lots (default 1.0).
    contract_size : float
        Contract size (default 1.0).

    Returns
    -------
    TradeChartMetrics
        Computed metrics for chart rendering.
    """
    t_open = pd.Timestamp(open_row["time"])
    t_close = pd.Timestamp(close_row["time"])
    direction = int(open_row["direction"]) if pd.notna(open_row.get("direction")) else 0
    entry_price = float(open_row["price"]) if pd.notna(open_row.get("price")) else np.nan
    exit_price = float(close_row["price"]) if pd.notna(close_row.get("price")) else np.nan

    # Extract bars within [t_open, t_close]
    trade_bars = bars.loc[(bars.index >= t_open) & (bars.index <= t_close)]

    # Fallback: if exit_price is nan, use close price from the close bar
    if np.isnan(exit_price) and len(trade_bars) > 0:
        exit_price = trade_bars["close"].iloc[-1]

    # Compute MAE and MFE
    mfe_idx: pd.Timestamp
    mae_idx: pd.Timestamp
    if len(trade_bars) > 0:
        if direction == 1:  # Long
            mfe_price = trade_bars["high"].max()
            mae_price = trade_bars["low"].min()
            mfe_idx = pd.Timestamp(trade_bars["high"].idxmax())
            mae_idx = pd.Timestamp(trade_bars["low"].idxmin())
        else:  # Short (-1)
            mfe_price = trade_bars["low"].min()
            mae_price = trade_bars["high"].max()
            mfe_idx = pd.Timestamp(trade_bars["low"].idxmin())
            mae_idx = pd.Timestamp(trade_bars["high"].idxmax())
    else:
        # Fallback if no bars in window
        mfe_price = entry_price
        mae_price = entry_price
        mfe_idx = t_open
        mae_idx = t_open

    # Compute SL and TP levels
    sl_price = None
    tp_price = None

    # Prefer per-trade SL/TP from trade log (structure-aware)
    if pd.notna(open_row.get("sl_price")):
        sl_price = float(open_row["sl_price"])
    elif max_loss_per_trade_usd is not None and not np.isnan(entry_price):
        # Fallback: compute from USD-based config
        price_move = max_loss_per_trade_usd / (lots * contract_size)
        if direction == 1:  # Long
            sl_price = entry_price - price_move
        else:  # Short
            sl_price = entry_price + price_move

    # Prefer per-trade TP from trade log (structure-aware)
    if pd.notna(open_row.get("tp_price")):
        tp_price = float(open_row["tp_price"])
    elif take_profit_per_trade_usd is not None and not np.isnan(entry_price):
        # Fallback: compute from USD-based config
        price_move = take_profit_per_trade_usd / (lots * contract_size)
        if direction == 1:  # Long
            tp_price = entry_price + price_move
        else:  # Short
            tp_price = entry_price - price_move

    return TradeChartMetrics(
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        mae_price=mae_price,
        mfe_price=mfe_price,
        mae_time=mae_idx,
        mfe_time=mfe_idx,
        sl_price=sl_price,
        tp_price=tp_price,
    )
