"""Event-driven backtester engine.

Fill-price semantics
--------------------
Every order fill uses a ``(bid, ask)`` quote obtained in this priority:

1. **Tick data** (``tickbook`` is not None): ``TickBook.quote_at(fill_instant)``
   where ``fill_instant`` is the open time of the *next* bar.
2. **Bar-spread fallback**: ``CostModel.bar_quote(close)``.

Mark-to-market and per-trade stop checks use the bar-close quote (long @ bid,
short @ ask).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from .account import AccountState
from .broker import Broker, Position
from .costs import COST_US100, CostModel
from .guardrails import FTMOGuardrails
from .risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short

if TYPE_CHECKING:
    from ..data.ticks import TickBook

ActionFn = Callable[[np.ndarray], int]


def _bar_spread_price_units(row: pd.Series, cost_model: CostModel) -> float | None:
    """Convert the bar's raw MT5 ``spread`` column (broker points) to price units.

    MT5 CSVs report ``spread`` in integer broker points (e.g. ``80``), not
    price units. ``CostModel.bar_quote``/``fill_price`` expect the spread
    already in price units (e.g. ``0.6`` for US100), so it must be scaled by
    ``point_size`` here. Without this conversion a raw value like 80 was
    being added directly to the close price, producing bid/ask quotes tens
    to hundreds of points away from the visible candle.
    """
    if "spread" not in row.index or pd.isna(row["spread"]):
        return None
    return float(row["spread"]) * cost_model.point_size


def _bar_quote(row: pd.Series, cost_model: CostModel) -> tuple[float, float]:
    """Derive ``(bid, ask)`` from a bar row (MT5 close = bid)."""
    bar_spread = _bar_spread_price_units(row, cost_model)
    return cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread)


def _fill_quote(
    fill_instant: pd.Timestamp,
    row: pd.Series,
    cost_model: CostModel,
    tickbook: "TickBook | None",
) -> tuple[float, float]:
    """Return the fill quote for an order executed at *fill_instant*.

    *row* must be the bar whose time range contains *fill_instant* (i.e. the
    bar actually being executed into) — it is used both as the bar-quote
    fallback and as a sanity range for tick fills.

    Guards against tick-data gaps/outliers: a ``TickBook`` forward-fills
    quotes across data gaps, which can return a stale tick far from the
    current candle. If the tick quote's mid price falls outside this bar's
    ``[low, high]`` range (widened by a small buffer for spread), the tick
    fill is rejected in favour of a bar-based quote, so fills always land
    on/near a visible candle.
    """
    bar_spread = _bar_spread_price_units(row, cost_model)
    if tickbook is not None:
        q = tickbook.quote_at(fill_instant)
        if q is not None:
            bid, ask = q
            mid = (bid + ask) / 2.0
            lo = float(row["low"])
            hi = float(row["high"])
            buffer = max(hi - lo, bar_spread or 0.0, cost_model.spread_points)
            if lo - buffer <= mid <= hi + buffer:
                return q
            # Outlier/gap tick (mid far outside the executing bar's range):
            # fall through to the bar-based quote below instead of returning
            # a fill price disconnected from the visible candles.
    return cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread)


def run_backtest(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    policy: ActionFn,
    obs_window: int = 60,
    cost_model: CostModel = COST_US100,
    broker_kwargs: dict | None = None,
    guardrail_kwargs: dict | None = None,
    initial_balance: float = 100_000.0,
    lots: float = 1.0,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    tickbook: "TickBook | None" = None,
    use_structure_sl_tp: bool = False,
    risk_frac: float = 0.01,
    rr_ratio: float = 2.0,
    swing_buffer_pts: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    contract_size: float = 1.0,
    hold_on_zero: bool = False,
    exit_action: int | None = None,
) -> dict[str, Any]:
    """Run a full backtest over aligned bars + features.
    
    Parameters
    ----------
    use_structure_sl_tp : bool
        If True, compute SL/TP from last_swing_low/high in features.
    risk_frac : float
        Risk fraction of equity for lot sizing.
    rr_ratio : float
        Reward/risk ratio for TP calculation.
    swing_buffer_pts : float
        Buffer in price points beyond swing for SL placement.
    min_lot, max_lot : float
        Lot size bounds.
    contract_size : float
        Contract multiplier.
    hold_on_zero : bool
        If True, action=0 means hold current position (don't close).
    exit_action : int | None
        If set, policy returning this value closes position (e.g., 2 for MACD exit).
    """
    broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
    guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))
    acc = AccountState(initial_balance=initial_balance)

    equity_curve: list[float] = []
    trade_log: list[dict] = []
    breach_log: list[str] = []
    breach_events: list[dict] = []
    breached_sessions: set[int] = set()
    session_set: set[int] = set()
    sessions_with_trades: set[int] = set()

    position: Position | None = None
    prev_session: int | None = None

    common_idx = bars.index.intersection(features.index)
    bars = bars.loc[common_idx]
    features = features.loc[common_idx]

    bar_times = bars.index
    feat_array = features.values.astype(np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0)
    n_bars = len(bars)

    for i in range(obs_window, n_bars):
        row = bars.iloc[i]
        bar_time = bar_times[i]
        session = int(row["session_id"]) if "session_id" in row.index else 0
        session_set.add(session)

        if session != prev_session:
            acc.reset_daily()
            prev_session = session

        bq = _bar_quote(row, cost_model)

        if position is not None:
            broker.mark_to_market(acc, position, bq)

        if i + 1 < n_bars:
            fill_instant = bar_times[i + 1]
            fill_row = bars.iloc[i + 1]
        else:
            fill_instant = bar_time + pd.Timedelta("1min")
            fill_row = row
        fq = _fill_quote(fill_instant, fill_row, cost_model, tickbook)

        if position is not None and max_loss_per_trade_usd is not None:
            # Check intrabar high/low vs SL
            sl_hit = False
            if position.sl_price is not None:
                if position.direction == 1 and float(row["low"]) <= position.sl_price:
                    sl_hit = True
                elif position.direction == -1 and float(row["high"]) >= position.sl_price:
                    sl_hit = True

            if sl_hit:
                pnl, fill_price = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "stop_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "structure_sl",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None
            # Fallback: check safety cap
            elif acc.open_pnl <= -abs(max_loss_per_trade_usd):
                pnl, fill_price = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "stop_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "max_loss_cap",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        # Check TP (structure-based if available, else use global)
        tp_hit = False
        if position is not None:
            if position.tp_price is not None:
                if position.direction == 1 and float(row["high"]) >= position.tp_price:
                    tp_hit = True
                elif position.direction == -1 and float(row["low"]) <= position.tp_price:
                    tp_hit = True

        if tp_hit and position is not None:
            pnl, fill_price = broker.close_position(acc, position, fq)
            trade_log.append({
                "type": "tp_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_tp",
                "bar": i,
                "time": bar_time,
                "equity": acc.equity,
            })
            sessions_with_trades.add(session)
            position = None
        elif not use_structure_sl_tp and take_profit_per_trade_usd is not None:
            # Fallback: global USD-based TP
            if acc.open_pnl >= abs(take_profit_per_trade_usd):
                pnl, fill_price = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "tp_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "take_profit_per_trade",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        reason = guardrails.breach_reason(acc)
        if reason and session not in breached_sessions:
            breached_sessions.add(session)
            breach_log.append(reason)
            breach_events.append({
                "time": bar_time,
                "session_id": session,
                "reason": reason,
                "equity": acc.equity,
            })
            if position is not None:
                pnl, fill_price = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "forced_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": reason,
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        equity_curve.append(acc.equity)

        if session in breached_sessions:
            continue

        obs = feat_array[i - obs_window : i].copy()
        action = policy(obs)

        if action == exit_action and position is not None:
            # Explicit exit action (e.g., 2 for MACD)
            pnl, fill_price = broker.close_position(acc, position, fq)
            trade_log.append({
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "bar": i,
                "time": bar_time,
                "equity": acc.equity,
            })
            sessions_with_trades.add(session)
            position = None

        elif action in (1, -1):
            if position is not None and position.direction != action:
                pnl, fill_price = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "close",
                    "pnl": pnl,
                    "price": fill_price,
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

            if position is None:
                # Extract structure features if available
                sl_price = None
                tp_price = None
                actual_lots = lots
                
                if use_structure_sl_tp:
                    # Try to extract swing levels from features
                    feat_row = features.iloc[i]
                    last_swing_low = (
                        feat_row.get("last_swing_low")
                        if "last_swing_low" in feat_row.index
                        else np.nan
                    )
                    last_swing_high = (
                        feat_row.get("last_swing_high")
                        if "last_swing_high" in feat_row.index
                        else np.nan
                    )

                    entry_price_for_calc = float(fq[1] if action == 1 else fq[0])

                    # Compute SL/TP based on direction
                    if action == 1 and not np.isnan(last_swing_low):
                        # Long
                        sl_price, tp_price = compute_sl_tp_long(
                            entry_price_for_calc,
                            last_swing_low,
                            buffer_pts=swing_buffer_pts,
                            rr_ratio=rr_ratio,
                        )
                        actual_lots = compute_lots(
                            acc.equity,
                            risk_frac,
                            entry_price_for_calc,
                            sl_price,
                            contract_size=contract_size,
                            min_lot=min_lot,
                            max_lot=max_lot,
                            max_loss_cap=max_loss_per_trade_usd,
                        )
                    elif action == -1 and not np.isnan(last_swing_high):
                        # Short
                        sl_price, tp_price = compute_sl_tp_short(
                            entry_price_for_calc,
                            last_swing_high,
                            buffer_pts=swing_buffer_pts,
                            rr_ratio=rr_ratio,
                        )
                        actual_lots = compute_lots(
                            acc.equity,
                            risk_frac,
                            entry_price_for_calc,
                            sl_price,
                            contract_size=contract_size,
                            min_lot=min_lot,
                            max_lot=max_lot,
                            max_loss_cap=max_loss_per_trade_usd,
                        )

                position = broker.open_position(acc, fq, actual_lots, action)
                if position:
                    # Attach SL/TP/risk info
                    position.sl_price = sl_price
                    position.tp_price = tp_price
                    position.risk_frac = risk_frac
                    position.rr_ratio = rr_ratio
                    
                    trade_log.append({
                        "type": "open",
                        "direction": action,
                        "price": position.entry_price,
                        "lots": position.size,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "risk_frac": risk_frac,
                        "rr_ratio": rr_ratio,
                        "bar": i,
                        "time": bar_time,
                        "equity": acc.equity,
                    })
                    sessions_with_trades.add(session)

        elif action == 0 and position is not None and not hold_on_zero:
            # Close position on action=0 (unless hold_on_zero is True)
            pnl, fill_price = broker.close_position(acc, position, fq)
            trade_log.append({
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "bar": i,
                "time": bar_time,
                "equity": acc.equity,
            })
            sessions_with_trades.add(session)
            position = None

    if position is not None:
        last_row = bars.iloc[-1]
        last_time = bar_times[-1]
        last_fq = _fill_quote(last_time, last_row, cost_model, tickbook)
        pnl, fill_price = broker.close_position(acc, position, last_fq)
        trade_log.append({
            "type": "eod_close",
            "pnl": pnl,
            "price": fill_price,
            "bar": n_bars - 1,
            "time": last_time,
            "equity": acc.equity,
        })

    trades_df = pd.DataFrame(trade_log)
    equity_series = pd.Series(equity_curve, index=bar_times[obs_window:])

    return {
        "equity": equity_series,
        "trades": trades_df,
        "account": acc,
        "breaches": breach_log,
        "breach_events": breach_events,
        "n_sessions": len(session_set),
        "n_breach_sessions": len(breached_sessions),
        "n_sessions_with_trades": len(sessions_with_trades),
        "n_sessions_skipped": len(breached_sessions),
    }
