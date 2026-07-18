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
from .risk import compute_sl_price, compute_tp_price, compute_lots

if TYPE_CHECKING:
    from ..data.ticks import TickBook

ActionFn = Callable[[np.ndarray], int]


def _bar_quote(row: pd.Series, cost_model: CostModel) -> tuple[float, float]:
    """Derive ``(bid, ask)`` from a bar row (MT5 close = bid)."""
    bar_spread = float(row["spread"]) if "spread" in row.index else None
    return cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread)


def _fill_quote(
    fill_instant: pd.Timestamp,
    row: pd.Series,
    cost_model: CostModel,
    tickbook: "TickBook | None",
) -> tuple[float, float]:
    """Return the fill quote for an order executed at *fill_instant*."""
    if tickbook is not None:
        q = tickbook.quote_at(fill_instant)
        if q is not None:
            return q
    bar_spread = float(row["spread"]) if "spread" in row.index else None
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
    use_structure_sl_tp: bool = True,
    contract_size: float = 1.0,
    default_risk_frac: float = 0.01,
) -> dict[str, Any]:
    """Run a full backtest over aligned bars + features."""
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
        else:
            fill_instant = bar_time + pd.Timedelta("1min")
        fq = _fill_quote(fill_instant, row, cost_model, tickbook)

        # --- Per-trade SL/TP enforcement from structure ---
        if position is not None and use_structure_sl_tp and position.sl_price is not None:
            high = float(row.get("high", row.get("close")))
            low = float(row.get("low", row.get("close")))
            
            if position.direction == 1 and low <= position.sl_price:
                # Long hit SL
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "stop_close",
                    "pnl": pnl,
                    "reason": "structure_sl",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None
            elif position.direction == -1 and high >= position.sl_price:
                # Short hit SL
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "stop_close",
                    "pnl": pnl,
                    "reason": "structure_sl",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None
        
        # --- Per-trade TP enforcement from structure ---
        if position is not None and use_structure_sl_tp and position.tp_price is not None:
            high = float(row.get("high", row.get("close")))
            low = float(row.get("low", row.get("close")))
            
            if position.direction == 1 and high >= position.tp_price:
                # Long hit TP
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "tp_close",
                    "pnl": pnl,
                    "reason": "structure_tp",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None
            elif position.direction == -1 and low <= position.tp_price:
                # Short hit TP
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "tp_close",
                    "pnl": pnl,
                    "reason": "structure_tp",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        if position is not None and max_loss_per_trade_usd is not None:
            if acc.open_pnl <= -abs(max_loss_per_trade_usd):
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "stop_close",
                    "pnl": pnl,
                    "reason": "max_loss_per_trade",
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        if position is not None and take_profit_per_trade_usd is not None:
            if acc.open_pnl >= abs(take_profit_per_trade_usd):
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "tp_close",
                    "pnl": pnl,
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
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "forced_close",
                    "pnl": pnl,
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

        if action != 0:
            if position is not None and position.direction != action:
                pnl = broker.close_position(acc, position, fq)
                trade_log.append({
                    "type": "close",
                    "pnl": pnl,
                    "bar": i,
                    "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

            if position is None:
                position = broker.open_position(acc, fq, lots, action)
                if position:
                    # Compute structure-based SL/TP if available
                    if use_structure_sl_tp and "last_swing_low_price" in features.columns and "last_swing_high_price" in features.columns:
                        last_swing_low = float(features.iloc[i].get("last_swing_low_price", np.nan))
                        last_swing_high = float(features.iloc[i].get("last_swing_high_price", np.nan))
                        
                        position.sl_price = compute_sl_price(
                            position.entry_price,
                            action,
                            last_swing_low,
                            last_swing_high,
                            buffer_pts=1.0,
                        )
                        
                        # Compute TP with default R:R ratio
                        rr_ratio = 2.0
                        position.tp_price = compute_tp_price(
                            position.entry_price,
                            position.sl_price,
                            action,
                            rr_ratio=rr_ratio,
                        )
                    
                    trade_log.append({
                        "type": "open",
                        "direction": action,
                        "price": position.entry_price,
                        "lots": lots,
                        "bar": i,
                        "time": bar_time,
                        "equity": acc.equity,
                        "sl_price": position.sl_price if position.sl_price else np.nan,
                        "tp_price": position.tp_price if position.tp_price else np.nan,
                        "last_swing_low": float(features.iloc[i].get("last_swing_low_price", np.nan)) if use_structure_sl_tp else np.nan,
                        "last_swing_high": float(features.iloc[i].get("last_swing_high_price", np.nan)) if use_structure_sl_tp else np.nan,
                    })
                    sessions_with_trades.add(session)

        elif action == 0 and position is not None:
            pnl = broker.close_position(acc, position, fq)
            trade_log.append({
                "type": "close",
                "pnl": pnl,
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
        pnl = broker.close_position(acc, position, last_fq)
        trade_log.append({
            "type": "eod_close",
            "pnl": pnl,
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
