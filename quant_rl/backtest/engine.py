"""Event-driven backtester engine.

Iterates bar by bar over a feature + price DataFrame, calls a policy for
actions, manages the broker/account/guardrails, and collects an equity curve.
"""
from __future__ import annotations

from typing import Callable, Any

import numpy as np
import pandas as pd

from .account import AccountState
from .broker import Broker, Position
from .costs import CostModel, COST_US100
from .guardrails import FTMOGuardrails
from ..data.session import add_session_id


ActionFn = Callable[[np.ndarray], int]   # obs → discrete action {-1, 0, +1}


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
) -> dict[str, Any]:
    """Run a full backtest.

    Parameters
    ----------
    bars:
        Price DataFrame (must contain ``close``, ``session_id``).
    features:
        Feature matrix aligned to *bars* index.
    policy:
        Function (obs_array) → int action in {-1, 0, +1}.
    obs_window:
        Number of bars in the rolling observation window fed to policy.
    max_loss_per_trade_usd:
        Hard per-trade stop in USD (unrealised). None = disabled.
    """
    broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
    guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))
    acc = AccountState(initial_balance=initial_balance)

    equity_curve: list[float] = []
    trade_log: list[dict] = []
    breach_log: list[str] = []
    breach_events: list[dict] = []          # NEW: rich breach event records
    breached_sessions: set[int] = set()
    session_set: set[int] = set()
    sessions_with_trades: set[int] = set()  # NEW: diagnostic counter

    position: Position | None = None
    prev_session: int | None = None
    common_idx = bars.index.intersection(features.index)
    bars = bars.loc[common_idx]
    features = features.loc[common_idx]

    bar_times = bars.index
    feat_array = features.values.astype(np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0)

    for i in range(obs_window, len(bars)):
        row = bars.iloc[i]
        price = row["close"]
        bar_time = bar_times[i]
        session = int(row["session_id"]) if "session_id" in row.index else 0
        session_set.add(session)

        # Session reset
        if session != prev_session:
            acc.reset_daily()
            prev_session = session

        # Mark-to-market
        if position is not None:
            broker.mark_to_market(acc, position, price)

        # Per-trade hard stop (independent of global guardrails)
        if position is not None and max_loss_per_trade_usd is not None:
            unrealised = (
                (price - position.entry_price)
                * position.direction
                * position.size
                * broker.contract_size
            )
            if unrealised <= -abs(max_loss_per_trade_usd):
                pnl = broker.close_position(acc, position, price)
                trade_log.append({
                    "type": "stop_close", "pnl": pnl,
                    "reason": "max_loss_per_trade", "bar": i, "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        # Guardrail check — one breach event per session
        reason = guardrails.breach_reason(acc)
        if reason and session not in breached_sessions:
            breached_sessions.add(session)
            breach_log.append(reason)
            breach_events.append({          # NEW: real timestamp, not synthetic
                "time": bar_time,
                "session_id": session,
                "reason": reason,
                "equity": acc.equity,
            })
            if position is not None:
                pnl = broker.close_position(acc, position, price)
                trade_log.append({
                    "type": "forced_close", "pnl": pnl,
                    "reason": reason, "bar": i, "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

        equity_curve.append(acc.equity)

        # Skip trading for rest of breached session
        if session in breached_sessions:
            continue

        # Build observation
        obs = feat_array[i - obs_window : i].copy()   # shape [T, F]

        # Get action
        action = policy(obs)  # {-1, 0, +1}

        # Execute action
        if action != 0:
            if position is not None and position.direction != action:
                # Reverse: close then reopen
                pnl = broker.close_position(acc, position, price)
                trade_log.append({
                    "type": "close", "pnl": pnl, "bar": i, "time": bar_time,
                    "equity": acc.equity,
                })
                sessions_with_trades.add(session)
                position = None

            if position is None:
                position = broker.open_position(acc, price, lots, action)
                if position:
                    trade_log.append({
                        "type": "open", "direction": action,
                        "price": position.entry_price, "bar": i, "time": bar_time,
                        "equity": acc.equity,
                    })
                    sessions_with_trades.add(session)
        elif action == 0 and position is not None:
            pnl = broker.close_position(acc, position, price)
            trade_log.append({
                "type": "close", "pnl": pnl, "bar": i, "time": bar_time,
                "equity": acc.equity,
            })
            sessions_with_trades.add(session)
            position = None

    # Close any remaining position at the last bar
    if position is not None:
        last_price = bars.iloc[-1]["close"]
        pnl = broker.close_position(acc, position, last_price)
        trade_log.append({
            "type": "eod_close", "pnl": pnl,
            "bar": len(bars) - 1, "time": bar_times[-1],
            "equity": acc.equity,
        })

    trades_df = pd.DataFrame(trade_log)
    equity_series = pd.Series(equity_curve, index=bars.index[obs_window:])
    n_sessions = len(session_set)
    n_breach_sessions = len(breached_sessions)

    return {
        "equity": equity_series,
        "trades": trades_df,
        "account": acc,
        "breaches": breach_log,
        "breach_events": breach_events,          # NEW: one dict per breached session
        "n_sessions": n_sessions,
        "n_breach_sessions": n_breach_sessions,
        "n_sessions_with_trades": len(sessions_with_trades),   # NEW: diagnostic
        "n_sessions_skipped": n_breach_sessions,               # NEW: alias
    }
