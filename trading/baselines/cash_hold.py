"""Continuous long hold sized to the account loss rules and the broker swap.

The lot is chosen on the training path only. A long is opened at the first
traded bar's ask and kept until the last bid, the $10,000 max loss, or the
$5,000 daily loss. Overnight financing uses the MT5 points swap, with
Wednesday counted three times. Margin is checked at 1:50 during the session
and 1:15 while the position is carried overnight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# US100 specification used by the thesis account. Swap long is in MT5 points.
POINT_SIZE = 0.01
CONTRACT_SIZE = 1.0
SPREAD = 0.6
SWAP_LONG_POINTS = -675.12
INITIAL_BALANCE = 100_000.0
MAX_LOSS_USD = 10_000.0
DAILY_LOSS_USD = 5_000.0
LEVERAGE_INTRADAY = 50
LEVERAGE_SWING = 15
OBS_WINDOW = 60


def swap_usd_per_lot(
    swap_points: float = SWAP_LONG_POINTS,
    point_size: float = POINT_SIZE,
    contract_size: float = CONTRACT_SIZE,
) -> float:
    """Dollar swap for one lot for one night. Negative is a charge."""
    return float(swap_points) * float(point_size) * float(contract_size)


def _slice_from_warmup(bars: pd.DataFrame, obs_window: int = OBS_WINDOW) -> pd.DataFrame:
    if len(bars) <= obs_window:
        raise ValueError("need bars after the observation window")
    return bars.iloc[obs_window:]


def _night_multiplier(previous_day: pd.Timestamp) -> int:
    """MT5 posts three nights of swap on Wednesday."""
    return 3 if int(previous_day.dayofweek) == 2 else 1


def fit_hold_lots(
    bars: pd.DataFrame,
    *,
    max_loss_usd: float = MAX_LOSS_USD,
    daily_loss_usd: float = DAILY_LOSS_USD,
    swap_per_lot: float | None = None,
    spread: float = SPREAD,
    obs_window: int = OBS_WINDOW,
) -> dict[str, float]:
    """Lot size whose train path reaches, and does not pass, the loss caps.

    Args:
        bars: Training bars. The entry is the first bar after ``obs_window``.
        max_loss_usd: Account loss from the initial balance.
        daily_loss_usd: Loss allowed from the session's starting equity.
        swap_per_lot: Overnight charge per lot. Defaults to the US100 long swap.
        spread: Bid/ask width in price units. Close is the bid.
        obs_window: Bars skipped before the first fill.

    Returns:
        Lots and the train moves that set them. The used lot is the tighter
        of the max-loss lot and the daily-loss lot.
    """
    path = _slice_from_warmup(bars, obs_window)
    per_lot = swap_usd_per_lot() if swap_per_lot is None else float(swap_per_lot)
    charge = abs(per_lot)
    ts = pd.DatetimeIndex(path.index).tz_convert("Etc/GMT-3")
    entry = float(path["close"].iloc[0]) + spread
    low = path["low"].astype(float).to_numpy()
    worst_i = int(np.argmin(low))
    adverse = entry - float(low[worst_i])
    nights_to_worst = _nights_between(ts[0], ts[worst_i])
    worst_day = _worst_session_drop(path, ts)
    if adverse <= 0.0:
        raise ValueError("training path has no adverse move to size against")
    lots_max = max_loss_usd / (adverse + charge * nights_to_worst)
    lots_day = daily_loss_usd / (worst_day + charge) if worst_day + charge > 0.0 else lots_max
    return {
        "lots": float(min(lots_max, lots_day)),
        "lots_from_max_loss": float(lots_max),
        "lots_from_daily_loss": float(lots_day),
        "adverse_points": float(adverse),
        "nights_to_worst": float(nights_to_worst),
        "worst_day_points": float(worst_day),
        "swap_usd_per_lot": float(per_lot),
    }


def simulate_hold(
    bars: pd.DataFrame,
    lots: float,
    *,
    initial_balance: float = INITIAL_BALANCE,
    max_loss_usd: float = MAX_LOSS_USD,
    daily_loss_usd: float = DAILY_LOSS_USD,
    swap_per_lot: float | None = None,
    spread: float = SPREAD,
    obs_window: int = OBS_WINDOW,
    leverage_intraday: int = LEVERAGE_INTRADAY,
    leverage_swing: int = LEVERAGE_SWING,
) -> dict[str, Any]:
    """Mark a fixed-lot long from the first traded ask to the last bid.

    The position closes early when equity has fallen by ``max_loss_usd`` from
    the start, or by ``daily_loss_usd`` from that session's starting equity.
    """
    path = _slice_from_warmup(bars, obs_window)
    per_lot = swap_usd_per_lot() if swap_per_lot is None else float(swap_per_lot)
    ts = pd.DatetimeIndex(path.index).tz_convert("Etc/GMT-3")
    close = path["close"].astype(float).to_numpy()
    low = path["low"].astype(float).to_numpy()
    entry = float(close[0]) + spread
    balance = float(initial_balance)
    day_key = ts[0].normalize()
    day_start_equity = balance
    swap_total = 0.0
    max_margin = 0.0
    carried_overnight = False
    reason = "held_to_end"
    exit_price = float(close[-1])
    closed = False
    for i in range(len(path)):
        stamp = ts[i]
        this_day = stamp.normalize()
        if this_day != day_key:
            mult = _night_multiplier(day_key)
            swap = per_lot * lots * mult
            balance += swap
            swap_total += swap
            carried_overnight = True
            day_key = this_day
            day_start_equity = balance + (float(close[i - 1]) - entry) * lots
        equity_low = balance + (float(low[i]) - entry) * lots
        leverage = leverage_swing if carried_overnight else leverage_intraday
        max_margin = max(max_margin, _margin(float(close[i]), lots, leverage))
        if initial_balance - equity_low >= max_loss_usd:
            exit_price = entry + (initial_balance - max_loss_usd - balance) / lots
            reason = "max_loss"
            closed = True
            break
        if day_start_equity - equity_low >= daily_loss_usd:
            exit_price = entry + (day_start_equity - daily_loss_usd - balance) / lots
            reason = "daily_loss"
            closed = True
            break
    if not closed:
        exit_price = float(close[-1])
        balance += (exit_price - entry) * lots
    else:
        balance += (exit_price - entry) * lots
    price_pnl = (exit_price - entry) * lots
    net = price_pnl + swap_total
    return {
        "lots": float(lots),
        "entry": float(entry),
        "exit": float(exit_price),
        "reason": reason,
        "price_pnl": float(price_pnl),
        "swap_pnl": float(swap_total),
        "total_pnl": float(net),
        "total_return_pct": float(net / initial_balance * 100.0),
        "max_margin_usd": float(max_margin),
        "margin_ok": bool(max_margin < initial_balance),
        "n_trades": 1,
        "breach_count": int(reason != "held_to_end"),
        "index_return_pct": float((float(close[-1]) / float(close[0]) - 1.0) * 100.0),
    }


def _nights_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz=start.tz)
    total = 0
    for day in days[:-1]:
        total += _night_multiplier(day)
    return int(total)


def _worst_session_drop(path: pd.DataFrame, ts: pd.DatetimeIndex) -> float:
    day = ts.normalize()
    worst = 0.0
    for _, group in path.groupby(day):
        start = float(group["close"].iloc[0])
        drop = start - float(group["low"].min())
        worst = max(worst, drop)
    return float(worst)


def _margin(price: float, lots: float, leverage: int) -> float:
    return float(price) * float(lots) * CONTRACT_SIZE / float(leverage)
