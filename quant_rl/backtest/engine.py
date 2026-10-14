"""Event-driven backtester engine.

Fill-price semantics
--------------------
Every order fill uses a ``(bid, ask)`` quote obtained in this priority:

1. **Tick data**: a valid quote at the next bar's open timestamp.
2. **Bar-spread fallback**: the next bar's close quote, timestamped at its
   close rather than at its open.

Mark-to-market and per-trade stop checks use the bar-close quote (long @ bid,
short @ ask).

Bar indexes are left-labeled. Equity observations are marked at each bar's
close timestamp, after current-bar protective events and session liquidation,
but before policy orders that execute on a later quote. OHLC-only protective
fills retain their triggering bar interval because their exact intrabar time
is not observable without ticks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from .account import AccountState
from .broker import Broker, Position
from .costs import COST_US100, CostModel
from .guardrails import FTMOGuardrails
from .risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short

if TYPE_CHECKING:
    from ..data.ticks import TickBook

ActionFn = Callable[[np.ndarray[Any, Any]], int]


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


def _sl_tp_fill_quote(
    direction: int,
    hit_price: float,
    cost_model: CostModel,
) -> tuple[float, float]:
    """Build a fill quote for an intrabar SL/TP hit on the triggering bar.

    The fill price is pinned to the SL/TP level; the opposite side is
    extended by the model spread so that ``Broker.close_position`` can
    deduct the round-turn spread cost.
    """
    # CostModel spread_points is configured in price units; point_size is
    # reserved for converting raw MT5 spread columns and price precision.
    spread = float(cost_model.spread_points)
    slippage = float(cost_model.slippage_points)
    if direction == 1:
        bid = hit_price - slippage
        ask = bid + spread
    else:
        ask = hit_price + slippage
        bid = ask - spread
    return bid, ask


def _bar_quote(row: pd.Series, cost_model: CostModel) -> tuple[float, float]:
    """Derive ``(bid, ask)`` from a bar row (MT5 close = bid)."""
    bar_spread = _bar_spread_price_units(row, cost_model)
    return cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread)


def _bar_duration(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Estimate bar duration from positive index deltas, defaulting to M1."""
    if len(index) < 2:
        return pd.Timedelta(minutes=1)
    deltas = np.diff(index.to_numpy(dtype="datetime64[ns]").astype("int64"))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(int(np.median(positive)), unit="ns")


def _fill_quote(
    fill_instant: pd.Timestamp,
    row: pd.Series,
    cost_model: CostModel,
    tickbook: TickBook | None,
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
    return _fill_quote_event(fill_instant, row, cost_model, tickbook)[0]


def _fill_quote_event(
    fill_instant: pd.Timestamp,
    row: pd.Series,
    cost_model: CostModel,
    tickbook: TickBook | None,
    *,
    bar_close_time: pd.Timestamp | None = None,
) -> tuple[tuple[float, float], pd.Timestamp, str]:
    """Return quote, execution timestamp and source for a delayed fill."""
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
                return q, fill_instant, "tick"
            # Outlier/gap tick (mid far outside the executing bar's range):
            # fall through to the bar-based quote below instead of returning
            # a fill price disconnected from the visible candles.
    return (
        cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread),
        fill_instant if bar_close_time is None else bar_close_time,
        "bar_close",
    )


def _is_last_ny_bar(bars: pd.DataFrame, i: int, is_ny: bool) -> bool:
    """True on the last NY bar of a session (next bar is non-NY or a time jump)."""
    if not is_ny:
        return False
    if i + 1 >= len(bars):
        return True
    nxt = bars.iloc[i + 1]
    next_ny = "session" not in nxt.index or str(nxt["session"]) == "ny"
    if not next_ny:
        return True
    return pd.Timestamp(bars.index[i + 1]) - pd.Timestamp(bars.index[i]) > pd.Timedelta(minutes=1)


def run_backtest(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    policy: ActionFn,
    obs_window: int = 60,
    cost_model: CostModel = COST_US100,
    broker_kwargs: dict[str, Any] | None = None,
    guardrail_kwargs: dict[str, Any] | None = None,
    initial_balance: float = 100_000.0,
    lots: float = 1.0,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    tickbook: TickBook | None = None,
    fill_delay_ms: int = 0,
    use_structure_sl_tp: bool = False,
    risk_frac: float = 0.01,
    rr_ratio: float = 2.0,
    swing_buffer_pts: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    contract_size: float = 1.0,
    hold_on_zero: bool = False,
    exit_action: int | None = None,
    block_overnight: bool = True,
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
    block_overnight : bool
        If True (default), force-close on the last NY bar of a session so
        positions do not carry the overnight gap. Matches ``TradingEnv``.
    """
    broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
    guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))
    acc = AccountState(initial_balance=initial_balance)

    equity_curve: list[float] = []
    equity_times: list[pd.Timestamp] = []
    trade_log: list[dict[str, Any]] = []
    breach_log: list[str] = []
    breach_events: list[dict[str, Any]] = []
    breached_sessions: set[int] = set()
    session_set: set[int] = set()
    sessions_with_trades: set[int] = set()

    position: Position | None = None
    prev_session: int | None = None

    common_idx = bars.index.intersection(features.index)
    bars = bars.loc[common_idx]
    features = features.loc[common_idx]

    bar_times = pd.DatetimeIndex(bars.index)
    bar_duration = _bar_duration(bar_times)
    feat_array: np.ndarray[Any, Any] = features.values.astype(np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0).astype(np.float32)
    n_bars = len(bars)
    position_sequence = 0

    def stamp_exit(pos: Position, record: dict[str, Any], fill_price: float) -> None:
        """Attach one exactly-once fill and completed-position accounting row."""
        record.setdefault(
            "order_submission_time",
            record.get("decision_time", pos.entry_timestamp),
        )
        if record.get("time_precision") == "bar_interval":
            # OHLC data locates the trigger in an interval, not at a known
            # instant. Keep the legacy event time and explicit bounds, while
            # leaving the precise fill time unknown.
            record.setdefault("fill_time", None)
        else:
            record.setdefault("fill_time", record.get("time"))
        initial = float(pos.initial_size or pos.size)
        close_size = max(0.0, initial - float(pos.filled_size) - float(pos.size))
        pos.filled_size += close_size
        pos.fill_count += 1
        pos.position_id = pos.position_id or f"bt:p{position_sequence}"
        record["position_id"] = pos.position_id
        record["fill_id"] = f"{pos.position_id}:f{pos.fill_count}"
        record["size"] = close_size
        record["price"] = float(fill_price)
        record["position_complete"] = bool(pos.size <= 1e-9)
        entry_commission = float(cost_model.total_cost(close_size))
        net_pnl = float(record.get("pnl", 0.0)) - entry_commission
        record["entry_commission_allocated"] = entry_commission
        record["net_pnl"] = net_pnl
        pos.position_net_pnl += net_pnl
        if record["position_complete"]:
            record["position_net_pnl"] = pos.position_net_pnl

    for i in range(obs_window, n_bars):
        row = bars.iloc[i]
        bar_time = bar_times[i]
        session = int(row["session_id"]) if "session_id" in row.index else 0
        session_set.add(session)
        is_ny = "session" not in row.index or str(row["session"]) == "ny"

        if is_ny and session != prev_session:
            acc.reset_daily()
            prev_session = session
        elif prev_session is None:
            prev_session = session

        bq = _bar_quote(row, cost_model)

        if position is not None:
            broker.mark_to_market(acc, position, bq)
        bar_end = pd.Timestamp(bar_time) + bar_duration
        entry_time = (
            None
            if position is None or position.entry_timestamp is None
            else pd.Timestamp(cast(Any, position.entry_timestamp))
        )
        position_active_in_bar = entry_time is None or entry_time < bar_end
        full_bar_available_after_entry = entry_time is None or entry_time <= pd.Timestamp(bar_time)

        if i + 1 < n_bars:
            fill_instant = bar_times[i + 1]
            fill_row = bars.iloc[i + 1]
            fill_bar_close = fill_instant + bar_duration
        else:
            fill_instant = bar_time
            fill_row = row
            fill_bar_close = bar_time + bar_duration
        fq, execution_time, fill_source = _fill_quote_event(
            fill_instant,
            fill_row,
            cost_model,
            tickbook,
            bar_close_time=fill_bar_close,
        )

        tick_hit = None
        if (
            position is not None
            and position_active_in_bar
            and tickbook is not None
            and (position.sl_price is not None or position.tp_price is not None)
        ):
            tick_hit = tickbook.delayed_exit_event(
                bar_time,
                int(position.direction),
                None if position.sl_price is None else float(position.sl_price),
                None if position.tp_price is None else float(position.tp_price),
                pd.Timedelta(milliseconds=fill_delay_ms),
                earliest_time=entry_time,
            )
        if tick_hit is not None and position is not None:
            kind, fill_time, bid, ask = tick_hit
            closed_position = position
            pnl, fill_price = broker.close_position(acc, closed_position, (bid, ask))
            record = {
                "type": "stop_close" if kind == "sl" else "tp_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_sl" if kind == "sl" else "structure_tp",
                "bar": i,
                "time": fill_time,
                "equity": acc.equity,
            }
            stamp_exit(closed_position, record, fill_price)
            trade_log.append(record)
            sessions_with_trades.add(session)
            position = None

        if (
            position is not None
            and position_active_in_bar
            and full_bar_available_after_entry
            and tick_hit is None
        ):
            # Structural stops are independent of the optional USD safety cap.
            sl_hit = False
            if position.sl_price is not None:
                if position.direction == 1 and float(row["low"]) <= position.sl_price:
                    sl_hit = True
                elif position.direction == -1 and float(row["high"]) >= position.sl_price:
                    sl_hit = True

            if sl_hit:
                assert position is not None
                assert position.sl_price is not None
                stop_price = float(position.sl_price)
                bar_spread = _bar_spread_price_units(row, cost_model)
                opening_spread = cost_model.spread_points if bar_spread is None else bar_spread
                if position.direction == 1:
                    stop_price = min(stop_price, float(row["open"]))
                else:
                    stop_price = max(stop_price, float(row["open"]) + opening_spread)
                sl_fq = _sl_tp_fill_quote(position.direction, stop_price, cost_model)
                closed_position = position
                pnl, fill_price = broker.close_position(acc, closed_position, sl_fq)
                record = {
                    "type": "stop_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "structure_sl",
                    "bar": i,
                    "time": bar_time,
                    "event_time_start": bar_time,
                    "event_time_end": bar_time + bar_duration,
                    "time_precision": "bar_interval",
                    "equity": acc.equity,
                }
                stamp_exit(closed_position, record, fill_price)
                trade_log.append(record)
                sessions_with_trades.add(session)
                position = None
            # Fallback: check the optional USD safety cap after the structural stop.
            elif max_loss_per_trade_usd is not None and acc.open_pnl <= -abs(
                max_loss_per_trade_usd
            ):
                assert position is not None
                closed_position = position
                pnl, fill_price = broker.close_position(acc, closed_position, fq)
                record = {
                    "type": "stop_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "max_loss_cap",
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                    "equity": acc.equity,
                }
                stamp_exit(closed_position, record, fill_price)
                trade_log.append(record)
                sessions_with_trades.add(session)
                position = None

        # Check TP (structure-based if available, else use global).
        # A tick fill already closed the position above.
        tp_hit = False
        if (
            position is not None
            and position_active_in_bar
            and full_bar_available_after_entry
            and tick_hit is None
        ):
            if position.tp_price is not None:
                if position.direction == 1 and float(row["high"]) >= position.tp_price:
                    tp_hit = True
                elif position.direction == -1 and float(row["low"]) <= position.tp_price:
                    tp_hit = True

        if tp_hit and position is not None:
            assert position is not None
            assert position.tp_price is not None
            tp_fq = _sl_tp_fill_quote(position.direction, position.tp_price, cost_model)
            closed_position = position
            pnl, fill_price = broker.close_position(acc, closed_position, tp_fq)
            record = {
                "type": "tp_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_tp",
                "bar": i,
                "time": bar_time,
                "event_time_start": bar_time,
                "event_time_end": bar_time + bar_duration,
                "time_precision": "bar_interval",
                "equity": acc.equity,
            }
            stamp_exit(closed_position, record, fill_price)
            trade_log.append(record)
            sessions_with_trades.add(session)
            position = None
        elif not use_structure_sl_tp and take_profit_per_trade_usd is not None:
            # Fallback: global USD-based TP
            if acc.open_pnl >= abs(take_profit_per_trade_usd):
                assert position is not None
                closed_position = position
                pnl, fill_price = broker.close_position(acc, closed_position, fq)
                record = {
                    "type": "tp_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "take_profit_per_trade",
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                    "equity": acc.equity,
                }
                stamp_exit(closed_position, record, fill_price)
                trade_log.append(record)
                sessions_with_trades.add(session)
                position = None

        reason = guardrails.breach_reason(acc)
        if reason and session not in breached_sessions:
            breached_sessions.add(session)
            breach_log.append(reason)
            breach_events.append(
                {
                    "time": bar_time,
                    "session_id": session,
                    "reason": reason,
                    "equity": acc.equity,
                }
            )
            if position is not None:
                assert position is not None
                closed_position = position
                pnl, fill_price = broker.close_position(acc, closed_position, fq)
                record = {
                    "type": "forced_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": reason,
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                    "equity": acc.equity,
                }
                stamp_exit(closed_position, record, fill_price)
                trade_log.append(record)
                sessions_with_trades.add(session)
                position = None

        last_ny = _is_last_ny_bar(bars, i, is_ny)
        if block_overnight and last_ny and position is not None:
            closed_position = position
            session_close_time = bar_time + bar_duration
            pnl, fill_price = broker.close_position(acc, closed_position, bq)
            record = {
                "type": "eod_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "session_end",
                "bar": i,
                "time": session_close_time,
                "fill_time": session_close_time,
                "fill_quote_source": "permitted_session_bar_close",
                "equity": acc.equity,
            }
            stamp_exit(closed_position, record, fill_price)
            trade_log.append(record)
            sessions_with_trades.add(session)
            position = None

        # Input bar indexes are left-labeled.  The mark above uses the bar
        # close quote, so label its valuation at the bar close.  Session-end
        # liquidation is also performed at that close and must be reflected
        # in this same valuation point.  Policy orders below execute later
        # and therefore first affect the next valuation point.
        equity_curve.append(acc.equity)
        equity_times.append(pd.Timestamp(bar_time) + bar_duration)

        if session in breached_sessions or not is_ny or (block_overnight and last_ny):
            continue

        # A policy decision needs a subsequent executable bar. The data-end
        # row may close an existing position, but cannot open a new one.
        if i + 1 >= n_bars:
            continue

        obs = feat_array[i - obs_window : i].copy()
        action = policy(obs)

        if action == exit_action and position is not None:
            # Explicit exit action (e.g., 2 for MACD)
            assert position is not None
            closed_position = position
            pnl, fill_price = broker.close_position(acc, closed_position, fq)
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "bar": i,
                "time": execution_time,
                "decision_time": bar_time,
                "fill_time": execution_time,
                "fill_quote_source": fill_source,
                "equity": acc.equity,
            }
            stamp_exit(closed_position, record, fill_price)
            trade_log.append(record)
            sessions_with_trades.add(session)
            position = None

        elif action in (1, -1):
            if position is not None and position.direction != action:
                assert position is not None
                closed_position = position
                pnl, fill_price = broker.close_position(acc, closed_position, fq)
                record = {
                    "type": "close",
                    "pnl": pnl,
                    "price": fill_price,
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                    "equity": acc.equity,
                }
                stamp_exit(closed_position, record, fill_price)
                trade_log.append(record)
                sessions_with_trades.add(session)
                position = None

            if position is None:
                if block_overnight and _is_last_ny_bar(
                    bars,
                    i + 1,
                    "session" not in bars.iloc[i + 1].index
                    or str(bars.iloc[i + 1]["session"]) == "ny",
                ):
                    continue
                # Extract structure features if available
                sl_price = None
                tp_price = None
                actual_lots = lots

                if use_structure_sl_tp:
                    # Try to extract swing levels from features
                    feat_row = features.iloc[i]
                    last_swing_low_raw = (
                        feat_row["last_swing_low"] if "last_swing_low" in feat_row.index else np.nan
                    )
                    last_swing_high_raw = (
                        feat_row["last_swing_high"]
                        if "last_swing_high" in feat_row.index
                        else np.nan
                    )
                    last_swing_low = (
                        float(last_swing_low_raw) if pd.notna(last_swing_low_raw) else None
                    )
                    last_swing_high = (
                        float(last_swing_high_raw) if pd.notna(last_swing_high_raw) else None
                    )

                    entry_price_for_calc = float(fq[1] if action == 1 else fq[0])

                    # Compute SL/TP based on direction
                    if action == 1 and last_swing_low is not None:
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
                            point_value=cost_model.point_value,
                            min_lot=min_lot,
                            max_lot=max_lot,
                            max_loss_cap=max_loss_per_trade_usd,
                        )
                    elif action == -1 and last_swing_high is not None:
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
                            point_value=cost_model.point_value,
                            min_lot=min_lot,
                            max_lot=max_lot,
                            max_loss_cap=max_loss_per_trade_usd,
                        )

                if sl_price is not None:
                    sl_distance = abs(entry_price_for_calc - sl_price)
                    trade_risk = sl_distance * contract_size * actual_lots * cost_model.point_value
                elif max_loss_per_trade_usd is not None:
                    trade_risk = max_loss_per_trade_usd
                else:
                    trade_risk = None

                if trade_risk is not None and guardrails.check_trade_risk(trade_risk):
                    continue

                position = broker.open_position(acc, fq, actual_lots, action)
                if position:
                    position_sequence += 1
                    position.position_id = f"bt:p{position_sequence}"
                    position.entry_timestamp = execution_time
                    # Attach SL/TP/risk info
                    position.sl_price = sl_price
                    position.tp_price = tp_price
                    position.risk_frac = risk_frac
                    position.rr_ratio = rr_ratio

                    trade_log.append(
                        {
                            "type": "open",
                            "position_id": position.position_id,
                            "direction": action,
                            "price": position.entry_price,
                            "lots": position.size,
                            "entry_commission": cost_model.total_cost(position.size),
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "risk_frac": risk_frac,
                            "rr_ratio": rr_ratio,
                            "bar": i,
                            "time": execution_time,
                            "decision_time": bar_time,
                            "order_submission_time": bar_time,
                            "fill_time": execution_time,
                            "fill_quote_source": fill_source,
                            "equity": acc.equity,
                        }
                    )
                    sessions_with_trades.add(session)

        elif action == 0 and position is not None and not hold_on_zero:
            # Close position on action=0 (unless hold_on_zero is True)
            closed_position = position
            pnl, fill_price = broker.close_position(acc, closed_position, fq)
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "bar": i,
                "time": execution_time,
                "decision_time": bar_time,
                "fill_time": execution_time,
                "fill_quote_source": fill_source,
                "equity": acc.equity,
            }
            stamp_exit(closed_position, record, fill_price)
            trade_log.append(record)
            sessions_with_trades.add(session)
            position = None

    if position is not None:
        last_row = bars.iloc[-1]
        last_time = bar_times[-1]
        last_fq = _bar_quote(last_row, cost_model)
        terminal_close_time = last_time + bar_duration
        assert position is not None
        closed_position = position
        pnl, fill_price = broker.close_position(acc, closed_position, last_fq)
        record = {
            "type": "eod_close",
            "pnl": pnl,
            "price": fill_price,
            "bar": n_bars - 1,
            "time": terminal_close_time,
            "fill_time": terminal_close_time,
            "fill_quote_source": "terminal_bar_close",
            "equity": acc.equity,
        }
        stamp_exit(closed_position, record, fill_price)
        trade_log.append(record)
        # The last bar has no later valuation point.  Its terminal close is
        # priced at this bar's close, so replace the pre-liquidation mark at
        # that same timestamp with the realized account value.
        if equity_curve:
            equity_curve[-1] = acc.equity
        else:
            equity_curve.append(acc.equity)
            equity_times.append(pd.Timestamp(last_time) + bar_duration)

    trades_df = pd.DataFrame(trade_log)
    equity_series = pd.Series(
        equity_curve,
        index=pd.DatetimeIndex(equity_times, name=bar_times.name),
        dtype=float,
    )

    return {
        "equity": equity_series,
        "equity_valuation_times": pd.DatetimeIndex(equity_times, name=bar_times.name),
        "trades": trades_df,
        "account": acc,
        "breaches": breach_log,
        "breach_events": breach_events,
        "n_sessions": len(session_set),
        "n_breach_sessions": len(breached_sessions),
        "n_sessions_with_trades": len(sessions_with_trades),
        "n_sessions_skipped": len(session_set) - len(sessions_with_trades),
    }
