"""Stateful lifecycle helpers for the event-driven backtest engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from market_data.loading.ticks import TickExecutionUnavailable, resolve_tick_exit
from trading.account.state import AccountState
from trading.execution.broker import Broker, Position
from trading.execution.costs import CostModel
from trading.risk.guardrails import FTMOGuardrails
from trading.risk.sizing import compute_lots, compute_sl_tp_long, compute_sl_tp_short

if TYPE_CHECKING:
    from market_data.loading.ticks import TickBook

ExecutionEvent = tuple[tuple[float, float], pd.Timestamp, str, dict[str, Any]]
ResolveExecution = Callable[[], ExecutionEvent]


@dataclass(frozen=True)
class BacktestSettings:
    """Execution and strategy settings shared across backtest lifecycle stages."""

    cost_model: CostModel
    lots: float
    max_loss_per_trade_usd: float | None
    take_profit_per_trade_usd: float | None
    tickbook: TickBook | None
    fill_delay_ms: int
    randomized_fill_latency_ms: float
    max_tick_quote_wait_ms: int
    tick_execution_required: bool
    use_structure_sl_tp: bool
    risk_frac: float
    rr_ratio: float
    swing_buffer_pts: float
    min_lot: float
    max_lot: float
    contract_size: float
    hold_on_zero: bool
    exit_action: int | None
    block_overnight: bool


@dataclass
class BacktestState:
    """Mutable accounting, position, and reporting state for one backtest."""

    account: AccountState
    position: Position | None = None
    prev_session: int | None = None
    position_sequence: int = 0
    current_execution_meta: dict[str, Any] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    equity_times: list[pd.Timestamp] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    breach_log: list[str] = field(default_factory=list)
    breach_events: list[dict[str, Any]] = field(default_factory=list)
    breached_sessions: set[int] = field(default_factory=set)
    session_set: set[int] = field(default_factory=set)
    sessions_with_trades: set[int] = field(default_factory=set)


def stamp_exit(
    state: BacktestState,
    settings: BacktestSettings,
    pos: Position,
    record: dict[str, Any],
    fill_price: float,
) -> None:
    """Attach one exactly-once fill and completed-position accounting row."""
    for key, value in state.current_execution_meta.items():
        record.setdefault(key, value)
    record.setdefault("order_submission_time", record.get("decision_time", pos.entry_timestamp))
    if record.get("time_precision") == "bar_interval":
        record.setdefault("fill_time", None)
    else:
        record.setdefault("fill_time", record.get("time"))
    initial = float(pos.initial_size or pos.size)
    close_size = max(0.0, initial - float(pos.filled_size) - float(pos.size))
    pos.filled_size += close_size
    pos.fill_count += 1
    pos.position_id = pos.position_id or f"bt:p{state.position_sequence}"
    record["position_id"] = pos.position_id
    record["fill_id"] = f"{pos.position_id}:f{pos.fill_count}"
    record["size"] = close_size
    record["price"] = float(fill_price)
    record["position_complete"] = bool(pos.size <= 1e-9)
    entry_commission = float(settings.cost_model.total_cost(close_size))
    net_pnl = float(record.get("pnl", 0.0)) - entry_commission
    record["entry_commission_allocated"] = entry_commission
    record["net_pnl"] = net_pnl
    pos.position_net_pnl += net_pnl
    if record["position_complete"]:
        record["position_net_pnl"] = pos.position_net_pnl


def _record_close(
    state: BacktestState,
    settings: BacktestSettings,
    broker: Broker,
    position: Position,
    quote: tuple[float, float],
    record: dict[str, Any],
    session: int,
) -> None:
    pnl, fill_price = broker.close_position(state.account, position, quote)
    record["pnl"] = pnl
    record["price"] = fill_price
    record["equity"] = state.account.equity
    stamp_exit(state, settings, position, record, fill_price)
    state.trade_log.append(record)
    state.sessions_with_trades.add(session)
    state.position = None


def process_protective_and_account_events(
    state: BacktestState,
    settings: BacktestSettings,
    broker: Broker,
    guardrails: FTMOGuardrails,
    bars: pd.DataFrame,
    row: pd.Series,
    *,
    i: int,
    bar_time: pd.Timestamp,
    bar_duration: pd.Timedelta,
    session: int,
    is_ny: bool,
    bar_quote: tuple[float, float],
    entry_time: pd.Timestamp | None,
    position_active_in_bar: bool,
    full_bar_available_after_entry: bool,
    execution_rng: np.random.Generator,
    resolve_bar_execution: ResolveExecution,
    fill_quote_event: Callable[..., ExecutionEvent],
    sl_tp_fill_quote: Callable[[int, float, CostModel], tuple[float, float]],
    bar_spread_price_units: Callable[[pd.Series, CostModel], float | None],
    is_last_ny_bar: Callable[[pd.DataFrame, int, bool], bool],
) -> bool:
    """Apply protective exits, guardrail liquidation, and session-end close."""
    position = state.position
    tick_hit: tuple[str, pd.Timestamp, float, float, dict[str, Any]] | None = None
    if (
        position is not None
        and position_active_in_bar
        and settings.tickbook is not None
        and (position.sl_price is not None or position.tp_price is not None)
    ):
        if settings.tick_execution_required or settings.randomized_fill_latency_ms > 0:
            resolved_exit = resolve_tick_exit(
                settings.tickbook,
                bar_time,
                int(position.direction),
                None if position.sl_price is None else float(position.sl_price),
                None if position.tp_price is None else float(position.tp_price),
                execution_rng,
                bar_minutes=max(1, int(round(bar_duration.total_seconds() / 60.0))),
                max_latency_ns_exclusive=int(settings.randomized_fill_latency_ms * 1_000_000),
                max_quote_wait_ns=settings.max_tick_quote_wait_ms * 1_000_000,
                earliest_time=entry_time,
            )
            if resolved_exit is not None:
                quote = resolved_exit.quote
                tick_hit = (
                    resolved_exit.kind,
                    quote.tick_time,
                    quote.bid,
                    quote.ask,
                    {
                        "order_submission_time": resolved_exit.trigger_time,
                        "execution_order_time": quote.order_time,
                        "execution_arrival_time": quote.arrival_time,
                        "execution_latency_ns": quote.latency_ns,
                        "actual_tick_time": quote.tick_time,
                        "fill_quote_source": "tick",
                    },
                )
        else:
            legacy_hit = settings.tickbook.delayed_exit_event(
                bar_time,
                int(position.direction),
                None if position.sl_price is None else float(position.sl_price),
                None if position.tp_price is None else float(position.tp_price),
                pd.Timedelta(milliseconds=settings.fill_delay_ms),
                earliest_time=entry_time,
            )
            if legacy_hit is not None:
                tick_hit = (*legacy_hit, {"fill_quote_source": "tick"})

    if tick_hit is not None and state.position is not None:
        kind, fill_time, bid, ask, tick_execution_meta = tick_hit
        state.current_execution_meta = tick_execution_meta
        _record_close(
            state,
            settings,
            broker,
            state.position,
            (bid, ask),
            {
                "type": "stop_close" if kind == "sl" else "tp_close",
                "reason": "structure_sl" if kind == "sl" else "structure_tp",
                "bar": i,
                "time": fill_time,
                **tick_execution_meta,
            },
            session,
        )

    position = state.position
    if (
        (settings.tick_execution_required or settings.randomized_fill_latency_ms > 0)
        and position is not None
        and tick_hit is None
    ):
        sl_touched = position.sl_price is not None and (
            (position.direction == 1 and float(row["low"]) <= position.sl_price)
            or (position.direction == -1 and float(row["high"]) >= position.sl_price)
        )
        tp_touched = position.tp_price is not None and (
            (position.direction == 1 and float(row["high"]) >= position.tp_price)
            or (position.direction == -1 and float(row["low"]) <= position.tp_price)
        )
        if sl_touched or tp_touched:
            raise TickExecutionUnavailable(
                "OHLC indicates a protective level was reached, but no matching "
                "tick trigger was found; candle fallback is disabled"
            )

    position = state.position
    if (
        position is not None
        and position_active_in_bar
        and full_bar_available_after_entry
        and tick_hit is None
    ):
        sl_hit = False
        if position.sl_price is not None:
            if position.direction == 1 and float(row["low"]) <= position.sl_price:
                sl_hit = True
            elif position.direction == -1 and float(row["high"]) >= position.sl_price:
                sl_hit = True

        if sl_hit:
            assert position.sl_price is not None
            stop_price = float(position.sl_price)
            raw_spread = bar_spread_price_units(row, settings.cost_model)
            opening_spread = settings.cost_model.spread_points if raw_spread is None else raw_spread
            if position.direction == 1:
                stop_price = min(stop_price, float(row["open"]))
            else:
                stop_price = max(stop_price, float(row["open"]) + opening_spread)
            _record_close(
                state,
                settings,
                broker,
                position,
                sl_tp_fill_quote(position.direction, stop_price, settings.cost_model),
                {
                    "type": "stop_close",
                    "reason": "structure_sl",
                    "bar": i,
                    "time": bar_time,
                    "event_time_start": bar_time,
                    "event_time_end": bar_time + bar_duration,
                    "time_precision": "bar_interval",
                },
                session,
            )
        elif settings.max_loss_per_trade_usd is not None and state.account.open_pnl <= -abs(
            settings.max_loss_per_trade_usd
        ):
            fq, execution_time, fill_source, _ = resolve_bar_execution()
            _record_close(
                state,
                settings,
                broker,
                position,
                fq,
                {
                    "type": "stop_close",
                    "reason": "max_loss_cap",
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                },
                session,
            )

    position = state.position
    tp_hit = False
    if (
        position is not None
        and position_active_in_bar
        and full_bar_available_after_entry
        and tick_hit is None
        and position.tp_price is not None
    ):
        if position.direction == 1 and float(row["high"]) >= position.tp_price:
            tp_hit = True
        elif position.direction == -1 and float(row["low"]) <= position.tp_price:
            tp_hit = True

    if tp_hit and state.position is not None:
        position = state.position
        assert position.tp_price is not None
        _record_close(
            state,
            settings,
            broker,
            position,
            sl_tp_fill_quote(position.direction, position.tp_price, settings.cost_model),
            {
                "type": "tp_close",
                "reason": "structure_tp",
                "bar": i,
                "time": bar_time,
                "event_time_start": bar_time,
                "event_time_end": bar_time + bar_duration,
                "time_precision": "bar_interval",
            },
            session,
        )
    elif (
        not settings.use_structure_sl_tp
        and settings.take_profit_per_trade_usd is not None
        and state.position is not None
        and state.account.open_pnl >= abs(settings.take_profit_per_trade_usd)
    ):
        fq, execution_time, fill_source, _ = resolve_bar_execution()
        _record_close(
            state,
            settings,
            broker,
            state.position,
            fq,
            {
                "type": "tp_close",
                "reason": "take_profit_per_trade",
                "bar": i,
                "time": execution_time,
                "decision_time": bar_time,
                "fill_time": execution_time,
                "fill_quote_source": fill_source,
            },
            session,
        )

    reason = guardrails.breach_reason(state.account)
    if reason and session not in state.breached_sessions:
        state.breached_sessions.add(session)
        state.breach_log.append(reason)
        state.breach_events.append(
            {
                "time": bar_time,
                "session_id": session,
                "reason": reason,
                "equity": state.account.equity,
            }
        )
        if state.position is not None:
            fq, execution_time, fill_source, _ = resolve_bar_execution()
            _record_close(
                state,
                settings,
                broker,
                state.position,
                fq,
                {
                    "type": "forced_close",
                    "reason": reason,
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                },
                session,
            )

    last_ny = is_last_ny_bar(bars, i, is_ny)
    if settings.block_overnight and last_ny and state.position is not None:
        session_close_time = bar_time + bar_duration
        if settings.tick_execution_required or settings.randomized_fill_latency_ms > 0:
            session_quote, session_fill_time, session_source, session_execution_meta = (
                fill_quote_event(
                    session_close_time,
                    row,
                    settings.cost_model,
                    settings.tickbook,
                    rng=execution_rng,
                    randomized_fill_latency_ms=settings.randomized_fill_latency_ms,
                    max_tick_quote_wait_ms=settings.max_tick_quote_wait_ms,
                    tick_execution_required=settings.tick_execution_required,
                )
            )
            state.current_execution_meta = session_execution_meta
        else:
            session_quote = bar_quote
            session_fill_time = session_close_time
            session_source = "permitted_session_bar_close"
            session_execution_meta = {}
        _record_close(
            state,
            settings,
            broker,
            state.position,
            session_quote,
            {
                "type": "eod_close",
                "reason": "session_end",
                "bar": i,
                "time": session_fill_time,
                "fill_time": session_fill_time,
                "fill_quote_source": session_source,
                **session_execution_meta,
            },
            session,
        )
    return last_ny


def apply_policy_action(
    state: BacktestState,
    settings: BacktestSettings,
    broker: Broker,
    guardrails: FTMOGuardrails,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    i: int,
    bar_time: pd.Timestamp,
    session: int,
    action: int,
    resolve_bar_execution: ResolveExecution,
    is_last_ny_bar: Callable[[pd.DataFrame, int, bool], bool],
) -> None:
    """Apply one policy action using the next executable quote."""
    if action == settings.exit_action and state.position is not None:
        fq, execution_time, fill_source, _ = resolve_bar_execution()
        _record_close(
            state,
            settings,
            broker,
            state.position,
            fq,
            {
                "type": "close",
                "bar": i,
                "time": execution_time,
                "decision_time": bar_time,
                "fill_time": execution_time,
                "fill_quote_source": fill_source,
            },
            session,
        )
        return

    if action in (1, -1):
        if state.position is not None and state.position.direction != action:
            fq, execution_time, fill_source, _ = resolve_bar_execution()
            _record_close(
                state,
                settings,
                broker,
                state.position,
                fq,
                {
                    "type": "close",
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                },
                session,
            )

        if state.position is not None:
            return
        if settings.block_overnight and is_last_ny_bar(
            bars,
            i + 1,
            "session" not in bars.iloc[i + 1].index or str(bars.iloc[i + 1]["session"]) == "ny",
        ):
            return

        fq, execution_time, fill_source, execution_meta = resolve_bar_execution()
        sl_price = None
        tp_price = None
        actual_lots = settings.lots
        entry_price_for_calc = float(fq[1] if action == 1 else fq[0])

        if settings.use_structure_sl_tp:
            feat_row = features.iloc[i]
            low_raw = feat_row["last_swing_low"] if "last_swing_low" in feat_row.index else np.nan
            high_raw = (
                feat_row["last_swing_high"] if "last_swing_high" in feat_row.index else np.nan
            )
            last_swing_low = float(low_raw) if pd.notna(low_raw) else None
            last_swing_high = float(high_raw) if pd.notna(high_raw) else None

            if action == 1 and last_swing_low is not None:
                sl_price, tp_price = compute_sl_tp_long(
                    entry_price_for_calc,
                    last_swing_low,
                    buffer_pts=settings.swing_buffer_pts,
                    rr_ratio=settings.rr_ratio,
                )
                actual_lots = compute_lots(
                    state.account.equity,
                    settings.risk_frac,
                    entry_price_for_calc,
                    sl_price,
                    contract_size=settings.contract_size,
                    point_value=settings.cost_model.point_value,
                    min_lot=settings.min_lot,
                    max_lot=settings.max_lot,
                    max_loss_cap=settings.max_loss_per_trade_usd,
                )
            elif action == -1 and last_swing_high is not None:
                sl_price, tp_price = compute_sl_tp_short(
                    entry_price_for_calc,
                    last_swing_high,
                    buffer_pts=settings.swing_buffer_pts,
                    rr_ratio=settings.rr_ratio,
                )
                actual_lots = compute_lots(
                    state.account.equity,
                    settings.risk_frac,
                    entry_price_for_calc,
                    sl_price,
                    contract_size=settings.contract_size,
                    point_value=settings.cost_model.point_value,
                    min_lot=settings.min_lot,
                    max_lot=settings.max_lot,
                    max_loss_cap=settings.max_loss_per_trade_usd,
                )

        if sl_price is not None:
            trade_risk = (
                abs(entry_price_for_calc - sl_price)
                * settings.contract_size
                * actual_lots
                * settings.cost_model.point_value
            )
        elif settings.max_loss_per_trade_usd is not None:
            trade_risk = settings.max_loss_per_trade_usd
        else:
            trade_risk = None

        if trade_risk is not None and guardrails.check_trade_risk(trade_risk, state.account):
            return

        position = broker.open_position(state.account, fq, actual_lots, action)
        if position:
            state.position_sequence += 1
            position.position_id = f"bt:p{state.position_sequence}"
            position.entry_timestamp = execution_time
            position.sl_price = sl_price
            position.tp_price = tp_price
            position.risk_frac = settings.risk_frac
            position.rr_ratio = settings.rr_ratio
            state.position = position
            state.trade_log.append(
                {
                    "type": "open",
                    "position_id": position.position_id,
                    "direction": action,
                    "price": position.entry_price,
                    "lots": position.size,
                    "entry_commission": settings.cost_model.total_cost(position.size),
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "risk_frac": settings.risk_frac,
                    "rr_ratio": settings.rr_ratio,
                    "bar": i,
                    "time": execution_time,
                    "decision_time": bar_time,
                    "order_submission_time": bar_time,
                    "fill_time": execution_time,
                    "fill_quote_source": fill_source,
                    "equity": state.account.equity,
                    **execution_meta,
                }
            )
            state.sessions_with_trades.add(session)
        return

    if action == 0 and state.position is not None and not settings.hold_on_zero:
        fq, execution_time, fill_source, _ = resolve_bar_execution()
        _record_close(
            state,
            settings,
            broker,
            state.position,
            fq,
            {
                "type": "close",
                "bar": i,
                "time": execution_time,
                "decision_time": bar_time,
                "fill_time": execution_time,
                "fill_quote_source": fill_source,
            },
            session,
        )


def liquidate_terminal_position(
    state: BacktestState,
    settings: BacktestSettings,
    broker: Broker,
    bars: pd.DataFrame,
    bar_times: pd.DatetimeIndex,
    bar_duration: pd.Timedelta,
    execution_rng: np.random.Generator,
    *,
    fill_quote_event: Callable[..., ExecutionEvent],
    bar_quote: Callable[[pd.Series, CostModel], tuple[float, float]],
) -> None:
    """Realize any remaining position at the final bar-close valuation point."""
    if state.position is None:
        return

    last_row = bars.iloc[-1]
    last_time = bar_times[-1]
    terminal_close_time = last_time + bar_duration
    if settings.tick_execution_required or settings.randomized_fill_latency_ms > 0:
        last_fq, terminal_close_time, terminal_source, terminal_execution_meta = fill_quote_event(
            terminal_close_time,
            last_row,
            settings.cost_model,
            settings.tickbook,
            rng=execution_rng,
            randomized_fill_latency_ms=settings.randomized_fill_latency_ms,
            max_tick_quote_wait_ms=settings.max_tick_quote_wait_ms,
            tick_execution_required=settings.tick_execution_required,
        )
        state.current_execution_meta = terminal_execution_meta
    else:
        last_fq = bar_quote(last_row, settings.cost_model)
        terminal_source = "terminal_bar_close"
        terminal_execution_meta = {}

    position = state.position
    pnl, fill_price = broker.close_position(state.account, position, last_fq)
    record = {
        "type": "eod_close",
        "pnl": pnl,
        "price": fill_price,
        "bar": len(bars) - 1,
        "time": terminal_close_time,
        "fill_time": terminal_close_time,
        "fill_quote_source": terminal_source,
        "equity": state.account.equity,
        **terminal_execution_meta,
    }
    stamp_exit(state, settings, position, record, fill_price)
    state.trade_log.append(record)
    state.position = None

    if state.equity_curve:
        state.equity_curve[-1] = state.account.equity
    else:
        state.equity_curve.append(state.account.equity)
        state.equity_times.append(pd.Timestamp(last_time) + bar_duration)
