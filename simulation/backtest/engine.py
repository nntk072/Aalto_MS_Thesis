"""Event-driven backtester engine.

Fill-price semantics
--------------------
Every order fill uses a ``(bid, ask)`` quote obtained in this priority:

1. **Tick data**: a valid quote at the next bar's open timestamp.
2. **Bar-spread fallback**: the next bar's close quote, timestamped at its
   close rather than at its open.

Mark-to-market and per-trade stop checks use the bar-close quote (long @ bid,
short @ ask).

When ``randomized_fill_latency_ms`` is positive, submitted orders instead use
the first recorded quote at or after a seeded random arrival time strictly
below the configured latency bound. That mode fails closed when no timely tick
exists and never substitutes a candle quote for an execution.

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

from market_data.loading.ticks import (
    TickExecutionUnavailable,
    resolve_tick_execution,
)
from trading.account.state import AccountState
from trading.execution.broker import Broker
from trading.execution.costs import COST_US100, CostModel
from trading.risk.guardrails import FTMOGuardrails

if TYPE_CHECKING:
    from market_data.loading.ticks import TickBook

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
    rng: np.random.Generator | None = None,
    randomized_fill_latency_ms: float = 0.0,
    max_tick_quote_wait_ms: int = 60_000,
    tick_execution_required: bool = False,
) -> tuple[tuple[float, float], pd.Timestamp, str, dict[str, Any]]:
    """Return a delayed execution quote and timing/source metadata."""
    if randomized_fill_latency_ms > 30.0 or randomized_fill_latency_ms < 0.0:
        raise ValueError("randomized_fill_latency_ms must be in [0, 30]")
    if max_tick_quote_wait_ms <= 0:
        raise ValueError("max_tick_quote_wait_ms must be positive")
    if tick_execution_required or randomized_fill_latency_ms > 0:
        if tickbook is None or rng is None:
            raise TickExecutionUnavailable("tick-only fills require a TickBook and RNG")
        execution = resolve_tick_execution(
            tickbook,
            fill_instant,
            rng,
            max_latency_ns_exclusive=int(randomized_fill_latency_ms * 1_000_000),
            max_quote_wait_ns=max_tick_quote_wait_ms * 1_000_000,
        )
        if execution is None:
            raise TickExecutionUnavailable(
                f"no executable tick within {max_tick_quote_wait_ms}ms of {fill_instant}"
            )
        return (
            (execution.bid, execution.ask),
            execution.tick_time,
            "tick",
            {
                "order_submission_time": execution.order_time,
                "execution_order_time": execution.order_time,
                "execution_arrival_time": execution.arrival_time,
                "execution_latency_ns": execution.latency_ns,
                "actual_tick_time": execution.tick_time,
                "fill_quote_source": "tick",
            },
        )

    bar_spread = _bar_spread_price_units(row, cost_model)
    if tickbook is not None:
        q = tickbook.quote_at(fill_instant)
        if q is not None:
            bid, ask = q
            mid = (bid + ask) / 2.0
            lo = float(row["low"])
            hi = float(row["high"])
            buffer = max(hi - lo, bar_spread or 0.0, cost_model.spread_points)
            if tick_execution_required or lo - buffer <= mid <= hi + buffer:
                return (
                    q,
                    fill_instant,
                    "tick",
                    {
                        "order_submission_time": fill_instant,
                        "execution_order_time": fill_instant,
                        "execution_arrival_time": fill_instant,
                        "execution_latency_ns": 0,
                        "actual_tick_time": fill_instant,
                        "fill_quote_source": "tick",
                    },
                )
            # In legacy mode, reject outlier ticks and use the candle quote.
    if tick_execution_required:
        raise TickExecutionUnavailable(
            f"no executable tick within {max_tick_quote_wait_ms}ms of {fill_instant}"
        )
    return (
        cost_model.bar_quote(float(row["close"]), bar_spread=bar_spread),
        fill_instant if bar_close_time is None else bar_close_time,
        "bar_close",
        {},
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
    randomized_fill_latency_ms: float = 0.0,
    max_tick_quote_wait_ms: int = 60_000,
    tick_execution_required: bool = False,
    execution_seed: int = 0,
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

    Stateful protective/account handling and policy-action execution live in
    simulation.backtest.runtime so this public function remains the readable
    composition root for the timing-sensitive execution contract.
    """
    from simulation.backtest.runtime import (
        BacktestSettings,
        BacktestState,
        apply_policy_action,
        liquidate_terminal_position,
        process_protective_and_account_events,
    )

    broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
    guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))
    settings = BacktestSettings(
        cost_model=cost_model,
        lots=lots,
        max_loss_per_trade_usd=max_loss_per_trade_usd,
        take_profit_per_trade_usd=take_profit_per_trade_usd,
        tickbook=tickbook,
        fill_delay_ms=fill_delay_ms,
        randomized_fill_latency_ms=randomized_fill_latency_ms,
        max_tick_quote_wait_ms=max_tick_quote_wait_ms,
        tick_execution_required=tick_execution_required,
        use_structure_sl_tp=use_structure_sl_tp,
        risk_frac=risk_frac,
        rr_ratio=rr_ratio,
        swing_buffer_pts=swing_buffer_pts,
        min_lot=min_lot,
        max_lot=max_lot,
        contract_size=contract_size,
        hold_on_zero=hold_on_zero,
        exit_action=exit_action,
        block_overnight=block_overnight,
    )
    state = BacktestState(account=AccountState(initial_balance=initial_balance))

    common_idx = bars.index.intersection(features.index)
    bars = bars.loc[common_idx]
    features = features.loc[common_idx]

    bar_times = pd.DatetimeIndex(bars.index)
    bar_duration = _bar_duration(bar_times)
    feat_array: np.ndarray[Any, Any] = features.values.astype(np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0).astype(np.float32)
    n_bars = len(bars)
    execution_rng = np.random.default_rng(int(execution_seed))
    if (tick_execution_required or randomized_fill_latency_ms > 0) and tickbook is None:
        raise TickExecutionUnavailable("tick-only execution requires a TickBook")

    for i in range(obs_window, n_bars):
        row = bars.iloc[i]
        bar_time = bar_times[i]
        session = int(row["session_id"]) if "session_id" in row.index else 0
        state.session_set.add(session)
        is_ny = "session" not in row.index or str(row["session"]) == "ny"

        if is_ny and session != state.prev_session:
            state.account.reset_daily()
            state.prev_session = session
        elif state.prev_session is None:
            state.prev_session = session

        bq = _bar_quote(row, cost_model)
        if state.position is not None:
            broker.mark_to_market(state.account, state.position, bq)

        bar_end = pd.Timestamp(bar_time) + bar_duration
        entry_time = (
            None
            if state.position is None or state.position.entry_timestamp is None
            else pd.Timestamp(cast(Any, state.position.entry_timestamp))
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

        execution_event: tuple[tuple[float, float], pd.Timestamp, str, dict[str, Any]] | None = None

        def resolve_bar_execution() -> tuple[
            tuple[float, float], pd.Timestamp, str, dict[str, Any]
        ]:
            nonlocal execution_event
            if execution_event is None:
                execution_event = _fill_quote_event(
                    fill_instant,
                    fill_row,
                    cost_model,
                    tickbook,
                    bar_close_time=fill_bar_close,
                    rng=execution_rng,
                    randomized_fill_latency_ms=randomized_fill_latency_ms,
                    max_tick_quote_wait_ms=max_tick_quote_wait_ms,
                    tick_execution_required=tick_execution_required,
                )
                state.current_execution_meta = execution_event[3]
            return execution_event

        if randomized_fill_latency_ms == 0 and not tick_execution_required:
            resolve_bar_execution()

        last_ny = process_protective_and_account_events(
            state,
            settings,
            broker,
            guardrails,
            bars,
            row,
            i=i,
            bar_time=bar_time,
            bar_duration=bar_duration,
            session=session,
            is_ny=is_ny,
            bar_quote=bq,
            entry_time=entry_time,
            position_active_in_bar=position_active_in_bar,
            full_bar_available_after_entry=full_bar_available_after_entry,
            execution_rng=execution_rng,
            resolve_bar_execution=resolve_bar_execution,
            fill_quote_event=_fill_quote_event,
            sl_tp_fill_quote=_sl_tp_fill_quote,
            bar_spread_price_units=_bar_spread_price_units,
            is_last_ny_bar=_is_last_ny_bar,
        )

        state.equity_curve.append(state.account.equity)
        state.equity_times.append(pd.Timestamp(bar_time) + bar_duration)

        if session in state.breached_sessions or not is_ny or (block_overnight and last_ny):
            continue
        if i + 1 >= n_bars:
            continue

        obs = feat_array[i - obs_window : i].copy()
        action = policy(obs)
        apply_policy_action(
            state,
            settings,
            broker,
            guardrails,
            bars,
            features,
            i=i,
            bar_time=bar_time,
            session=session,
            action=action,
            resolve_bar_execution=resolve_bar_execution,
            is_last_ny_bar=_is_last_ny_bar,
        )

    liquidate_terminal_position(
        state,
        settings,
        broker,
        bars,
        bar_times,
        bar_duration,
        execution_rng,
        fill_quote_event=_fill_quote_event,
        bar_quote=_bar_quote,
    )

    trades_df = pd.DataFrame(state.trade_log)
    equity_series = pd.Series(
        state.equity_curve,
        index=pd.DatetimeIndex(state.equity_times, name=bar_times.name),
        dtype=float,
    )
    return {
        "equity": equity_series,
        "equity_valuation_times": pd.DatetimeIndex(state.equity_times, name=bar_times.name),
        "trades": trades_df,
        "account": state.account,
        "breaches": state.breach_log,
        "breach_events": state.breach_events,
        "n_sessions": len(state.session_set),
        "n_breach_sessions": len(state.breached_sessions),
        "n_sessions_with_trades": len(state.sessions_with_trades),
        "n_sessions_skipped": len(state.session_set) - len(state.sessions_with_trades),
    }
