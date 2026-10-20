"""Causal, policy-visible calendar and quote context for each completed M1 bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_data.loading.ticks import TickBook

POLICY_MARKET_CONTEXT_VERSION = "m1-calendar-and-tick-context-v1"


def build_policy_market_context(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    tickbook: TickBook | None,
    *,
    schedule_path: str | Path,
    session_start: str = "16:30",
    session_end: str = "23:00",
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Return model inputs using schedule-known and completed-bar quote data.

    Schedule flags are keyed by broker calendar date and therefore known before
    that date's first bar. Quote aggregates use ticks in ``[bar open, bar end)``;
    the environment exposes the feature row only after that bar, preserving
    decision-time causality.
    """
    if bars.empty or not bars.index.equals(features.index):
        raise ValueError("bars and features must have the same nonempty index")
    if tickbook is None or len(tickbook._ts) == 0:
        raise ValueError("policy quote features require the configured bid/ask tick book")
    if "close" not in bars or "atr_5" not in features:
        raise ValueError("policy quote features require bar close and feature atr_5")

    schedule = pd.read_csv(schedule_path, dtype={"session_date": str})
    required = {
        "session_date",
        "scheduled_ny_close_broker_time",
        "scheduled_short",
    }
    if not required.issubset(schedule.columns):
        raise ValueError(f"session schedule is missing columns: {sorted(required - set(schedule))}")
    if schedule["session_date"].duplicated().any():
        raise ValueError("session schedule contains duplicate dates")

    start_minute = _minute_of_day(session_start)
    default_close_minute = _minute_of_day(session_end)
    close_by_date = {
        str(row.session_date): _minute_of_day(str(row.scheduled_ny_close_broker_time))
        for row in schedule.itertuples(index=False)
    }
    short_by_date = {
        str(row.session_date): str(row.scheduled_short).lower() in {"1", "true", "yes"}
        for row in schedule.itertuples(index=False)
    }

    index = pd.DatetimeIndex(bars.index)
    dates = np.asarray(index.strftime("%Y-%m-%d"), dtype=object)
    minute_of_day = np.asarray(index.hour * 60 + index.minute, dtype=np.float64)
    close_minutes = np.fromiter(
        (close_by_date.get(date, default_close_minute) for date in dates),
        dtype=np.float64,
        count=len(index),
    )
    scheduled_short = np.fromiter(
        (short_by_date.get(date, False) for date in dates),
        dtype=np.float32,
        count=len(index),
    )

    index_ns = index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    end_ns = index_ns + int(pd.Timedelta(minutes=bar_minutes).value)
    tick_ts = tickbook._ts
    left = np.searchsorted(tick_ts, index_ns, side="left")
    right = np.searchsorted(tick_ts, end_ns, side="left")
    count = right - left
    last = right - 1
    has_quote = count > 0
    bid = np.full(len(index), np.nan, dtype=np.float64)
    ask = np.full(len(index), np.nan, dtype=np.float64)
    age_minutes = np.full(len(index), np.nan, dtype=np.float64)
    bid[has_quote] = tickbook._bid[last[has_quote]]
    ask[has_quote] = tickbook._ask[last[has_quote]]
    age_minutes[has_quote] = (end_ns[has_quote] - 1 - tick_ts[last[has_quote]]) / 60_000_000_000.0

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(features["atr_5"], errors="coerce").to_numpy(dtype=np.float64)
    scale = np.where(np.isfinite(atr) & (atr > 0.0), atr, np.maximum(np.abs(close) * 1e-6, 1e-6))
    valid_quote = has_quote & np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask >= bid)
    for values in (bid, ask, age_minutes):
        values[~valid_quote] = np.nan

    session_minutes = np.maximum(close_minutes - start_minute, 1.0)
    session_clock = np.maximum(close_minutes - np.maximum(minute_of_day, start_minute), 0.0)
    out = pd.DataFrame(
        {
            "ny_scheduled_short": scheduled_short,
            "ny_scheduled_close_fraction": close_minutes / 1440.0,
            "ny_minutes_to_scheduled_close": np.minimum(session_clock / session_minutes, 1.0),
            "tick_bid_offset_atr": (bid - close) / scale,
            "tick_ask_offset_atr": (ask - close) / scale,
            "tick_spread_atr": (ask - bid) / scale,
            "tick_quote_age_fraction": age_minutes,
            "tick_quote_activity": np.log1p(count.astype(np.float64)),
            "tick_quote_available": valid_quote.astype(np.float32),
        },
        index=index,
    )
    # A missing quote is explicit in tick_quote_available; sequence NaNs are
    # zero-filled by the existing observation constructor.
    return out


def attach_policy_market_context(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    tickbook: TickBook | None,
    *,
    schedule_path: str | Path,
    session_start: str = "16:30",
    session_end: str = "23:00",
) -> pd.DataFrame:
    context = build_policy_market_context(
        bars,
        features,
        tickbook,
        schedule_path=schedule_path,
        session_start=session_start,
        session_end=session_end,
    )
    collisions = set(context.columns).intersection(features.columns)
    if collisions:
        raise ValueError(f"policy market-context columns already exist: {sorted(collisions)}")
    return pd.concat([features, context], axis=1)


def _minute_of_day(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid HH:MM time {value!r}")
    hour, minute = (int(part) for part in parts)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"invalid HH:MM time {value!r}")
    return hour * 60 + minute
