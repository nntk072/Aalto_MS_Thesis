"""Real-tick bid/ask execution book.

``TickBook`` provides ``quote_at(ts) → (bid, ask)`` using the first tick at or
after a given timestamp. It never forward-fills beyond the last recorded tick;
callers must treat a missing quote as unavailable execution data.

Typical use
-----------
::

    tick_book = build_tick_book(path, tz="Etc/GMT-3",
                                cache_path="cache/US100.cash_ticks_v3-validated-full-day.parquet")
    train_ticks = tick_book.slice(pd.Timestamp("2025-01-01", tz="Etc/GMT-3"),
                                  pd.Timestamp("2025-12-31 23:59", tz="Etc/GMT-3"))
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .loader import iter_ticks_chunks

log = logging.getLogger(__name__)
TICK_CACHE_VERSION = "v3-validated-full-day"


@dataclass(frozen=True, slots=True)
class TickExecutionQuote:
    """A causal tick quote selected after a sampled order-arrival latency."""

    order_time: pd.Timestamp
    arrival_time: pd.Timestamp
    tick_time: pd.Timestamp
    bid: float
    ask: float
    latency_ns: int


@dataclass(frozen=True, slots=True)
class TickExitExecution:
    """A protective exit trigger and its latency-adjusted executable quote."""

    kind: str
    trigger_time: pd.Timestamp
    quote: TickExecutionQuote


class TickExecutionUnavailable(RuntimeError):
    """Raised when an order cannot be matched to an admissible later tick."""


def resolve_tick_execution(
    book: TickBook,
    order_time: pd.Timestamp,
    rng: np.random.Generator,
    *,
    max_latency_ns_exclusive: int = 30_000_000,
    max_quote_wait_ns: int | None = 60_000_000_000,
) -> TickExecutionQuote | None:
    """Apply optional sub-30 ms latency and select the first eligible tick.

    A positive latency bound samples uniformly over integer nanoseconds in
    ``[0, max_latency_ns_exclusive)``; a zero bound means zero delay. The
    selected quote is the first tick at or after arrival; a pre-arrival quote
    is never eligible.
    The caller must treat ``None`` as unavailable tick execution, not substitute
    a candle quote.
    """
    if max_latency_ns_exclusive < 0:
        raise ValueError("max_latency_ns_exclusive cannot be negative")
    submitted = pd.Timestamp(order_time)
    # A zero-latency order still executes from the first observed tick at or
    # after submission. Zero disables random delay; it never enables candle
    # pricing.
    latency_ns = (
        0 if max_latency_ns_exclusive == 0 else int(rng.integers(0, max_latency_ns_exclusive))
    )
    arrival = submitted + pd.Timedelta(latency_ns, unit="ns")
    # Execution differs from a valuation lookup: if the order arrives just
    # before the first observed tick, that first tick is still the first
    # executable quote at or after arrival. ``quote_event_at`` intentionally
    # treats pre-coverage timestamps as unavailable for ordinary lookups.
    event = book.execution_quote_event_at(arrival)
    if event is None:
        return None
    tick_time, bid, ask = event
    if tick_time < arrival:
        raise RuntimeError("TickBook returned a pre-arrival quote for execution")
    if max_quote_wait_ns is not None and tick_time.value - arrival.value > max_quote_wait_ns:
        return None
    return TickExecutionQuote(
        order_time=submitted,
        arrival_time=arrival,
        tick_time=tick_time,
        bid=bid,
        ask=ask,
        latency_ns=latency_ns,
    )


def resolve_tick_exit(
    book: TickBook,
    bar_start: pd.Timestamp,
    direction: int,
    sl: float | None,
    tp: float | None,
    rng: np.random.Generator,
    *,
    bar_minutes: int = 1,
    earliest_time: pd.Timestamp | None = None,
    max_latency_ns_exclusive: int = 30_000_000,
    max_quote_wait_ns: int | None = 60_000_000_000,
) -> TickExitExecution | None:
    """Resolve the first tick-triggered SL/TP after randomized execution latency.

    Stop wins if a single tick is beyond both levels. Once a trigger is found,
    execution uses the first quote at or after the sampled arrival time. A
    triggered order without a timely subsequent quote raises rather than
    silently becoming a candle-priced fill.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    start = pd.Timestamp(bar_start)
    start_ns = start.value
    if earliest_time is not None:
        start_ns = max(start_ns, pd.Timestamp(earliest_time).value)
    end_ns = start.value + int(pd.Timedelta(minutes=bar_minutes).value)
    i0 = int(np.searchsorted(book._ts, start_ns, side="left"))
    i1 = int(np.searchsorted(book._ts, end_ns, side="left"))
    if i0 >= i1:
        return None

    side = book._bid if direction == 1 else book._ask
    for i in range(i0, i1):
        price = float(side[i])
        sl_cross = sl is not None and (
            (direction == 1 and price <= sl) or (direction == -1 and price >= sl)
        )
        tp_cross = tp is not None and (
            (direction == 1 and price >= tp) or (direction == -1 and price <= tp)
        )
        if not sl_cross and not tp_cross:
            continue
        kind = "sl" if sl_cross else "tp"
        trigger_time = pd.Timestamp(int(book._ts[i]), unit="ns", tz="UTC")
        quote = resolve_tick_execution(
            book,
            trigger_time,
            rng,
            max_latency_ns_exclusive=max_latency_ns_exclusive,
            max_quote_wait_ns=max_quote_wait_ns,
        )
        if quote is None:
            raise TickExecutionUnavailable(
                f"{kind.upper()} triggered at {trigger_time}, but no timely quote "
                "exists after its sampled order-arrival time"
            )
        return TickExitExecution(kind=kind, trigger_time=trigger_time, quote=quote)
    return None


def _ffill_inplace(arr: np.ndarray[Any, Any]) -> None:
    """In-place forward-fill of NaNs in a 1-D float array (no pandas)."""
    mask = np.isnan(arr)
    if not mask.any():
        return
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[:] = arr[idx]


class TickBook:
    """Sorted bid/ask lookup table backed by numpy arrays."""

    __slots__ = ("_ts", "_bid", "_ask")

    def __init__(
        self,
        ts_ns: np.ndarray[Any, Any],
        bid: np.ndarray[Any, Any],
        ask: np.ndarray[Any, Any],
    ) -> None:
        self._ts = ts_ns
        self._bid = bid
        self._ask = ask

    def quote_at(self, ts: pd.Timestamp) -> tuple[float, float] | None:
        """Return ``(bid, ask)`` for the first tick at or after *ts*.

        Returns ``None`` when *ts* is outside the recorded tick interval,
        including after the final recorded tick. A caller must not reuse a
        stale final quote as an execution price.
        """
        event = self.quote_event_at(ts)
        if event is None:
            return None
        _quote_time, bid, ask = event
        return bid, ask

    def quote_event_at(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, float, float] | None:
        """Return the selected quote and its recorded tick timestamp."""
        n = len(self._ts)
        if n == 0:
            return None
        ts_ns = ts.value
        if ts_ns < self._ts[0]:
            return None
        i = int(np.searchsorted(self._ts, ts_ns, side="left"))
        if i >= n:
            return None
        return (
            pd.Timestamp(int(self._ts[i]), unit="ns", tz="UTC"),
            float(self._bid[i]),
            float(self._ask[i]),
        )

    def execution_quote_event_at(
        self, ts: pd.Timestamp
    ) -> tuple[pd.Timestamp, float, float] | None:
        """Return the first quote at or after an execution arrival time.

        Unlike ``quote_event_at``, this permits an arrival timestamp earlier
        than the first stored tick. Callers still need a finite quote-wait
        limit so that a distant first observation cannot masquerade as a fill.
        """
        n = len(self._ts)
        if n == 0:
            return None
        i = int(np.searchsorted(self._ts, pd.Timestamp(ts).value, side="left"))
        if i >= n:
            return None
        return (
            pd.Timestamp(int(self._ts[i]), unit="ns", tz="UTC"),
            float(self._bid[i]),
            float(self._ask[i]),
        )

    def __len__(self) -> int:
        return len(self._ts)

    def bar_quotes(
        self, bar_start: pd.Timestamp, bar_minutes: int = 1
    ) -> Iterator[tuple[pd.Timestamp, float, float]]:
        """Yield recorded quotes in timestamp order within a half-open bar interval."""
        start_ns = pd.Timestamp(bar_start).value
        end_ns = start_ns + int(pd.Timedelta(minutes=bar_minutes).value)
        i0 = int(np.searchsorted(self._ts, start_ns, side="left"))
        i1 = int(np.searchsorted(self._ts, end_ns, side="left"))
        for idx in range(i0, i1):
            yield (
                pd.Timestamp(int(self._ts[idx]), unit="ns", tz="UTC"),
                float(self._bid[idx]),
                float(self._ask[idx]),
            )

    def delayed_exit_quote(
        self,
        bar_start: pd.Timestamp,
        direction: int,
        sl: float | None,
        tp: float | None,
        delay: pd.Timedelta,
        bar_minutes: int = 1,
    ) -> tuple[str, float, float] | None:
        """Quote ``delay`` after the first tick that trades through SL or TP.

        A long exit watches the bid; a short exit watches the ask. The first
        tick beyond either level is the trigger, so a target that prints
        before the stop wins. Delay 0 fills that crossing tick, which is
        already through the level. Returns ``None`` when the bar has no
        crossing tick.
        """
        event = self.delayed_exit_event(
            bar_start, direction, sl, tp, delay, bar_minutes=bar_minutes
        )
        if event is None:
            return None
        kind, _fill_time, bid, ask = event
        return kind, bid, ask

    def delayed_exit_event(
        self,
        bar_start: pd.Timestamp,
        direction: int,
        sl: float | None,
        tp: float | None,
        delay: pd.Timedelta,
        bar_minutes: int = 1,
        earliest_time: pd.Timestamp | None = None,
    ) -> tuple[str, pd.Timestamp, float, float] | None:
        """Return trigger kind, actual fill time, and delayed executable quote."""
        n = len(self._ts)
        if n == 0 or direction not in (1, -1):
            return None
        start_ns = pd.Timestamp(bar_start).value
        if earliest_time is not None:
            start_ns = max(start_ns, pd.Timestamp(earliest_time).value)
        end_ns = pd.Timestamp(bar_start).value + int(pd.Timedelta(minutes=bar_minutes).value)
        i0 = int(np.searchsorted(self._ts, start_ns, side="left"))
        i1 = int(np.searchsorted(self._ts, end_ns, side="left"))
        if i0 >= i1:
            return None
        side = self._bid if direction == 1 else self._ask
        trigger = -1
        kind = ""
        for i in range(i0, i1):
            px = float(side[i])
            sl_cross = sl is not None and (
                (direction == 1 and px <= sl) or (direction == -1 and px >= sl)
            )
            tp_cross = tp is not None and (
                (direction == 1 and px >= tp) or (direction == -1 and px <= tp)
            )
            if sl_cross:
                trigger = i
                kind = "sl"
                break
            if tp_cross:
                trigger = i
                kind = "tp"
                break
        if trigger < 0:
            return None
        fill_ns = int(self._ts[trigger]) + int(pd.Timedelta(delay).value)
        fill_time = pd.Timestamp(fill_ns, unit="ns", tz="UTC")
        quote_event = self.quote_event_at(fill_time)
        if quote_event is None:
            return None
        quote_time, bid, ask = quote_event
        return kind, max(fill_time, quote_time), bid, ask

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> TickBook:
        """Return a sub-book covering the half-open interval [start, end)."""
        i_s = int(np.searchsorted(self._ts, start.value, side="left"))
        i_e = int(np.searchsorted(self._ts, end.value, side="left"))
        return TickBook(
            self._ts[i_s:i_e].copy(),
            self._bid[i_s:i_e].copy(),
            self._ask[i_s:i_e].copy(),
        )


def ticks_covering(book: TickBook | None, bars: pd.DataFrame) -> TickBook | None:
    """Slice *book* to the bars, plus one extra minute for the fill delay.

    An empty slice means this window has no tape, so the caller keeps the
    level fill.
    """
    if book is None or len(book) == 0 or bars.empty:
        return None
    start = pd.Timestamp(bars.index[0])
    end = pd.Timestamp(bars.index[-1]) + pd.Timedelta(minutes=2)
    sliced = book.slice(start, end)
    if len(sliced) == 0:
        return None
    return sliced


def tick_execution_coverage(
    book: TickBook | None,
    order_times: pd.DatetimeIndex,
    *,
    max_latency_ns_exclusive: int,
    max_quote_wait_ms: int,
) -> dict[str, Any]:
    """Measure whether every candidate order time has a timely executable tick.

    For a positive latency bound, coverage is guaranteed for every latency in
    ``[0, max_latency)`` when a valid quote follows the latest possible
    arrival within ``max_quote_wait``. A zero bound checks the immediate order
    time. This is a conservative preflight for environments requiring tick
    quotes at each candidate market step.
    """
    if max_latency_ns_exclusive < 0 or max_latency_ns_exclusive > 30_000_000:
        raise ValueError("max_latency_ns_exclusive must be in [0, 30_000_000]")
    if max_quote_wait_ms <= 0:
        raise ValueError("max_quote_wait_ms must be positive")
    if order_times.tz is None:
        raise ValueError("order_times must be timezone-aware")

    order_ns = order_times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if order_ns.size == 0:
        return {
            "order_count": 0,
            "covered_count": 0,
            "unavailable_count": 0,
            "coverage_fraction": 1.0,
            "first_unavailable_order_time": None,
        }

    covered = np.zeros(order_ns.size, dtype=bool)
    if book is not None and len(book) > 0:
        latest_arrival_ns = (
            order_ns if max_latency_ns_exclusive == 0 else order_ns + max_latency_ns_exclusive - 1
        )
        candidate_idx = np.searchsorted(book._ts, latest_arrival_ns, side="left")
        has_quote = candidate_idx < len(book)
        if has_quote.any():
            covered[has_quote] = (
                book._ts[candidate_idx[has_quote]] - order_ns[has_quote]
                <= max_quote_wait_ms * 1_000_000
            )

    unavailable = np.flatnonzero(~covered)
    first_missing = (
        None
        if unavailable.size == 0
        else pd.Timestamp(int(order_ns[unavailable[0]]), unit="ns", tz="UTC").isoformat()
    )
    return {
        "order_count": int(order_ns.size),
        "covered_count": int(covered.sum()),
        "unavailable_count": int(unavailable.size),
        "coverage_fraction": float(covered.mean()),
        "first_unavailable_order_time": first_missing,
    }


def build_tick_book(
    path: str | Path,
    tz: str = "Etc/GMT-3",
    session_start: str | None = None,
    session_end: str | None = None,
    cache_path: str | Path | None = None,
    force: bool = False,
    chunksize: int = 2_000_000,
) -> TickBook:
    """Load an MT5 tick CSV, optionally session-filter, forward-fill, build a TickBook.

    Pass ``session_start=None`` and ``session_end=None`` (the default) to keep
    ticks outside NY so overnight replay can quote skipped bars.

    Memory-safe by design
    ----------------------
    MT5 tick exports can be multiple GB.  This streams the file in bounded
    chunks (:func:`iter_ticks_chunks`) instead of materialising it whole:

    * Only ``date``/``time``/``bid``/``ask`` are read — ``last``/``volume``/
      ``flags`` are dropped at the CSV-parser level via ``usecols``.
    * ``bid``/``ask`` are read as ``float32``.
    * The session-hours filter is applied **per chunk**, discarding the
      ~70%+ of rows outside the trading window before they ever accumulate.
    * Forward-fill state (the last valid bid/ask) is carried across chunk
      boundaries so results are identical to a whole-file ffill.

    Peak memory is therefore roughly one chunk's worth of raw CSV data plus
    the final (already session-filtered, much smaller) tick arrays — not the
    full file size multiplied by pandas' usual 3-5x parsing overhead.
    """
    path = Path(path)
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            log.info("Loading tick cache: %s", cache_path)
            df = pd.read_parquet(cache_path)
            return TickBook(
                cast(np.ndarray[Any, Any], df["ts_ns"].to_numpy()),
                cast(np.ndarray[Any, Any], df["bid"].to_numpy()),
                cast(np.ndarray[Any, Any], df["ask"].to_numpy()),
            )

    log.info("Building TickBook from %s (chunked, memory-safe) …", path)

    start_t = (
        pd.Timestamp(f"2000-01-01 {session_start}").time() if session_start is not None else None
    )
    end_t = pd.Timestamp(f"2000-01-01 {session_end}").time() if session_end is not None else None

    ts_chunks: list[np.ndarray[Any, Any]] = []
    bid_chunks: list[np.ndarray[Any, Any]] = []
    ask_chunks: list[np.ndarray[Any, Any]] = []

    last_bid = np.nan
    last_ask = np.nan
    n_rows_total = 0
    n_rows_kept = 0

    for raw_chunk in iter_ticks_chunks(path, chunksize=chunksize):
        n_rows_total += len(raw_chunk)

        # Chunk-local datetime parse (bounded memory) — Etc/GMT-3 etc. are
        # fixed-offset zones (no DST), so tz_localize here is unambiguous
        # regardless of how the file happens to be chunked.  MT5 tick time
        # includes milliseconds (HH:MM:SS.fff), unlike bar files.
        dt = pd.to_datetime(
            raw_chunk["date"].to_numpy() + " " + raw_chunk["time"].to_numpy(),
            format="%Y.%m.%d %H:%M:%S.%f",
        )
        dt = dt.tz_localize(tz)

        if start_t is not None and end_t is not None:
            t = dt.time
            mask = (t >= start_t) & (t <= end_t)
            if not mask.any():
                continue
        else:
            mask = np.ones(len(dt), dtype=bool)

        bid = raw_chunk["bid"].to_numpy(dtype=np.float32, copy=True)[mask]
        ask = raw_chunk["ask"].to_numpy(dtype=np.float32, copy=True)[mask]
        ts_ns = dt.values.astype("int64")[mask]

        # Treat non-finite and non-positive sides as missing before carrying
        # the last usable side forward. Infinity is not handled by
        # ``_ffill_inplace`` and must never enter the executable quote book.
        bid[~np.isfinite(bid) | (bid <= 0.0)] = np.nan
        ask[~np.isfinite(ask) | (ask <= 0.0)] = np.nan

        # Patch the first element with the carried-forward quote so ffill
        # is continuous across the chunk boundary.
        if len(bid):
            if np.isnan(bid[0]) and not np.isnan(last_bid):
                bid[0] = last_bid
            if np.isnan(ask[0]) and not np.isnan(last_ask):
                ask[0] = last_ask

        _ffill_inplace(bid)
        _ffill_inplace(ask)

        # A side can update independently, so temporarily crossed pairs are
        # possible while the other side catches up. Keep those side values as
        # ffill state, but do not expose an invalid pair to execution.
        valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask >= bid)
        if len(bid):
            last_bid, last_ask = float(bid[-1]), float(ask[-1])
        if not valid.all():
            bid, ask, ts_ns = bid[valid], ask[valid], ts_ns[valid]

        if len(bid):
            ts_chunks.append(ts_ns)
            bid_chunks.append(bid)
            ask_chunks.append(ask)
            n_rows_kept += len(bid)

    log.info(
        "Tick scan complete: kept %d / %d rows (%.1f%%)",
        n_rows_kept,
        n_rows_total,
        100.0 * n_rows_kept / max(n_rows_total, 1),
    )

    if not ts_chunks:
        ts_ns = np.array([], dtype="int64")
        bid = np.array([], dtype=np.float64)
        ask = np.array([], dtype=np.float64)
    else:
        ts_ns = np.concatenate(ts_chunks)
        bid = np.concatenate(bid_chunks).astype(np.float64)
        ask = np.concatenate(ask_chunks).astype(np.float64)
        # MT5 tick exports are already chronological, but sort defensively.
        order = np.argsort(ts_ns, kind="stable")
        ts_ns, bid, ask = ts_ns[order], bid[order], ask[order]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ts_ns": ts_ns, "bid": bid, "ask": ask}).to_parquet(cache_path, index=False)
        log.info("Tick cache written: %s (%d ticks)", cache_path, len(ts_ns))

    return TickBook(ts_ns, bid, ask)
