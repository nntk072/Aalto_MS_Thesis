"""Real-tick bid/ask execution book.

``TickBook`` provides ``quote_at(ts) → (bid, ask)`` using the first tick at or
after a given timestamp.  It is the primary fill-price source; the bar-spread
fallback in ``CostModel.bar_quote`` is used when no tick is available.

Typical use
-----------
::

    tick_book = build_tick_book(path, tz="Etc/GMT-3",
                                session_start="16:30", session_end="23:00",
                                cache_path="cache/US100.cash_ticks.parquet")
    train_ticks = tick_book.slice(pd.Timestamp("2025-01-01", tz="Etc/GMT-3"),
                                  pd.Timestamp("2025-12-31 23:59", tz="Etc/GMT-3"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .loader import iter_ticks_chunks

log = logging.getLogger(__name__)


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

        Returns ``None`` when *ts* is before the first recorded tick (no
        coverage yet — caller should fall back to ``CostModel.bar_quote``).
        Forward-fills with the last quote when *ts* is after the final tick.
        """
        n = len(self._ts)
        if n == 0:
            return None
        ts_ns = ts.value
        if ts_ns < self._ts[0]:
            return None
        i = int(np.searchsorted(self._ts, ts_ns, side="left"))
        if i >= n:
            i = n - 1
        return float(self._bid[i]), float(self._ask[i])

    def __len__(self) -> int:
        return len(self._ts)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> TickBook:
        """Return a sub-book covering the half-open interval [start, end)."""
        i_s = int(np.searchsorted(self._ts, start.value, side="left"))
        i_e = int(np.searchsorted(self._ts, end.value, side="left"))
        return TickBook(
            self._ts[i_s:i_e].copy(),
            self._bid[i_s:i_e].copy(),
            self._ask[i_s:i_e].copy(),
        )


def build_tick_book(
    path: str | Path,
    tz: str = "Etc/GMT-3",
    session_start: str = "16:30",
    session_end: str = "23:00",
    cache_path: str | Path | None = None,
    force: bool = False,
    chunksize: int = 2_000_000,
) -> TickBook:
    """Load an MT5 tick CSV, session-filter, forward-fill, and build a TickBook.

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

    start_t = pd.Timestamp(f"2000-01-01 {session_start}").time()
    end_t = pd.Timestamp(f"2000-01-01 {session_end}").time()

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

        # Session filter FIRST — drops the majority of rows immediately.
        t = dt.time
        mask = (t >= start_t) & (t <= end_t)
        if not mask.any():
            continue

        bid = raw_chunk["bid"].to_numpy(dtype=np.float32, copy=True)[mask]
        ask = raw_chunk["ask"].to_numpy(dtype=np.float32, copy=True)[mask]
        ts_ns = dt.values.astype("int64")[mask]

        # Patch the first element with the carried-forward quote so ffill
        # is continuous across the chunk boundary.
        if len(bid):
            if np.isnan(bid[0]) and not np.isnan(last_bid):
                bid[0] = last_bid
            if np.isnan(ask[0]) and not np.isnan(last_ask):
                ask[0] = last_ask

        _ffill_inplace(bid)
        _ffill_inplace(ask)

        valid = ~(np.isnan(bid) | np.isnan(ask))
        if not valid.all():
            bid, ask, ts_ns = bid[valid], ask[valid], ts_ns[valid]

        if len(bid):
            last_bid, last_ask = float(bid[-1]), float(ask[-1])

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
