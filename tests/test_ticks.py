"""Tests: TickBook quote lookup + chunked builder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_rl.data.ticks import TickBook, build_tick_book, _ffill_inplace


def _make_book() -> TickBook:
    times = pd.to_datetime([
        "2025-01-06 16:30:00",
        "2025-01-06 16:30:05",
        "2025-01-06 16:30:10",
    ], utc=True)
    ts_ns = times.values.astype("int64")
    bid = np.array([100.0, 100.5, 101.0])
    ask = np.array([100.6, 101.1, 101.6])
    return TickBook(ts_ns, bid, ask)


def test_quote_at_exact_tick():
    book = _make_book()
    ts = pd.Timestamp("2025-01-06 16:30:05", tz="UTC")
    bid, ask = book.quote_at(ts)
    assert bid == pytest.approx(100.5)
    assert ask == pytest.approx(101.1)


def test_quote_at_before_first_tick_returns_none():
    book = _make_book()
    ts = pd.Timestamp("2025-01-06 16:29:00", tz="UTC")
    assert book.quote_at(ts) is None


def test_ffill_inplace_fills_internal_nans():
    arr = np.array([1.0, np.nan, np.nan, 2.0, np.nan], dtype=np.float32)
    _ffill_inplace(arr)
    np.testing.assert_allclose(arr, [1.0, 1.0, 1.0, 2.0, 2.0])


def test_ffill_inplace_leading_nan_stays_nan():
    arr = np.array([np.nan, np.nan, 3.0], dtype=np.float32)
    _ffill_inplace(arr)
    assert np.isnan(arr[0]) and np.isnan(arr[1])
    assert arr[2] == pytest.approx(3.0)


def _write_tick_csv(path: Path, rows: list[tuple[str, str, float, float]]) -> None:
    """Write a synthetic MT5 tick CSV. ``time`` must include milliseconds
    (HH:MM:SS.fff), matching the real MT5 export format."""
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"]
    for date, time, bid, ask in rows:
        lines.append(f"{date}\t{time}\t{bid}\t{ask}\t0\t0\t0")
    path.write_text("\n".join(lines) + "\n")


def test_build_tick_book_chunked_matches_expected(tmp_path: Path):
    """Small synthetic CSV, forced to a 2-row chunksize to exercise the
    cross-chunk forward-fill / session-filter boundary logic."""
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(csv_path, [
        ("2025.01.06", "10:00:00.000", 1.0, 1.1),   # outside session -> dropped
        ("2025.01.06", "16:30:00.618", 100.0, 100.6),
        ("2025.01.06", "16:30:05.020", np.nan, 101.1),  # bid missing -> ffill from prev row
        ("2025.01.06", "16:30:10.500", 101.0, 101.6),
        ("2025.01.06", "23:30:00.000", 200.0, 200.6),  # outside session -> dropped
    ])

    book = build_tick_book(
        csv_path, tz="Etc/GMT-3",
        session_start="16:30", session_end="23:00",
        cache_path=None, chunksize=2,  # forces multiple small chunks
    )

    assert len(book) == 3
    ts = pd.Timestamp("2025-01-06 16:30:05", tz="Etc/GMT-3")
    bid, ask = book.quote_at(ts)
    assert bid == pytest.approx(100.0)  # forward-filled across the chunk boundary
    assert ask == pytest.approx(101.1)


def test_build_tick_book_uses_and_writes_cache(tmp_path: Path):
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(csv_path, [
        ("2025.01.06", "16:30:00.000", 100.0, 100.6),
        ("2025.01.06", "16:30:05.000", 100.5, 101.1),
    ])
    cache_path = tmp_path / "ticks.parquet"

    book1 = build_tick_book(csv_path, cache_path=cache_path, chunksize=1)
    assert cache_path.exists()
    assert len(book1) == 2

    # Second call should hit the cache path (force=False) — same result.
    book2 = build_tick_book(csv_path, cache_path=cache_path, chunksize=1)
    assert len(book2) == len(book1)
