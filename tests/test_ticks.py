"""Tests: TickBook quote lookup + chunked builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.data.pipeline import build_tick_books
from quant_rl.data.ticks import (
    TICK_CACHE_VERSION,
    TickBook,
    _ffill_inplace,
    build_tick_book,
)


def _make_book() -> TickBook:
    times = pd.to_datetime(
        [
            "2025-01-06 16:30:00",
            "2025-01-06 16:30:05",
            "2025-01-06 16:30:10",
        ],
        utc=True,
    )
    ts_ns = times.values.astype("int64")
    bid = np.array([100.0, 100.5, 101.0])
    ask = np.array([100.6, 101.1, 101.6])
    return TickBook(ts_ns, bid, ask)


def test_quote_at_exact_tick():
    book = _make_book()
    ts = pd.Timestamp("2025-01-06 16:30:05", tz="UTC")
    quote = book.quote_at(ts)
    assert quote is not None
    bid, ask = quote
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


def test_build_tick_books_bypasses_legacy_tick_cache(tmp_path: Path):
    symbol = "US100.cash"
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(csv_path, [("2025.01.06", "16:30:00.000", 100.0, 100.6)])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    legacy_cache = cache_dir / f"{symbol}_ticks_v2-full-day.parquet"
    pd.DataFrame(
        {
            "ts_ns": [pd.Timestamp("2025-01-06 13:30:00", tz="UTC").value],
            "bid": [np.inf],
            "ask": [100.6],
        }
    ).to_parquet(legacy_cache, index=False)
    legacy_cache_contents = legacy_cache.read_bytes()
    config = OmegaConf.create(
        {
            "costs": {"use_tick_execution": True},
            "data": {
                "raw_dir": str(tmp_path),
                "cache_dir": str(cache_dir),
                "tick_files": {symbol: csv_path.name},
                "symbols": [symbol],
                "tz": "Etc/GMT-3",
            },
        }
    )

    books = build_tick_books(config)

    book = books[symbol]
    assert book is not None
    quote = book.quote_at(pd.Timestamp("2025-01-06 16:30:00", tz="Etc/GMT-3"))
    assert quote is not None
    assert quote[0] == pytest.approx(100.0)
    assert quote[1] == pytest.approx(100.6)
    assert legacy_cache.is_file()
    assert legacy_cache.read_bytes() == legacy_cache_contents
    assert (cache_dir / f"{symbol}_ticks_{TICK_CACHE_VERSION}.parquet").is_file()


def test_build_tick_book_chunked_matches_expected(tmp_path: Path):
    """Small synthetic CSV, forced to a 2-row chunksize to exercise the
    cross-chunk forward-fill / session-filter boundary logic."""
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(
        csv_path,
        [
            ("2025.01.06", "10:00:00.000", 1.0, 1.1),  # outside session -> dropped
            ("2025.01.06", "16:30:00.618", 100.0, 100.6),
            ("2025.01.06", "16:30:05.020", np.nan, 101.1),  # bid missing -> ffill from prev row
            ("2025.01.06", "16:30:10.500", 101.0, 101.6),
            ("2025.01.06", "23:30:00.000", 200.0, 200.6),  # outside session -> dropped
        ],
    )

    book = build_tick_book(
        csv_path,
        tz="Etc/GMT-3",
        session_start="16:30",
        session_end="23:00",
        cache_path=None,
        chunksize=2,  # forces multiple small chunks
    )

    assert len(book) == 3
    ts = pd.Timestamp("2025-01-06 16:30:05", tz="Etc/GMT-3")
    quote = book.quote_at(ts)
    assert quote is not None
    bid, ask = quote
    assert bid == pytest.approx(100.0)  # forward-filled across the chunk boundary
    assert ask == pytest.approx(101.1)


def test_build_tick_book_filters_nonfinite_nonpositive_and_crossed_quotes(tmp_path: Path):
    csv_path = tmp_path / "invalid_quotes.csv"
    _write_tick_csv(
        csv_path,
        [
            ("2025.01.06", "16:30:00.000", 100.0, 100.6),
            ("2025.01.06", "16:30:01.000", float("inf"), 101.2),
            ("2025.01.06", "16:30:02.000", 102.0, float("inf")),
            ("2025.01.06", "16:30:03.000", 102.0, 102.5),
            ("2025.01.06", "16:30:04.000", 0.0, 0.0),
            ("2025.01.06", "16:30:05.000", 104.0, 103.0),
            ("2025.01.06", "16:30:06.000", np.nan, 104.5),
        ],
    )

    book = build_tick_book(csv_path, tz="Etc/GMT-3", chunksize=3)
    quotes = list(book.bar_quotes(pd.Timestamp("2025-01-06 16:30:00", tz="Etc/GMT-3")))

    assert len(quotes) == 5
    assert all(np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask for _, bid, ask in quotes)
    assert quotes[1][1:] == pytest.approx((100.0, 101.2))
    assert quotes[-1][1:] == pytest.approx((104.0, 104.5))


def test_ticks_available_outside_ny(tmp_path: Path):
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(
        csv_path,
        [
            ("2025.01.06", "10:00:00.000", 1.0, 1.1),
            ("2025.01.06", "16:30:00.000", 100.0, 100.6),
        ],
    )
    book = build_tick_book(
        csv_path,
        tz="Etc/GMT-3",
        session_start=None,
        session_end=None,
        cache_path=None,
        chunksize=10,
    )
    assert len(book) == 2
    q = book.quote_at(pd.Timestamp("2025-01-06 10:00:00", tz="Etc/GMT-3"))
    assert q is not None
    assert q[0] == pytest.approx(1.0)


def test_build_tick_book_uses_and_writes_cache(tmp_path: Path):
    csv_path = tmp_path / "ticks.csv"
    _write_tick_csv(
        csv_path,
        [
            ("2025.01.06", "16:30:00.000", 100.0, 100.6),
            ("2025.01.06", "16:30:05.000", 100.5, 101.1),
        ],
    )
    cache_path = tmp_path / "ticks.parquet"

    book1 = build_tick_book(csv_path, cache_path=cache_path, chunksize=1)
    assert cache_path.exists()
    assert len(book1) == 2

    book2 = build_tick_book(csv_path, cache_path=cache_path, chunksize=1)
    assert len(book2) == len(book1)
