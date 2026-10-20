"""Tests: TickBook quote lookup + chunked builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from market_data.loading.pipeline import build_tick_books
from market_data.loading.ticks import (
    TICK_CACHE_VERSION,
    TickBook,
    TickExecutionUnavailable,
    _ffill_inplace,
    build_tick_book,
    resolve_tick_execution,
    resolve_tick_execution_by_deadline,
    resolve_tick_exit,
    tick_execution_coverage,
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


def test_equal_timestamp_quote_uses_first_source_row_across_chunks(tmp_path: Path):
    csv_path = tmp_path / "equal_timestamp_ticks.csv"
    _write_tick_csv(
        csv_path,
        [
            ("2025.01.06", "16:30:00.000", 100.0, 100.6),
            ("2025.01.06", "16:30:00.000", 100.2, 100.8),
            ("2025.01.06", "16:30:01.000", 101.0, 101.6),
        ],
    )
    book = build_tick_book(csv_path, session_start=None, session_end=None, chunksize=1)
    tick_time = pd.Timestamp("2025-01-06 16:30:00", tz="Etc/GMT-3")

    event = book.quote_event_at(tick_time)
    fill = resolve_tick_execution(
        book,
        tick_time,
        np.random.default_rng(1729),
        max_latency_ns_exclusive=0,
    )

    assert event is not None
    assert event[1:] == pytest.approx((100.0, 100.6))
    assert fill is not None
    assert fill.tick_time == pd.Timestamp("2025-01-06 13:30:00", tz="UTC")
    assert (fill.bid, fill.ask) == pytest.approx((100.0, 100.6))


def test_quote_at_before_first_tick_returns_none():
    book = _make_book()
    ts = pd.Timestamp("2025-01-06 16:29:00", tz="UTC")
    assert book.quote_at(ts) is None


def test_quote_at_after_last_tick_returns_none_instead_of_stale_quote():
    book = _make_book()
    ts = pd.Timestamp("2025-01-06 16:30:11", tz="UTC")
    assert book.quote_at(ts) is None
    assert book.quote_event_at(ts) is None


def test_randomized_tick_execution_is_reproducible_and_never_uses_prearrival_tick():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    tick_times = pd.to_datetime(
        [start + pd.Timedelta(milliseconds=10), start + pd.Timedelta(milliseconds=40)],
        utc=True,
    )
    book = TickBook(
        tick_times.values.astype("int64"),
        np.array([100.0, 101.0]),
        np.array([100.5, 101.5]),
    )

    first = resolve_tick_execution(book, start, np.random.default_rng(1729))
    repeated = resolve_tick_execution(book, start, np.random.default_rng(1729))

    assert first == repeated
    assert first is not None
    assert 0 <= first.latency_ns < 30_000_000
    assert first.tick_time >= first.arrival_time


def test_randomized_execution_can_use_first_tick_after_coverage_start():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    first_tick = start + pd.Timedelta(seconds=1)
    book = TickBook(
        np.array([first_tick.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    fill = resolve_tick_execution(book, start, np.random.default_rng(1729))

    assert fill is not None
    assert fill.tick_time == first_tick
    assert fill.tick_time >= fill.arrival_time
    assert fill.bid == pytest.approx(100.0)


def test_zero_latency_execution_uses_first_tick_at_or_after_order_time():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    tick_time = start + pd.Timedelta(milliseconds=2)
    book = TickBook(
        np.array([tick_time.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    fill = resolve_tick_execution(
        book,
        start,
        np.random.default_rng(1729),
        max_latency_ns_exclusive=0,
    )

    assert fill is not None
    assert fill.latency_ns == 0
    assert fill.arrival_time == start
    assert fill.tick_time == tick_time
    assert fill.tick_time >= fill.arrival_time


def test_randomized_execution_rejects_first_tick_beyond_quote_wait_limit():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    book = TickBook(
        np.array([(start + pd.Timedelta(seconds=2)).value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )
    assert (
        resolve_tick_execution(
            book,
            start,
            np.random.default_rng(1729),
            max_quote_wait_ns=1_000_000_000,
        )
        is None
    )


def test_scheduled_close_execution_uses_first_tick_after_arrival():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    latency_ns = int(np.random.default_rng(1729).integers(0, 30_000_000))
    eligible_tick = deadline + pd.Timedelta(latency_ns + 1_000_000, unit="ns")
    next_session_tick = deadline + pd.Timedelta(hours=5)
    book = TickBook(
        np.array([eligible_tick.value, next_session_tick.value], dtype=np.int64),
        np.array([100.0, 101.0]),
        np.array([100.5, 101.5]),
    )

    fill = resolve_tick_execution(
        book,
        deadline,
        np.random.default_rng(1729),
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ns=60_000_000_000,
    )

    assert fill is not None
    assert 0 <= fill.latency_ns < 30_000_000
    assert fill.arrival_time >= deadline
    assert fill.tick_time > fill.arrival_time
    assert fill.tick_time == eligible_tick
    assert (fill.bid, fill.ask) == pytest.approx((100.0, 100.5))


def test_scheduled_close_rejects_quotes_only_before_arrival():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    prior_tick = deadline - pd.Timedelta(milliseconds=1)
    book = TickBook(
        np.array([prior_tick.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    fill = resolve_tick_execution(
        book,
        deadline,
        np.random.default_rng(1729),
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ns=60_000_000_000,
    )

    assert fill is None


def test_scheduled_close_uses_first_quote_after_arrival_before_deadline():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    order_time = deadline - pd.Timedelta(minutes=1)
    latency_ns = int(np.random.default_rng(1729).integers(0, 30_000_000))
    arrival = order_time + pd.Timedelta(latency_ns, unit="ns")
    eligible_tick = arrival + pd.Timedelta(milliseconds=1)
    book = TickBook(
        np.array([eligible_tick.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    fill = resolve_tick_execution_by_deadline(
        book,
        order_time,
        deadline,
        np.random.default_rng(1729),
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ns=60_000_000_000,
    )

    assert fill is not None
    assert fill.order_time == order_time
    assert order_time <= fill.arrival_time < fill.tick_time <= deadline
    assert fill.tick_time == eligible_tick


def test_scheduled_close_rejects_quote_before_sampled_arrival():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    order_time = deadline - pd.Timedelta(minutes=1)
    seed = 1729
    latency_ns = int(np.random.default_rng(seed).integers(0, 30_000_000))
    assert latency_ns > 0
    pre_arrival_tick = order_time + pd.Timedelta(latency_ns - 1, unit="ns")
    next_session_tick = deadline + pd.Timedelta(hours=5)
    book = TickBook(
        np.array([pre_arrival_tick.value, next_session_tick.value], dtype=np.int64),
        np.array([100.0, 101.0]),
        np.array([100.5, 101.5]),
    )

    fill = resolve_tick_execution_by_deadline(
        book,
        order_time,
        deadline,
        np.random.default_rng(seed),
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ns=60_000_000_000,
    )

    assert fill is None


def test_scheduled_close_rejects_first_quote_after_wait_bound():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    order_time = deadline - pd.Timedelta(minutes=1)
    seed = 1729
    latency_ns = int(np.random.default_rng(seed).integers(0, 30_000_000))
    arrival = order_time + pd.Timedelta(latency_ns, unit="ns")
    late_tick = arrival + pd.Timedelta(seconds=1)
    book = TickBook(
        np.array([late_tick.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    assert (
        resolve_tick_execution_by_deadline(
            book,
            order_time,
            deadline,
            np.random.default_rng(seed),
            max_latency_ns_exclusive=30_000_000,
            max_quote_wait_ns=500_000_000,
        )
        is None
    )


def test_scheduled_close_rejects_last_quote_outside_age_limit():
    deadline = pd.Timestamp("2025-01-06 20:00:00", tz="UTC")
    order_time = deadline - pd.Timedelta(minutes=1)
    stale_tick = order_time - pd.Timedelta(seconds=1)
    book = TickBook(
        np.array([stale_tick.value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    assert (
        resolve_tick_execution_by_deadline(
            book,
            order_time,
            deadline,
            np.random.default_rng(1729),
            max_latency_ns_exclusive=30_000_000,
            max_quote_wait_ns=60_000_000_000,
        )
        is None
    )


def test_tick_execution_coverage_requires_quote_for_all_allowed_latencies():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    order_times = pd.date_range(start, periods=2, freq="1min")
    book = TickBook(
        np.array([(order_times[0] + pd.Timedelta(milliseconds=30)).value], dtype=np.int64),
        np.array([100.0]),
        np.array([100.5]),
    )

    report = tick_execution_coverage(
        book,
        order_times,
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ms=1_000,
    )

    assert report["order_count"] == 2
    assert report["covered_count"] == 1
    assert report["unavailable_count"] == 1
    assert report["coverage_fraction"] == 0.5
    assert report["first_unavailable_order_time"] == order_times[1].isoformat()


def test_tick_execution_coverage_accepts_quote_at_latest_arrival_edge():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    order_time = pd.DatetimeIndex([start])
    book = TickBook(
        np.array([(start + pd.Timedelta(nanoseconds=29_999_999)).value]),
        np.array([100.0]),
        np.array([100.5]),
    )

    report = tick_execution_coverage(
        book,
        order_time,
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ms=30,
    )

    assert report["coverage_fraction"] == 1.0
    assert report["unavailable_count"] == 0


def test_randomized_tick_exit_records_trigger_arrival_and_executable_quote():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    trigger = start + pd.Timedelta(seconds=5)
    tick_times = pd.to_datetime(
        [trigger, trigger + pd.Timedelta(milliseconds=10), trigger + pd.Timedelta(milliseconds=40)],
        utc=True,
    )
    book = TickBook(
        tick_times.values.astype("int64"),
        np.array([94.0, 95.0, 96.0]),
        np.array([94.5, 95.5, 96.5]),
    )

    result = resolve_tick_exit(
        book,
        start,
        direction=1,
        sl=95.0,
        tp=None,
        rng=np.random.default_rng(1729),
    )

    assert result is not None
    assert result.kind == "sl"
    assert result.trigger_time == trigger
    assert 0 <= result.quote.latency_ns < 30_000_000
    assert result.quote.tick_time >= result.quote.arrival_time
    assert result.quote.bid in (94.0, 95.0, 96.0)


def test_tick_exit_respects_multi_minute_bar_window():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    trigger = start + pd.Timedelta(minutes=3)
    book = TickBook(
        np.array([trigger.value], dtype=np.int64),
        np.array([94.0]),
        np.array([94.5]),
    )

    result = resolve_tick_exit(
        book,
        start,
        direction=1,
        sl=95.0,
        tp=None,
        rng=np.random.default_rng(7),
        bar_minutes=5,
        max_latency_ns_exclusive=0,
    )

    assert result is not None
    assert result.kind == "sl"
    assert result.trigger_time == trigger
    assert result.quote.tick_time == trigger
    assert result.quote.bid == 94.0


def test_randomized_tick_exit_fails_closed_when_trigger_has_no_later_quote():
    start = pd.Timestamp("2025-01-06 16:30:00", tz="UTC")
    book = TickBook(
        np.array([(start + pd.Timedelta(seconds=5)).value], dtype=np.int64),
        np.array([94.0]),
        np.array([94.5]),
    )

    with pytest.raises(TickExecutionUnavailable, match="no timely quote"):
        resolve_tick_exit(
            book,
            start,
            direction=1,
            sl=95.0,
            tp=None,
            rng=np.random.default_rng(1729),
        )


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


def test_build_tick_books_can_load_only_requested_symbols(tmp_path: Path) -> None:
    primary, secondary = "US100.cash", "US500.cash"
    primary_path = tmp_path / f"{primary}.csv"
    secondary_path = tmp_path / f"{secondary}.csv"
    _write_tick_csv(primary_path, [("2025.01.06", "16:30:00.000", 100.0, 100.6)])
    _write_tick_csv(secondary_path, [("2025.01.06", "16:30:00.000", 200.0, 200.6)])
    cache_dir = tmp_path / "cache"
    config = OmegaConf.create(
        {
            "costs": {"use_tick_execution": True},
            "data": {
                "raw_dir": str(tmp_path),
                "cache_dir": str(cache_dir),
                "tick_files": {primary: primary_path.name, secondary: secondary_path.name},
                "symbols": [primary, secondary],
                "tz": "Etc/GMT-3",
            },
        }
    )

    books = build_tick_books(config, symbols=[primary])

    assert set(books) == {primary}
    assert books[primary] is not None
    assert (cache_dir / f"{primary}_ticks_{TICK_CACHE_VERSION}.parquet").is_file()
    assert not (cache_dir / f"{secondary}_ticks_{TICK_CACHE_VERSION}.parquet").exists()


def test_build_tick_books_rejects_unconfigured_symbol(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "costs": {"use_tick_execution": True},
            "data": {
                "raw_dir": str(tmp_path),
                "cache_dir": str(tmp_path / "cache"),
                "tick_files": {},
                "symbols": ["US100.cash"],
                "tz": "Etc/GMT-3",
            },
        }
    )

    with pytest.raises(ValueError, match="not configured"):
        build_tick_books(config, symbols=["US500.cash"])


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
