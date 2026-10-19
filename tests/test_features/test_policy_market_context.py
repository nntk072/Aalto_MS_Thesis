from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.data.ticks import TickBook
from quant_rl.features.policy_market_context import build_policy_market_context


def _book(index: pd.DatetimeIndex, seconds: list[int]) -> TickBook:
    stamps = np.asarray(
        [index[0].value + second * 1_000_000_000 for second in seconds], dtype=np.int64
    )
    return TickBook(
        stamps,
        np.asarray([100.0 + i for i in range(len(stamps))]),
        np.asarray([100.2 + i for i in range(len(stamps))]),
    )


def test_short_schedule_is_known_at_first_bar_and_quotes_stop_at_bar_close(tmp_path) -> None:
    index = pd.date_range("2025-11-28 16:30", periods=2, freq="min", tz="Etc/GMT-3")
    bars = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
    features = pd.DataFrame({"atr_5": [2.0, 2.0]}, index=index)
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(
        "session_date,scheduled_ny_close_broker_time,scheduled_short\n2025-11-28,20:15,true\n",
        encoding="utf-8",
    )
    book = _book(index, [5, 30, 60, 90])

    context = build_policy_market_context(
        bars,
        features,
        book,
        schedule_path=schedule,
    )

    first = context.iloc[0]
    assert first["ny_scheduled_short"] == 1.0
    assert np.isclose(first["ny_scheduled_close_fraction"], 1215 / 1440)
    assert np.isclose(first["ny_minutes_to_scheduled_close"], 1.0)
    assert first["tick_quote_available"] == 1.0
    assert first["tick_quote_activity"] == np.log1p(2)
    # The quote stamped exactly at 16:31 belongs to the next bar.
    assert np.isclose(first["tick_bid_offset_atr"], 0.5)
    assert np.isclose(first["tick_ask_offset_atr"], 0.6)


def test_future_ticks_do_not_change_completed_bar_context(tmp_path) -> None:
    index = pd.date_range("2025-11-28 16:30", periods=2, freq="min", tz="Etc/GMT-3")
    bars = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
    features = pd.DataFrame({"atr_5": [2.0, 2.0]}, index=index)
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(
        "session_date,scheduled_ny_close_broker_time,scheduled_short\n",
        encoding="utf-8",
    )
    full = build_policy_market_context(
        bars,
        features,
        _book(index, [5, 30, 60, 90]),
        schedule_path=schedule,
    )
    prefix = build_policy_market_context(
        bars.iloc[:1],
        features.iloc[:1],
        _book(index[:1], [5, 30, 60, 90]),
        schedule_path=schedule,
    )

    pd.testing.assert_series_equal(full.iloc[0], prefix.iloc[0])
