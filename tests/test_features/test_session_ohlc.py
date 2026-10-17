"""Tests for CT-anchored session OHLC levels, prior-period high/low,
range quadrants and raw VWAP level (quant_rl.features.session_ohlc).

The session windows are defined in America/Chicago wall-clock time. Native
fixed-offset tests preserve legacy behavior; the FTMO server-clock regression
uses the documented GMT+2/GMT+3 US-DST model to check the production mapping.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from quant_rl.features.indicators import vwap_from_session, vwap_level
from quant_rl.features.session_ohlc import (
    prev_ny_high_low,
    previous_labeled_session_ohlc,
    prior_period_high_low,
    range_quadrants,
    session_mask,
    session_ohlc,
)

DATA_TZ = "Etc/GMT-3"
SESSION_TZ = "America/Chicago"
NY_SESSION_START = "08:30"
NY_SESSION_END = "15:00"


def _sawtooth_bars(
    start: str, periods: int, freq: str = "5min", amplitude: int = 20
) -> pd.DataFrame:
    """Deterministic sawtooth bars in the fixed-offset data tz."""
    idx = pd.date_range(start, periods=periods, freq=freq, tz=DATA_TZ)
    close = 100.0 + (np.arange(periods) % amplitude) * 0.1
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# session_mask
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# session_ohlc
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_ohlc_basic() -> None:
    """Live running O/H/L/C inside the Asian window, held after it closes."""
    df = _sawtooth_bars("2026-01-06 00:00", 3 * 288)
    ohlc = session_ohlc(df, "19:00", "23:00", session_tz=SESSION_TZ, prefix="asian_ct")

    assert list(ohlc.columns) == [
        "asian_ct_open",
        "asian_ct_high",
        "asian_ct_low",
        "asian_ct_close",
    ]
    assert ohlc.index.equals(df.index)

    idx = pd.DatetimeIndex(df.index, tz=DATA_TZ)
    # January: CT = broker - 9h, so 19:00-23:00 CT = 04:00-08:00 broker.
    sess = (idx.hour >= 4) & (idx.hour < 8)

    # NaN before the first session ever starts.
    pre_first = idx < pd.Timestamp("2026-01-06 04:00", tz=DATA_TZ)
    assert ohlc.loc[pre_first].isna().all().all()

    # Running high/low inside the session (causal cummax/cummin), constant open.
    day1 = (idx.date == pd.Timestamp("2026-01-06").date()) & sess
    run_high = df["high"].where(day1).groupby(idx.date).cummax().where(day1)
    run_low = df["low"].where(day1).groupby(idx.date).cummin().where(day1)
    first_open = df["open"].where(day1).groupby(idx.date).transform("first").where(day1)

    pd.testing.assert_series_equal(
        ohlc.loc[day1, "asian_ct_high"], run_high.loc[day1], check_names=False
    )
    pd.testing.assert_series_equal(
        ohlc.loc[day1, "asian_ct_low"], run_low.loc[day1], check_names=False
    )
    pd.testing.assert_series_equal(
        ohlc.loc[day1, "asian_ct_open"], first_open.loc[day1], check_names=False
    )

    # Held after the session ends: final session O/H/L/C persists.
    after = (idx >= pd.Timestamp("2026-01-06 08:00", tz=DATA_TZ)) & (
        idx < pd.Timestamp("2026-01-06 16:00", tz=DATA_TZ)
    )
    sess_day1 = df.loc[day1]
    assert (ohlc.loc[after, "asian_ct_open"] == sess_day1["open"].iloc[0]).all()
    assert (ohlc.loc[after, "asian_ct_high"] == sess_day1["high"].max()).all()
    assert (ohlc.loc[after, "asian_ct_low"] == sess_day1["low"].min()).all()
    assert (ohlc.loc[after, "asian_ct_close"] == sess_day1["close"].iloc[-1]).all()

    # Next session resets: first bar of day 2's window starts fresh.
    day2_first = pd.Timestamp("2026-01-07 04:00", tz=DATA_TZ)
    assert ohlc.loc[day2_first, "asian_ct_high"] == pytest.approx(df.loc[day2_first, "high"])
    assert ohlc.loc[day2_first, "asian_ct_open"] == pytest.approx(df.loc[day2_first, "open"])


# ---------------------------------------------------------------------------
# DST regression tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_ohlc_dst_spring_forward() -> None:
    """2026-03-08 spring forward: NY window stays 08:30-15:00 CT wall clock.

    The broker-tz clock times of the selected bars shift by one hour across
    the boundary (17:30 broker start on 03-07 vs 16:30 on 03-09) — a naive
    fixed-offset implementation selects the same broker times both days and
    fails this test.
    """
    idx = pd.date_range("2026-03-06 00:00", periods=5 * 1440, freq="1min", tz=DATA_TZ)
    mask = session_mask(idx, "08:30", "15:00", session_tz=SESSION_TZ)
    masked = idx[mask.to_numpy()]

    pre = masked[masked.date == pd.Timestamp("2026-03-07").date()]
    post = masked[masked.date == pd.Timestamp("2026-03-09").date()]
    # Same wall-clock CT window either side of the transition: 6.5h of bars.
    assert len(pre) == 390
    assert len(post) == 390
    assert {t.replace(tzinfo=None).time() for t in pre.tz_convert(SESSION_TZ)} == {
        t.replace(tzinfo=None).time() for t in post.tz_convert(SESSION_TZ)
    }
    # Broker-side start times differ by exactly the DST shift.
    assert pre[0].hour == 17 and pre[0].minute == 30  # CST: UTC-6 -> +9h broker
    assert post[0].hour == 16 and post[0].minute == 30  # CDT: UTC-5 -> +8h broker

    # Levels line up too: the first in-session bar carries its own open.
    df = _sawtooth_bars("2026-03-06 00:00", 5 * 1440, freq="1min")
    ohlc = session_ohlc(df, "08:30", "15:00", session_tz=SESSION_TZ, prefix="ny_ct")
    first_of_day = mask.to_numpy() & ~np.concatenate(([False], mask.to_numpy()[:-1]))
    pd.testing.assert_series_equal(
        ohlc.loc[first_of_day, "ny_ct_open"],
        df.loc[first_of_day, "open"],
        check_names=False,
    )


@pytest.mark.unit
def test_session_ohlc_dst_fall_back() -> None:
    """2026-11-01 fall back: repeated 01:00-02:00 CT hour handled correctly."""
    idx = pd.date_range("2026-10-30 00:00", periods=5 * 1440, freq="1min", tz=DATA_TZ)
    mask = session_mask(idx, "08:30", "15:00", session_tz=SESSION_TZ)
    masked = idx[mask.to_numpy()]

    pre = masked[masked.date == pd.Timestamp("2026-10-31").date()]  # CDT (UTC-5)
    post = masked[masked.date == pd.Timestamp("2026-11-02").date()]  # CST (UTC-6)
    assert len(pre) == 390
    assert len(post) == 390
    assert pre[0].hour == 16 and pre[0].minute == 30
    assert post[0].hour == 17 and post[0].minute == 30

    # The repeated local hour resolves via the underlying UTC instants: the
    # two 01:00-01:59 CT passes on 11-01 map to distinct broker-tz times.
    ct = idx.tz_convert(SESSION_TZ)
    transition_mask = np.array([ts.date() == pd.Timestamp("2026-11-01").date() for ts in ct])
    ct_times = pd.Series([t.time() for t in ct[transition_mask]])
    duplicated = ct_times.duplicated().sum()
    assert duplicated > 0  # the fold produces repeated wall-clock minutes


@pytest.mark.unit
def test_session_ohlc_causal() -> None:
    """Session levels at bar t must not change when future bars are appended."""
    df = _sawtooth_bars("2026-01-06 00:00", 3 * 288)
    full = session_ohlc(df, "19:00", "23:00", session_tz=SESSION_TZ, prefix="asian_ct")
    trunc = session_ohlc(df.iloc[:500], "19:00", "23:00", session_tz=SESSION_TZ, prefix="asian_ct")
    overlap = full.index[:500]
    pd.testing.assert_frame_equal(full.loc[overlap], trunc.loc[overlap])


# ---------------------------------------------------------------------------
# session_mask
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_mask_january_broker_times() -> None:
    """Asian window 19:00-23:00 CT = 04:00-08:00 broker time in January (CST)."""
    idx = pd.date_range("2026-01-06 00:00", periods=288, freq="5min", tz=DATA_TZ)
    mask = session_mask(idx, "19:00", "23:00", session_tz=SESSION_TZ)
    masked = idx[mask.to_numpy()]
    assert len(masked) == 48
    assert masked[0].hour == 4 and masked[0].minute == 0
    assert masked[-1].hour == 7 and masked[-1].minute == 55


@pytest.mark.unit
def test_ftmo_server_clock_maps_winter_and_summer_to_same_ct_session() -> None:
    """FTMO MT5 server wall time remains NY-open aligned across US DST."""
    idx = pd.DatetimeIndex(
        [
            "2025-01-15 16:29",
            "2025-01-15 16:30",
            "2025-01-15 16:31",
            "2025-01-15 16:32",
            "2025-07-15 16:29",
            "2025-07-15 16:30",
            "2025-07-15 16:31",
            "2025-07-15 16:32",
            "2025-11-03 16:29",
            "2025-11-03 16:30",
            "2025-11-03 16:31",
            "2025-11-03 16:32",
        ],
        tz=DATA_TZ,
    )

    native = session_mask(idx, "08:30", "08:32", SESSION_TZ)
    ftmo = session_mask(
        idx,
        "08:30",
        "08:32",
        SESSION_TZ,
        source_clock_model="ftmo_mt5_us_dst",
    )

    assert native.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert ftmo.tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]
    assert ftmo.index.equals(idx)


@pytest.mark.unit
def test_prev_ny_high_low_uses_only_completed_ny_sessions() -> None:
    """Future bars in an open NY session cannot alter its visible prior range."""
    idx = pd.date_range("2026-01-05 00:00", periods=4 * 1440, freq="1min", tz=DATA_TZ)
    ny_mask = session_mask(idx, NY_SESSION_START, NY_SESSION_END, session_tz=SESSION_TZ).to_numpy()
    local_dates = np.asarray(idx.tz_convert(SESSION_TZ).date)
    session_dates = sorted(set(local_dates[ny_mask]))
    first_date, second_date = session_dates[:2]
    first_rows = ny_mask & (local_dates == first_date)
    second_rows = ny_mask & (local_dates == second_date)
    bars = pd.DataFrame({"high": 100.0, "low": 95.0}, index=idx)
    bars.loc[first_rows, "high"] = 110.0
    bars.loc[first_rows, "low"] = 90.0
    bars.loc[second_rows, "high"] = 120.0
    bars.loc[second_rows, "low"] = 80.0

    poisoned = bars.copy()
    second_positions = np.flatnonzero(second_rows)
    poisoned.loc[idx[second_positions[1:]], "high"] = 999.0
    poisoned.loc[idx[second_positions[1:]], "low"] = 1.0

    original_features = prev_ny_high_low(bars, session_tz=SESSION_TZ)
    poisoned_features = prev_ny_high_low(poisoned, session_tz=SESSION_TZ)

    # The first session has no earlier NY range. Throughout session two, the
    # visible range must remain session one's completed 110/90 despite poisoning
    # every later bar in session two.
    assert original_features.loc[first_rows].isna().all().all()
    second_index = idx[second_positions]
    expected_previous = pd.DataFrame(
        {"prev_ny_high": 110.0, "prev_ny_low": 90.0},
        index=second_index,
    )
    pd.testing.assert_frame_equal(
        original_features.loc[second_index], expected_previous, check_freq=False
    )
    pd.testing.assert_frame_equal(
        poisoned_features.loc[second_index], expected_previous, check_freq=False
    )

    # The new range becomes available on the first decision bar at session
    # close, after all of its in-session bars have completed.
    close_time = pd.Timestamp(f"{second_date} {NY_SESSION_END}", tz=SESSION_TZ).tz_convert(DATA_TZ)
    assert original_features.loc[close_time].to_dict() == {
        "prev_ny_high": 120.0,
        "prev_ny_low": 80.0,
    }
    assert poisoned_features.loc[close_time].to_dict() == {
        "prev_ny_high": 999.0,
        "prev_ny_low": 1.0,
    }


@pytest.mark.unit
def test_prev_ny_high_low_promotes_observed_range_after_clock_close_with_data_gap() -> None:
    """A clock-completed session uses its observed bars; G1 audits gaps separately."""
    idx = pd.date_range("2026-01-05 00:00", periods=3 * 1440, freq="1min", tz=DATA_TZ)
    ny_mask = session_mask(idx, NY_SESSION_START, NY_SESSION_END, session_tz=SESSION_TZ).to_numpy()
    local_dates = np.asarray(idx.tz_convert(SESSION_TZ).date)
    session_dates = sorted(set(local_dates[ny_mask]))
    first_date, second_date = session_dates[:2]
    first_rows = ny_mask & (local_dates == first_date)

    bars = pd.DataFrame({"high": 100.0, "low": 95.0}, index=idx)
    bars.loc[first_rows, "high"] = 110.0
    bars.loc[first_rows, "low"] = 90.0
    missing = idx[np.flatnonzero(first_rows)[100]]
    bars = bars.drop(index=missing)

    result = prev_ny_high_low(bars, session_tz=SESSION_TZ)
    second_session_start = pd.Timestamp(
        f"{second_date} {NY_SESSION_START}", tz=SESSION_TZ
    ).tz_convert(DATA_TZ)
    assert result.loc[second_session_start].to_dict() == {
        "prev_ny_high": 110.0,
        "prev_ny_low": 90.0,
    }


@pytest.mark.unit
def test_prev_ny_high_low_uses_local_close_across_dst_transition() -> None:
    idx = pd.date_range("2026-03-06 00:00", periods=5 * 1440, freq="1min", tz=DATA_TZ)
    ny_mask = session_mask(idx, NY_SESSION_START, NY_SESSION_END, session_tz=SESSION_TZ).to_numpy()
    local_dates = np.asarray(idx.tz_convert(SESSION_TZ).date)
    bars = pd.DataFrame({"high": 100.0, "low": 90.0}, index=idx)
    for date_index, session_date in enumerate(sorted(set(local_dates[ny_mask]))):
        session_rows = ny_mask & (local_dates == session_date)
        bars.loc[session_rows, "high"] = 200.0 + date_index
        bars.loc[session_rows, "low"] = 80.0 - date_index

    result = prev_ny_high_low(bars, session_tz=SESSION_TZ)
    prior_close = pd.Timestamp(f"2026-03-08 {NY_SESSION_END}", tz=SESSION_TZ).tz_convert(DATA_TZ)
    next_session = ny_mask & (local_dates == pd.Timestamp("2026-03-09").date())
    prior_session = ny_mask & (local_dates == pd.Timestamp("2026-03-08").date())
    expected_high = bars.loc[prior_session, "high"].iloc[0]
    expected_low = bars.loc[prior_session, "low"].iloc[0]

    assert result.loc[prior_close].to_dict() == {
        "prev_ny_high": expected_high,
        "prev_ny_low": expected_low,
    }
    visible = result.loc[idx[next_session]]
    assert visible["prev_ny_high"].eq(expected_high).all()
    assert visible["prev_ny_low"].eq(expected_low).all()


# ---------------------------------------------------------------------------
# prior_period_high_low
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prior_period_high_low_yesterday() -> None:
    """yesterday_high/low on CT day N equal the full-day extremes of day N-1."""
    # Start at broker 09:00 = CT 00:00 (January: broker = CT + 9h) so CT days
    # align with fixture days.
    idx = pd.date_range("2026-01-06 09:00", periods=3 * 288, freq="5min", tz=DATA_TZ)
    close = 100.0 + (np.arange(len(idx)) % 40) * 0.1
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    out = prior_period_high_low(df, freq="D", tz=SESSION_TZ)

    assert list(out.columns) == ["yesterday_high", "yesterday_low"]

    ct_dates = pd.Series([ts.date() for ts in idx.tz_convert(SESSION_TZ)], index=idx)
    day_high = df["high"].groupby(ct_dates.to_numpy()).max().sort_index()
    day_low = df["low"].groupby(ct_dates.to_numpy()).min().sort_index()

    # NaN on the first CT day (no completed prior day yet).
    first_day = ct_dates == ct_dates.iloc[0]
    assert out.loc[first_day.to_numpy()].isna().all().all()

    # Every later bar sees the previous CT day's completed extremes.
    expected_high = ct_dates.map(day_high.shift(1))
    expected_low = ct_dates.map(day_low.shift(1))
    pd.testing.assert_series_equal(out["yesterday_high"], expected_high, check_names=False)
    pd.testing.assert_series_equal(out["yesterday_low"], expected_low, check_names=False)


@pytest.mark.unit
def test_prior_period_high_low_lastweek() -> None:
    """lastweek_high/low are constant within a week and reset at the ISO boundary."""
    # Two full ISO weeks starting Monday 2026-01-05 (broker 09:00 = CT 00:00).
    idx = pd.date_range("2026-01-05 09:00", periods=10 * 96, freq="15min", tz=DATA_TZ)
    close = 100.0 + (np.arange(len(idx)) % 80) * 0.1
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    out = prior_period_high_low(df, freq="W", tz=SESSION_TZ)

    assert list(out.columns) == ["lastweek_high", "lastweek_low"]

    ct = idx.tz_convert(SESSION_TZ)
    iso = ct.isocalendar()
    week_keys = pd.Series(
        [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)],
        index=idx,
    )
    # Week 1 has no prior week: all NaN.
    first_week = week_keys == week_keys.iloc[0]
    assert out.loc[first_week.to_numpy()].isna().all().all()

    # Week 2 bars all share week 1's completed extremes.
    week1 = df.loc[first_week.to_numpy()]
    week2_mask = ~first_week.to_numpy()
    assert (out.loc[week2_mask, "lastweek_high"] == week1["high"].max()).all()
    assert (out.loc[week2_mask, "lastweek_low"] == week1["low"].min()).all()


# ---------------------------------------------------------------------------
# range_quadrants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_range_quadrants() -> None:
    """Quarter-band boundaries nearest equilibrium for high=110 / low=100."""
    high = pd.Series([110.0, 210.0])
    low = pd.Series([100.0, 200.0])
    q = range_quadrants(high, low, prefix="yesterday")

    assert list(q.columns) == [
        "yesterday_q_upper_outer",
        "yesterday_q_upper_inner",
        "yesterday_q_lower_inner",
        "yesterday_q_lower_outer",
    ]
    assert q["yesterday_q_upper_outer"].tolist() == [107.5, 207.5]
    assert q["yesterday_q_upper_inner"].tolist() == [105.0, 205.0]
    assert q["yesterday_q_lower_inner"].tolist() == [105.0, 205.0]
    assert q["yesterday_q_lower_outer"].tolist() == [102.5, 202.5]


@pytest.mark.unit
def test_range_quadrants_nan_propagates() -> None:
    """NaN range (e.g. no prior week yet) stays NaN in every quadrant."""
    high = pd.Series([np.nan, 110.0])
    low = pd.Series([np.nan, 100.0])
    q = range_quadrants(high, low, prefix="lastweek")
    assert q.iloc[0].isna().all()
    assert q.iloc[1].notna().all()


# ---------------------------------------------------------------------------
# vwap_level
# ---------------------------------------------------------------------------


def _vwap_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 03:00", periods=2 * 288, freq="5min", tz=DATA_TZ)
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "tickvol": rng.integers(10, 200, len(idx)).astype(float),
        },
        index=idx,
    )
    df["session_id"] = pd.factorize(idx.normalize())[0]
    return df


@pytest.mark.unit
def test_vwap_level_matches_normalized_vwap() -> None:
    """(close - vwap_level) / vwap_level == vwap_from_session output."""
    df = _vwap_df()
    raw = vwap_level(df)
    normalized = vwap_from_session(df)
    expected = (df["close"] - raw) / raw.replace(0, np.nan)
    pd.testing.assert_series_equal(normalized, expected, check_names=False)


@pytest.mark.unit
def test_vwap_level_resets_per_session() -> None:
    """VWAP restarts from the first bar of each session_id group."""
    df = _vwap_df()
    raw = vwap_level(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    # First bar of day 2: cumsum restarts, so VWAP == that bar's typical price.
    second_day_pos = int((df["session_id"].to_numpy() != df["session_id"].iloc[0]).argmax())
    expected = typical.iloc[second_day_pos]
    assert raw.iloc[second_day_pos] == pytest.approx(expected)
    assert raw.iloc[:second_day_pos].notna().all()
    assert raw.iloc[second_day_pos:].notna().all()


@pytest.mark.unit
def test_vwap_level_requires_session_id() -> None:
    df = _vwap_df().drop(columns=["session_id"])
    with pytest.raises(ValueError, match="session_id"):
        vwap_level(df)


# ---------------------------------------------------------------------------
# build_features integration
# ---------------------------------------------------------------------------


def _sess_base_cfg() -> dict[str, Any]:
    return {
        "features": {
            "ema_periods": [9],
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "atr_period": 14,
            "adx_period": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "return_horizons": [1],
            "realized_vol_period": 20,
            "smt_swing_period": 5,
            "smt_corr_window": 20,
            "zscore_window": 100,
            "htf_timeframes": [],
            "include_session_ohlc": False,
            "sessions": {
                "tz": "America/Chicago",
                "asian": {"start": "19:00", "end": "23:00"},
                "london": {"start": "02:00", "end": "05:00"},
                "ny": {"start": "08:30", "end": "15:00"},
                "week_start": "MON",
                "quadrant_ranges": ["yesterday", "lastweek"],
            },
        }
    }


def _dst_week_m1_bars() -> pd.DataFrame:
    """Multi-day M1 bars spanning the 2026-03-08 US spring-forward transition."""
    idx = pd.date_range("2026-03-06 00:00", periods=5 * 288, freq="5min", tz=DATA_TZ)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    df = pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "tickvol": rng.integers(10, 200, len(idx)).astype(float),
            "volume": rng.integers(1000, 5000, len(idx)).astype(float),
        },
        index=idx,
    )
    df["session_id"] = pd.factorize(idx.normalize())[0]
    return df


@pytest.mark.integration
def test_build_features_session_ohlc_additive() -> None:
    """include_session_ohlc=true adds exactly the new level columns, stays numeric."""
    from quant_rl.features.build import build_features

    bars = _dst_week_m1_bars()
    cfg_base = OmegaConf.create(_sess_base_cfg())
    base = build_features(bars, cfg=cfg_base)

    cfg_sess = OmegaConf.merge(
        cfg_base,
        OmegaConf.create(
            {
                "features": {
                    "include_session_ohlc": True,
                    "sessions": {"ny": {"start": "09:00", "end": "14:00"}},
                }
            }
        ),
    )
    sess = build_features(bars, cfg=cast(DictConfig, cfg_sess))

    expected = {
        "asian_ct_open",
        "asian_ct_high",
        "asian_ct_low",
        "asian_ct_close",
        "london_ct_open",
        "london_ct_high",
        "london_ct_low",
        "london_ct_close",
        "ny_ct_open",
        "ny_ct_high",
        "ny_ct_low",
        "ny_ct_close",
        "yesterday_high",
        "yesterday_low",
        "prev_ny_high",
        "prev_ny_low",
        "lastweek_high",
        "lastweek_low",
        "yesterday_q_upper_outer",
        "yesterday_q_upper_inner",
        "yesterday_q_lower_inner",
        "yesterday_q_lower_outer",
        "lastweek_q_upper_outer",
        "lastweek_q_upper_inner",
        "lastweek_q_lower_inner",
        "lastweek_q_lower_outer",
    }
    assert expected <= set(map(str, sess.columns))
    assert set(map(str, base.columns)) < set(map(str, sess.columns))
    assert all(pd.api.types.is_numeric_dtype(sess[c]) for c in sess.columns)
    assert "vwap" not in sess.columns  # vwap_session off by default

    # Levels are finite from day 3 onward (prior day/week exist by then).
    day3 = sess.loc[sess.index >= pd.Timestamp("2026-03-09 00:00", tz=DATA_TZ)]
    for col in ("yesterday_high", "lastweek_high", "ny_ct_high"):
        assert day3[col].notna().any(), col

    # Prior-NY levels use the configured CT window, even when it differs from
    # the defaults. In March, 14:00 CT closes at 22:00 in the data's fixed UTC+3
    # clock; that range becomes visible at the close timestamp.
    ny_start = pd.Timestamp("2026-03-09 09:00", tz=SESSION_TZ).tz_convert(DATA_TZ)
    ny_close = pd.Timestamp("2026-03-09 14:00", tz=SESSION_TZ).tz_convert(DATA_TZ)
    ny_bars = bars.loc[(bars.index >= ny_start) & (bars.index < ny_close)]
    assert sess.loc[ny_close, "prev_ny_high"] == pytest.approx(ny_bars["high"].max())
    assert sess.loc[ny_close, "prev_ny_low"] == pytest.approx(ny_bars["low"].min())

    # Legacy broker-tz levels are untouched (additive block).
    assert {"asian_high", "asian_low", "london_high", "london_low"} <= set(map(str, sess.columns))


@pytest.mark.integration
def test_build_features_uses_ftmo_clock_model_for_ct_sessions() -> None:
    """The resolved config must carry FTMO clock correction into the CT block."""
    from quant_rl.features.build import build_features

    idx = pd.DatetimeIndex(
        [
            "2025-01-15 16:29",
            "2025-01-15 16:30",
            "2025-01-15 16:31",
            "2025-01-15 16:32",
            "2025-07-15 16:29",
            "2025-07-15 16:30",
            "2025-07-15 16:31",
            "2025-07-15 16:32",
            "2025-11-03 16:29",
            "2025-11-03 16:30",
            "2025-11-03 16:31",
            "2025-11-03 16:32",
        ],
        tz=DATA_TZ,
    )
    close = np.arange(100.0, 112.0)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tickvol": np.full(len(idx), 10.0),
            "volume": np.zeros(len(idx)),
        },
        index=idx,
    )
    cfg = OmegaConf.merge(
        OmegaConf.create(_sess_base_cfg()),
        OmegaConf.create(
            {
                "data": {"timestamp_clock": "ftmo_mt5_us_dst"},
                "features": {
                    "include_session_ohlc": True,
                    "include_previous_session_pd": True,
                    "sessions": {"ny": {"start": "08:30", "end": "08:32"}},
                },
            }
        ),
    )

    features = build_features(bars, cfg=cast(DictConfig, cfg))

    starts = idx[idx.time == pd.Timestamp("2000-01-01 16:30").time()]
    assert features.loc[starts, "ny_ct_open"].tolist() == [101.0, 105.0, 109.0]
    closes = idx[idx.time == pd.Timestamp("2000-01-01 16:32").time()]
    assert features.loc[closes, "previous_session_high"].tolist() == [103.0, 107.0, 111.0]


@pytest.mark.integration
def test_default_config_carries_sessions_block() -> None:
    """default.yaml ships the sessions block, flag off by default."""
    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    assert cfg.data.timestamp_clock == "ftmo_mt5_us_dst"
    assert cfg.features.include_session_ohlc is False
    assert cfg.features.sessions.tz == "America/Chicago"
    assert cfg.features.sessions.asian.start == "19:00"
    assert cfg.features.sessions.asian.end == "23:00"
    assert cfg.features.sessions.london.start == "02:00"
    assert cfg.features.sessions.london.end == "05:00"
    assert cfg.features.sessions.ny.start == "08:30"
    assert cfg.features.sessions.ny.end == "15:00"
    assert list(cfg.features.sessions.quadrant_ranges) == ["yesterday", "lastweek"]


@pytest.mark.unit
def test_previous_labeled_session_uses_latest_completed_session_and_dst_clock() -> None:
    """After the DST switch, the latest completed label is NY, then Asia."""
    idx = pd.date_range("2026-03-09 00:00", periods=2 * 1440, freq="1min", tz=DATA_TZ)
    values = np.arange(len(idx), dtype=float)
    bars = pd.DataFrame({"high": values + 2.0, "low": values - 2.0}, index=idx)
    windows = {
        "asian": {"start": "19:00", "end": "23:00"},
        "london": {"start": "02:00", "end": "05:00"},
        "ny": {"start": "08:30", "end": "15:00"},
    }
    result = previous_labeled_session_ohlc(bars, windows, session_tz=SESSION_TZ)

    ny_close = pd.Timestamp("2026-03-09 23:00", tz=DATA_TZ)
    ny_close_pos = int(np.flatnonzero(idx == ny_close)[0])
    ny_rows = (idx >= pd.Timestamp("2026-03-09 16:30", tz=DATA_TZ)) & (idx < ny_close)
    london_positions = np.flatnonzero(session_mask(idx, "02:00", "05:00", SESSION_TZ).to_numpy())
    assert result.loc[ny_rows, "previous_session_code"].eq(2).all()
    result_values = result.to_numpy(dtype=float)
    assert result_values[ny_close_pos, 3] == 3
    ny_session = session_mask(idx, "08:30", "15:00", SESSION_TZ).to_numpy()
    ny_session &= idx < ny_close
    assert result_values[ny_close_pos, 0] == bars["high"].loc[ny_session].max()

    # At the next configured session close, Asia is now the most recent label.
    asia_close = pd.Timestamp("2026-03-10 07:00", tz=DATA_TZ)
    asia_close_pos = int(np.flatnonzero(idx == asia_close)[0])
    assert result_values[asia_close_pos, 3] == 1
    previous_high = float(result_values[asia_close_pos, 0])
    previous_low = float(result_values[asia_close_pos, 1])
    previous_midpoint = float(result_values[asia_close_pos, 2])
    assert previous_midpoint == pytest.approx((previous_high + previous_low) / 2.0)
    assert london_positions.size > 0


@pytest.mark.unit
def test_previous_labeled_session_does_not_promote_truncated_tail() -> None:
    idx = pd.date_range("2026-03-09 00:00", "2026-03-09 22:58", freq="1min", tz=DATA_TZ)
    bars = pd.DataFrame({"high": 110.0, "low": 90.0}, index=idx)
    result = previous_labeled_session_ohlc(
        bars,
        {
            "asian": {"start": "19:00", "end": "23:00"},
            "london": {"start": "02:00", "end": "05:00"},
            "ny": {"start": "08:30", "end": "15:00"},
        },
        session_tz=SESSION_TZ,
    )
    # The current NY window is truncated before its last expected bar, so the
    # completed London session remains the reference.
    assert result.iloc[-1]["previous_session_code"] == 2
    assert result.iloc[-1]["previous_session_midpoint"] == pytest.approx(100.0)


@pytest.mark.unit
def test_previous_labeled_session_skips_session_with_missing_interior_bar() -> None:
    idx = pd.date_range("2026-03-09 00:00", "2026-03-10 00:00", freq="1min", tz=DATA_TZ)
    bars = pd.DataFrame({"high": 110.0, "low": 90.0}, index=idx)
    missing = pd.Timestamp("2026-03-09 18:00", tz=DATA_TZ)
    bars = bars.drop(index=missing)
    result = previous_labeled_session_ohlc(
        bars,
        {
            "asian": {"start": "19:00", "end": "23:00"},
            "london": {"start": "02:00", "end": "05:00"},
            "ny": {"start": "08:30", "end": "15:00"},
        },
        session_tz=SESSION_TZ,
    )
    # Despite its final bar being present, NY has a hole and is not a complete
    # reference. London remains the latest fully observed session.
    assert result.loc[pd.Timestamp("2026-03-09 23:00", tz=DATA_TZ), "previous_session_code"] == 2
