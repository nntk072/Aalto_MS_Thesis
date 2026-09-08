"""Tests for CT-anchored session OHLC levels, prior-period high/low,
range quadrants and raw VWAP level (quant_rl.features.session_ohlc).

The DST tests are the critical ones: session windows are defined in
America/Chicago wall-clock time while the data index is fixed Etc/GMT-3
(no DST rules), so a naive fixed-offset implementation shifts windows by
one hour for weeks around each US transition. See the implementation plan
(§0) for the failure mode these tests pin down.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from quant_rl.features.indicators import vwap_from_session, vwap_level
from quant_rl.features.session_ohlc import (
    prior_period_high_low,
    range_quadrants,
    session_mask,
    session_ohlc,
)

DATA_TZ = "Etc/GMT-3"
SESSION_TZ = "America/Chicago"


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
        cfg_base, OmegaConf.create({"features": {"include_session_ohlc": True}})
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
        "vwap",
    }
    assert expected <= set(map(str, sess.columns))
    assert set(map(str, base.columns)) < set(map(str, sess.columns))
    assert all(pd.api.types.is_numeric_dtype(sess[c]) for c in sess.columns)

    # Levels are finite from day 3 onward (prior day/week exist by then).
    day3 = sess.loc[sess.index >= pd.Timestamp("2026-03-09 00:00", tz=DATA_TZ)]
    for col in ("yesterday_high", "lastweek_high", "ny_ct_high", "vwap"):
        assert day3[col].notna().any(), col

    # Legacy broker-tz levels are untouched (additive block).
    assert {"asian_high", "asian_low", "london_high", "london_low"} <= set(map(str, sess.columns))


@pytest.mark.integration
def test_default_config_carries_sessions_block() -> None:
    """default.yaml ships the sessions block, flag off by default."""
    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    assert cfg.features.include_session_ohlc is False
    assert cfg.features.sessions.tz == "America/Chicago"
    assert cfg.features.sessions.asian.start == "19:00"
    assert cfg.features.sessions.asian.end == "23:00"
    assert cfg.features.sessions.london.start == "02:00"
    assert cfg.features.sessions.london.end == "05:00"
    assert cfg.features.sessions.ny.start == "08:30"
    assert cfg.features.sessions.ny.end == "15:00"
    assert list(cfg.features.sessions.quadrant_ranges) == ["yesterday", "lastweek"]
