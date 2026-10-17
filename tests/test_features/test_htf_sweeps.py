"""Causality and reference-selection tests for HTF liquidity sweeps."""

from __future__ import annotations

import pandas as pd
import pytest

import quant_rl.features.htf_sweeps as htf_sweeps


def _bars(index: pd.DatetimeIndex, *, lows: list[float] | None = None) -> pd.DataFrame:
    close = [101.0] * len(index)
    low = lows or [100.5] * len(index)
    high = [101.5] * len(index)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=index)


def _ref(
    tf: str,
    level: float,
    confirmed_at: pd.Timestamp,
    pivot_minute: int,
    side: str = "low",
) -> dict[str, object]:
    return {
        "side": side,
        "timeframe": tf,
        "level": level,
        "pivot_time": confirmed_at - pd.Timedelta(minutes=pivot_minute),
        "confirmed_at": confirmed_at,
    }


def test_unconfirmed_reference_is_not_available_and_audit_keeps_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=4, freq="min", tz="UTC")
    bars = _bars(idx, lows=[100.5, 100.5, 99.5, 100.5])
    refs = [
        _ref("M5", 100.0, idx[2], 5),
        _ref("M15", 100.2, idx[2], 15),
    ]
    monkeypatch.setattr(
        htf_sweeps,
        "_pivot_references",
        lambda _bars, tf: [ref for ref in refs if ref["timeframe"] == tf],
    )

    features, audit = htf_sweeps.detect_confirmed_htf_sweeps(
        bars, timeframes=("M5", "M15"), atr_period=1, return_audit=True
    )

    assert features["sweep_low"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert features["sweep_low_level"].iloc[2] == pytest.approx(100.0)
    assert features["sweep_low_ref_tf_code"].iloc[2] == 5
    assert features["sweep_low_candidate_count"].iloc[2] == 2
    candidates = audit.loc[audit["sweep_time"] == idx[2]]
    assert len(candidates) == 2
    assert candidates["selected"].sum() == 1
    assert candidates.loc[candidates["selected"], "reference_timeframe"].item() == "M5"


def test_confirmation_after_sweep_cannot_change_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=5, freq="min", tz="UTC")
    bars = _bars(idx, lows=[100.5, 99.5, 100.5, 100.5, 100.5])
    late_ref = _ref("M5", 100.0, idx[3], 5)
    monkeypatch.setattr(htf_sweeps, "_pivot_references", lambda _bars, _tf: [late_ref])

    full = htf_sweeps.detect_confirmed_htf_sweeps(bars, timeframes=("M5",), atr_period=1)
    prefix = htf_sweeps.detect_confirmed_htf_sweeps(bars.iloc[:3], timeframes=("M5",), atr_period=1)

    pd.testing.assert_frame_equal(full.iloc[:3], prefix)
    assert full["sweep_low"].sum() == 0


@pytest.mark.parametrize("timeframe", ["M5", "M10", "M15", "M30", "H1", "H4", "D1"])
@pytest.mark.parametrize("side", ["high", "low"])
def test_each_confirmed_timeframe_can_initiate_a_sweep(
    monkeypatch: pytest.MonkeyPatch, timeframe: str, side: str
) -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=4, freq="min", tz="UTC")
    bars = _bars(idx, lows=[100.5, 100.5, 99.5, 100.5])
    level = 100.0 if side == "low" else 101.0
    ref = _ref(timeframe, level, idx[2], 5, side=side)
    monkeypatch.setattr(htf_sweeps, "_pivot_references", lambda _bars, _tf: [ref])

    features = htf_sweeps.detect_confirmed_htf_sweeps(bars, timeframes=(timeframe,), atr_period=1)

    assert features[f"sweep_{side}"].iloc[2] == 1
    assert (
        features[f"sweep_{side}_ref_tf_code"].iloc[2]
        == {"M5": 5, "M10": 10, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[timeframe]
    )


def test_duplicate_or_unsupported_timeframes_are_rejected() -> None:
    bars = _bars(pd.date_range("2024-01-02", periods=2, freq="min", tz="UTC"))
    with pytest.raises(ValueError, match="must be unique"):
        htf_sweeps.detect_confirmed_htf_sweeps(bars, timeframes=("M5", "M5"))
    with pytest.raises(ValueError, match="Unsupported"):
        htf_sweeps.detect_confirmed_htf_sweeps(bars, timeframes=("M2",))


def test_actual_resampled_pivot_uses_close_time_and_ignores_partial_bar() -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=40, freq="min", tz="UTC")
    group_closes = [110.0, 109.0, 108.0, 100.0, 108.0, 109.0, 110.0, 111.0]
    close = [group_closes[i // 5] for i in range(len(idx))]
    bars = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
        },
        index=idx,
    )

    refs = htf_sweeps._pivot_references(bars, "M5")
    pivot_lows = [ref for ref in refs if ref["side"] == "low"]

    assert len(pivot_lows) == 1
    assert pivot_lows[0]["pivot_time"] == idx[15]
    assert pivot_lows[0]["confirmed_at"] == idx[35]


def test_actual_resampled_m10_pivot_uses_m10_close_time() -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=80, freq="min", tz="UTC")
    group_closes = [110.0, 109.0, 108.0, 100.0, 108.0, 109.0, 110.0, 111.0]
    close = [group_closes[i // 10] for i in range(len(idx))]
    bars = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
        },
        index=idx,
    )

    refs = htf_sweeps._pivot_references(bars, "M10")
    pivot_lows = [ref for ref in refs if ref["side"] == "low"]

    assert len(pivot_lows) == 1
    assert pivot_lows[0]["pivot_time"] == idx[30]
    assert pivot_lows[0]["confirmed_at"] == idx[70]


def test_real_m1_sweep_waits_until_resampled_m5_pivot_is_confirmed() -> None:
    idx = pd.date_range("2024-01-02 00:00", periods=40, freq="min", tz="UTC")
    group_closes = [110.0, 109.0, 108.0, 100.0, 108.0, 109.0, 110.0, 111.0]
    close = [group_closes[i // 5] for i in range(len(idx))]
    low = [value - 0.2 for value in close]
    low[25] = 99.0  # Crosses the eventual reference while its M5 confirmation is pending.
    low[35] = 99.0  # Crosses again after the confirming M5 candle has closed.
    bars = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": low,
            "close": close,
        },
        index=idx,
    )

    features = htf_sweeps.detect_confirmed_htf_sweeps(bars, timeframes=("M5",), atr_period=1)

    assert features["sweep_low"].iloc[:35].sum() == 0
    assert features["sweep_low"].iloc[35] == 1
    assert features["sweep_low_confirmed_at_ns"].iloc[35] == idx[35].value


def test_dst_boundary_resampling_preserves_timezone_aware_confirmation_times() -> None:
    idx = pd.date_range("2024-03-30 20:00", "2024-03-31 06:00", freq="min", tz="Europe/Helsinki")
    close = [100.0 + ((i // 5) % 8) for i in range(len(idx))]
    bars = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
        },
        index=idx,
    )

    refs = htf_sweeps._pivot_references(bars, "M5")

    assert refs
    assert all(pd.Timestamp(ref["confirmed_at"]).tz is not None for ref in refs)
    assert all(str(pd.Timestamp(ref["pivot_time"]).tz) == "Europe/Helsinki" for ref in refs)


def test_daily_pivot_confirmation_uses_next_local_midnight_across_dst() -> None:
    idx = pd.date_range(
        "2024-03-25 00:00",
        "2024-04-10 00:00",
        freq="min",
        tz="Europe/Helsinki",
        inclusive="left",
    )
    days = pd.DatetimeIndex(idx.normalize().unique())
    daily_close = [110, 109, 108, 107, 106, 100, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    by_day = dict(zip(days, daily_close, strict=True))
    close = [float(by_day[day]) for day in idx.normalize()]
    bars = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
        },
        index=idx,
    )

    lows = [ref for ref in htf_sweeps._pivot_references(bars, "D1") if ref["side"] == "low"]

    assert lows
    assert lows[0]["pivot_time"] == days[5]
    assert lows[0]["confirmed_at"] == days[11]
    assert lows[0]["confirmed_at"].utcoffset() != days[5].utcoffset()
