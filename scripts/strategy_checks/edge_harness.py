"""Leakage-safe conditional forward-return measurement.

Scores any boolean condition on M1 NY bars against matched controls, so every
result is measured *in excess of market drift* rather than in absolute points.
US100 rose 41% across this sample, so an unconditional long prints a large
positive number that means nothing.

Causality rules enforced here:

* context columns come from the same broker date's Asia/London sessions and the
  previous calendar day -- all strictly before the NY open;
* forward returns never cross a session boundary (the ``bars_left`` guard);
* controls are drawn from the same minute-of-NY-session and the same direction,
  so time-of-day and drift cancel out.

Uncertainty is a cluster bootstrap over NY dates. Intraday observations are
autocorrelated, so resampling individual bars would understate the interval.

Diagnostic only: changes no training code and launches no run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "cache" / "US100.cash_M1_v2-full-day.parquet"
OUT = REPO / "outputs"
NY_TZ = "America/New_York"
HORIZONS = (5, 15, 30, 60, 120)
SEED = 20261003
N_BOOT = 2000

# The US100 cash session, stated in New York wall-clock time.
#
# The broker file is stamped ``Etc/GMT-3``, a FIXED UTC+3 offset with no DST, so
# the ``session == "ny"`` label (16:30-23:00 broker) does not track the exchange
# clock year-round. Measured against ``America/New_York`` the same label covers
# 09:30-16:00 during EDT but 08:30-15:00 during EST, so winter sessions carried
# an hour of pre-open bars and lost the last hour of the session. Deriving the
# window from the ET clock makes it DST-correct by construction.
NY_OPEN_ET = 9 * 60 + 30
NY_CLOSE_ET = 16 * 60


def load_bars() -> pd.DataFrame:
    """M1 OHLCV with a broker-tz session label."""
    from quant_rl.data.session import add_session_labels

    bars = pd.read_parquet(BARS)
    if "session" not in bars.columns:
        bars = add_session_labels(bars)
    return bars


def build_ny_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """NY-session M1 bars carrying only strictly pre-NY context.

    Session membership is decided by the ``America/New_York`` wall clock, not by
    the broker ``session`` label, so it stays 09:30-16:00 across the DST change.
    """
    index = pd.DatetimeIndex(bars.index)
    date = pd.Series(index.date, index=index)

    prev_high = bars["high"].groupby(date).max().shift(1)
    prev_low = bars["low"].groupby(date).min().shift(1)

    def _sess_hl(name: str) -> tuple[pd.Series, pd.Series]:
        mask = bars["session"].eq(name)
        key = date[mask]
        return (
            bars.loc[mask, "high"].groupby(key).max(),
            bars.loc[mask, "low"].groupby(key).min(),
        )

    asia_high, asia_low = _sess_hl("asia")
    london_high, london_low = _sess_hl("london")

    local_all = index.tz_convert(NY_TZ)
    minutes_all = (local_all.hour * 60 + local_all.minute).to_numpy()
    in_cash = (minutes_all >= NY_OPEN_ET) & (minutes_all < NY_CLOSE_ET)

    ny = bars.loc[in_cash].copy()
    # The ET session day, not the broker day: they agree inside this window only
    # because the broker offset is positive and fixed, so assert rather than assume.
    ny["date"] = pd.Series(pd.DatetimeIndex(ny.index).tz_convert(NY_TZ).date, index=ny.index)
    ny["broker_date"] = pd.Series(pd.DatetimeIndex(ny.index).date, index=ny.index)
    mismatched = ny["date"].to_numpy() != ny["broker_date"].to_numpy()
    if mismatched.any():
        raise AssertionError(
            f"{int(mismatched.sum())} cash-session bars straddle an ET date boundary; "
            "the broker date cannot be used as the session id"
        )
    ny = ny.loc[ny["date"].isin(asia_high.index)].sort_index()

    for name, series in (
        ("asia_high", asia_high),
        ("asia_low", asia_low),
        ("london_high", london_high),
        ("london_low", london_low),
        ("prev_day_high", prev_high),
        ("prev_day_low", prev_low),
    ):
        ny[name] = ny["date"].map(series)

    local = pd.DatetimeIndex(ny.index).tz_convert(NY_TZ)
    ny["ny_min"] = local.hour * 60 + local.minute
    ny["year"] = local.year
    ny["ny_open"] = ny.groupby("date")["open"].transform("first")
    ny["bars_left"] = ny.groupby("date").cumcount(ascending=False)
    for stem in ("asia", "london", "prev_day"):
        ny[f"{stem}_mid"] = (ny[f"{stem}_high"] + ny[f"{stem}_low"]) / 2.0
    return ny


def _fwd(
    ny: pd.DataFrame, horizon: int, direction: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Direction-aware net forward points, valid only inside one session."""
    close = ny["close"].to_numpy(dtype=float)
    spread = ny["spread"].to_numpy(dtype=float) / 100.0
    valid = ny["bars_left"].to_numpy() >= horizon + 1
    moved = np.roll(close, -horizon) - close
    return np.where(valid, direction * moved - 2.0 * spread, np.nan), valid


def _pools(ny: pd.DataFrame, valid: np.ndarray[Any, Any]) -> dict[int, np.ndarray[Any, Any]]:
    """Non-signal bar indices grouped by minute-of-NY, the control pool."""
    minutes = ny["ny_min"].to_numpy(dtype=np.int64)
    pools: dict[int, list[int]] = {}
    for i in np.flatnonzero(valid):
        pools.setdefault(int(minutes[i]), []).append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}


def _tail_stats(per_day: np.ndarray[Any, Any]) -> dict[str, float]:
    """Robustness checks that a fat-tail sample cannot fake.

    A mean over ~300 sessions is hostage to a handful of extreme days: the
    2025-04-07 crash rebound alone supplied 14% of one candidate's total P&L.
    A real edge survives dropping the best days, so these are reported next to
    the mean and must agree with it before anything is believed.
    """
    ordered = np.sort(per_day)[::-1]
    gross = float(per_day[per_day > 0].sum())
    top1 = float(ordered[0]) if len(ordered) else float("nan")
    top5 = float(ordered[:5].sum())
    # Share of GROSS PROFIT, not of net. Dividing by the net total explodes
    # when gains and losses nearly cancel, reading as a red flag rather than
    # the concentration it actually measures.
    top5_share = (100.0 * top5 / gross) if gross > 0 else float("nan")
    # Mean with the single most profitable session removed: the harshest
    # leave-one-out sanity check.
    without_best = float(np.delete(per_day, int(np.argmax(per_day))).mean())
    lo, hi = np.percentile(per_day, [10, 90])
    trimmed = per_day[(per_day >= lo) & (per_day <= hi)]
    return {
        "trimmed": float(trimmed.mean()) if trimmed.size else float("nan"),
        "median_day": float(np.median(per_day)),
        "win_days": float((per_day > 0).mean()),
        "top1_day": top1,
        "top5_share": top5_share,
        "drop_best": without_best,
    }


def _daily_means(
    diff: np.ndarray[Any, Any], sel_dates: np.ndarray[Any, Any], *, min_count: int
) -> np.ndarray[Any, Any]:
    """Per-session means, discarding sessions with too few signals.

    Without ``min_count`` a session holding a single signal votes as heavily as
    one holding thirty. One signal on 2026-01-29 contributed +398 pts and pulled
    the day-weighted mean to +24.8 while the bar-weighted mean was only +3.6.
    A candidate measured on one observation per session is not measurable.
    """
    finite = np.isfinite(diff)
    if not finite.any():
        return np.zeros(0)
    dates_f, diff_f = sel_dates[finite], diff[finite]
    keys, counts = np.unique(dates_f, return_counts=True)
    keep = counts >= min_count
    if not keep.any():
        return np.zeros(0)
    sums = np.asarray([diff_f[dates_f == k].sum() for k in keys[keep]])
    return np.asarray(sums / counts[keep], dtype=float)


def _cluster_boot(
    dates: np.ndarray[Any, Any],
    diffs: np.ndarray[Any, Any],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile CI for the mean of ``diffs``, resampling whole NY dates."""
    good = np.isfinite(diffs)
    if not good.any():
        return (float("nan"), float("nan"))
    uniq = np.unique(dates[good])
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    per_day = np.asarray([diffs[good & (dates == d)].mean() for d in uniq])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    lo, hi = np.percentile(per_day[draw].mean(axis=1), [2.5, 97.5])
    return (float(lo), float(hi))


def sweep_flags(ny: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Asia-range sweep and reclaim: the core ICT liquidity premise.

    Long: price trades below the Asia low and closes back above it. Short is
    the mirror. Both use only the current bar plus levels fixed before the open.
    """
    valid = ny["asia_low"].notna() & ny["asia_high"].notna()
    long_side = valid & (ny["low"] < ny["asia_low"]) & (ny["close"] > ny["asia_low"])
    short_side = valid & (ny["high"] > ny["asia_high"]) & (ny["close"] < ny["asia_high"])
    return long_side, short_side


def displacement_flags(
    ny: pd.DataFrame, *, lookback: int = 20, atr_window: int = 60
) -> pd.DataFrame:
    """Impulsive expansion away from a swept Asia level: displacement.

    This is the ICT leg the sweep test could not isolate. A sweep-and-reclaim
    with no expansion behind it is a failed take; the tradable event should be
    a genuine impulse. All inputs are the current bar plus strictly past bars.
    """
    close = ny["close"]
    high = ny["high"]
    low = ny["low"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(atr_window, min_periods=20).mean()

    # Impulse measured over the last few bars versus the recent quiet range.
    up_move = close - close.rolling(lookback, min_periods=5).min()
    dn_move = close.rolling(lookback, min_periods=5).max() - close
    unit = atr.where(atr > 0)
    out = pd.DataFrame(index=ny.index)
    out["up_atr"] = up_move / unit
    out["dn_atr"] = dn_move / unit

    # Expansion must be happening NOW, not already finished. Without this the
    # signal stays true for many bars after the move, which is not a trigger.
    recent = close.diff(3)
    out["recent_up_atr"] = recent / unit
    out["recent_dn_atr"] = -recent / unit

    valid = ny["asia_low"].notna() & ny["asia_high"].notna()
    # Swept the Asia level, reclaimed it, and is expanding now by >= 1 ATR
    # measured over the last three bars.
    out["disp_long"] = (
        valid
        & (low <= ny["asia_low"])
        & (close > ny["asia_low"])
        & (out["up_atr"] >= 1.0)
        & (out["recent_up_atr"] >= 1.0)
    )
    out["disp_short"] = (
        valid
        & (high >= ny["asia_high"])
        & (close < ny["asia_high"])
        & (out["dn_atr"] >= 1.0)
        & (out["recent_dn_atr"] >= 1.0)
    )
    return out


def swing_levels(ny: pd.DataFrame, *, lookback: int = 120) -> pd.DataFrame:
    """Buy-side/sell-side liquidity: the prior swing extreme, causal.

    The target is the last swing high before the setup. Pre-NY Asia/London
    highs count as liquidity too, so the rolling extreme is combined with the
    completed session highs.
    """
    high = ny["high"]
    low = ny["low"]
    out = pd.DataFrame(index=ny.index)
    # Strictly past bars only: a setup bar cannot be its own target.
    out["prior_high"] = high.rolling(lookback, min_periods=20).max().shift(1)
    out["prior_low"] = low.rolling(lookback, min_periods=20).min().shift(1)
    out["buy_side"] = pd.concat(
        [out["prior_high"], ny["asia_high"], ny["london_high"]], axis=1
    ).max(axis=1)
    out["sell_side"] = pd.concat([out["prior_low"], ny["asia_low"], ny["london_low"]], axis=1).min(
        axis=1
    )
    return out


def htf_swing_levels(bars: pd.DataFrame, ny: pd.DataFrame) -> pd.DataFrame:
    """Prior-trading-session swing highs/lows: the levels ICT liquidity really means.

    ``sweep_flags`` sweeps the *intraday* Asia range, which is not what the
    liquidity premise describes. Buy-side and sell-side liquidity are pools at
    swing extremes left by previous sessions, so the level a sweep must take is
    the prior D1 swing -- not a 2-hour rolling extreme.

    Fractal swings are detected on D1 bars and shifted one full day, so every
    level used by a session was already printed before that session opened. No
    M1 bar of the current day can contribute.
    """
    index = pd.DatetimeIndex(bars.index)
    broker_date = pd.Series(index.date, index=index)

    daily = bars.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    high = daily["high"].to_numpy()
    low = daily["low"].to_numpy()

    def _fractal_extreme(
        src: np.ndarray[Any, Any], *, order: int, mode: str
    ) -> np.ndarray[Any, Any]:
        """Confirmed swing highs (mode='high') / lows, using `order` bars each side."""
        n = len(src)
        out = np.full(n, np.nan)
        for i in range(order, n - order):
            window = src[i - order : i + order + 1]
            if mode == "high" and src[i] == window.max() and (window == src[i]).sum() == 1:
                out[i] = src[i]
            elif mode == "low" and src[i] == window.min() and (window == src[i]).sum() == 1:
                out[i] = src[i]
        return out

    order = 2
    swing_high = _fractal_extreme(high, order=order, mode="high")
    swing_low = _fractal_extreme(low, order=order, mode="low")

    # A fractal at day X is only *confirmed* `order` days later, when the bars that
    # must be lower/higher have printed. Publishing it unlagged would be lookahead,
    # so the swing columns carry their own confirmation lag. prior_day_* need no
    # lag beyond the single session shift applied below, since a D1 bar is final at
    # its close.
    prior = pd.DataFrame(
        {
            "htf_swing_high": pd.Series(swing_high, index=daily.index).shift(order),
            "htf_swing_low": pd.Series(swing_low, index=daily.index).shift(order),
            "prior_day_high": daily["high"],
            "prior_day_low": daily["low"],
        },
        index=daily.index,
    )

    out = pd.DataFrame(index=ny.index)
    # Each NY session must be scored against the D1 bar BEFORE its own day, so a
    # swing that today's price printed cannot be used as "prior" liquidity.
    # `prior` is indexed by D1 Timestamp while the session keys are dates, so the
    # lookup index is normalised to dates before mapping.
    first_bar = broker_date.groupby(broker_date).first()
    prev_day = first_bar.groupby(first_bar).first().shift(1)
    prior_by_date = pd.DataFrame(index=pd.Index(pd.DatetimeIndex(prior.index).date))
    prior_by_date.index.name = "d1_date"
    prior_by_date["htf_swing_high"] = prior["htf_swing_high"].to_numpy()
    prior_by_date["htf_swing_low"] = prior["htf_swing_low"].to_numpy()
    prior_by_date["prior_day_high"] = prior["prior_day_high"].to_numpy()
    prior_by_date["prior_day_low"] = prior["prior_day_low"].to_numpy()

    ny_prev_day = ny["broker_date"].map(prev_day)
    for col in ("htf_swing_high", "htf_swing_low", "prior_day_high", "prior_day_low"):
        out[col] = ny_prev_day.map(prior_by_date[col])

    # Liquidity = the nearest untraded extreme above/below. Falls back to the
    # prior-day extreme when no fractal printed recently, so the column is never
    # silently empty and quietly drops setups.
    out["buy_side_htf"] = out["htf_swing_high"].combine_first(out["prior_day_high"])
    out["sell_side_htf"] = out["htf_swing_low"].combine_first(out["prior_day_low"])
    # Buy-side liquidity must sit above price to be a target; otherwise the
    # "prior swing high" is already broken and the premise does not apply.
    out["buy_side_htf"] = out["buy_side_htf"].where(out["buy_side_htf"] > ny["close"])
    out["sell_side_htf"] = out["sell_side_htf"].where(out["sell_side_htf"] < ny["close"])
    return out


def audit_htf_causality(bars: pd.DataFrame, ny: pd.DataFrame, lv: pd.DataFrame) -> dict[str, int]:
    """Count sessions whose HTF levels could not have existed at the open.

    Two independent failure modes are counted separately, because they have
    different fixes:

    ``prior_day``
        the level does not equal the D1 extreme of the day before the session;
    ``fractal``
        the level exceeds every high printed up to and including that day, i.e.
        it leaked from a future swing.

    Both should be zero.
    """
    daily = bars.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    daily_by_date = daily.copy()
    daily_by_date.index = pd.Index(pd.DatetimeIndex(daily.index).date)

    # One row per session, taken from the session's first bar.
    per_session = lv.groupby(ny["broker_date"]).first()
    days = pd.Index(per_session.index)
    # The D1 bar immediately preceding each session, by position in the sorted
    # union of bar dates and D1 labels.
    all_days = pd.Index(sorted(set(days) | set(daily_by_date.index)))
    pos = all_days.get_indexer(days)
    prev_pos = pos - 1
    valid = prev_pos >= 0
    prev_days = pd.Series(
        [all_days[p] if v else None for p, v in zip(prev_pos, valid)],
        index=days,
    )

    # Positional access keeps this a plain ndarray walk; `.loc` on a grouped frame
    # with a tuple key is both slow here and opaque to the type checker.
    level_col = per_session["prior_day_high"].to_numpy(dtype=float)
    fractal_col = per_session["htf_swing_high"].to_numpy(dtype=float)

    prior_bad = 0
    fract_bad = 0
    high_by_day = daily["high"]
    daily_tz = pd.DatetimeIndex(daily.index).tz
    ceiling_cache: dict[object, float] = {}
    for i in range(len(days)):
        prev = prev_days.iloc[i]
        if prev is None or prev not in daily_by_date.index:
            continue
        reference = daily_by_date.loc[prev]
        level = level_col[i]
        if not np.isnan(level) and not np.isclose(level, float(reference["high"])):
            prior_bad += 1
        if prev not in ceiling_cache:
            bound = pd.Timestamp(prev)
            if daily_tz is not None:
                bound = bound.tz_localize(daily_tz)
            ceiling_cache[prev] = float(high_by_day.loc[:bound].max())
        fractal = fractal_col[i]
        if not np.isnan(fractal) and fractal > ceiling_cache[prev] + 1e-9:
            fract_bad += 1

    return {"sessions": int(len(per_session)), "prior_day": prior_bad, "fractal": fract_bad}


MANIP_WINDOW_END_ET = 10 * 60  # 10:00 ET, i.e. 30 minutes after the cash open


def session_phase(ny: pd.DataFrame) -> pd.DataFrame:
    """Label the ICT intraday phases of the NY cash session.

    The premise is sequential, and the earlier tests ignored that. Price opens at
    09:30 ET, manipulates the pools sitting just inside the overnight range during
    roughly the first half hour, and only then distributes towards the side that
    was swept. A setup that fires at 14:00 ET has no manipulation left to fade and
    no distribution phase to join, yet the unrestricted test kept trading it.

    Phases, in ET minutes from the open:

    ``manipulation``  09:30-10:00  pools above/below the open are taken
    ``distribution``  10:00-13:00  the directional move the premise describes
    ``late``          13:00-16:00  outside the described sequence
    """
    minutes = ny["ny_min"].to_numpy(dtype=np.int64)
    elapsed = minutes - NY_OPEN_ET
    phase = np.where(elapsed < 0, "pre_open", "")
    phase = np.where((elapsed >= 0) & (elapsed < 30), "manipulation", phase)
    phase = np.where((elapsed >= 30) & (elapsed < 210), "distribution", phase)
    phase = np.where(elapsed >= 210, "late", phase)
    return pd.DataFrame({"elapsed_min": elapsed, "phase": phase}, index=ny.index)


def liquidity_pools(bars: pd.DataFrame, ny: pd.DataFrame) -> pd.DataFrame:
    """Every level a session may sweep, not just the Asia range.

    Liquidity pools on this instrument are not one level. They are the extremes
    left by each prior session and by the HTF swings: the Asia and London ranges,
    the previous day's high and low, and the confirmed D1 swing highs/lows. A
    sweep-and-reclaim premise has to be tested against all of them, because
    "take liquidity" is a statement about a pool, not about Asia in particular.

    Every pool here is fixed before the 09:30 ET open. Nothing in this frame is
    derived from a bar of the session being traded, so it cannot leak.

    Returns one column per pool holding the level, plus ``pool_names`` listing
    which pools are populated for that bar so a chart can label them.
    """
    index = pd.DatetimeIndex(bars.index)
    broker_date = pd.Series(index.date, index=index)

    daily = bars.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    high_d = daily["high"].to_numpy(dtype=float)
    low_d = daily["low"].to_numpy(dtype=float)

    def _fractal(src: np.ndarray[Any, Any], *, order: int, mode: str) -> np.ndarray[Any, Any]:
        out = np.full(len(src), np.nan)
        for i in range(order, len(src) - order):
            win = src[i - order : i + order + 1]
            if mode == "high" and src[i] == win.max() and (win == src[i]).sum() == 1:
                out[i] = src[i]
            elif mode == "low" and src[i] == win.min() and (win == src[i]).sum() == 1:
                out[i] = src[i]
        return out

    order = 2
    d1 = pd.DataFrame(
        {
            "swing_high": pd.Series(
                _fractal(high_d, order=order, mode="high"), index=daily.index
            ).shift(order),
            "swing_low": pd.Series(
                _fractal(low_d, order=order, mode="low"), index=daily.index
            ).shift(order),
            "d_high": daily["high"],
            "d_low": daily["low"],
        }
    )

    first_bar = broker_date.groupby(broker_date).first()
    prev_day = first_bar.groupby(first_bar).first().shift(1)
    by_date = pd.DataFrame(index=pd.Index(pd.DatetimeIndex(d1.index).date))
    for col in d1.columns:
        by_date[col] = d1[col].to_numpy()
    ny_prev = ny["broker_date"].map(prev_day)

    out = pd.DataFrame(index=ny.index)
    for col in ("swing_high", "swing_low", "d_high", "d_low"):
        out[col] = ny_prev.map(by_date[col])

    # Session ranges come straight from build_ny_frame's strictly pre-open context.
    out["asia_high"] = ny["asia_high"]
    out["asia_low"] = ny["asia_low"]
    out["london_high"] = ny["london_high"]
    out["london_low"] = ny["london_low"]
    out["prev_day_high"] = ny["prev_day_high"]
    out["prev_day_low"] = ny["prev_day_low"]

    # Buy-side pools are highs; sell-side pools are lows. Both are referenced to
    # the session open, never to the current bar: masking against the same bar's
    # close would make the sweep test self-contradictory, since sweeping a high
    # needs close > level while the mask would demand level > close.
    reference = ny["ny_open"]
    highs = ["asia_high", "london_high", "prev_day_high", "swing_high", "d_high"]
    lows = ["asia_low", "london_low", "prev_day_low", "swing_low", "d_low"]
    for col in highs:
        out[f"bs_{col}"] = out[col].where(out[col] > reference)
    for col in lows:
        out[f"ss_{col}"] = out[col].where(out[col] < reference)

    out["pool_names"] = [
        "|".join(c for c in [*highs, *lows] if pd.notna(row[c]))
        for _, row in out[highs + lows].iterrows()
    ]
    # Nearest pool above / below the session open: the levels actually in play.
    # Computed against the open rather than the live close, so it does not drift
    # bar to bar and cannot mask a pool that price is currently sweeping.
    bs_cols = [f"bs_{c}" for c in highs]
    ss_cols = [f"ss_{c}" for c in lows]
    out["nearest_bs"] = out[bs_cols].min(axis=1)
    out["nearest_ss"] = out[ss_cols].max(axis=1)
    return out


def sweep_pools(ny: pd.DataFrame, pools: pd.DataFrame) -> dict[str, pd.Series]:
    """Sweep-and-reclaim flags against every pool, per side.

    A sweep pierces a pool and closes back inside it. Each pool is tested
    independently so a chart and the summary can attribute a move to the level
    that actually mattered, instead of attributing everything to Asia.
    """
    flags: dict[str, pd.Series] = {}
    for col in pools.columns:
        if not (col.startswith("bs_") or col.startswith("ss_")):
            continue
        level = pools[col]
        name = col
        if col.startswith("bs_"):
            # Buy-side pool taken from below, then reclaimed below.
            flags[f"long_{name}"] = level.notna() & (ny["low"] <= level) & (ny["close"] > level)
        else:
            flags[f"short_{name}"] = level.notna() & (ny["high"] >= level) & (ny["close"] < level)
    return flags


def fvg_zones(ny: pd.DataFrame) -> pd.DataFrame:
    """Three-bar fair value gaps and their inversion state.

    A bearish FVG at bar ``k`` is ``low[k-2] > high[k]``: price left a gap
    between ``high[k]`` (bottom) and ``low[k-2]`` (top). When price later
    closes back above that top, the gap inverts into a bullish zone whose
    edges are unchanged -- the ICT inverse FVG. Both are reported so the
    test can compare trading the original gap against trading its inversion.
    """
    high = ny["high"].to_numpy(dtype=float)
    low = ny["low"].to_numpy(dtype=float)
    close = ny["close"].to_numpy(dtype=float)

    bear_top = np.full(len(ny), np.nan)  # low[k-2]: upper edge of the gap
    bear_bot = np.full(len(ny), np.nan)  # high[k]: lower edge of the gap
    bear_top[2:] = low[:-2]
    bear_bot[2:] = high[2:]
    is_bear = bear_top > bear_bot

    bull_top = np.full(len(ny), np.nan)
    bull_bot = np.full(len(ny), np.nan)
    bull_top[2:] = high[:-2]
    bull_bot[2:] = low[2:]
    is_bull = bull_top < bull_bot

    # Inversion: a later close beyond a gap's far edge flips its polarity.
    # A rolling max of prior gap tops keeps this O(n) and fully causal.
    frame_top = pd.Series(bear_top).where(is_bear).ffill()
    rolling_top = frame_top.rolling(240, min_periods=1).max().shift(1)
    inverted_bear = (
        pd.Series(~is_bear) & rolling_top.notna() & (pd.Series(close) > rolling_top)
    ).to_numpy(dtype=bool)

    out = pd.DataFrame(
        {
            "fvg_bear_top": np.where(is_bear, bear_top, np.nan),
            "fvg_bear_bot": np.where(is_bear, bear_bot, np.nan),
            "fvg_bull_top": np.where(is_bull, bull_top, np.nan),
            "fvg_bull_bot": np.where(is_bull, bull_bot, np.nan),
            "ifvg_bull_top": np.where(inverted_bear, bear_top, np.nan),
            "ifvg_bull_bot": np.where(inverted_bear, bear_bot, np.nan),
        },
        index=ny.index,
    )
    return out


def atr_series(ny: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average true range for stop buffers; past bars only."""
    prev_close = ny["close"].shift(1)
    tr = pd.concat(
        [ny["high"] - ny["low"], (ny["high"] - prev_close).abs(), (ny["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=20).mean()


def evaluate(
    ny: pd.DataFrame,
    mask: pd.Series,
    direction: pd.Series,
    label: str,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    by_year: bool = True,
    min_count_per_day: int = 5,
) -> pd.DataFrame:
    """Condition versus matched control across horizons; returns a tidy frame."""
    mask_a = mask.reindex(ny.index).to_numpy(dtype=bool)
    if isinstance(direction, pd.Series):
        dir_a = direction.reindex(ny.index).to_numpy(dtype=np.int64)
    else:
        dir_a = np.full(len(ny), int(direction), dtype=np.int64)
    # A NaN direction would silently become a huge int64. Fail loudly instead.
    if not np.isin(dir_a, (-1, 0, 1)).all():
        raise ValueError("direction must be +/-1 aligned to the NY index")
    dates = ny["date"].to_numpy()
    close = ny["close"].to_numpy(dtype=float)
    spread = ny["spread"].to_numpy(dtype=float) / 100.0
    minutes = ny["ny_min"].to_numpy(dtype=np.int64)
    years = ny["year"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []

    for horizon in horizons:
        net, valid = _fwd(ny, horizon, dir_a)
        pools = _pools(ny, valid & ~mask_a)
        take = np.flatnonzero(mask_a & valid)
        if len(take) == 0:
            rows.append({"label": label, "group": "all", "horizon": horizon, "n": 0})
            continue

        picks = np.array(
            [
                int(rng.choice(pools[int(minutes[i])]))
                for i in take
                if int(minutes[i]) in pools and len(pools[int(minutes[i])])
            ],
            dtype=np.int64,
        )
        take = take[: len(picks)]
        ctrl = dir_a[take] * (close[picks + horizon] - close[picks]) - 2.0 * spread[picks]

        obs_dates, obs_net, obs_years = dates[take], net[take], years[take]
        groups = [("all", np.ones(len(take), dtype=bool))]
        if by_year:
            groups += [(str(y), obs_years == y) for y in sorted(set(obs_years.tolist()))]

        for name, sel in groups:
            diff = obs_net[sel] - ctrl[sel]
            sel_dates = obs_dates[sel]
            per_day = _daily_means(diff, sel_dates, min_count=min_count_per_day)
            if len(per_day) < 2:
                continue
            mean = float(per_day.mean())
            n = int(per_day.size)
            sd = float(per_day.std(ddof=1))
            lo, hi = _cluster_boot(obs_dates[sel], diff)
            tail = _tail_stats(per_day)
            rows.append(
                {
                    "label": label,
                    "group": name,
                    "horizon": horizon,
                    "n_days_used": n,
                    "net": float(np.nanmean(obs_net[sel])),
                    "ctrl": float(np.nanmean(ctrl[sel])),
                    "diff": mean,
                    "lo": lo,
                    "hi": hi,
                    "tstat": mean / (sd / np.sqrt(n)) if sd > 0 else float("nan"),
                    **tail,
                }
            )
    return pd.DataFrame(rows)


def cash_hold(ny: pd.DataFrame) -> pd.DataFrame:
    """Net points from holding long all session: the drift benchmark to beat."""
    grouped = ny.groupby("date")
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "close": grouped["close"].last(),
            "spread": grouped["spread"].max() / 100.0,
        }
    )
    out["net"] = out["close"] - out["open"] - 2.0 * out["spread"]
    out["year"] = out.index.map(lambda d: d.year)
    return out.reset_index(names="date")


COLS = [
    "label",
    "group",
    "horizon",
    "n_days_used",
    "net",
    "ctrl",
    "diff",
    "lo",
    "hi",
    "tstat",
    "trimmed",
    "median_day",
    "win_days",
    "top1_day",
    "top5_share",
    "drop_best",
]


def show(frame: pd.DataFrame, *, cols: list[str] | None = None) -> None:
    """Print evaluate() output; a CI spanning zero is visibly not an edge."""
    have = [c for c in (cols or COLS) if c in frame.columns]
    with pd.option_context("display.width", 240, "display.max_columns", 60):
        print(frame[have].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


# A candidate must clear all of these. They are deliberately strict: the
# thresholds are set above what pure noise produces, not at the edge.
GATE = {
    "min_t": 2.0,
    "min_win_days": 0.55,
    "max_top5_share": 35.0,
    "min_drop_best": 0.0,
}


def verdict(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag which rows clear every robustness gate. Explain each failure."""
    df = frame.copy()
    checks = {
        "t_ok": df["tstat"].abs() >= GATE["min_t"],
        "days_ok": (100.0 * df["win_days"]) >= GATE["min_win_days"] * 100.0,
        "tail_ok": df["top5_share"].abs() <= GATE["max_top5_share"],
        "robust_ok": df["drop_best"] > GATE["min_drop_best"],
        "trim_ok": np.sign(df["trimmed"]) == np.sign(df["diff"]),
    }
    for name, ok in checks.items():
        df[name] = ok
    df["passes"] = np.logical_and.reduce(list(checks.values()))
    df["why"] = [
        ", ".join(
            name.removesuffix("_ok") for name, ok in zip(checks, row, strict=True) if not bool(ok)
        )
        for row in zip(*(checks[k] for k in checks), strict=True)
    ]
    return df
