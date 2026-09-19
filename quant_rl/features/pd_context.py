"""Session-conditional premium/discount (PD) context features.

Routing (broker-tz sessions via :func:`quant_rl.data.session.get_session`):

- **ny**: completed Asia + completed London ranges active; prev-day/week summaries
- **london**: completed Asia active; London self-range is live/running; prev-day/week
- **asia**: prev-day + prev-week summaries only (same-day Europe masked to zero)

All session extremes are causal: a session's completed H/L is only visible after
that session has finished. Raw absolute levels are emitted for env/diagnostics
but should be stripped from the model ``seq`` via ``MTF_RAW_SUFFIXES``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.session import add_session_labels

# Model-facing column stems (ATR distances / flags). Raw level stems are listed
# in build.MTF_RAW_SUFFIXES so TradingEnv drops them from ``seq``.
PD_CONTEXT_FEATURE_COLUMNS = (
    "active_session_asia",
    "active_session_london",
    "active_session_ny",
    "ctx_asia_active",
    "ctx_asia_high_dist_atr",
    "ctx_asia_low_dist_atr",
    "ctx_london_active",
    "ctx_london_high_dist_atr",
    "ctx_london_low_dist_atr",
    "ctx_pd_mid_dist_atr",
    "ctx_in_premium",
    "ctx_in_discount",
    "ctx_prev_day_high_dist_atr",
    "ctx_prev_day_low_dist_atr",
    "ctx_prev_day_in_premium",
    "ctx_prev_day_in_discount",
    "ctx_prev_week_high_dist_atr",
    "ctx_prev_week_low_dist_atr",
    "ctx_prev_week_in_premium",
    "ctx_prev_week_in_discount",
)

PD_CONTEXT_RAW_COLUMNS = (
    "ctx_asia_high",
    "ctx_asia_low",
    "ctx_london_high",
    "ctx_london_low",
    "ctx_prev_day_high",
    "ctx_prev_day_low",
    "ctx_prev_week_high",
    "ctx_prev_week_low",
)


def _atr_dist(level: pd.Series, close: pd.Series, atr: pd.Series, *, side: str) -> pd.Series:
    """ATR-normalised distance; ``side='high'`` → (level-close)/atr, else (close-level)/atr."""
    safe = atr.where(atr > 0)
    if side == "high":
        return (level - close) / safe
    return (close - level) / safe


def _prior_calendar_hl(
    bars: pd.DataFrame,
    *,
    freq: str,
) -> pd.DataFrame:
    """Previous completed calendar day/week H/L in the index timezone (causal)."""
    idx = pd.DatetimeIndex(bars.index)
    if freq == "D":
        keys = pd.Series(idx.date, index=idx)
        prefix = "ctx_prev_day"
    elif freq == "W":
        iso = idx.isocalendar()
        keys = pd.Series([f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)], index=idx)
        prefix = "ctx_prev_week"
    else:
        raise ValueError(f"unsupported freq {freq!r}")

    period_high = bars["high"].groupby(keys.to_numpy()).max().sort_index()
    period_low = bars["low"].groupby(keys.to_numpy()).min().sort_index()
    return pd.DataFrame(
        {
            f"{prefix}_high": keys.map(period_high.shift(1)),
            f"{prefix}_low": keys.map(period_low.shift(1)),
        },
        index=idx,
    )


def _session_completed_and_running(
    bars: pd.DataFrame,
    session: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Completed Asia H/L + completed/live London H/L per bar (causal).

    Asia completed extremes for calendar date D become visible from the first
    London (or later) bar of D. London completed extremes become visible from
    the first NY bar of D. During London, London H/L are the running session
    extremes so far.
    """
    idx = pd.DatetimeIndex(bars.index)
    dates = pd.Series(idx.date, index=idx)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    sess = session.to_numpy()
    date_arr = dates.to_numpy()

    n = len(bars)
    asia_c_hi = np.full(n, np.nan)
    asia_c_lo = np.full(n, np.nan)
    lon_hi = np.full(n, np.nan)
    lon_lo = np.full(n, np.nan)

    # Per-date accumulators for the forming session extremes.
    day_asia_hi: dict[object, float] = {}
    day_asia_lo: dict[object, float] = {}
    day_asia_done: dict[object, bool] = {}
    day_lon_hi: dict[object, float] = {}
    day_lon_lo: dict[object, float] = {}
    day_lon_done: dict[object, bool] = {}

    for i in range(n):
        d = date_arr[i]
        s = sess[i]

        if s == "asia":
            h, l = high[i], low[i]
            if d not in day_asia_hi:
                day_asia_hi[d] = h
                day_asia_lo[d] = l
            else:
                day_asia_hi[d] = max(day_asia_hi[d], h)
                day_asia_lo[d] = min(day_asia_lo[d], l)
            # Same-day Asia context is masked by the caller; leave NaN here.
        elif s == "london":
            # Asia is finished once London starts.
            if d in day_asia_hi and not day_asia_done.get(d, False):
                day_asia_done[d] = True
            if day_asia_done.get(d, False) or d in day_asia_hi:
                day_asia_done[d] = True
                asia_c_hi[i] = day_asia_hi[d]
                asia_c_lo[i] = day_asia_lo[d]

            h, l = high[i], low[i]
            if d not in day_lon_hi:
                day_lon_hi[d] = h
                day_lon_lo[d] = l
            else:
                day_lon_hi[d] = max(day_lon_hi[d], h)
                day_lon_lo[d] = min(day_lon_lo[d], l)
            # Live/running London range during London.
            lon_hi[i] = day_lon_hi[d]
            lon_lo[i] = day_lon_lo[d]
        elif s == "ny":
            if d in day_asia_hi:
                day_asia_done[d] = True
                asia_c_hi[i] = day_asia_hi[d]
                asia_c_lo[i] = day_asia_lo[d]
            if d in day_lon_hi and not day_lon_done.get(d, False):
                day_lon_done[d] = True
            if d in day_lon_hi:
                day_lon_done[d] = True
                lon_hi[i] = day_lon_hi[d]
                lon_lo[i] = day_lon_lo[d]
        else:
            # closed / overnight: hold last completed levels if already known
            if day_asia_done.get(d, False):
                asia_c_hi[i] = day_asia_hi[d]
                asia_c_lo[i] = day_asia_lo[d]
            if day_lon_done.get(d, False):
                lon_hi[i] = day_lon_hi[d]
                lon_lo[i] = day_lon_lo[d]

    return (
        pd.Series(asia_c_hi, index=idx, dtype=float),
        pd.Series(asia_c_lo, index=idx, dtype=float),
        pd.Series(lon_hi, index=idx, dtype=float),
        pd.Series(lon_lo, index=idx, dtype=float),
    )


def build_pd_context_features(
    bars: pd.DataFrame,
    atr: pd.Series,
    *,
    tz: str = "Etc/GMT-3",
    ny_end: str = "23:00",
) -> pd.DataFrame:
    """Build fixed-width session-routed PD context columns.

    Parameters
    ----------
    bars:
        M1 (or any) OHLC frame with a tz-aware DatetimeIndex.
    atr:
        ATR series aligned to ``bars`` (zeros/NaN → distance features NaN).
    tz:
        Broker timezone for session labels (must match ``get_session``).
    ny_end:
        Inclusive NY end HH:MM forwarded to session labelling.
    """
    labeled = add_session_labels(bars[["open", "high", "low", "close"]].copy(), tz=tz, ny_end=ny_end)
    session = labeled["session"]
    close = bars["close"]
    atr_s = atr.reindex(bars.index)

    asia_hi, asia_lo, lon_hi, lon_lo = _session_completed_and_running(bars, session)

    is_asia = session == "asia"
    is_london = session == "london"
    is_ny = session == "ny"

    # Routing masks: which session ranges may appear in the observation.
    asia_active = is_london | is_ny
    london_active = is_london | is_ny

    ctx_asia_high = asia_hi.where(asia_active)
    ctx_asia_low = asia_lo.where(asia_active)
    ctx_london_high = lon_hi.where(london_active)
    ctx_london_low = lon_lo.where(london_active)

    # Active composite PD range for premium/discount flags.
    # NY: union of Asia+London; London: Asia (completed) ∪ live London; Asia: prev day.
    prev_day = _prior_calendar_hl(bars, freq="D")
    prev_week = _prior_calendar_hl(bars, freq="W")

    comp_hi = pd.Series(np.nan, index=bars.index, dtype=float)
    comp_lo = pd.Series(np.nan, index=bars.index, dtype=float)

    # London: max/min of completed Asia and running London (when present).
    lon_comp_hi = pd.concat([ctx_asia_high, ctx_london_high], axis=1).max(axis=1)
    lon_comp_lo = pd.concat([ctx_asia_low, ctx_london_low], axis=1).min(axis=1)
    comp_hi = comp_hi.where(~is_london, lon_comp_hi)
    comp_lo = comp_lo.where(~is_london, lon_comp_lo)

    ny_comp_hi = pd.concat([ctx_asia_high, ctx_london_high], axis=1).max(axis=1)
    ny_comp_lo = pd.concat([ctx_asia_low, ctx_london_low], axis=1).min(axis=1)
    comp_hi = comp_hi.where(~is_ny, ny_comp_hi)
    comp_lo = comp_lo.where(~is_ny, ny_comp_lo)

    comp_hi = comp_hi.where(~is_asia, prev_day["ctx_prev_day_high"])
    comp_lo = comp_lo.where(~is_asia, prev_day["ctx_prev_day_low"])

    mid = (comp_hi + comp_lo) / 2.0
    in_prem = (close > mid) & mid.notna()
    in_disc = (close < mid) & mid.notna()

    out = pd.DataFrame(index=bars.index)
    out["active_session_asia"] = is_asia.astype(float)
    out["active_session_london"] = is_london.astype(float)
    out["active_session_ny"] = is_ny.astype(float)

    out["ctx_asia_active"] = asia_active.astype(float)
    out["ctx_asia_high"] = ctx_asia_high
    out["ctx_asia_low"] = ctx_asia_low
    out["ctx_asia_high_dist_atr"] = _atr_dist(ctx_asia_high, close, atr_s, side="high").fillna(0.0)
    out["ctx_asia_low_dist_atr"] = _atr_dist(ctx_asia_low, close, atr_s, side="low").fillna(0.0)
    # Zero-mask inactive sessions (distances already NaN→0; force zeros when masked).
    out.loc[~asia_active, ["ctx_asia_high_dist_atr", "ctx_asia_low_dist_atr"]] = 0.0
    out.loc[~asia_active, ["ctx_asia_high", "ctx_asia_low"]] = 0.0

    out["ctx_london_active"] = london_active.astype(float)
    out["ctx_london_high"] = ctx_london_high
    out["ctx_london_low"] = ctx_london_low
    out["ctx_london_high_dist_atr"] = _atr_dist(ctx_london_high, close, atr_s, side="high").fillna(
        0.0
    )
    out["ctx_london_low_dist_atr"] = _atr_dist(ctx_london_low, close, atr_s, side="low").fillna(0.0)
    out.loc[~london_active, ["ctx_london_high_dist_atr", "ctx_london_low_dist_atr"]] = 0.0
    out.loc[~london_active, ["ctx_london_high", "ctx_london_low"]] = 0.0

    out["ctx_pd_mid_dist_atr"] = ((close - mid) / atr_s.where(atr_s > 0)).fillna(0.0)
    out["ctx_in_premium"] = in_prem.astype(float)
    out["ctx_in_discount"] = in_disc.astype(float)

    for block, prefix in ((prev_day, "ctx_prev_day"), (prev_week, "ctx_prev_week")):
        hi = block[f"{prefix}_high"]
        lo = block[f"{prefix}_low"]
        out[f"{prefix}_high"] = hi
        out[f"{prefix}_low"] = lo
        out[f"{prefix}_high_dist_atr"] = _atr_dist(hi, close, atr_s, side="high")
        out[f"{prefix}_low_dist_atr"] = _atr_dist(lo, close, atr_s, side="low")
        p_mid = (hi + lo) / 2.0
        out[f"{prefix}_in_premium"] = ((close > p_mid) & p_mid.notna()).astype(float)
        out[f"{prefix}_in_discount"] = ((close < p_mid) & p_mid.notna()).astype(float)

    return out


def build_htf_pd_distance_features(
    bars: pd.DataFrame,
    atr_m1: pd.Series,
    *,
    swing_period: int = 5,
    atr_mult: float = 0.0,
) -> pd.DataFrame:
    """HTF swing + session-range PD distances on a single timeframe's bars.

    Returns columns without a TF prefix; callers align via ``align_timeframes``.
    Distances use the HTF bar's close and the caller's M1 ATR after alignment
    is not available here — we normalise by an ATR computed on ``bars`` so the
    values are scale-free before causal ffill onto M1.
    """
    from .indicators import atr as atr_fn
    from .structure import detect_session_levels, structure_levels

    lv = structure_levels(bars, swing_period=swing_period, atr_mult=atr_mult)
    sess = detect_session_levels(bars)
    atr_tf = atr_fn(bars, period=5).where(lambda s: s > 0)
    close = bars["close"]

    out = pd.DataFrame(index=bars.index)
    out["pd_swing_high_dist_atr"] = (lv["last_swing_high"] - close) / atr_tf
    out["pd_swing_low_dist_atr"] = (close - lv["last_swing_low"]) / atr_tf

    # Session-range mid premium/discount on this TF (always-on levels, not routed).
    asia_mid = (sess["asian_high"] + sess["asian_low"]) / 2.0
    lon_mid = (sess["london_high"] + sess["london_low"]) / 2.0
    out["pd_asia_mid_dist_atr"] = (close - asia_mid) / atr_tf
    out["pd_london_mid_dist_atr"] = (close - lon_mid) / atr_tf
    out["pd_asia_in_premium"] = ((close > asia_mid) & asia_mid.notna()).astype(float)
    out["pd_asia_in_discount"] = ((close < asia_mid) & asia_mid.notna()).astype(float)
    out["pd_london_in_premium"] = ((close > lon_mid) & lon_mid.notna()).astype(float)
    out["pd_london_in_discount"] = ((close < lon_mid) & lon_mid.notna()).astype(float)

    # Unused here but keeps signature compatible with callers that pass M1 ATR.
    _ = atr_m1
    return out
