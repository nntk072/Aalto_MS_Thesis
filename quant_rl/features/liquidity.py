"""Causal liquidity sweep and break-of-structure (BOS) detection.

All detectors use only information available at decision time ``t``:
sweep/BOS events reference the most recent *confirmed* swing level as of the
previous bar (``structure_levels(...)`` output shifted by one), so adding
future bars never changes historical feature values (Agent.md §3, §19, §26).

Sweep semantics (Agent.md §5):
- ``sweep_high = 1``: current bar takes liquidity above a previously known
  swing-high level (fresh take: the previous bar did not already sit above
  the same level, and the overshoot is within ``max_sweep_distance_atr`` ×
  ATR — deeper breaks are treated as breakouts, not sweeps).
- ``sweep_high_reclaimed = 1``: after taking the level, price closes back
  below it (same bar or later); mirrored for the low side.
- ``sweep_high_level`` persists for the lifetime of the active sweep setup;
  ``sweep_high_age`` counts bars since the event (0 on the event bar, NaN
  when no sweep is active). A reclaim ends the setup.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from .indicators import atr
from .structure import structure_levels
from .swings import MAX_RETAINED_LEVELS


def _detect_side_sweeps(
    level: npt.NDArray[np.float64],
    attack: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    atr_arr: npt.NDArray[np.float64],
    *,
    side: int,
    max_sweep_distance_atr: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Run the causal sweep state machine for one side.

    ``level`` is the previously-known swing level (shifted, NaN before the
    first confirmation). ``side=1`` detects takes above a swing high (attack
    = high, reclaim = close back below the level); ``side=-1`` detects takes
    below a swing low (attack = low, reclaim = close back above the level).
    """
    n = len(level)
    event = np.zeros(n)
    reclaimed = np.zeros(n)
    level_out = np.full(n, np.nan)
    age_out = np.full(n, np.nan)

    active = False
    state_level = np.nan
    age = 0

    for t in range(n):
        lv = level[t]
        a = atr_arr[t]
        fires = False
        takes_liquidity = attack[t] > lv if side == 1 else attack[t] < lv
        if not np.isnan(lv) and not np.isnan(a) and takes_liquidity:
            prev_lv = level[t - 1] if t > 0 else np.nan
            prev_attack = attack[t - 1] if t > 0 else np.nan
            if side == 1:
                fresh = t == 0 or np.isnan(prev_lv) or prev_attack <= prev_lv or lv != prev_lv
                dist = attack[t] - lv
            else:
                fresh = t == 0 or np.isnan(prev_lv) or prev_attack >= prev_lv or lv != prev_lv
                dist = lv - attack[t]
            if fresh and dist <= max_sweep_distance_atr * a:
                fires = True
        if fires:
            active = True
            state_level = lv
            age = 0
        if active:
            event[t] = 1.0 if fires else 0.0
            level_out[t] = state_level
            age_out[t] = float(age)
            back_through = close[t] < state_level if side == 1 else close[t] > state_level
            if back_through:
                reclaimed[t] = 1.0
                active = False
        if active:
            age += 1
    return event, reclaimed, level_out, age_out


def detect_liquidity_sweeps(
    bars: pd.DataFrame,
    *,
    swing_period: int = 5,
    max_sweep_distance_atr: float = 2.0,
    atr_period: int = 5,
) -> pd.DataFrame:
    """Detect causal liquidity sweep events and associated levels.

    Parameters
    ----------
    bars:
        OHLC DataFrame with 'high', 'low', 'close' columns and a DatetimeIndex.
    swing_period:
        Bars on each side used to confirm swing levels (see ``structure_levels``).
    max_sweep_distance_atr:
        Maximum distance beyond the level (in ATRs) for the take to count as
        a sweep rather than a breakout.
    atr_period:
        ATR period for the distance filter.

    Returns
    -------
    pd.DataFrame
        Indexed like ``bars`` with columns ``sweep_high``, ``sweep_low``,
        ``sweep_high_level``, ``sweep_low_level``, ``sweep_high_reclaimed``,
        ``sweep_low_reclaimed``, ``sweep_high_age``, ``sweep_low_age``.
    """
    levels = structure_levels(bars, swing_period)
    # Level known *before* the current bar (strict causality).
    prev_sh = levels["last_swing_high"].shift(1).to_numpy(dtype=float)
    prev_sl = levels["last_swing_low"].shift(1).to_numpy(dtype=float)
    atr_arr = atr(bars, atr_period).to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    ev_h, rec_h, lv_h, age_h = _detect_side_sweeps(
        prev_sh, high, close, atr_arr, side=1, max_sweep_distance_atr=max_sweep_distance_atr
    )
    ev_l, rec_l, lv_l, age_l = _detect_side_sweeps(
        prev_sl, low, close, atr_arr, side=-1, max_sweep_distance_atr=max_sweep_distance_atr
    )

    return pd.DataFrame(
        {
            "sweep_high": ev_h,
            "sweep_low": ev_l,
            "sweep_high_level": lv_h,
            "sweep_low_level": lv_l,
            "sweep_high_reclaimed": rec_h,
            "sweep_low_reclaimed": rec_l,
            "sweep_high_age": age_h,
            "sweep_low_age": age_l,
        },
        index=bars.index,
    )


def detect_liquidity_sweeps_retained(
    bars: pd.DataFrame,
    *,
    swing_period: int = 5,
    max_sweep_distance_atr: float = 2.0,
    atr_period: int = 5,
    max_levels: int = MAX_RETAINED_LEVELS,
) -> pd.DataFrame:
    """Liquidity sweeps against *every* retained unswept swing level.

    Unlike :func:`detect_liquidity_sweeps`, which only ever references the
    single ``last_swing_*`` level, this fires a sweep event for any retained
    level price takes out. So a sweep "at this swing low but not lower swing
    lows" is representable: each retained level is swept independently.

    A level is swept when ``high[t]`` exceeds a retained high (or ``low[t]``
    falls below a retained low), within ``max_sweep_distance_atr`` ATR of the
    level. Deeper breaks count as breakouts, not sweeps.

    All retained levels are causally observable: they enter only on their
    confirmation bar, and the previous bar's retained set drives detection.

    Args:
        bars: OHLC frame with high/low/close and a DatetimeIndex.
        swing_period: Bars on each side used to confirm swing levels.
        max_sweep_distance_atr: ATR cap on the overshoot for a sweep.
        atr_period: ATR period for the distance filter.
        max_levels: Cap on retained levels per side.

    Returns:
        Frame indexed like ``bars`` with per-level sweep events:
        ``sweep_high`` / ``sweep_low`` (1 on a sweep bar, else 0),
        ``swept_high_level`` / ``swept_low_level`` (the level taken out),
        ``swept_high_loc`` / ``swept_low_loc`` (its bar location),
        ``n_swept_high`` / ``n_swept_low`` (levels swept this bar).
    """
    levels = structure_levels(bars, swing_period, retained_max=max_levels)
    k = max(int(max_levels), 1)

    # Retained set as of the previous bar (strict causality).
    prev_highs = [levels[f"retained_high_{i + 1}"].shift(1).to_numpy(dtype=float) for i in range(k)]
    prev_lows = [levels[f"retained_low_{i + 1}"].shift(1).to_numpy(dtype=float) for i in range(k)]
    atr_arr = atr(bars, atr_period).to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    n = len(bars)
    sweep_h = np.zeros(n)
    sweep_l = np.zeros(n)
    swept_h_level = np.full(n, np.nan)
    swept_l_level = np.full(n, np.nan)
    swept_h_loc = np.full(n, np.nan)
    swept_l_loc = np.full(n, np.nan)
    n_swept_h = np.zeros(n, dtype=int)
    n_swept_l = np.zeros(n, dtype=int)

    for t in range(n):
        a = atr_arr[t]
        for i in range(k):
            lv = prev_highs[i][t]
            if np.isnan(lv):
                continue
            dist = high[t] - lv
            if dist > 0.0 and (a <= 0.0 or dist <= max_sweep_distance_atr * a):
                sweep_h[t] = 1.0
                swept_h_level[t] = lv
                swept_h_loc[t] = levels[f"retained_high_{i + 1}_loc"].iloc[t]
                n_swept_h[t] += 1
                break  # one sweep event per side per bar
        for i in range(k):
            lv = prev_lows[i][t]
            if np.isnan(lv):
                continue
            dist = lv - low[t]
            if dist > 0.0 and (a <= 0.0 or dist <= max_sweep_distance_atr * a):
                sweep_l[t] = 1.0
                swept_l_level[t] = lv
                swept_l_loc[t] = levels[f"retained_low_{i + 1}_loc"].iloc[t]
                n_swept_l[t] += 1
                break

    return pd.DataFrame(
        {
            "sweep_high": sweep_h,
            "sweep_low": sweep_l,
            "swept_high_level": swept_h_level,
            "swept_low_level": swept_l_level,
            "swept_high_loc": swept_h_loc,
            "swept_low_loc": swept_l_loc,
            "n_swept_high": n_swept_h,
            "n_swept_low": n_swept_l,
        },
        index=bars.index,
    )


def detect_bos(bars: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Detect causal break-of-structure events from confirmed swings.

    ``bos_up = close > last_confirmed_swing_high`` and
    ``bos_down = close < last_confirmed_swing_low``, both evaluated against
    the level as of the previous bar. Wick penetration without close-through
    does not fire a BOS.

    Parameters
    ----------
    bars:
        OHLC DataFrame with a 'close' column.
    structure:
        Output of :func:`quant_rl.features.structure.structure_levels`.

    Returns
    -------
    pd.DataFrame
        Columns ``bos_up``, ``bos_down`` (0/1) and ``bos_up_level``,
        ``bos_down_level`` (the level crossed, NaN when no BOS on that bar).
    """
    prev_sh = structure["last_swing_high"].shift(1)
    prev_sl = structure["last_swing_low"].shift(1)
    close = bars["close"]

    bos_up = (close > prev_sh) & prev_sh.notna()
    bos_down = (close < prev_sl) & prev_sl.notna()

    return pd.DataFrame(
        {
            "bos_up": bos_up.astype(int),
            "bos_down": bos_down.astype(int),
            "bos_up_level": prev_sh.where(bos_up),
            "bos_down_level": prev_sl.where(bos_down),
        },
        index=bars.index,
    )


def detect_mss(
    bars: pd.DataFrame,
    structure: pd.DataFrame,
    bos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect a causal market-structure shift (MSS) after an opposite BOS.

    Minimal LTF decision feature: an MSS fires when a BOS flips the prior BOS
    direction (structure shift against the previous swing break). Wick-only
    penetration does not count — MSS inherits BOS close-through semantics.

    - ``mss_up = 1``: ``bos_up`` while the last completed BOS was ``bos_down``
    - ``mss_down = 1``: ``bos_down`` while the last completed BOS was ``bos_up``

    Levels are the swing level crossed on the MSS bar (same as the BOS level).
    """
    if bos is None:
        bos = detect_bos(bars, structure)

    n = len(bars)
    bos_up = bos["bos_up"].to_numpy(dtype=float)
    bos_down = bos["bos_down"].to_numpy(dtype=float)
    bos_up_lv = bos["bos_up_level"].to_numpy(dtype=float)
    bos_down_lv = bos["bos_down_level"].to_numpy(dtype=float)

    mss_up = np.zeros(n)
    mss_down = np.zeros(n)
    mss_up_level = np.full(n, np.nan)
    mss_down_level = np.full(n, np.nan)

    last_dir = 0  # +1 last bos_up, -1 last bos_down
    for t in range(n):
        if bos_up[t] > 0:
            if last_dir == -1:
                mss_up[t] = 1.0
                mss_up_level[t] = bos_up_lv[t]
            last_dir = 1
        elif bos_down[t] > 0:
            if last_dir == 1:
                mss_down[t] = 1.0
                mss_down_level[t] = bos_down_lv[t]
            last_dir = -1

    return pd.DataFrame(
        {
            "mss_up": mss_up.astype(int),
            "mss_down": mss_down.astype(int),
            "mss_up_level": mss_up_level,
            "mss_down_level": mss_down_level,
        },
        index=bars.index,
    )
