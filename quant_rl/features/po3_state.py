"""PO3 manipulation/distribution state machine and IFVG active zones.

All outputs are causal: they are computed from the current and earlier bars
only, so adding future bars never changes historical values (Agent.md §3).

Manipulation/distribution semantics (Agent.md §6.3), long side:
1. Asian accumulation context is available (``asian_high``/``asian_low``).
2. A sell-side liquidity sweep fires (``sweep_low`` event).
3. The manipulation leg starts: ``po3_manipulation_low/high`` track the
   running extremes from the sweep bar onward.
4. The first bar whose close exceeds the manipulation-leg high as of the
   previous bar confirms the post-manipulation direction — that bar marks
   ``po3_manipulation_end = 1`` and starts the distribution phase
   (``po3_distribution = 1``, ``po3_distribution_direction = +1``).
5. The manipulation extremes freeze at the end bar and persist while the
   distribution phase is active (the environment needs the raw price for
   structural SL placement). A fresh same-side sweep re-arms the setup with
   new extremes; an opposite-side sweep flips the setup to the mirror side.
The short side mirrors all of the above.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from .indicators import atr
from .po3_config import FVGConfig, IFVGConfig, detect_fvg, detect_ifvg_confirmation

# State-machine states.
_IDLE, _MANIP_LONG, _MANIP_SHORT, _DIST_LONG, _DIST_SHORT = 0, 1, 2, 3, 4


def _asian_context(asian_levels: pd.DataFrame | None, n: int) -> npt.NDArray[np.bool_]:
    """Per-bar mask: True where Asian accumulation levels are available."""
    if asian_levels is None or "asian_high" not in asian_levels.columns:
        return np.ones(n, dtype=bool)
    ah = asian_levels["asian_high"].to_numpy(dtype=float)
    al = asian_levels["asian_low"].to_numpy(dtype=float)
    out: npt.NDArray[np.bool_] = ~(np.isnan(ah) | np.isnan(al))
    return out


def build_po3_state(
    bars: pd.DataFrame,
    sweeps: pd.DataFrame,
    asian_levels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal PO3 manipulation/distribution state columns.

    Parameters
    ----------
    bars:
        OHLC DataFrame with 'high', 'low', 'close' columns.
    sweeps:
        Output of :func:`quant_rl.features.liquidity.detect_liquidity_sweeps`.
    asian_levels:
        DataFrame with ``asian_high``/``asian_low`` columns (e.g. the output
        of ``detect_session_levels``). Rows with NaN levels carry no context.

    Returns
    -------
    pd.DataFrame
        Columns ``po3_manipulation_active``, ``po3_manipulation_high``,
        ``po3_manipulation_low``, ``po3_manipulation_end``,
        ``po3_distribution``, ``po3_distribution_direction``.
    """
    n = len(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    sw_high = sweeps["sweep_high"].to_numpy(dtype=float)
    sw_low = sweeps["sweep_low"].to_numpy(dtype=float)
    ctx = _asian_context(asian_levels, n)

    manip_active = np.zeros(n)
    manip_h = np.full(n, np.nan)
    manip_l = np.full(n, np.nan)
    end = np.zeros(n)
    dist = np.zeros(n)
    direction = np.zeros(n)

    state = _IDLE
    run_h = np.nan
    run_l = np.nan

    for t in range(n):
        new_low = sw_low[t] == 1 and ctx[t]
        new_high = sw_high[t] == 1 and ctx[t]

        if state in (_IDLE, _DIST_LONG, _DIST_SHORT):
            if new_low:
                state, run_h, run_l = _MANIP_LONG, high[t], low[t]
            elif new_high:
                state, run_h, run_l = _MANIP_SHORT, high[t], low[t]
        elif state == _MANIP_LONG:
            if new_high:  # opposite sweep re-arms the mirror setup
                state, run_h, run_l = _MANIP_SHORT, high[t], low[t]
            elif new_low:  # fresh sell-side sweep re-arms with new extremes
                run_h, run_l = high[t], low[t]
            elif close[t] > run_h:  # reclaim of the manipulation-leg high
                end[t] = 1.0
                state = _DIST_LONG
            else:
                run_h = max(run_h, high[t])
                run_l = min(run_l, low[t])
        elif state == _MANIP_SHORT:
            if new_low:
                state, run_h, run_l = _MANIP_LONG, high[t], low[t]
            elif new_high:
                run_h, run_l = high[t], low[t]
            elif close[t] < run_l:  # reclaim of the manipulation-leg low
                end[t] = 1.0
                state = _DIST_SHORT
            else:
                run_h = max(run_h, high[t])
                run_l = min(run_l, low[t])

        if state in (_MANIP_LONG, _MANIP_SHORT):
            manip_active[t] = 1.0
            manip_h[t] = run_h
            manip_l[t] = run_l
        elif state == _DIST_LONG:
            manip_h[t] = run_h
            manip_l[t] = run_l
            dist[t] = 1.0
            direction[t] = 1.0
        elif state == _DIST_SHORT:
            manip_h[t] = run_h
            manip_l[t] = run_l
            dist[t] = 1.0
            direction[t] = -1.0

    return pd.DataFrame(
        {
            "po3_manipulation_active": manip_active,
            "po3_manipulation_high": manip_h,
            "po3_manipulation_low": manip_l,
            "po3_manipulation_end": end,
            "po3_distribution": dist,
            "po3_distribution_direction": direction,
        },
        index=bars.index,
    )


def build_ifvg_zone_features(
    bars: pd.DataFrame,
    *,
    max_age_bars: int = 50,
    fvg_config: FVGConfig | None = None,
    ifvg_config: IFVGConfig | None = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Model/environment-facing IFVG active-zone features.

    Reuses :func:`quant_rl.features.po3_config.detect_fvg` +
    :func:`quant_rl.features.po3_config.detect_ifvg_confirmation` (no second
    IFVG implementation) and gives each confirmed zone an explicit lifetime
    ``start <= t < start + max_age_bars`` — zones never forward-fill forever
    (Agent.md §7). The most recent confirmed zone per side is the active one.

    Parameters
    ----------
    bars:
        OHLC DataFrame with 'high', 'low', 'close' columns.
    max_age_bars:
        Zone lifetime in bars after confirmation.
    fvg_config / ifvg_config:
        Optional detector configurations.
    atr_period:
        ATR period used to normalise the price-to-zone distance.

    Returns
    -------
    pd.DataFrame
        Columns ``ifvg_bull_active``, ``ifvg_bull_low``, ``ifvg_bull_high``,
        ``price_in_ifvg_bull``, ``ifvg_bull_distance_atr`` and the bearish
        mirrors ``ifvg_bear_*`` / ``price_in_ifvg_bear``.
    """
    n = len(bars)
    idx = bars.index
    out = pd.DataFrame(
        0.0,
        index=idx,
        columns=[
            "ifvg_bull_active",
            "ifvg_bull_low",
            "ifvg_bull_high",
            "price_in_ifvg_bull",
            "ifvg_bull_distance_atr",
            "ifvg_bear_active",
            "ifvg_bear_low",
            "ifvg_bear_high",
            "price_in_ifvg_bear",
            "ifvg_bear_distance_atr",
        ],
    )
    if n == 0:
        return out

    fvg = detect_fvg(bars, fvg_config)
    conf = detect_ifvg_confirmation(bars, fvg, ifvg_config)
    atr_arr = atr(bars, atr_period).to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    conf_bull = conf["ifvg_bullish_confirmed"].to_numpy(dtype=int)
    conf_bear = conf["ifvg_bearish_confirmed"].to_numpy(dtype=int)
    bull_low = conf["ifvg_bullish_low"].to_numpy(dtype=float)
    bull_high = conf["ifvg_bullish_high"].to_numpy(dtype=float)
    bear_low = conf["ifvg_bearish_low"].to_numpy(dtype=float)
    bear_high = conf["ifvg_bearish_high"].to_numpy(dtype=float)

    # Latest confirmed zone per side; expires max_age_bars after confirmation.
    bull_start, bull_zl, bull_zh = -1, np.nan, np.nan
    bear_start, bear_zl, bear_zh = -1, np.nan, np.nan
    for i in range(n):
        if conf_bull[i] == 1:
            bull_start, bull_zl, bull_zh = i, bull_low[i], bull_high[i]
        if conf_bear[i] == 1:
            bear_start, bear_zl, bear_zh = i, bear_low[i], bear_high[i]

        a = atr_arr[i] if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else 1.0
        if bull_start >= 0 and i - bull_start < max_age_bars:
            out.iat[i, 0] = 1.0
            out.iat[i, 1] = bull_zl
            out.iat[i, 2] = bull_zh
            if bull_zl <= close[i] <= bull_zh:
                out.iat[i, 3] = 1.0
            out.iat[i, 4] = min(abs(close[i] - bull_zl), abs(close[i] - bull_zh)) / a
        if bear_start >= 0 and i - bear_start < max_age_bars:
            out.iat[i, 5] = 1.0
            out.iat[i, 6] = bear_zl
            out.iat[i, 7] = bear_zh
            if bear_zl <= close[i] <= bear_zh:
                out.iat[i, 8] = 1.0
            out.iat[i, 9] = min(abs(close[i] - bear_zl), abs(close[i] - bear_zh)) / a

    return out
