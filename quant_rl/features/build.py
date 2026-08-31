"""Feature build pipeline: indicators + SMT + normalisation → feature matrix.

Chain A: per-timeframe technical features. For every timeframe listed in
``cfg.features.htf_timeframes`` the same ``build_indicators()`` call used
for M1 is run on that timeframe's own bars, then causally forward-filled
onto the M1 spine via :func:`quant_rl.data.align.align_timeframes` with
``{TF}_{indicator}`` column names.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..data.align import align_timeframes
from ..data.resample import resample
from .indicators import atr, build_indicators, sweep_velocity, volume_spike, wick_ratio
from .normalize import rolling_zscore
from .po3_config import (
    FVGConfig,
    build_fvg_zones,
    detect_fvg,
    detect_po3_entries,
)
from .smt import smt_divergence
from .structure import detect_session_levels, structure_levels

# Bump whenever build_features() output schema changes so stale caches are
# not silently reused by the {symbol}_features.parquet call sites.
# v4: Chain F + causal PO3 mapping — detect_htf_fvg/detect_ltf_ifvg now shift
# signals one HTF/primary period forward (no within-period lookahead).
FEATURE_CACHE_VERSION = "v4-po3causal"

_DEFAULT_HTF_TIMEFRAMES = ("M5", "M15", "H1")
# Capped normalised FVG distance: value used when no zone is active nearby.
_FVG_DIST_CAP = 5.0

# Numeric encoding of detect_po3_entries' entry_trigger_type string column so
# the feature matrix stays homogeneous float (TradingEnv casts to float32).
_PO3_TRIGGER_CODE = {"": 0, "retest": 1, "close_through": 2, "ltf_fvg": 3}


def build_po3_phase_features(
    index: pd.DatetimeIndex,
    session_start: str = "16:30",
    session_end: str = "23:00",
) -> pd.DataFrame:
    """PO3 (Power of 3) daily-cycle phase tag, causally time-derived.

    Maps each bar's time-of-day position inside the trading session onto the
    classic PO3 arc: accumulation (session open) → manipulation (middle) →
    distribution (session close). This is a function of the timestamp only,
    so it is identical across timeframes once aligned onto the M1 spine —
    computed once here rather than duplicated per TF.

    Returns a DataFrame indexed like ``index`` with:
    - ``po3_phase``: ordinal 0/1/2 (accumulation/manipulation/distribution)
    - ``session_progress``: continuous fraction of session elapsed in [0, 1]
    """
    tz = index.tz
    start_ts = pd.Timestamp(f"2000-01-01 {session_start}").tz_localize(tz)
    end_ts = pd.Timestamp(f"2000-01-01 {session_end}").tz_localize(tz)
    session_len = (end_ts - start_ts).total_seconds()

    tod = index.hour * 3600 + index.minute * 60 + index.second
    start_secs = start_ts.hour * 3600 + start_ts.minute * 60
    elapsed = tod - start_secs
    progress = pd.Series(np.clip(elapsed / session_len, 0.0, 1.0), index=index)
    phase = pd.Series(np.digitize(progress.to_numpy(), [1.0 / 3.0, 2.0 / 3.0]), index=index)
    phase = phase.astype("float64")
    phase[elapsed < 0] = np.nan  # outside session → unknown phase
    return pd.DataFrame({"po3_phase": phase, "session_progress": progress})


def build_fvg_zone_features(
    bars: pd.DataFrame,
    fvg_config: FVGConfig | None = None,
    max_zone_bars: int = 50,
) -> pd.DataFrame:
    """Model-facing FVG zone features for one timeframe's own bars.

    Uses :func:`quant_rl.features.po3_config.detect_fvg` +
    :func:`quant_rl.features.po3_config.build_fvg_zones`, then per bar reports:

    - ``fvg_in_bull`` / ``fvg_in_bear``: 1 if close is inside an active zone
    - ``fvg_bull_dist`` / ``fvg_bear_dist``: distance from close to the nearest
      active zone edge, normalised by ATR and capped at 5.0 (cap value = no
      active zone nearby)

    All values are computed from the current and earlier bars only (zone
    activity windows end at fill/expire time), so the block is causal.
    """
    n = len(bars)
    idx = bars.index
    out = pd.DataFrame(
        0.0, index=idx, columns=["fvg_in_bull", "fvg_in_bear", "fvg_bull_dist", "fvg_bear_dist"]
    )
    out["fvg_bull_dist"] = _FVG_DIST_CAP
    out["fvg_bear_dist"] = _FVG_DIST_CAP

    if n == 0:
        return out

    signals = detect_fvg(bars, fvg_config)
    # build_fvg_zones() matches on the htf_fvg_* column family; rename the
    # plain detect_fvg() output so the zone builder recognises it.
    signals = signals.rename(
        columns={
            "fvg_bullish": "htf_fvg_bullish",
            "fvg_bullish_low": "htf_fvg_bullish_low",
            "fvg_bullish_high": "htf_fvg_bullish_high",
            "fvg_bearish": "htf_fvg_bearish",
            "fvg_bearish_low": "htf_fvg_bearish_low",
            "fvg_bearish_high": "htf_fvg_bearish_high",
        }
    )
    zones = build_fvg_zones(bars, signals, max_zone_bars=max_zone_bars)
    if not zones:
        return out

    atr_s = atr(bars, period=14).to_numpy()
    close = bars["close"].to_numpy()

    for zone in zones:
        start_i = int(idx.searchsorted(zone.start_ts, side="left"))
        end_i = int(idx.searchsorted(zone.end_ts, side="right"))  # exclusive
        if start_i >= n:
            continue
        mid = 0.5 * (zone.zone_low + zone.zone_high)
        bull_in = cast(int, out.columns.get_loc("fvg_in_bull"))
        bear_in = cast(int, out.columns.get_loc("fvg_in_bear"))
        bull_dist = cast(int, out.columns.get_loc("fvg_bull_dist"))
        bear_dist = cast(int, out.columns.get_loc("fvg_bear_dist"))
        for i in range(start_i, min(end_i, n)):
            if zone.side == "bullish":
                edge = zone.zone_high if close[i] >= mid else zone.zone_low
            else:
                edge = zone.zone_low if close[i] <= mid else zone.zone_high
            d = abs(close[i] - edge)
            a = atr_s[i] if not np.isnan(atr_s[i]) and atr_s[i] > 0 else 1.0
            in_col, dist_col = (
                (bull_in, bull_dist) if zone.side == "bullish" else (bear_in, bear_dist)
            )
            dist = min(d / a, out.iat[i, dist_col])
            out.iat[i, dist_col] = dist
            # "in zone" = close inside the zone boundaries (OR across zones)
            if zone.zone_low <= close[i] <= zone.zone_high:
                out.iat[i, in_col] = 1.0
    return out


def build_features(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None = None,
    cfg: DictConfig | None = None,
    train_mask: pd.Series | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Build the full feature matrix for the primary instrument.

    Parameters
    ----------
    primary:
        Cleaned, session-filtered M1 DataFrame (US100).
    secondary:
        US500 M1 DataFrame for SMT divergence (optional but recommended).
    cfg:
        Full OmegaConf config; ``cfg.features`` sub-node is used.
    train_mask:
        Boolean Series aligned to primary index; True = training bar.
    cache_path:
        If given, save/load feature parquet here.
    force:
        Ignore existing cache.
    """
    if cache_path and Path(cache_path).exists() and not force:
        return pd.read_parquet(cache_path)

    feat_cfg = cfg.features if cfg is not None else None

    # --- base indicators ---
    if feat_cfg is not None:
        feat = build_indicators(primary, feat_cfg)
    else:
        # fallback: basic returns only
        feat = pd.DataFrame(
            {"ret_1": np.log(primary["close"]).diff()},
            index=primary.index,
        )

    # --- SMT divergence ---
    if secondary is not None and feat_cfg is not None:
        smt = smt_divergence(
            primary,
            secondary,
            swing_period=feat_cfg.smt_swing_period,
            corr_window=feat_cfg.smt_corr_window,
        )
        feat = pd.concat([feat, smt], axis=1)

    # --- Higher-timeframe technical features (Chain A) ---
    # Run the same indicator build on each HTF's own bars, then causally
    # forward-fill onto the M1 spine. Done BEFORE normalisation so HTF
    # columns are z-scored like the M1 ones.
    if feat_cfg is not None:
        htf_cfg_tfs = getattr(feat_cfg, "htf_timeframes", None)
        htf_tfs = list(htf_cfg_tfs) if htf_cfg_tfs is not None else list(_DEFAULT_HTF_TIMEFRAMES)
        if htf_tfs:
            htf_blocks: dict[str, pd.DataFrame] = {}
            for tf in htf_tfs:
                tf_bars = resample(primary, tf)  # type: ignore[arg-type]
                htf_blocks[str(tf)] = build_indicators(tf_bars, feat_cfg)
            feat = align_timeframes(feat, htf_blocks)

    # --- Chain B variant blocks (opt-in via config flags) ---
    if feat_cfg is not None:
        # PO3 phase tag: time-derived, identical across TFs once aligned.
        if bool(getattr(feat_cfg, "include_po3", False)):
            session = OmegaConf.select(cfg, "session") if cfg is not None else None
            po3 = build_po3_phase_features(
                pd.DatetimeIndex(primary.index),
                session_start=str(session.get("start", "16:30")) if session else "16:30",
                session_end=str(session.get("end", "23:00")) if session else "23:00",
            )
            feat = pd.concat([feat, po3], axis=1)

        # FVG zone features per timeframe (Chain A per-TF pattern).
        if bool(getattr(feat_cfg, "include_fvg_ifvg", False)):
            fvg_blocks: dict[str, pd.DataFrame] = {}
            for tf in htf_tfs or ["M5", "M15", "H1"]:
                tf_bars = resample(primary, tf)  # type: ignore[arg-type]
                fvg_blocks[str(tf)] = build_fvg_zone_features(tf_bars)
            feat = align_timeframes(feat, fvg_blocks)

    # --- normalisation ---
    window = feat_cfg.zscore_window if feat_cfg is not None else 252
    feat = rolling_zscore(feat, window=window, train_mask=train_mask)

    # --- Structure levels (swings) - add AFTER normalization to keep raw prices ---
    if feat_cfg is not None:
        structure = structure_levels(primary, swing_period=feat_cfg.smt_swing_period)
        # Drop time columns (not needed in feature matrix)
        structure = structure[["last_swing_high", "last_swing_low"]]
        feat = pd.concat([feat, structure], axis=1)

    # --- Full PO3 pipeline (Chain F): HTF FVG -> LTF IFVG -> entry triggers ---
    # Added AFTER normalization so the 0/1 entry signals and the numeric
    # trigger code keep their raw meaning (exactly like structure levels).
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_po3_full", False)):
        po3_signals = detect_po3_entries(
            primary,
            htf=str(getattr(feat_cfg, "po3_htf", "M15")),
            primary_tf=str(getattr(feat_cfg, "po3_ltf", "M5")),
        )
        po3_feats = po3_signals[["entry_long", "entry_short", "entry_trigger_type"]].copy()
        # entry_trigger_type is a string column; encode numerically so the
        # matrix stays homogeneous float for TradingEnv's float32 cast.
        po3_feats["entry_trigger_type"] = (
            po3_feats["entry_trigger_type"].map(_PO3_TRIGGER_CODE).fillna(0.0).astype("float64")
        )
        feat = pd.concat([feat, po3_feats], axis=1)

    # --- Liquidity Levels + Volume Spike + ATR - add AFTER normalization ---
    levels = detect_session_levels(primary)
    feat = pd.concat([feat, levels], axis=1)
    # Only compute volume_spike if volume column exists
    if "volume" in primary.columns:
        feat["volume_spike"] = volume_spike(primary["volume"], window=20)
    feat["atr_5"] = atr(primary, period=5)

    # --- Sweep Velocity and Wick Ratio - for PLAN 3 ---
    # Add sweep velocity (uses liquidity levels from levels)
    sweep_vel = sweep_velocity(
        primary,
        london_high=levels.get("london_high"),
        london_low=levels.get("london_low"),
        asian_high=levels.get("asian_high"),
        asian_low=levels.get("asian_low"),
        atr_period=5,
    )
    feat = pd.concat([feat, sweep_vel], axis=1)

    # Add wick ratio
    feat["wick_ratio"] = wick_ratio(primary)

    # Drop leading NaNs from warmup
    feat = feat.dropna(how="all")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(cache_path)

    return feat
