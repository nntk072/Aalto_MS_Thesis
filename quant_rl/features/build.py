"""Feature build pipeline: indicators + SMT + normalisation → feature matrix.

Chain A: per-timeframe technical features. For every timeframe listed in
``cfg.features.htf_timeframes`` the same ``build_indicators()`` call used
for M1 is run on that timeframe's own bars, then causally forward-filled
onto the M1 spine via :func:`quant_rl.data.align.align_timeframes` with
``{TF}_{indicator}`` column names.

MTF extensions (mtf_feature_expansion_plan.md §§4.1-4.6): SMT divergence
(§4.1), structure levels + BOS (§4.2) and liquidity sweeps (§4.3) follow the
same resample → per-TF compute → ``align_timeframes`` pattern; FVG zones
(§4.5) run over the full ladder (M1 + HTFs); PO3 manipulation/distribution
state (§4.4) and IFVG active zones (§4.6) are scoped to M1/M5/M15.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..data.align import align_timeframes
from ..data.resample import resample
from .indicators import atr, build_indicators, sweep_velocity, volume_spike, vwap_level, wick_ratio
from .liquidity import detect_bos, detect_liquidity_sweeps
from .normalize import rolling_zscore
from .po3_config import (
    FVGConfig,
    build_fvg_zones,
    detect_fvg,
    detect_po3_entries,
)
from .po3_state import build_ifvg_zone_features, build_po3_state
from .session_ohlc import prior_period_high_low, range_quadrants, session_ohlc
from .smt import smt_divergence
from .structure import (
    TIMEFRAME_CONFIG,
    classify_structure,
    detect_pivots,
    detect_session_levels,
    detect_swings,
    structure_levels,
    swing_features,
)

# Bump whenever build_features() output schema changes so stale caches are
# not silently reused by the {symbol}_features.parquet call sites.
# v4: Chain F + causal PO3 mapping — detect_htf_fvg/detect_ltf_ifvg now shift
# signals one HTF/primary period forward (no within-period lookahead).
# v5: Idea 1 strategy state — liquidity sweeps, PO3 manipulation/distribution
# state and IFVG active zones (include_strategy_state opt-in block).
# v6: CT-anchored session levels — session OHLC, yesterday/last-week H/L,
# range quadrants and raw vwap (include_session_ohlc opt-in block).
# v7: MTF expansion (plan §§4.1-4.6) — per-TF SMT divergence (§4.1),
# structure/BOS (§4.2) and liquidity sweeps (§4.3) via the Chain-A
# resample → per-TF compute → align_timeframes pattern; FVG zones over the
# full M1+HTF ladder (§4.5); PO3 state (§4.4) and IFVG zones (§4.6) on
# M1/M5/M15.
# v8: full-day M1 spine, completed-bar HTF ffill, confirmed ATR swings.
FEATURE_CACHE_VERSION = "v8-full-day-completed-htf-swings"

_DEFAULT_HTF_TIMEFRAMES = ("M5", "M15", "H1")
# Capped normalised FVG distance: value used when no zone is active nearby.
_FVG_DIST_CAP = 5.0

# Numeric encoding of detect_po3_entries' entry_trigger_type string column so
# the feature matrix stays homogeneous float (TradingEnv casts to float32).
_PO3_TRIGGER_CODE = {"": 0, "retest": 1, "close_through": 2, "ltf_fvg": 3}

# Raw price-level suffixes the environment must exclude from the model
# sequence (Agent.md §11): the M1 convention is ``strategy.raw_columns``
# (exact names, e.g. ``last_swing_high``). MTF blocks reuse the same stems
# with a ``{TF}_`` prefix (``M5_last_swing_high``), so the env matches
# ``{TF}_{stem}`` for every stem below. ATR-normalised distances
# (``*_dist_to_swing_*_atr``, ``*_distance_atr``) and binary flags stay in.
MTF_RAW_SUFFIXES = (
    "last_swing_high",
    "last_swing_low",
    "sweep_high_level",
    "sweep_low_level",
    "bos_up_level",
    "bos_down_level",
    "po3_manipulation_high",
    "po3_manipulation_low",
    "ifvg_bull_low",
    "ifvg_bull_high",
    "ifvg_bear_low",
    "ifvg_bear_high",
    "asian_high",
    "asian_low",
    "london_high",
    "london_low",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
)

# Scoped per-TF subsets for the state-heavy blocks (plan §8, option (b)):
# PO3 (§4.4) and IFVG active-zones (§4.6) run only on M1/M5/M15 by default —
# H1 rarely sweeps within one NY session and adds mostly-collinear columns.
_MTF_STATE_TFS = ("M1", "M5", "M15")

# MTF subset for the lightweight per-TF blocks (plan §3): SMT divergence
# (§4.1), structure/BOS (§4.2) and liquidity sweeps (§4.3) run on M5/M15/H1
# by default (M1 already has the unprefixed block). M30 stays excluded until
# htf_timeframes gains it, keeping column growth bounded.
_MTF_LIGHT_TFS = ("M5", "M15", "H1")


def _htf_list(feat_cfg: object, key: str, default: tuple[str, ...]) -> list[str]:
    """Read a TF subset list (plan §5 nested block) with a safe default.

    Accepts ``features.<key>`` stored either as a nested block
    (``features.smt: {timeframes: [...]}``) or as a flat key
    (``features.smt_timeframes: [...]``); both merge cleanly through
    OmegaConf. Falls back to ``default`` when unset/empty.
    """
    raw: object = None
    if feat_cfg is not None:
        block = getattr(feat_cfg, key, None)
        if block is not None and hasattr(block, "timeframes"):
            raw = getattr(block, "timeframes")
        else:
            raw = getattr(feat_cfg, f"{key}_timeframes", None)
    if raw is None:
        return list(default)
    tfs = [str(t) for t in list(cast(Any, raw))]
    return tfs or list(default)


def _resample_pair(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None,
    tf: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Resample both symbols to ``tf`` (M1 = identity copy).

    The secondary symbol is resampled from its own M1 bars (never from
    primary's bars) so SMT divergence on TF compares like-for-like bars.
    Returns ``(primary_tf, secondary_tf_or_None)``.
    """
    pri_tf = resample(primary, tf) if tf != "M1" else primary  # type: ignore[arg-type]
    sec_tf: pd.DataFrame | None = None
    if secondary is not None:
        sec_tf = resample(secondary, tf) if tf != "M1" else secondary  # type: ignore[arg-type]
    return pri_tf, sec_tf


def _enabled(feat_cfg: object, key: str, default: bool) -> bool:
    """Read ``features.<key>.enabled`` (plan §5 nested block) or flat key."""
    if feat_cfg is None:
        return default
    block = getattr(feat_cfg, key, None)
    if block is not None and hasattr(block, "enabled"):
        return bool(getattr(block, "enabled"))
    flat = getattr(feat_cfg, f"{key}_enabled", None)
    if flat is not None:
        return bool(flat)
    return default


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
    secondary_m1: pd.DataFrame | None = secondary  # M1 copy kept for per-TF resampling

    # --- base indicators ---
    if feat_cfg is not None:
        feat = build_indicators(primary, feat_cfg)
    else:
        # fallback: basic returns only
        feat = pd.DataFrame(
            {"ret_1": np.log(primary["close"]).diff()},
            index=primary.index,
        )

    # --- SMT divergence (M1 always; MTF optional §4.1) ---
    # M1 block is z-scored with the rest; per-TF blocks join before
    # normalisation for the same treatment. Secondary is resampled from its
    # own M1 bars so each TF compares like-for-like bars.
    if secondary is not None and feat_cfg is not None:
        smt = smt_divergence(
            primary,
            secondary,
            swing_period=int(getattr(feat_cfg, "smt_swing_period", 5)),
            corr_window=int(getattr(feat_cfg, "smt_corr_window", 20)),
        )
        feat = pd.concat([feat, smt], axis=1)
        if _enabled(feat_cfg, "smt", False):
            smt_tfs = _htf_list(feat_cfg, "smt", _MTF_LIGHT_TFS)
            smt_blocks: dict[str, pd.DataFrame] = {}
            for tf in smt_tfs:
                if str(tf) == "M1":
                    continue  # M1 already present unprefixed
                pri_tf, sec_tf = _resample_pair(primary, secondary_m1, str(tf))
                if sec_tf is None or pri_tf.empty or sec_tf.empty:
                    continue
                smt_blocks[str(tf)] = smt_divergence(
                    pri_tf,
                    sec_tf,
                    swing_period=int(getattr(feat_cfg, "smt_swing_period", 5)),
                    corr_window=int(getattr(feat_cfg, "smt_corr_window", 20)),
                )
            if smt_blocks:
                feat = align_timeframes(feat, smt_blocks)

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

        # FVG zone features (plan §4.5): full M1+HTF ladder, not HTFs only.
        # build_fvg_zone_features is TF-agnostic, so this is purely what goes
        # into the loop: ["M1"] + htf_timeframes (config-overridable via
        # fvg_timeframes), giving M1_fvg_*/M5_fvg_*/... columns.
        if bool(getattr(feat_cfg, "include_fvg_ifvg", False)):
            fvg_cfg_tfs = getattr(feat_cfg, "fvg_timeframes", None)
            if fvg_cfg_tfs is not None:
                fvg_tfs = [str(t) for t in list(fvg_cfg_tfs)]
            else:
                fvg_tfs = ["M1", *[str(t) for t in htf_tfs]]
            fvg_blocks: dict[str, pd.DataFrame] = {}
            for tf in fvg_tfs:
                tf_bars = primary if str(tf) == "M1" else resample(primary, tf)  # type: ignore[arg-type]
                fvg_blocks[str(tf)] = build_fvg_zone_features(tf_bars)
            feat = align_timeframes(feat, fvg_blocks)

    # --- normalisation ---
    window = feat_cfg.zscore_window if feat_cfg is not None else 252
    feat = rolling_zscore(feat, window=window, train_mask=train_mask)

    # --- Structure levels (swings) + MTF structure/BOS (plan §4.2) ---
    # Raw price levels after normalization (existing convention). M1 stays
    # unprefixed; MTF variants add {TF}_last_swing_high/low + {TF}_bos_up/down
    # (close-through flags) + ATR-normalized distances. Per-TF BOS uses the
    # TF's own swings (detect_bos(tf_bars, structure_tf)), not M1 proxies.
    structure = None
    _mtf_struct_levels: dict[str, pd.DataFrame] = {}
    if feat_cfg is not None:
        m1_cfg = TIMEFRAME_CONFIG["M1"]
        left = int(m1_cfg["left"])
        atr_mult = float(m1_cfg["atr_mult"])
        structure = structure_levels(primary, swing_period=left, atr_mult=atr_mult)
        structure = structure[["last_swing_high", "last_swing_low"]]
        pivots = detect_pivots(primary, left=left, right=int(m1_cfg["right"]))
        swings_df = detect_swings(primary, pivots, atr_mult=atr_mult)
        struct_cls = classify_structure(swings_df)
        feat = pd.concat(
            [feat, structure, swing_features(primary, swings_df, struct_cls)],
            axis=1,
        )
        if _enabled(feat_cfg, "structure", False):
            struct_blocks: dict[str, pd.DataFrame] = {}
            struct_levels: dict[str, pd.DataFrame] = {}
            for tf in _htf_list(feat_cfg, "structure", _MTF_LIGHT_TFS):
                if str(tf) == "M1":
                    continue
                pri_tf, _ = _resample_pair(primary, None, str(tf))
                if pri_tf.empty:
                    continue
                tf_cfg = TIMEFRAME_CONFIG.get(str(tf), m1_cfg)
                lv_tf = structure_levels(
                    pri_tf,
                    swing_period=int(tf_cfg["left"]),
                    atr_mult=float(tf_cfg["atr_mult"]),
                )[["last_swing_high", "last_swing_low"]]
                struct_levels[str(tf)] = lv_tf
                bos_tf = detect_bos(pri_tf, lv_tf)[["bos_up", "bos_down"]]
                struct_blocks[str(tf)] = pd.concat([lv_tf, bos_tf], axis=1)
            if struct_blocks:
                feat = align_timeframes(feat, struct_blocks)
                _mtf_struct_levels = struct_levels

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
    vol_s = None
    if "volume" in primary.columns:
        vol_s = primary["volume"]
    elif "tickvol" in primary.columns:
        vol_s = primary["tickvol"]
    if vol_s is not None:
        feat["volume_spike"] = volume_spike(vol_s, window=20)
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

    # --- Liquidity sweep MTF (plan §4.3) ---
    # True per-TF sweeps on genuinely resampled OHLC (not M1 proxies with
    # wider windows). Opt-in; follows the structure/BOS placement rule.
    _mtf_sweep_levels: dict[str, pd.DataFrame] = {}
    if feat_cfg is not None and _enabled(feat_cfg, "liquidity", False):
        swing = int(getattr(feat_cfg, "smt_swing_period", 5))
        sweep_blocks: dict[str, pd.DataFrame] = {}
        for tf in _htf_list(feat_cfg, "liquidity", _MTF_LIGHT_TFS):
            if str(tf) == "M1":
                continue
            pri_tf, _ = _resample_pair(primary, None, str(tf))
            if pri_tf.empty:
                continue
            sweeps_tf = detect_liquidity_sweeps(pri_tf, swing_period=swing)
            sweep_blocks[str(tf)] = sweeps_tf
            _mtf_sweep_levels[str(tf)] = detect_session_levels(pri_tf)
        if sweep_blocks:
            feat = align_timeframes(feat, sweep_blocks)

    # --- MTF structure/BOS ATR distances (plan §4.2, second half) ---
    # Needs atr_5 (computed above) + aligned {TF}_swing levels via ffill.
    if feat_cfg is not None and _enabled(feat_cfg, "structure", False) and _mtf_struct_levels:
        atr5_m1 = feat["atr_5"].where(feat["atr_5"] > 0)
        for tf, lv_tf in _mtf_struct_levels.items():
            hi = lv_tf["last_swing_high"].reindex(feat.index, method="ffill")
            lo = lv_tf["last_swing_low"].reindex(feat.index, method="ffill")
            feat[f"{tf}_dist_to_swing_high_atr"] = (hi - primary["close"]) / atr5_m1
            feat[f"{tf}_dist_to_swing_low_atr"] = (primary["close"] - lo) / atr5_m1

    # --- Idea 1 (PO3 + IFVG) strategy state features (opt-in) ---
    # Assembled after session levels and ATR so the state machine can consume
    # the causal Asian levels and the distance features can use atr_5.
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_strategy_state", False)):
        swings = int(getattr(feat_cfg, "smt_swing_period", 5))
        sweeps = detect_liquidity_sweeps(primary, swing_period=swings)
        bos = (
            detect_bos(primary, structure)
            if structure is not None
            else detect_bos(
                primary,
                structure_levels(primary, swing_period=swings)[
                    ["last_swing_high", "last_swing_low"]
                ],
            )
        )
        po3_state = build_po3_state(primary, sweeps, levels)
        ifvg_zones = build_ifvg_zone_features(primary)
        feat = pd.concat([feat, sweeps, bos, po3_state, ifvg_zones], axis=1)

        # ATR-normalised distances for the model; the raw levels in the frame
        # above stay available for the environment's structural SL/TP logic
        # and must be excluded from the model sequence (env seq_exclude).
        atr5 = feat["atr_5"].where(feat["atr_5"] > 0)
        feat["asian_range"] = feat["asian_high"] - feat["asian_low"]
        feat["price_to_asian_high_atr"] = (feat["asian_high"] - primary["close"]) / atr5
        feat["price_to_asian_low_atr"] = (primary["close"] - feat["asian_low"]) / atr5
        feat["manipulation_low_distance_atr"] = (
            primary["close"] - feat["po3_manipulation_low"]
        ) / atr5
        feat["manipulation_high_distance_atr"] = (
            feat["po3_manipulation_high"] - primary["close"]
        ) / atr5

        # --- PO3 state MTF (plan §4.4, opt-in, default off) ---
        # One independent state machine per TF on that TF's own resampled
        # bars (own sweeps + own Asian levels), prefixed {TF}_po3_*, aligned
        # onto M1. Ships behind po3_state_mtf.enabled until causality tests
        # (§6.1) and an ablation run justify it as default.
        if _enabled(feat_cfg, "po3_state_mtf", False):
            po3_mtf_blocks: dict[str, pd.DataFrame] = {}
            for tf in _htf_list(feat_cfg, "po3_state_mtf", _MTF_STATE_TFS):
                if str(tf) == "M1":
                    continue  # M1 already present unprefixed
                pri_tf, _ = _resample_pair(primary, None, str(tf))
                if pri_tf.empty:
                    continue
                sweeps_tf = detect_liquidity_sweeps(pri_tf, swing_period=swings)
                cached_lv = _mtf_sweep_levels.get(str(tf))
                if cached_lv is not None and len(cached_lv) == len(pri_tf):
                    po3_lv_tf: pd.DataFrame = cached_lv
                else:
                    po3_lv_tf = detect_session_levels(pri_tf)
                po3_tf = build_po3_state(pri_tf, sweeps_tf, po3_lv_tf)
                po3_mtf_blocks[str(tf)] = po3_tf
            if po3_mtf_blocks:
                # Rename AFTER align: align_timeframes prefixes every column
                # with {TF}_ (e.g. M5_po3_manipulation_active), so renaming
                # per-TF frames beforehand would double-prefix (M5_M5_po3_*).
                feat = align_timeframes(feat, po3_mtf_blocks)

        # --- IFVG active-zone MTF (plan §4.6, opt-in, default off) ---
        # Same resample-loop treatment as FVG §4.5, scoped to M1/M5/M15
        # (IFVG confirmation is a two-step causal sequence, unlike the
        # memoryless FVG zone-distance features).
        if _enabled(feat_cfg, "ifvg_mtf", False):
            ifvg_blocks: dict[str, pd.DataFrame] = {}
            for tf in _htf_list(feat_cfg, "ifvg_mtf", _MTF_STATE_TFS):
                if str(tf) == "M1":
                    continue  # M1 already present unprefixed
                pri_tf, _ = _resample_pair(primary, None, str(tf))
                if pri_tf.empty:
                    continue
                ifvg_blocks[str(tf)] = build_ifvg_zone_features(pri_tf)
            if ifvg_blocks:
                feat = align_timeframes(feat, ifvg_blocks)

    # --- CT-anchored session levels (opt-in) ---
    # Raw price levels like the structure block above, so added AFTER
    # normalization. Windows are defined in America/Chicago wall-clock time
    # and resolved per bar (DST-safe); columns are additive to the legacy
    # broker-tz asian_*/london_* levels and carry a _ct suffix.
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_session_ohlc", False)):
        sess_cfg = feat_cfg.sessions
        sess_tz = str(sess_cfg.get("tz", "America/Chicago"))
        for name in ("asian", "london", "ny"):
            window = sess_cfg[name]
            ohlc = session_ohlc(
                primary,
                str(window["start"]),
                str(window["end"]),
                session_tz=sess_tz,
                prefix=f"{name}_ct",
            )
            feat = pd.concat([feat, ohlc], axis=1)

        yday = prior_period_high_low(primary, freq="D", tz=sess_tz)
        lweek = prior_period_high_low(primary, freq="W", tz=sess_tz)
        feat = pd.concat([feat, yday, lweek], axis=1)

        for range_name in list(sess_cfg.get("quadrant_ranges", ["yesterday", "lastweek"])):
            feat = pd.concat(
                [
                    feat,
                    range_quadrants(
                        feat[f"{range_name}_high"],
                        feat[f"{range_name}_low"],
                        prefix=str(range_name),
                    ),
                ],
                axis=1,
            )

        # Raw VWAP level; same tickvol/session_id contract as vwap_from_session.
        if "tickvol" in primary.columns and "session_id" in primary.columns:
            feat["vwap"] = vwap_level(primary)

    # Drop leading NaNs from warmup
    feat = feat.dropna(how="all")
    feat.columns = feat.columns.map(str)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(cache_path)

    return feat
