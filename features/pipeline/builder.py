"""Feature build pipeline: indicators + SMT + normalisation → feature matrix.

Chain A: per-timeframe technical features. For every timeframe listed in
``cfg.features.htf_timeframes`` the same ``build_indicators()`` call used
for M1 is run on that timeframe's own bars, then causally forward-filled
onto the M1 spine via :func:`market_data.transforms.align.align_timeframes` with
``{TF}_{indicator}`` column names.

MTF extensions (mtf_feature_expansion_plan.md §§4.1-4.6): SMT divergence
(§4.1), structure levels + BOS (§4.2) and liquidity sweeps (§4.3) follow the
same resample → per-TF compute → ``align_timeframes`` pattern; FVG zones
(§4.5) run over the full ladder (M1 + HTFs); PO3 manipulation/distribution
state (§4.4) and IFVG active zones (§4.6) are independently computed on
configured standard timeframes, including M10, then aligned to the M1 spine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configuration.paths import repository_root
from features.context.liquidity import detect_bos, detect_liquidity_sweeps, detect_mss
from features.context.premium_discount import (
    build_htf_pd_distance_features,
    build_pd_context_features,
    build_trader_context_features,
)
from features.context.session_levels import (
    prev_ny_high_low,
    previous_labeled_session_ohlc,
    prior_period_high_low,
    range_quadrants,
    session_ohlc,
)
from features.context.smt import smt_divergence
from features.pipeline.cache import (
    feature_cache_content_hash,
)
from features.pipeline.context import (
    ensure_london_atr_distances,
)
from features.pipeline.normalization import rolling_zscore
from features.po3.config import (
    IFVGConfig,
    detect_po3_entries,
)
from features.po3.entries import build_po3_phase_features
from features.po3.fvg import build_fvg_zone_features
from features.po3.ifvg_state import build_po3_ifvg_state
from features.po3.state import (
    align_ifvg_zone_timeframes,
    build_distribution_after_ifvg,
    build_ifvg_zone_features,
    build_po3_state,
)
from features.technical.htf_sweeps import detect_confirmed_htf_sweeps
from features.technical.indicators import (
    atr,
    build_indicators,
    ny_session_clock,
    sweep_velocity,
    volume_spike,
    vwap_level,
    wick_ratio,
)
from features.technical.structure import (
    TIMEFRAME_CONFIG,
    classify_structure,
    detect_pivots,
    detect_session_levels,
    detect_swings,
    structure_levels,
    swing_features,
)
from features.technical.swings import swing_history
from market_data.transforms.align import align_timeframes, timeframe_close_times
from market_data.transforms.resample import resample

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
# full M1+HTF ladder (§4.5); PO3 state (§4.4) and IFVG zones (§4.6) use
# configured supported timeframes and align onto the M1 spine.
# v8: full-day M1 spine, completed-bar HTF ffill, confirmed ATR swings.
# v10: PD context, VWAP off.
# v11: content-hash cache key (config + data identity + optional train_mask).
# v12: trader-context columns (htf_day_bias, manip_reverses_htf, context_trade_direction).
# v13: tick-count VWAP / activity (ignore CFD vol=0).
# v14: swing age in [0, 1], London ATR distances, NY session clock.
# v15: swing confirmation pulses and the distribution-after-IFVG latch.
# v16: SL modes (state machine + agent veto), prev-NY session H/L, swing
#      history columns (swing_high_{1..3}/swing_low_{1..3}), TF-SMT enablement.
# v22: causal completed prior-NY levels use the configured NY session window.


_DEFAULT_HTF_TIMEFRAMES = ("M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1")
# Capped normalised FVG distance: value used when no zone is active nearby.
_FVG_DIST_CAP = 5.0
# Kept off the z-score path: prices for the stop ladder, and the 0/1 retest
# and active flags the entry gate reads on any timeframe prefix.
_FVG_RAW_STEMS = (
    "fvg_retest_bull",
    "fvg_retest_bear",
    "fvg_bull_active",
    "fvg_bear_active",
    "fvg_bull_low",
    "fvg_bull_high",
    "fvg_bear_low",
    "fvg_bear_high",
    "fvg_bull_origin",
    "fvg_bear_origin",
)

# Numeric encoding of detect_po3_entries' entry_trigger_type string column so
# the feature matrix stays homogeneous float (TradingEnv casts to float32).
_PO3_TRIGGER_CODE = {"": 0, "retest": 1, "close_through": 2, "ltf_fvg": 3}

# Supported fallback timeframes for the state-heavy blocks (plan §8, option
# (b)). Normal configs provide explicit subsets; IFVG can use any timeframe
# supported by data.resample, while its M1 source is always included directly.
_MTF_STATE_TFS = ("M1", "M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1")

# Default MTF subset for SMT divergence (§4.1), structure/BOS (§4.2), and
# liquidity sweeps (§4.3). Each block accepts an explicit list of any standard
# resampled timeframe; the bounded default keeps column growth predictable.
_MTF_LIGHT_TFS = ("M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1")


def _split_fvg_raw(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull FVG prices and retest flags out before the rolling z-score."""
    cols = [
        c
        for c in feat.columns
        if any(str(c) == stem or str(c).endswith("_" + stem) for stem in _FVG_RAW_STEMS)
    ]
    if not cols:
        return feat, pd.DataFrame(index=feat.index)
    return feat.drop(columns=cols), feat.loc[:, cols].copy()


def _htf_list(feat_cfg: object, key: str, default: tuple[str, ...]) -> list[str]:
    """Read a TF subset list (plan §5 nested block) with a safe default.

    Accepts ``features.<key>`` stored either as a nested block
    (``features.context.smt: {timeframes: [...]}``) or as a flat key
    (``features.context.smt_timeframes: [...]``); both merge cleanly through
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
    primary_timeframes: dict[str, pd.DataFrame] | None = None,
    secondary_timeframes: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Select native bars when supplied; otherwise aggregate each symbol's M1.

    The secondary symbol is resampled from its own M1 bars (never from
    primary's bars) so SMT divergence on TF compares like-for-like bars.
    Returns ``(primary_tf, secondary_tf_or_None)``.
    """
    pri_tf = (primary_timeframes or {}).get(tf)
    if pri_tf is None:
        pri_tf = primary if tf == "M1" else resample(primary, tf)  # type: ignore[arg-type]
    sec_tf: pd.DataFrame | None = None
    if secondary is not None:
        sec_tf = (secondary_timeframes or {}).get(tf)
        if sec_tf is None:
            sec_tf = secondary if tf == "M1" else resample(secondary, tf)  # type: ignore[arg-type]
    return pri_tf, sec_tf


def _scheduled_close_map(cfg: DictConfig | None) -> dict[str, str]:
    if cfg is None:
        return {}
    schedule_value = OmegaConf.select(cfg, "data.session_schedule_file")
    if not schedule_value:
        return {}
    schedule_path = Path(str(schedule_value)).expanduser()
    if not schedule_path.is_absolute():
        schedule_path = repository_root(__file__) / schedule_path
    if not schedule_path.is_file():
        raise FileNotFoundError(f"configured session schedule is missing: {schedule_path}")
    schedule = pd.read_csv(schedule_path, dtype=str)
    required = {"session_date", "scheduled_ny_close_broker_time"}
    if not required.issubset(schedule.columns):
        raise ValueError(
            f"session schedule missing columns: {sorted(required - set(schedule.columns))}"
        )
    return dict(
        zip(
            schedule["session_date"].astype(str),
            schedule["scheduled_ny_close_broker_time"].astype(str),
            strict=True,
        )
    )


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


def build_features(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None = None,
    cfg: DictConfig | None = None,
    train_mask: pd.Series | None = None,
    cache_path: Path | None = None,
    force: bool = False,
    normalization_mode: str = "legacy_masked_window",
    timeframe_bars: dict[str, pd.DataFrame] | None = None,
    secondary_timeframe_bars: dict[str, pd.DataFrame] | None = None,
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
    expected_hash = feature_cache_content_hash(
        cfg,
        primary,
        train_mask=train_mask,
        secondary=secondary,
        normalization_mode=normalization_mode,
        timeframe_bars=timeframe_bars,
        secondary_timeframe_bars=secondary_timeframe_bars,
    )
    requested_reference_mode = str(
        (OmegaConf.select(cfg, "features.po3_reference_mode") if cfg is not None else None)
        or "m1_legacy"
    )
    sweep_audit_path = Path(str(cache_path) + ".htf_sweeps.parquet") if cache_path else None
    if cache_path and Path(cache_path).exists() and not force:
        meta_path = Path(str(cache_path) + ".hash")
        audit_ready = requested_reference_mode != "htf_confirmed" or (
            sweep_audit_path is not None and sweep_audit_path.exists()
        )
        if (
            audit_ready
            and meta_path.exists()
            and meta_path.read_text(encoding="utf-8").strip() == expected_hash
        ):
            return pd.read_parquet(cache_path)
        # Stale or pre-hash cache: rebuild rather than silently reuse.

    feat_cfg = cfg.features if cfg is not None else None
    secondary_m1: pd.DataFrame | None = secondary  # M1 copy kept for per-TF resampling
    primary_timeframes = {str(tf).upper(): frame for tf, frame in (timeframe_bars or {}).items()}
    primary_timeframes["M1"] = primary
    secondary_timeframe_frames = {
        str(tf).upper(): frame for tf, frame in (secondary_timeframe_bars or {}).items()
    }
    if secondary is not None:
        secondary_timeframe_frames["M1"] = secondary
    scheduled_closes = _scheduled_close_map(cfg)
    session_start = str(OmegaConf.select(cfg, "session.start") or "16:30") if cfg else "16:30"

    def primary_tf_bars(tf: str) -> pd.DataFrame:
        key = str(tf).upper()
        frame = primary_timeframes.get(key)
        if frame is None:
            frame = primary if key == "M1" else resample(primary, key)  # type: ignore[arg-type]
            primary_timeframes[key] = frame
        return frame

    def align_blocks(
        target: pd.DataFrame,
        blocks: dict[str, pd.DataFrame],
        *,
        completed_bars: bool = True,
    ) -> pd.DataFrame:
        availability = {
            str(tf): timeframe_close_times(
                primary_tf_bars(str(tf)), str(tf), scheduled_closes, session_start
            )
            for tf in blocks
        }
        return align_timeframes(
            target,
            blocks,
            completed_bars=completed_bars,
            availability_times=availability,
        )

    htf_sweep_audit = pd.DataFrame()

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
                pri_tf, sec_tf = _resample_pair(
                    primary,
                    secondary_m1,
                    str(tf),
                    primary_timeframes,
                    secondary_timeframe_frames,
                )
                if sec_tf is None or pri_tf.empty or sec_tf.empty:
                    continue
                smt_blocks[str(tf)] = smt_divergence(
                    pri_tf,
                    sec_tf,
                    swing_period=int(getattr(feat_cfg, "smt_swing_period", 5)),
                    corr_window=int(getattr(feat_cfg, "smt_corr_window", 20)),
                )
            if smt_blocks:
                feat = align_blocks(feat, smt_blocks)

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
                tf_bars = primary_tf_bars(str(tf))
                htf_blocks[str(tf)] = build_indicators(tf_bars, feat_cfg)
            feat = align_blocks(feat, htf_blocks)

    # --- Chain B variant blocks (opt-in via config flags) ---
    _fvg_raw = pd.DataFrame(index=primary.index)
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
                tf_bars = primary_tf_bars(str(tf))
                fvg_blocks[str(tf)] = build_fvg_zone_features(tf_bars)
            feat = align_blocks(feat, fvg_blocks)
            # Zone prices and the retest flag must stay raw. The distance
            # columns above are normalised with the rest of the matrix.
            feat, _fvg_raw = _split_fvg_raw(feat)

    # --- normalisation ---
    window = feat_cfg.zscore_window if feat_cfg is not None else 252
    feat = rolling_zscore(feat, window=window, train_mask=train_mask, mode=normalization_mode)
    if len(_fvg_raw.columns) > 0:
        feat = pd.concat([feat, _fvg_raw], axis=1)

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
        obs_window = 60
        if cfg is not None:
            selected = OmegaConf.select(cfg, "env.obs_window")
            if selected is not None:
                obs_window = int(selected)
        feat = pd.concat(
            [
                feat,
                structure,
                swing_features(primary, swings_df, struct_cls, obs_window=obs_window),
                swing_history(swings_df, n=3),
            ],
            axis=1,
        )
        if _enabled(feat_cfg, "structure", False):
            struct_blocks: dict[str, pd.DataFrame] = {}
            struct_levels: dict[str, pd.DataFrame] = {}
            swing_candidate_count = int(
                OmegaConf.select(cfg, "features.structure.swing_candidate_count", default=1)
                if cfg is not None
                else 1
            )
            if not 1 <= swing_candidate_count <= 3:
                raise ValueError("features.structure.swing_candidate_count must be between 1 and 3")
            candidate_price_increment = (
                OmegaConf.select(cfg, "features.structure.candidate_price_increment", default=None)
                if cfg is not None
                else None
            )
            if candidate_price_increment is None and cfg is not None:
                candidate_price_increment = OmegaConf.select(cfg, "costs.point_size", default=None)
            for tf in _htf_list(feat_cfg, "structure", _MTF_LIGHT_TFS):
                if str(tf) == "M1":
                    continue
                pri_tf, _ = _resample_pair(primary, None, str(tf), primary_timeframes)
                if pri_tf.empty:
                    continue
                tf_cfg = TIMEFRAME_CONFIG.get(str(tf), m1_cfg)
                tf_duration_ms = int(
                    pd.Timedelta(
                        {
                            "M1": "1min",
                            "M3": "3min",
                            "M5": "5min",
                            "M10": "10min",
                            "M15": "15min",
                            "M30": "30min",
                            "H1": "1h",
                            "H4": "4h",
                            "D1": "1d",
                        }[str(tf)]
                    ).total_seconds()
                    * 1000
                )
                lv_tf = structure_levels(
                    pri_tf,
                    swing_period=int(tf_cfg["left"]),
                    atr_mult=float(tf_cfg["atr_mult"]),
                    swing_candidate_count=swing_candidate_count,
                    price_increment=(
                        float(candidate_price_increment)
                        if candidate_price_increment is not None
                        else None
                    ),
                    confirmation_delay_ms=tf_duration_ms,
                )
                keep_columns = ["last_swing_high", "last_swing_low"]
                if swing_candidate_count > 1:
                    keep_columns.extend(
                        f"swing_{side}_candidate_{slot}{suffix}"
                        for side in ("high", "low")
                        for slot in range(1, swing_candidate_count + 1)
                        for suffix in ("", "_pivot_time_ms", "_confirmed_at_ms", "_age_bars")
                    )
                lv_tf = lv_tf[keep_columns]
                struct_levels[str(tf)] = lv_tf
                bos_tf = detect_bos(pri_tf, lv_tf)[["bos_up", "bos_down"]]
                struct_blocks[str(tf)] = pd.concat([lv_tf, bos_tf], axis=1)
            if struct_blocks:
                feat = align_blocks(feat, struct_blocks)
                _mtf_struct_levels = struct_levels

    # --- Full PO3 pipeline (Chain F): HTF FVG -> LTF IFVG -> entry triggers ---
    # Added AFTER normalization so the 0/1 entry signals and the numeric
    # trigger code keep their raw meaning (exactly like structure levels).
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_po3_full", False)):
        po3_htf = str(getattr(feat_cfg, "po3_htf", "M15")).upper()
        po3_ltf = str(getattr(feat_cfg, "po3_ltf", "M5")).upper()
        po3_signals = detect_po3_entries(
            primary,
            htf=po3_htf,
            primary_tf=po3_ltf,
            timeframe_bars=primary_timeframes,
            availability_times={
                po3_htf: timeframe_close_times(
                    primary_tf_bars(po3_htf), po3_htf, scheduled_closes, session_start
                ),
                po3_ltf: timeframe_close_times(
                    primary_tf_bars(po3_ltf), po3_ltf, scheduled_closes, session_start
                ),
            },
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
    from market_data.transforms.activity import activity_series

    vol_s = activity_series(primary)
    if vol_s is not None:
        feat["volume_spike"] = volume_spike(vol_s, window=20)
    feat["atr_5"] = atr(primary, period=5)

    # --- Session-conditional PD context (opt-in) ---
    include_pd = feat_cfg is not None and bool(getattr(feat_cfg, "include_pd_context", False))
    if include_pd:
        sess_tz = "Etc/GMT-3"
        if cfg is not None:
            sess_tz = str(
                OmegaConf.select(cfg, "strategy.session.timezone")
                or OmegaConf.select(cfg, "session.tz")
                or OmegaConf.select(cfg, "data.tz")
                or sess_tz
            )
        pd_ctx = build_pd_context_features(primary, feat["atr_5"], tz=sess_tz)
        feat = pd.concat([feat, pd_ctx], axis=1)

        # HTF swing / session-range PD distances onto M1 (beyond indicator copies).
        htf_cfg_tfs = getattr(feat_cfg, "htf_timeframes", None)
        htf_tfs = list(htf_cfg_tfs) if htf_cfg_tfs is not None else list(_DEFAULT_HTF_TIMEFRAMES)
        pd_htf_blocks: dict[str, pd.DataFrame] = {}
        for tf in htf_tfs:
            pri_tf, _ = _resample_pair(primary, None, str(tf), primary_timeframes)
            if pri_tf.empty:
                continue
            tf_cfg = TIMEFRAME_CONFIG.get(str(tf), TIMEFRAME_CONFIG["M1"])
            pd_htf_blocks[str(tf)] = build_htf_pd_distance_features(
                pri_tf,
                feat["atr_5"],
                swing_period=int(tf_cfg["left"]),
                atr_mult=float(tf_cfg["atr_mult"]),
            )
        if pd_htf_blocks:
            feat = align_blocks(feat, pd_htf_blocks)

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
            pri_tf, _ = _resample_pair(primary, None, str(tf), primary_timeframes)
            if pri_tf.empty:
                continue
            sweeps_tf = detect_liquidity_sweeps(pri_tf, swing_period=swing)
            tf_cfg = TIMEFRAME_CONFIG.get(str(tf), TIMEFRAME_CONFIG["M1"])
            pivots_tf = detect_pivots(pri_tf, left=int(tf_cfg["left"]), right=int(tf_cfg["right"]))
            swings_tf = detect_swings(pri_tf, pivots_tf, atr_mult=float(tf_cfg["atr_mult"]))
            events_tf = swings_tf[["swing_high_event", "swing_low_event"]].astype(float)
            sweep_blocks[str(tf)] = pd.concat([sweeps_tf, events_tf], axis=1)
            _mtf_sweep_levels[str(tf)] = detect_session_levels(pri_tf)
        if sweep_blocks:
            feat = align_blocks(feat, sweep_blocks)

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
        reference_mode = str(getattr(feat_cfg, "po3_reference_mode", "m1_legacy"))
        if reference_mode == "htf_confirmed":
            reference_timeframes = tuple(_htf_list(feat_cfg, "po3_reference", ("M5", "M15", "H1")))
            sweeps, htf_sweep_audit = detect_confirmed_htf_sweeps(
                primary,
                timeframes=reference_timeframes,
                return_audit=True,
            )
        elif reference_mode == "m1_legacy":
            sweeps = detect_liquidity_sweeps(primary, swing_period=swings)
        else:
            raise ValueError(
                "features.po3_reference_mode must be 'm1_legacy' or 'htf_confirmed', "
                f"got {reference_mode!r}"
            )
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
        phase_mode = str(getattr(feat_cfg, "po3_phase_mode", "legacy"))
        if phase_mode not in {"legacy", "reclaim_then_ifvg"}:
            raise ValueError(
                "features.po3_phase_mode must be 'legacy' or 'reclaim_then_ifvg', "
                f"got {phase_mode!r}"
            )
        close_threshold = (
            OmegaConf.select(cfg, "strategy.ifvg.close_through_threshold")
            if cfg is not None
            else None
        )
        inverse_opposing = bool(
            (
                OmegaConf.select(cfg, "strategy.ifvg.require_opposing_fvg")
                if cfg is not None
                else False
            )
            or False
        )
        ifvg_config = (
            IFVGConfig(
                close_through_threshold=float(close_threshold),
                inverse_opposing_fvg=inverse_opposing,
            )
            if close_threshold is not None
            else IFVGConfig(inverse_opposing_fvg=inverse_opposing)
        )
        ifvg_zones = build_ifvg_zone_features(primary, ifvg_config=ifvg_config)
        state_ifvg_zones = ifvg_zones.copy()
        state_ifvg_zones["_ifvg_bull_confirmation_tf_code"] = np.where(
            state_ifvg_zones["ifvg_bull_confirmed"] > 0.0, 1.0, 0.0
        )
        state_ifvg_zones["_ifvg_bear_confirmation_tf_code"] = np.where(
            state_ifvg_zones["ifvg_bear_confirmed"] > 0.0, 1.0, 0.0
        )
        ifvg_mtf_blocks = pd.DataFrame(index=primary.index)
        if _enabled(feat_cfg, "ifvg_mtf", False):
            higher_ifvg: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
            for tf in _htf_list(feat_cfg, "ifvg_mtf", _MTF_STATE_TFS):
                tf = str(tf).upper()
                if tf == "M1":
                    continue  # M1 zones are already the primary block.
                pri_tf, _ = _resample_pair(primary, None, tf, primary_timeframes)
                if pri_tf.empty:
                    continue
                higher_ifvg[tf] = (
                    pri_tf,
                    build_ifvg_zone_features(pri_tf, ifvg_config=ifvg_config),
                )
            state_ifvg_zones, ifvg_mtf_blocks = align_ifvg_zone_timeframes(
                primary,
                ifvg_zones,
                higher_ifvg,
                availability_times={
                    tf: timeframe_close_times(tf_bars, tf, scheduled_closes, session_start)
                    for tf, (tf_bars, _) in higher_ifvg.items()
                },
            )
        if phase_mode == "reclaim_then_ifvg":
            po3_state = build_po3_ifvg_state(
                primary,
                sweeps,
                state_ifvg_zones,
                max_ifvg_age_bars=int(
                    (
                        OmegaConf.select(cfg, "strategy.ifvg.max_age_bars")
                        if cfg is not None
                        else None
                    )
                    or 50
                ),
            )
        else:
            po3_state = build_po3_state(primary, sweeps, levels)
        bull_conf = state_ifvg_zones["ifvg_bull_confirmed"].to_numpy(dtype=float)
        bear_conf = state_ifvg_zones["ifvg_bear_confirmed"].to_numpy(dtype=float)
        zone_cols = [c for c in ifvg_zones.columns if not str(c).endswith("_confirmed")]
        feat = pd.concat(
            [feat, sweeps, bos, po3_state, ifvg_zones[zone_cols]],
            axis=1,
        )
        feat["ifvg_bull_confirmed"] = bull_conf
        feat["ifvg_bear_confirmed"] = bear_conf
        feat["ifvg_bull_confirmation_tf_code"] = state_ifvg_zones[
            "_ifvg_bull_confirmation_tf_code"
        ].to_numpy(dtype=float)
        feat["ifvg_bear_confirmation_tf_code"] = state_ifvg_zones[
            "_ifvg_bear_confirmation_tf_code"
        ].to_numpy(dtype=float)
        if not ifvg_mtf_blocks.empty:
            feat = pd.concat([feat, ifvg_mtf_blocks], axis=1)

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
                pri_tf, _ = _resample_pair(primary, None, str(tf), primary_timeframes)
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
                feat = align_blocks(feat, po3_mtf_blocks)

        if phase_mode == "legacy":
            feat["po3_distribution_after_ifvg"] = build_distribution_after_ifvg(
                po3_state["po3_distribution_direction"].to_numpy(dtype=float),
                po3_state["po3_manipulation_active"].to_numpy(dtype=float),
                po3_state["po3_manipulation_direction"].to_numpy(dtype=float),
                bull_conf,
                bear_conf,
            )

    # --- LTF BOS + MSS decision stack ---
    # With include_strategy_state, BOS is already present; with include_pd_context
    # alone we add BOS here. MSS is always paired when either flag is on.
    need_ltf_stack = feat_cfg is not None and (
        bool(getattr(feat_cfg, "include_strategy_state", False))
        or bool(getattr(feat_cfg, "include_pd_context", False))
    )
    if need_ltf_stack and structure is not None:
        atr5_ltf = feat["atr_5"].where(feat["atr_5"] > 0)
        if "bos_up" not in feat.columns:
            bos_m1 = detect_bos(primary, structure)
            feat = pd.concat([feat, bos_m1], axis=1)
        if "bos_up_dist_atr" not in feat.columns:
            feat["bos_up_dist_atr"] = (feat["bos_up_level"] - primary["close"]) / atr5_ltf
            feat["bos_down_dist_atr"] = (primary["close"] - feat["bos_down_level"]) / atr5_ltf
        if "mss_up" not in feat.columns:
            bos_for_mss = feat[["bos_up", "bos_down", "bos_up_level", "bos_down_level"]]
            mss_m1 = detect_mss(primary, structure, bos_for_mss)
            feat = pd.concat([feat, mss_m1], axis=1)
        if "mss_up_dist_atr" not in feat.columns:
            feat["mss_up_dist_atr"] = (feat["mss_up_level"] - primary["close"]) / atr5_ltf
            feat["mss_down_dist_atr"] = (primary["close"] - feat["mss_down_level"]) / atr5_ltf

    # --- CT-anchored session levels (opt-in) ---
    # Raw price levels like the structure block above, so added AFTER
    # normalization. Windows are defined in America/Chicago wall-clock time
    # and resolved per bar (DST-safe); columns are additive to the legacy
    # broker-tz asian_*/london_* levels and carry a _ct suffix.
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_session_ohlc", False)):
        sess_cfg = feat_cfg.sessions
        sess_tz = str(sess_cfg.get("tz", "America/Chicago"))
        source_clock_model = str(
            (
                OmegaConf.select(cfg, "data.timestamp_clock", default="native")
                if cfg is not None
                else "native"
            )
            or "native"
        )
        for name in ("asian", "london", "ny"):
            window = sess_cfg[name]
            ohlc = session_ohlc(
                primary,
                str(window["start"]),
                str(window["end"]),
                session_tz=sess_tz,
                prefix=f"{name}_ct",
                source_clock_model=source_clock_model,
            )
            feat = pd.concat([feat, ohlc], axis=1)

        yday = prior_period_high_low(
            primary, freq="D", tz=sess_tz, source_clock_model=source_clock_model
        )
        lweek = prior_period_high_low(
            primary, freq="W", tz=sess_tz, source_clock_model=source_clock_model
        )
        ny_window = sess_cfg["ny"]
        prev_ny = prev_ny_high_low(
            primary,
            start_ct=str(ny_window["start"]),
            end_ct=str(ny_window["end"]),
            session_tz=sess_tz,
            source_clock_model=source_clock_model,
        )
        feat = pd.concat([feat, yday, lweek, prev_ny], axis=1)

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

        # Raw VWAP level; tick-count weights via activity_series.
        from market_data.transforms.activity import activity_series

        if (
            bool(getattr(feat_cfg, "vwap_session", False))
            and activity_series(primary) is not None
            and "session_id" in primary.columns
        ):
            feat["vwap"] = vwap_level(primary)

    # Separate from running session levels: Idea 2 reads only the latest
    # configured session whose final expected bar has already closed.
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_previous_session_pd", False)):
        sess_cfg = feat_cfg.sessions
        sess_tz = str(sess_cfg.get("tz", "America/Chicago"))
        windows = {
            name: {"start": str(sess_cfg[name]["start"]), "end": str(sess_cfg[name]["end"])}
            for name in ("asian", "london", "ny")
        }
        feat = pd.concat(
            [
                feat,
                previous_labeled_session_ohlc(
                    primary,
                    windows,
                    session_tz=sess_tz,
                    source_clock_model=source_clock_model,
                ),
            ],
            axis=1,
        )

    # --- Trader-context HTF bias / manip reverse / trade direction ---
    # Wired when PD context or strategy state is on (Idea 1/2).
    need_trader_ctx = feat_cfg is not None and (
        bool(getattr(feat_cfg, "include_strategy_state", False))
        or bool(getattr(feat_cfg, "include_pd_context", False))
    )
    if need_trader_ctx:
        trader_ctx = build_trader_context_features(feat)
        for col in trader_ctx.columns:
            feat[col] = trader_ctx[col]

    feat = ensure_london_atr_distances(feat, primary["close"])
    # Raw decision price is retained only for the opt-in Idea 2 entry gate
    # and excluded from the model observation sequence.
    if feat_cfg is not None and bool(getattr(feat_cfg, "include_previous_session_pd", False)):
        feat["decision_close"] = primary["close"].reindex(feat.index)
    session = OmegaConf.select(cfg, "session") if cfg is not None else None
    session_start = str(session.get("start", "16:30")) if session else "16:30"
    session_end = str(session.get("end", "23:00")) if session else "23:00"
    clock = ny_session_clock(pd.DatetimeIndex(feat.index), session_start, session_end)
    feat = pd.concat([feat, clock], axis=1)

    # Drop leading NaNs from warmup
    feat = feat.dropna(how="all")
    feat.columns = feat.columns.map(str)

    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(path)
        if requested_reference_mode == "htf_confirmed" and sweep_audit_path is not None:
            htf_sweep_audit.to_parquet(sweep_audit_path)
        path.with_name(path.name + ".hash").write_text(expected_hash + "\n", encoding="utf-8")

    return feat
