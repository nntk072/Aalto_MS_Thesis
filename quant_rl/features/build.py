"""Feature build pipeline: indicators + SMT + normalisation → feature matrix.

Chain A: per-timeframe technical features. For every timeframe listed in
``cfg.features.htf_timeframes`` the same ``build_indicators()`` call used
for M1 is run on that timeframe's own bars, then causally forward-filled
onto the M1 spine via :func:`quant_rl.data.align.align_timeframes` with
``{TF}_{indicator}`` column names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from ..data.align import align_timeframes
from ..data.resample import resample
from .indicators import atr, build_indicators, sweep_velocity, volume_spike, wick_ratio
from .normalize import rolling_zscore
from .smt import smt_divergence
from .structure import detect_session_levels, structure_levels

# Bump whenever build_features() output schema changes so stale caches are
# not silently reused by the {symbol}_features.parquet call sites.
FEATURE_CACHE_VERSION = "v2-htf"

_DEFAULT_HTF_TIMEFRAMES = ("M5", "M15", "H1")


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
        import numpy as np

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

    # --- normalisation ---
    window = feat_cfg.zscore_window if feat_cfg is not None else 252
    feat = rolling_zscore(feat, window=window, train_mask=train_mask)

    # --- Structure levels (swings) - add AFTER normalization to keep raw prices ---
    if feat_cfg is not None:
        structure = structure_levels(primary, swing_period=feat_cfg.smt_swing_period)
        # Drop time columns (not needed in feature matrix)
        structure = structure[["last_swing_high", "last_swing_low"]]
        feat = pd.concat([feat, structure], axis=1)

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
