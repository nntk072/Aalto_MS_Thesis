"""Feature build pipeline: indicators + SMT + normalisation → feature matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from .indicators import build_indicators
from .normalize import rolling_zscore
from .smt import smt_divergence
from .structure import structure_levels


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

    # --- normalisation ---
    window = feat_cfg.zscore_window if feat_cfg is not None else 252
    feat = rolling_zscore(feat, window=window, train_mask=train_mask)

    # --- Structure levels (swings) - add AFTER normalization to keep raw prices ---
    if feat_cfg is not None:
        structure = structure_levels(primary, swing_period=feat_cfg.smt_swing_period)
        # Drop time columns (not needed in feature matrix)
        structure = structure[["last_swing_high", "last_swing_low"]]
        feat = pd.concat([feat, structure], axis=1)

    # Drop leading NaNs from warmup
    feat = feat.dropna(how="all")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(cache_path)

    return feat
