"""Feature context helpers used by model observations and walk-forward folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from features.pipeline.cache import feature_cache_path


def reachable_r_series(bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Causal reward room: max take-profit distance divided by the structural stop.

    The side is the distribution direction when it is set, otherwise the
    higher-timeframe bias. A flat bar uses the tighter of the two stops.
    The ratio is scale-free, so it can sit in the model sequence.
    """
    from trading.positions.tp_reach import max_tp_distance

    idx = bars.index
    if "atr_5" not in features.columns or "close" not in bars.columns:
        return pd.Series(0.0, index=idx, name="reachable_r")
    dist = max_tp_distance(bars, features["atr_5"]).reindex(idx)
    close = pd.to_numeric(bars["close"], errors="coerce")
    atr = pd.to_numeric(features["atr_5"].reindex(idx), errors="coerce")
    if "po3_manipulation_low" in features.columns and "po3_manipulation_high" in features.columns:
        risk_long = (
            close - pd.to_numeric(features["po3_manipulation_low"].reindex(idx), errors="coerce")
        ).clip(lower=0.0)
        risk_short = (
            pd.to_numeric(features["po3_manipulation_high"].reindex(idx), errors="coerce") - close
        ).clip(lower=0.0)
    else:
        risk_long = atr.clip(lower=0.0)
        risk_short = atr.clip(lower=0.0)
    if "po3_distribution_direction" in features.columns:
        direction = pd.to_numeric(
            features["po3_distribution_direction"].reindex(idx), errors="coerce"
        ).fillna(0.0)
    elif "htf_day_bias" in features.columns:
        direction = pd.to_numeric(features["htf_day_bias"].reindex(idx), errors="coerce").fillna(
            0.0
        )
    else:
        direction = pd.Series(0.0, index=idx)
    long_a = risk_long.to_numpy(dtype=float)
    short_a = risk_short.to_numpy(dtype=float)
    side = direction.to_numpy(dtype=float)
    risk = np.where(side < 0.0, short_a, long_a)
    tighter = np.minimum(
        np.where(long_a > 0.0, long_a, np.inf),
        np.where(short_a > 0.0, short_a, np.inf),
    )
    risk = np.where(side == 0.0, tighter, risk)
    atr_a = atr.to_numpy(dtype=float)
    missing = ~np.isfinite(risk) | (risk <= 0.0)
    risk = np.where(missing, atr_a, risk)
    room = dist.to_numpy(dtype=float)
    ratio = room / np.where(risk > 1e-8, risk, np.nan)
    ratio = np.clip(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 20.0)
    return pd.Series(ratio, index=idx, name="reachable_r")


def attach_reachable_r(
    obs: pd.DataFrame,
    bars: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Append ``reachable_r`` after raw price columns have been dropped."""
    out = obs.copy()
    out["reachable_r"] = (
        reachable_r_series(bars, features).reindex(obs.index).fillna(0.0).astype(float)
    )
    return out


def ensure_london_atr_distances(feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Add London ATR distances when the PD-context pair is absent.

    ``price_to_london_high_atr = (london_high - close) / atr_5``
    ``price_to_london_low_atr = (close - london_low) / atr_5``
    """
    if "ctx_london_high_dist_atr" in feat.columns and "ctx_london_low_dist_atr" in feat.columns:
        return feat
    needed = ("london_high", "london_low", "atr_5")
    if any(col not in feat.columns for col in needed):
        return feat
    atr5 = feat["atr_5"].where(feat["atr_5"] > 0)
    aligned_close = close.reindex(feat.index)
    feat = feat.copy()
    feat["price_to_london_high_atr"] = (feat["london_high"] - aligned_close) / atr5
    feat["price_to_london_low_atr"] = (aligned_close - feat["london_low"]) / atr5
    return feat


def build_walkforward_fold_features(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None,
    cfg: DictConfig,
    fold_train_bars: pd.DataFrame,
    fold_validation_bars: pd.DataFrame,
    cache_dir: Path | str,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Refit normalization on one walk-forward fold with causal warm-up bars.

    Keep all historical bars through the fold's validation end as indicator
    context, but fit normalization only on the fold's train indices. Later
    folds and the locked evaluation period cannot influence this fitted state.
    """
    if (
        primary.empty
        or fold_train_bars.empty
        or fold_validation_bars.empty
        or not primary.index.is_unique
        or not primary.index.is_monotonic_increasing
    ):
        raise ValueError("Walk-forward requires nonempty, unique, ordered bars")
    train_index = fold_train_bars.index
    validation_index = fold_validation_bars.index
    if (
        not train_index.isin(primary.index).all()
        or not validation_index.isin(primary.index).all()
        or train_index.max() >= validation_index.min()
        or train_index.intersection(validation_index).size
    ):
        raise ValueError("Walk-forward train/validation indices must be ordered and disjoint")

    last_validation_bar = validation_index.max()
    context_primary = primary.loc[:last_validation_bar]
    context_secondary = secondary.loc[:last_validation_bar] if secondary is not None else None
    fold_mask = pd.Series(context_primary.index.isin(train_index), index=context_primary.index)
    cache = feature_cache_path(
        cache_dir,
        symbol,
        cfg,
        context_primary,
        train_mask=fold_mask,
        secondary=context_secondary,
        normalization_mode="frozen_fold",
    )
    from features import build_features

    features = build_features(
        context_primary,
        secondary=context_secondary,
        cfg=cfg,
        train_mask=fold_mask,
        cache_path=cache,
        normalization_mode="frozen_fold",
    )
    return features.loc[train_index], features.loc[validation_index], cache
