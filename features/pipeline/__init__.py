"""Causal feature assembly, observation selection, normalization, and cache identity."""

from .builder import build_features
from .cache import FEATURE_CACHE_VERSION, feature_cache_content_hash, feature_cache_path
from .context import (
    attach_reachable_r,
    build_walkforward_fold_features,
    ensure_london_atr_distances,
)
from .normalization import rolling_zscore
from .schema import OBS_RAW_PRICE_COLUMNS, select_legacy_obs_columns, select_obs_columns

__all__ = [
    "FEATURE_CACHE_VERSION",
    "OBS_RAW_PRICE_COLUMNS",
    "attach_reachable_r",
    "build_features",
    "build_walkforward_fold_features",
    "ensure_london_atr_distances",
    "feature_cache_content_hash",
    "feature_cache_path",
    "rolling_zscore",
    "select_legacy_obs_columns",
    "select_obs_columns",
]
