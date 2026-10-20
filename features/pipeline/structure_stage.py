"""Swing, structure, and higher-timeframe BOS feature assembly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from features.context.liquidity import detect_bos
from features.technical.structure import (
    TIMEFRAME_CONFIG,
    classify_structure,
    detect_pivots,
    detect_swings,
    structure_levels,
    swing_features,
)
from features.technical.swings import swing_history

ResamplePair = Callable[[str], tuple[pd.DataFrame, pd.DataFrame | None]]
AlignBlocks = Callable[[pd.DataFrame, dict[str, pd.DataFrame]], pd.DataFrame]
TimeframeList = Callable[[object, str, tuple[str, ...]], list[str]]
FeatureEnabled = Callable[[object, str, bool], bool]


def append_structure_features(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_config: Any,
    cfg: DictConfig | None,
    resample_pair: ResamplePair,
    align_blocks: AlignBlocks,
    timeframe_list: TimeframeList,
    enabled: FeatureEnabled,
    default_timeframes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, pd.DataFrame]]:
    """Append M1 and optional MTF structure features in causal source order."""
    if feature_config is None:
        return features, None, {}

    m1_config = TIMEFRAME_CONFIG["M1"]
    left = int(m1_config["left"])
    atr_mult = float(m1_config["atr_mult"])
    structure = structure_levels(primary, swing_period=left, atr_mult=atr_mult)
    structure = structure[["last_swing_high", "last_swing_low"]]
    pivots = detect_pivots(primary, left=left, right=int(m1_config["right"]))
    swings = detect_swings(primary, pivots, atr_mult=atr_mult)
    structure_class = classify_structure(swings)
    observation_window = 60
    if cfg is not None:
        selected_window = OmegaConf.select(cfg, "env.obs_window")
        if selected_window is not None:
            observation_window = int(selected_window)
    features = pd.concat(
        [
            features,
            structure,
            swing_features(primary, swings, structure_class, obs_window=observation_window),
            swing_history(swings, n=3),
        ],
        axis=1,
    )

    mtf_levels: dict[str, pd.DataFrame] = {}
    if not enabled(feature_config, "structure", False):
        return features, structure, mtf_levels

    structure_blocks: dict[str, pd.DataFrame] = {}
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

    for timeframe in timeframe_list(feature_config, "structure", default_timeframes):
        if str(timeframe) == "M1":
            continue
        timeframe_bars, _ = resample_pair(str(timeframe))
        if timeframe_bars.empty:
            continue
        timeframe_config = TIMEFRAME_CONFIG.get(str(timeframe), m1_config)
        timeframe_duration_ms = int(
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
                }[str(timeframe)]
            ).total_seconds()
            * 1000
        )
        levels = structure_levels(
            timeframe_bars,
            swing_period=int(timeframe_config["left"]),
            atr_mult=float(timeframe_config["atr_mult"]),
            swing_candidate_count=swing_candidate_count,
            price_increment=(
                float(candidate_price_increment) if candidate_price_increment is not None else None
            ),
            confirmation_delay_ms=timeframe_duration_ms,
        )
        keep_columns = ["last_swing_high", "last_swing_low"]
        if swing_candidate_count > 1:
            keep_columns.extend(
                f"swing_{side}_candidate_{slot}{suffix}"
                for side in ("high", "low")
                for slot in range(1, swing_candidate_count + 1)
                for suffix in ("", "_pivot_time_ms", "_confirmed_at_ms", "_age_bars")
            )
        levels = levels[keep_columns]
        mtf_levels[str(timeframe)] = levels
        bos = detect_bos(timeframe_bars, levels)[["bos_up", "bos_down"]]
        structure_blocks[str(timeframe)] = pd.concat([levels, bos], axis=1)

    if structure_blocks:
        features = align_blocks(features, structure_blocks)
    return features, structure, mtf_levels


__all__ = ["append_structure_features"]
