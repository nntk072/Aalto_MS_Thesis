"""PO3 and IFVG state feature assembly after market context is available."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from features.context.liquidity import detect_bos, detect_liquidity_sweeps
from features.po3.ifvg_detection import IFVGConfig
from features.po3.ifvg_state import build_po3_ifvg_state
from features.po3.state import (
    align_ifvg_zone_timeframes,
    build_distribution_after_ifvg,
    build_ifvg_zone_features,
    build_po3_state,
)
from features.technical.htf_sweeps import detect_confirmed_htf_sweeps
from features.technical.structure import detect_session_levels, structure_levels
from market_data.transforms.align import timeframe_close_times
from market_data.transforms.resample import resample

_MTF_STATE_TFS = ("M1", "M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1")

AlignBlocks = Callable[[pd.DataFrame, dict[str, pd.DataFrame]], pd.DataFrame]


def append_po3_state_features(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_config: Any,
    cfg: DictConfig | None,
    primary_timeframes: dict[str, pd.DataFrame],
    scheduled_closes: dict[str, str],
    session_start: str,
    levels: pd.DataFrame,
    structure: pd.DataFrame | None,
    mtf_sweep_levels: dict[str, pd.DataFrame],
    align_blocks: AlignBlocks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append configured PO3/IFVG state and return its optional sweep audit."""
    sweep_audit = pd.DataFrame()
    if feature_config is None or not bool(getattr(feature_config, "include_strategy_state", False)):
        return features, sweep_audit

    def primary_tf_bars(timeframe: str) -> pd.DataFrame:
        key = str(timeframe).upper()
        frame = primary_timeframes.get(key)
        if frame is None:
            frame = primary if key == "M1" else resample(primary, key)  # type: ignore[arg-type]
            primary_timeframes[key] = frame
        return frame

    def htf_list(key: str, default: tuple[str, ...]) -> list[str]:
        block = getattr(feature_config, key, None)
        configured = (
            getattr(block, "timeframes")
            if block is not None and hasattr(block, "timeframes")
            else getattr(feature_config, f"{key}_timeframes", None)
        )
        if configured is None:
            return list(default)
        timeframes = [str(value) for value in list(configured)]
        return timeframes or list(default)

    def enabled(key: str, default: bool) -> bool:
        block = getattr(feature_config, key, None)
        if block is not None and hasattr(block, "enabled"):
            return bool(getattr(block, "enabled"))
        value = getattr(feature_config, f"{key}_enabled", None)
        return bool(value) if value is not None else default

    def resample_pair(timeframe: str) -> pd.DataFrame:
        key = str(timeframe).upper()
        frame = primary_timeframes.get(key)
        if frame is None:
            frame = resample(primary, key)  # type: ignore[arg-type]
            primary_timeframes[key] = frame
        return frame

    swings = int(getattr(feature_config, "smt_swing_period", 5))
    reference_mode = str(getattr(feature_config, "po3_reference_mode", "m1_legacy"))
    if reference_mode == "htf_confirmed":
        reference_timeframes = tuple(htf_list("po3_reference", ("M5", "M15", "H1")))
        sweeps, sweep_audit = detect_confirmed_htf_sweeps(
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
            structure_levels(primary, swing_period=swings)[["last_swing_high", "last_swing_low"]],
        )
    )
    phase_mode = str(getattr(feature_config, "po3_phase_mode", "legacy"))
    if phase_mode not in {"legacy", "reclaim_then_ifvg"}:
        raise ValueError(
            f"features.po3_phase_mode must be 'legacy' or 'reclaim_then_ifvg', got {phase_mode!r}"
        )
    close_threshold = (
        OmegaConf.select(cfg, "strategy.ifvg.close_through_threshold") if cfg else None
    )
    inverse_opposing = bool(
        (OmegaConf.select(cfg, "strategy.ifvg.require_opposing_fvg") if cfg else False) or False
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
    if enabled("ifvg_mtf", False):
        higher_ifvg: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for timeframe in htf_list("ifvg_mtf", _MTF_STATE_TFS):
            if timeframe == "M1":
                continue
            timeframe_bars = resample_pair(timeframe)
            if timeframe_bars.empty:
                continue
            higher_ifvg[timeframe] = (
                timeframe_bars,
                build_ifvg_zone_features(timeframe_bars, ifvg_config=ifvg_config),
            )
        state_ifvg_zones, ifvg_mtf_blocks = align_ifvg_zone_timeframes(
            primary,
            ifvg_zones,
            higher_ifvg,
            availability_times={
                timeframe: timeframe_close_times(
                    timeframe_bars, timeframe, scheduled_closes, session_start
                )
                for timeframe, (timeframe_bars, _) in higher_ifvg.items()
            },
        )

    if phase_mode == "reclaim_then_ifvg":
        po3_state = build_po3_ifvg_state(
            primary,
            sweeps,
            state_ifvg_zones,
            max_ifvg_age_bars=int(
                (OmegaConf.select(cfg, "strategy.ifvg.max_age_bars") if cfg else None) or 50
            ),
        )
    else:
        po3_state = build_po3_state(primary, sweeps, levels)

    bull_confirmation = state_ifvg_zones["ifvg_bull_confirmed"].to_numpy(dtype=float)
    bear_confirmation = state_ifvg_zones["ifvg_bear_confirmed"].to_numpy(dtype=float)
    zone_columns = [
        column for column in ifvg_zones.columns if not str(column).endswith("_confirmed")
    ]
    features = pd.concat([features, sweeps, bos, po3_state, ifvg_zones[zone_columns]], axis=1)
    features["ifvg_bull_confirmed"] = bull_confirmation
    features["ifvg_bear_confirmed"] = bear_confirmation
    features["ifvg_bull_confirmation_tf_code"] = state_ifvg_zones[
        "_ifvg_bull_confirmation_tf_code"
    ].to_numpy(dtype=float)
    features["ifvg_bear_confirmation_tf_code"] = state_ifvg_zones[
        "_ifvg_bear_confirmation_tf_code"
    ].to_numpy(dtype=float)
    if not ifvg_mtf_blocks.empty:
        features = pd.concat([features, ifvg_mtf_blocks], axis=1)

    atr5 = features["atr_5"].where(features["atr_5"] > 0)
    features["asian_range"] = features["asian_high"] - features["asian_low"]
    features["price_to_asian_high_atr"] = (features["asian_high"] - primary["close"]) / atr5
    features["price_to_asian_low_atr"] = (primary["close"] - features["asian_low"]) / atr5
    features["manipulation_low_distance_atr"] = (
        primary["close"] - features["po3_manipulation_low"]
    ) / atr5
    features["manipulation_high_distance_atr"] = (
        features["po3_manipulation_high"] - primary["close"]
    ) / atr5

    if enabled("po3_state_mtf", False):
        po3_mtf_blocks: dict[str, pd.DataFrame] = {}
        for timeframe in htf_list("po3_state_mtf", _MTF_STATE_TFS):
            if timeframe == "M1":
                continue
            timeframe_bars = resample_pair(timeframe)
            if timeframe_bars.empty:
                continue
            timeframe_sweeps = detect_liquidity_sweeps(timeframe_bars, swing_period=swings)
            cached_levels = mtf_sweep_levels.get(timeframe)
            po3_levels = (
                cached_levels
                if cached_levels is not None and len(cached_levels) == len(timeframe_bars)
                else detect_session_levels(timeframe_bars)
            )
            po3_mtf_blocks[timeframe] = build_po3_state(
                timeframe_bars, timeframe_sweeps, po3_levels
            )
        if po3_mtf_blocks:
            features = align_blocks(features, po3_mtf_blocks)

    if phase_mode == "legacy":
        features["po3_distribution_after_ifvg"] = build_distribution_after_ifvg(
            po3_state["po3_distribution_direction"].to_numpy(dtype=float),
            po3_state["po3_manipulation_active"].to_numpy(dtype=float),
            po3_state["po3_manipulation_direction"].to_numpy(dtype=float),
            bull_confirmation,
            bear_confirmation,
        )
    return features, sweep_audit


__all__ = ["append_po3_state_features"]
