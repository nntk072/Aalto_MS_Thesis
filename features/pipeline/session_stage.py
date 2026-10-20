"""Session-level feature assembly after normalized indicators are built."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from features.context.session_levels import (
    prev_ny_high_low,
    previous_labeled_session_ohlc,
    prior_period_high_low,
    range_quadrants,
    session_ohlc,
)
from features.technical.indicators import vwap_level
from market_data.transforms.activity import activity_series


def append_session_features(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    cfg: DictConfig | None,
    feature_config: Any,
) -> pd.DataFrame:
    """Add configured session levels and prior-session context.

    The source clock is resolved independently of the running-session feature
    flag because previous-session context can be enabled on its own.
    """
    if feature_config is None:
        return features

    source_clock_model = str(
        (
            OmegaConf.select(cfg, "data.timestamp_clock", default="native")
            if cfg is not None
            else None
        )
        or "native"
    )
    if bool(getattr(feature_config, "include_session_ohlc", False)):
        session_config = feature_config.sessions
        session_timezone = str(session_config.get("tz", "America/Chicago"))
        for name in ("asian", "london", "ny"):
            window = session_config[name]
            features = pd.concat(
                [
                    features,
                    session_ohlc(
                        primary,
                        str(window["start"]),
                        str(window["end"]),
                        session_tz=session_timezone,
                        prefix=f"{name}_ct",
                        source_clock_model=source_clock_model,
                    ),
                ],
                axis=1,
            )

        yesterday = prior_period_high_low(
            primary, freq="D", tz=session_timezone, source_clock_model=source_clock_model
        )
        last_week = prior_period_high_low(
            primary, freq="W", tz=session_timezone, source_clock_model=source_clock_model
        )
        ny_window = session_config["ny"]
        previous_ny = prev_ny_high_low(
            primary,
            start_ct=str(ny_window["start"]),
            end_ct=str(ny_window["end"]),
            session_tz=session_timezone,
            source_clock_model=source_clock_model,
        )
        features = pd.concat([features, yesterday, last_week, previous_ny], axis=1)

        for range_name in list(session_config.get("quadrant_ranges", ["yesterday", "lastweek"])):
            features = pd.concat(
                [
                    features,
                    range_quadrants(
                        features[f"{range_name}_high"],
                        features[f"{range_name}_low"],
                        prefix=str(range_name),
                    ),
                ],
                axis=1,
            )

        if (
            bool(getattr(feature_config, "vwap_session", False))
            and activity_series(primary) is not None
            and "session_id" in primary.columns
        ):
            features["vwap"] = vwap_level(primary)

    if bool(getattr(feature_config, "include_previous_session_pd", False)):
        session_config = feature_config.sessions
        session_timezone = str(session_config.get("tz", "America/Chicago"))
        windows = {
            name: {
                "start": str(session_config[name]["start"]),
                "end": str(session_config[name]["end"]),
            }
            for name in ("asian", "london", "ny")
        }
        features = pd.concat(
            [
                features,
                previous_labeled_session_ohlc(
                    primary,
                    windows,
                    session_tz=session_timezone,
                    source_clock_model=source_clock_model,
                ),
            ],
            axis=1,
        )
    return features


__all__ = ["append_session_features"]
