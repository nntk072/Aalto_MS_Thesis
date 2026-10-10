"""Convert training frames into the arrays stored in a shared env bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from ..data.ticks import TickBook
from ..features.build import attach_reachable_r, select_obs_columns
from .shared_bundle import build_bundle, load_bundle


def observation_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    raw_columns: list[str] | tuple[str, ...],
) -> tuple[NDArray[np.float32], list[str]]:
    """Build the same contiguous float32 observation matrix used by training."""
    frame = attach_reachable_r(select_obs_columns(features, raw_columns), bars, features)
    array = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
    return array, [str(column) for column in frame.columns]


def _numeric_feature_array(features: pd.DataFrame) -> tuple[NDArray[Any], list[str]]:
    """Return numeric feature values in the same column order as TradingEnv."""
    numeric = features.select_dtypes(include="number")
    array = numeric.to_numpy(copy=True)
    if array.dtype != np.dtype(np.float64):
        array = array.astype(np.float64, copy=False)
    return array, [str(column) for column in numeric.columns]


def _numeric_column(features: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Match pandas ``to_numeric(errors='coerce')`` semantics for one column."""
    if name not in features:
        return np.full(len(features), np.nan, dtype=np.float64)
    values = pd.to_numeric(features[name], errors="coerce")
    return values.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def _session_codes(bars: pd.DataFrame) -> tuple[NDArray[np.int32], list[str]]:
    """Encode session labels as int32 codes and return their manifest labels."""
    if "session" not in bars:
        return np.zeros(len(bars), dtype=np.int32), [""]
    codes, labels = pd.factorize(bars["session"], sort=True)
    return codes.astype(np.int32, copy=False), [str(label) for label in labels]


def _pre_ny_arrays(
    pre_ny_by_date: Mapping[Any, NDArray[Any]] | None,
) -> tuple[NDArray[np.float32], dict[str, int]]:
    """Stack day-keyed VAE inputs in a stable order and return their row map."""
    if not pre_ny_by_date:
        return np.empty((0, 0, 0), dtype=np.float32), {}
    entries = sorted(
        (
            (pd.Timestamp(day).date().isoformat(), np.asarray(values))
            for day, values in pre_ny_by_date.items()
        ),
        key=lambda item: item[0],
    )
    shapes = {values.shape for _, values in entries}
    if len(shapes) != 1:
        raise ValueError("pre-NY sequences must have one shared shape")
    mapping = {day: row for row, (day, _) in enumerate(entries)}
    return np.stack([values for _, values in entries]).astype(np.float32, copy=False), mapping


def _tick_arrays(tickbook: TickBook | None) -> dict[str, NDArray[Any]]:
    """Expose the source tick arrays, or correctly typed empty arrays."""
    if tickbook is None:
        return {
            "tick_ts": np.empty(0, dtype=np.int64),
            "tick_bid": np.empty(0, dtype=np.float32),
            "tick_ask": np.empty(0, dtype=np.float32),
        }
    return {
        "tick_ts": tickbook._ts,
        "tick_bid": tickbook._bid,
        "tick_ask": tickbook._ask,
    }


def _bundle_source_fingerprints(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """Fingerprint configured raw bar and tick files by size and mtime."""
    raw_dir = Path(str(cfg.data.raw_dir))
    fingerprints: dict[str, dict[str, Any]] = {}
    for field in ("m1_files", "tick_files"):
        files = cfg.data.get(field, {})
        if files is None:
            continue
        for symbol, filename in files.items():
            relative = str(filename)
            path = raw_dir / relative
            try:
                stat = path.stat()
            except FileNotFoundError:
                fingerprints[f"{field}/{symbol}"] = {"file": relative, "missing": True}
            else:
                fingerprints[f"{field}/{symbol}"] = {
                    "file": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
    return fingerprints


def build_env_bundle(
    cache_dir: Path | str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: DictConfig,
    *,
    raw_columns: list[str] | tuple[str, ...],
    feature_config_hash: str,
    tickbook: TickBook | None,
    pre_ny_by_date: Mapping[Any, NDArray[Any]] | None,
) -> Path:
    """Build the complete read-only data bundle for one train split."""
    obs_features, obs_columns = observation_features(bars, features, raw_columns)
    feature_values, feature_columns = _numeric_feature_array(features)
    session_codes, session_labels = _session_codes(bars)
    pre_ny_sequences, pre_ny_rows = _pre_ny_arrays(pre_ny_by_date)
    session_ids = (
        bars["session_id"].to_numpy(dtype=np.int64, copy=True)
        if "session_id" in bars
        else np.zeros(len(bars), dtype=np.int64)
    )
    index = pd.DatetimeIndex(bars.index)
    arrays: dict[str, NDArray[Any]] = {
        "obs_features": obs_features,
        "features_num": feature_values,
        "bar_ohlc": bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64, copy=True),
        "bar_spread": (
            bars["spread"].to_numpy(dtype=np.float64, copy=True)
            if "spread" in bars
            else np.full(len(bars), np.nan, dtype=np.float64)
        ),
        "bar_volume": (
            bars["volume"].to_numpy(dtype=np.float64, copy=True)
            if "volume" in bars
            else np.full(len(bars), np.nan, dtype=np.float64)
        ),
        "bar_tickvol": (
            bars["tickvol"].to_numpy(dtype=np.float64, copy=True)
            if "tickvol" in bars
            else np.full(len(bars), np.nan, dtype=np.float64)
        ),
        "bar_time_ns": index.to_numpy(dtype="datetime64[ns]").view(np.int64).copy(),
        "bar_session": session_codes,
        "bar_session_id": session_ids,
        "bar_day_ord": np.asarray([day.toordinal() for day in index.date], dtype=np.int32),
        "atr_5": _numeric_column(features, "atr_5"),
        "ema_21": _numeric_column(features, "ema_21"),
        "pre_ny_seq": pre_ny_sequences,
    }
    arrays.update(_tick_arrays(tickbook))
    if bool(cfg.env.get("strategy_actions", False)) and "atr_5" in features:
        from quant_rl.backtest.tp_reach import max_tp_distance

        max_tp = max_tp_distance(bars, features["atr_5"]).reindex(bars.index)
        arrays["max_tp_dist"] = max_tp.to_numpy(dtype=np.float64, copy=True)
    else:
        arrays["max_tp_dist"] = np.empty(0, dtype=np.float64)

    raw_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("resolved training config must be a mapping")
    config = cast(dict[str, Any], raw_config)
    strategy_config = dict(config.get("strategy", {}) or {})
    strategy_config["raw_columns"] = list(raw_columns)
    config["strategy"] = strategy_config
    config["feature_config_hash"] = feature_config_hash
    input_signature = {
        "rows": len(bars),
        "first_bar_ns": int(index[0].value) if len(index) else None,
        "last_bar_ns": int(index[-1].value) if len(index) else None,
    }
    fingerprints = _bundle_source_fingerprints(cfg)
    fingerprints["bundle_input"] = input_signature
    return build_bundle(
        cache_dir,
        arrays=arrays,
        config=config,
        source_fingerprints=fingerprints,
        column_names={"features_num": feature_columns, "obs_features": obs_columns},
        labels={
            "session": session_labels,
            "has_session": "session" in bars,
            "has_session_id": "session_id" in bars,
            "bar_timezone": str(bars.index.tz)
            if isinstance(bars.index, pd.DatetimeIndex)
            else None,
            "pre_ny_day_to_row": pre_ny_rows,
        },
    )


def bars_for_export(bundle_path: Path | str) -> pd.DataFrame:
    """Materialize the small OHLC view used by end-of-run chart exporters."""
    bundle = load_bundle(bundle_path)
    arrays = bundle.arrays
    labels = bundle.manifest.get("labels", {})
    index = pd.to_datetime(arrays["bar_time_ns"], utc=True)
    timezone = labels.get("bar_timezone")
    if timezone:
        index = index.tz_convert(str(timezone))
    ohlc = arrays["bar_ohlc"]
    values: dict[str, Any] = {
        "open": ohlc[:, 0],
        "high": ohlc[:, 1],
        "low": ohlc[:, 2],
        "close": ohlc[:, 3],
        "spread": arrays["bar_spread"],
    }
    # Guard for bundles built before volume/tickvol were stored (see
    # BUNDLE_VERSION): older caches lack these arrays and must still export.
    if "bar_volume" in arrays:
        values["volume"] = arrays["bar_volume"]
    if "bar_tickvol" in arrays:
        values["tickvol"] = arrays["bar_tickvol"]
    if labels.get("has_session_id", False):
        values["session_id"] = arrays["bar_session_id"]
    return pd.DataFrame(values, index=index, copy=False)
