"""Feature-cache identity and path construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configuration.paths import repository_root

FEATURE_CACHE_VERSION = "v24-native-source-timeframes"


def feature_cache_content_hash(
    cfg: DictConfig | None,
    primary: pd.DataFrame,
    train_mask: pd.Series | None = None,
    secondary: pd.DataFrame | None = None,
    normalization_mode: str = "legacy_masked_window",
    timeframe_bars: dict[str, pd.DataFrame] | None = None,
    secondary_timeframe_bars: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Stable short hash for feature-cache invalidation (TI-3).

    Inputs: ``FEATURE_CACHE_VERSION``, resolved ``cfg.features`` (plus tz keys
    that affect PD/session), both instruments' full indexes and OHLCV/spread
    values, and the full ``train_mask``.
    """
    h = hashlib.sha256()
    h.update(FEATURE_CACHE_VERSION.encode())
    # Retain legacy cache identities for existing experiments. A frozen-fold
    # fit must never consume a cache produced with legacy rolling statistics.
    if normalization_mode != "legacy_masked_window":
        h.update(f"normalization:{normalization_mode}".encode())
    if cfg is not None:
        payload: dict[str, Any] = {}
        if hasattr(cfg, "features"):
            payload["features"] = OmegaConf.to_container(cfg.features, resolve=True)
        cost_point_size = OmegaConf.select(cfg, "costs.point_size", default=None)
        if cost_point_size is not None:
            payload["costs.point_size"] = float(cost_point_size)
        for path in (
            "data.tz",
            "data.timestamp_clock",
            "data.session_schedule_file",
            "session.tz",
            "session.start",
            "session.end",
            "strategy.session.timezone",
        ):
            node: Any = cfg
            ok = True
            for part in path.split("."):
                if not hasattr(node, part):
                    ok = False
                    break
                node = getattr(node, part)
            if ok:
                payload[path] = str(node)
        h.update(json.dumps(payload, sort_keys=True, default=str).encode())
        schedule_value = OmegaConf.select(cfg, "data.session_schedule_file")
        if schedule_value:
            schedule_path = Path(str(schedule_value)).expanduser()
            if not schedule_path.is_absolute():
                schedule_path = repository_root(__file__) / schedule_path
            if schedule_path.is_file():
                h.update(hashlib.sha256(schedule_path.read_bytes()).digest())
            else:
                h.update(b"missing-session-schedule")
    h.update(str(len(primary)).encode())
    if len(primary) > 0:
        primary_index_ns = (
            pd.DatetimeIndex(primary.index).to_numpy(dtype="datetime64[ns]").view(np.int64)
        )
        h.update(np.ascontiguousarray(primary_index_ns).tobytes())
    # SMT and optional per-timeframe SMT use the secondary bars. Hash both
    # inputs so a corrected/missing/re-timestamped US500 bar invalidates cache.
    for label, frame in (("primary", primary), ("secondary", secondary)):
        if frame is None:
            h.update(f"{label}:none".encode())
            continue
        h.update(label.encode())
        h.update(str(len(frame)).encode())
        if len(frame) > 0:
            frame_index_ns = (
                pd.DatetimeIndex(frame.index).to_numpy(dtype="datetime64[ns]").view(np.int64)
            )
            h.update(np.ascontiguousarray(frame_index_ns).tobytes())
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tickvol",
            "spread",
            "tick_count",
            "activity",
        ):
            if column in frame:
                values = frame[column].to_numpy(dtype=np.float64, copy=False)
                h.update(column.encode())
                h.update(np.ascontiguousarray(values).tobytes())
    for label, frames in (
        ("primary_timeframe", timeframe_bars),
        ("secondary_timeframe", secondary_timeframe_bars),
    ):
        if frames is None:
            h.update(f"{label}:none".encode())
            continue
        for timeframe, frame in sorted(frames.items()):
            h.update(f"{label}:{timeframe}".encode())
            h.update(
                pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes()
            )
    if train_mask is not None:
        mask_arr = np.ascontiguousarray(np.asarray(train_mask, dtype=bool))
        h.update(str(int(mask_arr.sum())).encode())
        h.update(mask_arr.tobytes())
    return h.hexdigest()[:16]


def feature_cache_path(
    cache_dir: Path | str,
    symbol: str,
    cfg: DictConfig | None,
    primary: pd.DataFrame,
    train_mask: pd.Series | None = None,
    secondary: pd.DataFrame | None = None,
    normalization_mode: str = "legacy_masked_window",
    timeframe_bars: dict[str, pd.DataFrame] | None = None,
    secondary_timeframe_bars: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Build ``{symbol}_features_{version}_{hash}.parquet`` under ``cache_dir``."""
    digest = feature_cache_content_hash(
        cfg,
        primary,
        train_mask=train_mask,
        secondary=secondary,
        normalization_mode=normalization_mode,
        timeframe_bars=timeframe_bars,
        secondary_timeframe_bars=secondary_timeframe_bars,
    )
    return Path(cache_dir) / f"{symbol}_features_{FEATURE_CACHE_VERSION}_{digest}.parquet"
