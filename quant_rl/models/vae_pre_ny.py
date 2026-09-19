"""Pre-NY sequence helpers for the narrative VAE feature path."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from quant_rl.data.resample import resample
from quant_rl.models.vae import VAE

_PRE_NY_START = time(1, 5)
_PRE_NY_END = time(16, 29)


def _ohlcv_frame(m1_bars: pd.DataFrame) -> pd.DataFrame:
    """Return an OHLCV frame suitable for M5 resampling."""
    out = m1_bars.copy()
    if "volume" not in out.columns:
        if "tickvol" in out.columns:
            out["volume"] = out["tickvol"]
        else:
            out["volume"] = 0.0
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    return out.loc[:, cols]


def build_pre_ny_by_date(
    m1_bars: pd.DataFrame,
    *,
    seq_len: int = 186,
    n_features: int = 5,
) -> dict[date, np.ndarray[Any, Any]]:
    """Build one padded/truncated M5 OHLCV sequence per calendar day.

    Window is broker-local 01:05–16:29 (inclusive end via last M5 bar ≤ 16:29).
    Days with no bars in the window are omitted.
    """
    if m1_bars.empty:
        return {}
    work = _ohlcv_frame(m1_bars)
    m5 = resample(work, "M5")
    if m5.empty:
        return {}

    feature_cols = [c for c in ("open", "high", "low", "close", "volume") if c in m5.columns]
    if not feature_cols:
        feature_cols = list(m5.columns[:n_features])

    idx = pd.DatetimeIndex(m5.index)
    out: dict[date, np.ndarray[Any, Any]] = {}
    for day, day_df in m5.groupby(idx.date):
        local = pd.DatetimeIndex(day_df.index)
        times = local.time
        mask = np.array([(t >= _PRE_NY_START) and (t <= _PRE_NY_END) for t in times])
        window = day_df.loc[mask, feature_cols]
        if window.empty:
            continue
        arr_any: np.ndarray[Any, Any] = np.asarray(window.to_numpy(dtype=np.float64))
        if arr_any.ndim != 2:
            continue
        if arr_any.shape[1] > n_features:
            arr_any = arr_any[:, :n_features]
        elif arr_any.shape[1] < n_features:
            pad_f = np.zeros((arr_any.shape[0], n_features - arr_any.shape[1]), dtype=np.float64)
            arr_any = np.concatenate([arr_any, pad_f], axis=1)
        # Scale for VAE stability: OHLC as fraction of first close; volume as log1p.
        # Keeps train_vae and TradingEnv pre_ny_seq on the same footing.
        first_close = float(arr_any[0, 3]) if arr_any.shape[1] >= 4 else 0.0
        if first_close > 0.0:
            arr_any[:, : min(4, arr_any.shape[1])] = (
                arr_any[:, : min(4, arr_any.shape[1])] / first_close
            ) - 1.0
        if arr_any.shape[1] >= 5:
            vol = np.log1p(np.maximum(arr_any[:, 4], 0.0))
            vol_std = float(vol.std())
            if vol_std > 1e-8:
                arr_any[:, 4] = (vol - float(vol.mean())) / vol_std
            else:
                arr_any[:, 4] = 0.0
        if len(arr_any) >= seq_len:
            arr_any = arr_any[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(arr_any), n_features), dtype=np.float64)
            arr_any = np.concatenate([pad, arr_any], axis=0)
        day_key = day if isinstance(day, date) else pd.Timestamp(str(day)).date()
        out[day_key] = np.asarray(arr_any, dtype=np.float32)
    return out


def load_vae_from_checkpoint(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> VAE:
    """Construct a VAE and load ``state_dict`` from ``path``."""
    cfg: dict[str, Any] = {}
    if config_path is not None and Path(config_path).is_file():
        with open(config_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                cfg = loaded

    model = VAE(
        seq_len=int(cfg.get("seq_len", 186)),
        n_features=int(cfg.get("n_features", 5)),
        latent_dim=int(cfg.get("latent_dim", 16)),
        encoder_channels=tuple(cfg.get("encoder_channels", [64, 32, 16])),
        decoder_channels=tuple(cfg.get("decoder_channels", [16, 32, 64])),
        kernel_size=int(cfg.get("kernel_size", 3)),
        stride=int(cfg.get("stride", 2)),
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
