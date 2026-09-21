"""Shared train/live observation pieces: window pad and account vector."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def pad_observation_window(
    seq: np.ndarray[Any, Any],
    window: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Left-pad ``seq`` to ``window`` rows. Mask is 0 on the pad and 1 on real rows."""
    values = np.asarray(seq, dtype=np.float32)
    if values.ndim == 1:
        width = 1
        values = values.reshape(-1, 1)
    else:
        width = int(values.shape[1]) if values.shape[1:] else 1
    n_real = int(values.shape[0])
    mask = np.zeros(window, dtype=np.float32)
    if n_real >= window:
        return values[-window:], np.ones(window, dtype=np.float32)
    if n_real == 0:
        return np.zeros((window, width), dtype=np.float32), mask
    padded = np.pad(values, ((window - n_real, 0), (0, 0)), mode="constant", constant_values=0.0)
    mask[window - n_real :] = 1.0
    return cast(np.ndarray[Any, Any], padded.astype(np.float32)), mask


def normalized_account_vector(
    *,
    equity: float,
    initial_balance: float,
    pos_dir: float,
    open_pnl: float,
    dist_to_sl: float,
    trailing_dd: float,
    close: float,
) -> np.ndarray[Any, Any]:
    """The six account features the encoder MLP expects. Already O(1)."""
    norm_equity = float(np.log(equity / initial_balance)) if initial_balance > 0 else 0.0
    norm_pnl = open_pnl / initial_balance if initial_balance > 0 else 0.0
    unrealised_r = (open_pnl / equity * 100.0) if equity > 0 else 0.0
    norm_dist = dist_to_sl / close if close > 0 else 0.0
    return np.array(
        [norm_equity, pos_dir, norm_pnl, unrealised_r, norm_dist, float(trailing_dd)],
        dtype=np.float32,
    )
