"""Session / TZ diagnostic for PD windows (T-02.3)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quant_rl.data.session import add_session_labels
from quant_rl.features.pd_context import build_pd_context_features


def _bars(n: int = 3 * 24 * 60) -> pd.DataFrame:
    idx = pd.date_range("2025-06-02 00:00", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(2)
    close = 20000.0 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "tickvol": 10,
            "volume": 1000,
            "vol": 0,
            "spread": 0.6,
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )


def _shift_index(df: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
    """Return a copy whose DatetimeIndex is shifted by ``delta``."""
    out = df.copy()
    idx = cast(pd.DatetimeIndex, out.index)
    out.index = idx + delta
    return out


def test_plus_one_hour_shift_changes_session_labels() -> None:
    """A +1h clock shift (DST-like) changes broker-tz session labels."""
    bars = _bars()
    base = add_session_labels(bars, tz="Etc/GMT-3")
    shifted = _shift_index(bars, pd.Timedelta(hours=1))
    moved = add_session_labels(shifted, tz="Etc/GMT-3")
    # Align by position: same wall clock in broker tz after shift ≠ same labels
    disagree = (base["session"].to_numpy() != moved["session"].to_numpy()).mean()
    assert disagree > 0.05


def test_pd_context_sensitive_to_tz_shift() -> None:
    bars = _bars()
    atr = pd.Series(np.full(len(bars), 10.0), index=bars.index)
    pd_base = build_pd_context_features(bars, atr, tz="Etc/GMT-3")
    shifted = _shift_index(bars, -pd.Timedelta(hours=1))
    atr_s = pd.Series(np.full(len(shifted), 10.0), index=shifted.index)
    pd_shift = build_pd_context_features(shifted, atr_s, tz="Etc/GMT-3")
    cols = [c for c in pd_base.columns if c in pd_shift.columns]
    assert cols
    diffs = []
    for c in cols:
        a = pd_base[c].to_numpy(dtype=float)
        b = pd_shift[c].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            diffs.append(float(np.mean(np.abs(a[mask] - b[mask]) > 1e-9)))
    assert max(diffs) > 0.01
