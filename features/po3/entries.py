"""PO3 phase and entry-context features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_po3_phase_features(
    index: pd.DatetimeIndex,
    session_start: str = "16:30",
    session_end: str = "23:00",
) -> pd.DataFrame:
    """PO3 (Power of 3) daily-cycle phase tag, causally time-derived.

    Maps each bar's time-of-day position inside the trading session onto the
    classic PO3 arc: accumulation (session open) → manipulation (middle) →
    distribution (session close). This is a function of the timestamp only,
    so it is identical across timeframes once aligned onto the M1 spine —
    computed once here rather than duplicated per TF.

    Returns a DataFrame indexed like ``index`` with:
    - ``po3_phase``: ordinal 0/1/2 (accumulation/manipulation/distribution)
    - ``session_progress``: continuous fraction of session elapsed in [0, 1]
    """
    tz = index.tz
    start_ts = pd.Timestamp(f"2000-01-01 {session_start}").tz_localize(tz)
    end_ts = pd.Timestamp(f"2000-01-01 {session_end}").tz_localize(tz)
    session_len = (end_ts - start_ts).total_seconds()

    tod = index.hour * 3600 + index.minute * 60 + index.second
    start_secs = start_ts.hour * 3600 + start_ts.minute * 60
    elapsed = tod - start_secs
    progress = pd.Series(np.clip(elapsed / session_len, 0.0, 1.0), index=index)
    phase = pd.Series(np.digitize(progress.to_numpy(), [1.0 / 3.0, 2.0 / 3.0]), index=index)
    phase = phase.astype("float64")
    phase[elapsed < 0] = np.nan  # outside session → unknown phase
    return pd.DataFrame({"po3_phase": phase, "session_progress": progress})
