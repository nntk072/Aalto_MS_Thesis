"""Buy-and-hold baseline strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_returns(bars: pd.DataFrame) -> pd.Series:
    """Return a daily equity curve for a simple buy-and-hold.

    Parameters
    ----------
    bars:
        Bar DataFrame with a ``close`` column.
    """
    close = bars["close"]
    returns = close.pct_change().fillna(0.0)
    equity = (1 + returns).cumprod()
    return equity
