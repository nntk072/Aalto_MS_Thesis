"""Train-year survival gate, with profitability retained as reported metrics.

The replay fails on a risk-limit breach or excessive peak trailing drawdown.
Net PnL and equity slope are recorded but are not acceptance requirements.
"""

from __future__ import annotations

from simulation.environment.episode_equity import (
    assess_train_equity,
    equity_slope,
)

__all__ = ["assess_train_equity", "equity_slope"]
