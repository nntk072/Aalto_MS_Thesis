"""Trading strategy abstraction (Agent.md §13).

One :class:`~quant_rl.envs.trading_env.TradingEnv` hosts many strategies; a
strategy object owns only strategy semantics — which features it requires,
whether an entry is valid at bar ``t``, where the structural stop reference
sits, and which take-profit targets are candidates. The environment keeps
all mechanics (fills, risk, lot sizing, logging).
"""

from .base import TradingStrategy
from .baseline import BaselineStrategy
from .distribution import DistributionStrategy
from .po3_ifvg import PO3IFVGStrategy

__all__ = [
    "BaselineStrategy",
    "DistributionStrategy",
    "PO3IFVGStrategy",
    "TradingStrategy",
]
