"""FTMO-style account state tracker.

Tracks equity, balance, daily P&L, peak equity, daily loss, and max drawdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccountState:
    """Mutable account state for the backtester / environment."""

    initial_balance: float = 100_000.0
    balance: float = field(init=False)
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    session_start_balance: float = field(init=False)
    daily_loss: float = field(init=False)          # positive = loss so far today
    max_drawdown: float = field(init=False)        # running max drawdown (positive)
    open_pnl: float = field(init=False)

    def __post_init__(self) -> None:
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.session_start_balance = self.initial_balance
        self.daily_loss = 0.0
        self.max_drawdown = 0.0
        self.open_pnl = 0.0

    # ------------------------------------------------------------------

    def update_equity(self, open_pnl: float) -> None:
        """Recompute equity from balance + unrealised P&L."""
        self.open_pnl = open_pnl
        self.equity = self.balance + open_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def close_trade(self, pnl: float) -> None:
        """Realise P&L from a closed trade."""
        self.balance += pnl
        self.open_pnl = 0.0
        self.equity = self.balance
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        # Daily loss: only accumulates when it is a loss
        if pnl < 0:
            self.daily_loss += abs(pnl)

    def reset_daily(self) -> None:
        """Call at the start of each trading session / calendar day."""
        self.session_start_balance = self.balance
        self.daily_loss = 0.0

    def drawdown_pct(self) -> float:
        return self.max_drawdown / self.initial_balance if self.initial_balance else 0.0

    def to_array(self) -> list[float]:
        """Return a fixed-length array for the env observation."""
        return [
            self.balance / self.initial_balance - 1.0,
            self.equity / self.initial_balance - 1.0,
            self.open_pnl / self.initial_balance,
            self.daily_loss / self.initial_balance,
            self.max_drawdown / self.initial_balance,
        ]
