"""FTMO-style account state tracker.

Tracks equity, balance, daily P&L, and absolute loss from initial balance.
``max_drawdown`` here means peak loss from *initial* balance (not trailing HWM).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccountState:
    """Mutable account state for the backtester / environment."""

    initial_balance: float = 100_000.0
    balance: float = field(init=False)
    equity: float = field(init=False)
    peak_equity: float = field(init=False)  # informational HWM only
    session_start_balance: float = field(init=False)
    daily_loss: float = field(init=False)  # positive = loss so far today
    max_drawdown: float = field(init=False)  # peak loss from initial (positive $)
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

    def _update_loss_from_initial(self) -> None:
        """Track peak absolute loss vs initial balance (FTMO max-loss, not trailing)."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        loss = max(0.0, self.initial_balance - self.equity)
        if loss > self.max_drawdown:
            self.max_drawdown = loss

    def update_equity(self, open_pnl: float) -> None:
        """Recompute equity from balance + unrealised P&L."""
        self.open_pnl = open_pnl
        self.equity = self.balance + open_pnl
        self._update_loss_from_initial()

    def close_trade(self, pnl: float) -> None:
        """Realise P&L from a closed trade."""
        self.balance += pnl
        self.open_pnl = 0.0
        self.equity = self.balance
        self._update_loss_from_initial()
        # Daily loss: only accumulates when it is a loss
        if pnl < 0:
            self.daily_loss += abs(pnl)

    def reset_daily(self) -> None:
        """Call at the start of each trading session / calendar day."""
        self.session_start_balance = self.balance
        self.daily_loss = 0.0

    def loss_from_initial(self) -> float:
        """Current absolute loss vs initial balance (0 if equity ≥ initial)."""
        return max(0.0, self.initial_balance - self.equity)

    def drawdown_pct(self) -> float:
        """Peak loss from initial as a fraction of initial balance."""
        return self.max_drawdown / self.initial_balance if self.initial_balance else 0.0

    def trailing_drawdown_pct(self) -> float:
        """Current drawdown from peak equity (HWM) as a fraction of peak."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def to_array(self) -> list[float]:
        """Return a fixed-length array for the env observation."""
        return [
            self.balance / self.initial_balance - 1.0,
            self.equity / self.initial_balance - 1.0,
            self.open_pnl / self.initial_balance,
            self.daily_loss / self.initial_balance,
            self.max_drawdown / self.initial_balance,
        ]
