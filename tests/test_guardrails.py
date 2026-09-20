"""Tests: FTMO guardrail breach detection."""

from __future__ import annotations

from quant_rl.backtest.account import AccountState
from quant_rl.backtest.guardrails import FTMOGuardrails


def _make(daily_loss: float = 0.0, max_dd: float = 0.0) -> AccountState:
    acc = AccountState(initial_balance=100_000.0)
    acc.daily_loss = daily_loss
    acc.max_drawdown = max_dd
    return acc


def test_no_breach():
    g = FTMOGuardrails()
    acc = _make(daily_loss=1000.0, max_dd=500.0)
    assert not g.any_breach(acc)
    assert g.breach_reason(acc) is None


def test_daily_loss_breach():
    g = FTMOGuardrails(daily_loss_limit=5000.0)
    acc = _make(daily_loss=5000.0)
    assert g.check_daily(acc)
    assert g.any_breach(acc)
    assert g.breach_reason(acc) == "daily_loss"


def test_max_drawdown_breach():
    g = FTMOGuardrails(max_loss_limit=10_000.0)
    acc = _make()
    acc.equity = 90_000.0  # $10k below initial — absolute max loss, not trailing
    assert g.check_max_drawdown(acc)
    assert g.breach_reason(acc) == "max_drawdown"


def test_max_loss_not_trailing_hwm():
    """Profits raise equity; max-loss still measured from initial, not peak."""
    g = FTMOGuardrails(max_loss_limit=10_000.0)
    acc = AccountState(initial_balance=100_000.0)
    acc.update_equity(8_000.0)  # peak 108k
    acc.balance = 100_000.0
    acc.update_equity(-5_000.0)  # equity 95k — only $5k from initial
    assert not g.check_max_drawdown(acc)
    acc.balance = 100_000.0
    acc.update_equity(-10_000.0)  # equity 90k
    assert g.check_max_drawdown(acc)


def test_trade_risk_check():
    g = FTMOGuardrails(risk_per_trade_limit=1000.0)
    assert g.check_trade_risk(999.0) is False
    assert g.check_trade_risk(1001.0) is True


def test_soft_daily_does_not_hard_breach():
    g = FTMOGuardrails(daily_loss_limit=5000.0, soft_daily_loss_limit=2000.0)
    acc = _make(daily_loss=2500.0)
    assert g.check_soft_daily(acc)
    assert not g.check_daily(acc)
    assert not g.any_breach(acc)
    assert g.breach_reason(acc) is None
