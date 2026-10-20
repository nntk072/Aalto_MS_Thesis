"""Headless tests for stop-based live position and aggregate-risk limits."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

if "MetaTrader5" not in sys.modules:
    try:  # pragma: no cover - depends on platform
        import MetaTrader5  # noqa: F401
    except ImportError:
        sys.modules["MetaTrader5"] = ModuleType("MetaTrader5")

_mt5_stub = sys.modules["MetaTrader5"]
if not hasattr(_mt5_stub, "TIMEFRAME_H1"):
    setattr(_mt5_stub, "TIMEFRAME_H1", 60)
if not hasattr(_mt5_stub, "ORDER_TYPE_BUY"):
    setattr(_mt5_stub, "ORDER_TYPE_BUY", 0)
if not hasattr(_mt5_stub, "ORDER_TYPE_SELL"):
    setattr(_mt5_stub, "ORDER_TYPE_SELL", 1)

import MetaTrader5 as mt5  # noqa: E402

from integrations.mt5.domain.risk_manager import RiskManager  # noqa: E402


@pytest.fixture
def mt5_risk_api(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    info = SimpleNamespace(
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        digits=2,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    tick = SimpleNamespace(bid=100.0, ask=100.01)
    positions: list[Any] | None = []
    monkeypatch.setattr(mt5, "symbol_select", lambda *_args, **_kwargs: True, raising=False)
    monkeypatch.setattr(mt5, "symbol_info", lambda _symbol: info, raising=False)
    monkeypatch.setattr(mt5, "symbol_info_tick", lambda _symbol: tick, raising=False)
    monkeypatch.setattr(
        mt5,
        "account_info",
        lambda: SimpleNamespace(balance=100_000.0, equity=100_000.0),
        raising=False,
    )
    monkeypatch.setattr(mt5, "positions_get", lambda: positions, raising=False)
    return {"info": info, "tick": tick, "positions": positions}


def test_sizing_rounds_down_and_stop_price_uses_same_distance(
    mt5_risk_api: dict[str, Any],
) -> None:
    manager = RiskManager(risk_per_symbol=0.01, max_total_risk=0.02)

    size = manager.calculate_position_size("US100.cash", stop_loss_pips=50.0)
    buy_stop = manager.calculate_stop_loss_price("US100.cash", mt5.ORDER_TYPE_BUY, 50.0)
    sell_stop = manager.calculate_stop_loss_price("US100.cash", mt5.ORDER_TYPE_SELL, 50.0)

    assert size == pytest.approx(20.0)
    assert buy_stop == pytest.approx(99.51)
    assert sell_stop == pytest.approx(100.5)
    allowed, message = manager.check_risk_limits("US100.cash", size, 50.0)
    assert allowed is True
    assert "total stop risk: 1.00%" in message


def test_risk_manager_constructor_defaults_match_live_config() -> None:
    manager = RiskManager()

    assert manager.risk_per_symbol == pytest.approx(0.01)
    assert manager.max_total_risk == pytest.approx(0.02)


def test_volume_step_rounding_never_exceeds_cash_risk_budget(
    mt5_risk_api: dict[str, Any],
) -> None:
    manager = RiskManager(risk_per_symbol=0.01, max_total_risk=0.02)

    size = manager.calculate_position_size("US100.cash", stop_loss_pips=50.0, risk_amount=999.9)
    distance = 50.0 * mt5_risk_api["info"].trade_tick_size
    planned_risk = (
        distance
        / mt5_risk_api["info"].trade_tick_size
        * mt5_risk_api["info"].trade_tick_value
        * size
    )

    assert size == pytest.approx(19.99)
    assert planned_risk <= 999.9


def test_aggregate_limit_uses_open_stop_risk_not_floating_profit(
    mt5_risk_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    position = SimpleNamespace(
        symbol="US500.cash",
        volume=30.0,
        price_open=100.0,
        sl=99.5,
        profit=0.0,
    )
    monkeypatch.setattr(mt5, "positions_get", lambda: [position], raising=False)
    manager = RiskManager(risk_per_symbol=0.01, max_total_risk=0.02)

    exposure = manager.get_total_exposure()
    allowed, message = manager.check_risk_limits("US100.cash", 20.0, 50.0)

    assert exposure["total_open_risk"] == pytest.approx(1500.0)
    assert exposure["risk_unknown_positions"] == 0
    assert allowed is False
    assert "2.50%" in message


@pytest.mark.parametrize(
    "position",
    [
        SimpleNamespace(symbol="US500.cash", volume=1.0, price_open=100.0, sl=0.0, profit=0.0),
        SimpleNamespace(symbol="US500.cash", volume=1.0, price_open=100.0, profit=0.0),
    ],
)
def test_unknown_existing_stop_risk_blocks_new_trade(
    mt5_risk_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    position: Any,
) -> None:
    monkeypatch.setattr(mt5, "positions_get", lambda: [position], raising=False)
    manager = RiskManager(risk_per_symbol=0.01, max_total_risk=0.02)

    allowed, message = manager.check_risk_limits("US100.cash", 1.0, 50.0)

    assert allowed is False
    assert "risk is unknown" in message


def test_position_query_failure_blocks_new_trade(
    mt5_risk_api: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mt5, "positions_get", lambda: None, raising=False)
    manager = RiskManager(risk_per_symbol=0.01, max_total_risk=0.02)

    allowed, message = manager.check_risk_limits("US100.cash", 1.0, 50.0)

    assert allowed is False
    assert "risk is unknown" in message


def test_minimum_volume_is_not_forced_above_risk_budget(
    mt5_risk_api: dict[str, Any],
) -> None:
    manager = RiskManager(risk_per_symbol=0.000001, max_total_risk=0.02)

    assert manager.calculate_position_size("US100.cash", stop_loss_pips=50.0) == 0.0
