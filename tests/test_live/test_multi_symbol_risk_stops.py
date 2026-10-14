"""The multi-symbol risk robot must submit the stop it sized against."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

if "MetaTrader5" not in sys.modules:
    try:  # pragma: no cover - depends on platform
        import MetaTrader5  # noqa: F401
    except ImportError:
        sys.modules["MetaTrader5"] = ModuleType("MetaTrader5")

_mt5 = sys.modules["MetaTrader5"]
for _name, _value in {
    "TIMEFRAME_H1": 60,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "TRADE_RETCODE_DONE": 10009,
}.items():
    if not hasattr(_mt5, _name):
        setattr(_mt5, _name, _value)

import MetaTrader5 as mt5  # noqa: E402
from mt5_trading.domain.signal import Signal  # noqa: E402
from mt5_trading.robot.multi_symbol_robot import MultiSymbolRobot  # noqa: E402


class _Trader:
    def __init__(self) -> None:
        self.open_calls: list[dict[str, Any]] = []

    def open_position(
        self, symbol: str, volume: float, position_type: int, *args: Any, **kwargs: Any
    ):
        self.open_calls.append({"symbol": symbol, "type": position_type, "sl": kwargs.get("sl")})
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, order=1, volume=volume, price=100.0)

    def get_opened_positions(self, *_args: Any, **_kwargs: Any) -> tuple[int, pd.DataFrame]:
        return 0, pd.DataFrame()

    def close_positions(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Strategy:
    def __init__(self, symbol: str, signal: Signal) -> None:
        self.symbol = symbol
        self._signal = signal

    def signal(self) -> tuple[str, Signal]:
        return self.symbol, self._signal


class _Risk:
    def calculate_position_size(self, *_args: Any, **_kwargs: Any) -> float:
        return 0.1

    def check_risk_limits(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return True, "risk ok"

    def calculate_stop_loss_price(self, _symbol: str, order_type: int, _pips: float) -> float:
        return 99.0 if order_type == mt5.ORDER_TYPE_BUY else 101.0


class _Symbols:
    def get_volatility_metrics(self, _symbol: str) -> None:
        return None


@pytest.mark.parametrize(
    ("signal", "order_type", "expected_stop"),
    [(Signal.BUY, mt5.ORDER_TYPE_BUY, 99.0), (Signal.SELL, mt5.ORDER_TYPE_SELL, 101.0)],
)
def test_multi_symbol_robot_submits_broker_stop(
    signal: Signal, order_type: int, expected_stop: float
) -> None:
    trader = _Trader()
    symbol = "US100.cash"
    robot = MultiSymbolRobot(
        symbol_manager=cast(Any, _Symbols()),
        trader=trader,  # type: ignore[arg-type]
        strategies=cast(Any, {symbol: _Strategy(symbol, signal)}),
        risk_manager=_Risk(),  # type: ignore[arg-type]
        paper_trading=False,
    )

    robot.trade_symbol(symbol)

    assert trader.open_calls == [{"symbol": symbol, "type": order_type, "sl": expected_stop}]
