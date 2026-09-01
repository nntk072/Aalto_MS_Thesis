"""Headless tests for the RL live/paper robot (Master Roadmap Stage 8).

Runs without MT5 or SB3: the bridge/robot are exercised with a stub model,
a stub data source, and a recording trader. ``MetaTrader5`` (Windows-only)
is stubbed in ``sys.modules`` so the ``mt5_trading`` imports resolve.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
from mt5_trading.adapters import Trader, TradingData

# --- headless MetaTrader5 stub (Windows-only package, absent on Linux) -----
# Additive: several sibling test modules install their own partial stub; always
# top up whatever MetaTrader5 module is already present, never replace it.
if "MetaTrader5" not in sys.modules:
    try:  # pragma: no cover - depends on platform
        import MetaTrader5  # noqa: F401
    except ImportError:
        sys.modules["MetaTrader5"] = ModuleType("MetaTrader5")

_mt5_stub = sys.modules["MetaTrader5"]
for _attr in (
    "TIMEFRAME_M1",
    "TIMEFRAME_M5",
    "TIMEFRAME_M15",
    "TIMEFRAME_M30",
    "TIMEFRAME_H1",
    "TIMEFRAME_H4",
    "TIMEFRAME_D1",
):
    if not hasattr(_mt5_stub, _attr):
        setattr(_mt5_stub, _attr, 1)  # all timeframes equal is fine headless
if not hasattr(_mt5_stub, "ORDER_TYPE_BUY"):
    setattr(_mt5_stub, "ORDER_TYPE_BUY", 0)
if not hasattr(_mt5_stub, "ORDER_TYPE_SELL"):
    setattr(_mt5_stub, "ORDER_TYPE_SELL", 1)
if not hasattr(_mt5_stub, "TRADE_RETCODE_DONE"):
    setattr(_mt5_stub, "TRADE_RETCODE_DONE", 10009)
if not hasattr(_mt5_stub, "account_info"):
    setattr(_mt5_stub, "account_info", lambda: SimpleNamespace(balance=100_000.0, equity=100_000.0))
if not hasattr(_mt5_stub, "symbol_select"):
    setattr(_mt5_stub, "symbol_select", lambda *a, **k: True)
if not hasattr(_mt5_stub, "symbol_info"):
    setattr(
        _mt5_stub,
        "symbol_info",
        lambda *a, **k: SimpleNamespace(trade_tick_size=0.1, trade_tick_value=1.0, digits=5),
    )
if not hasattr(_mt5_stub, "symbol_info_tick"):
    setattr(_mt5_stub, "symbol_info_tick", lambda *a, **k: SimpleNamespace(bid=100.0, ask=100.1))
if not hasattr(_mt5_stub, "order_send"):
    setattr(_mt5_stub, "order_send", lambda *a, **k: None)
if not hasattr(_mt5_stub, "copy_rates_from_pos"):
    setattr(_mt5_stub, "copy_rates_from_pos", lambda *a, **k: None)


def _make_bars(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "tickvol": np.full(n, 100.0),
            "vol": np.zeros(n),
            "spread": np.full(n, 0.6),
        },
        index=idx,
    )


class _StubModel:
    """Duck-typed SB3 model; action = +1 (long) always."""

    def predict(self, obs: dict[str, Any], deterministic: bool = True) -> np.ndarray[Any, Any]:  # noqa: ARG002
        return np.array([1])


class _StubData(TradingData):
    """Duck-typed MT5Data returning a fixed bars frame."""

    def __init__(self, bars: pd.DataFrame, symbol: str = "US100.cash") -> None:
        self._bars = bars
        self._symbol = symbol

    def get_symbol(self) -> str:
        return self._symbol

    def get_data(self) -> pd.DataFrame:
        return self._bars


class RecordingTrader(Trader):
    """Duck-typed Trader that records calls instead of touching MT5."""

    def __init__(self, *, sells_open: bool = False) -> None:
        self.opened: list[tuple[Any, ...]] = []
        self.closed: list[tuple[Any, ...]] = []
        self.sells_open = sells_open

    def open_position(self, symbol, volume, position_type, comment, magic_number, sl=None, tp=None):
        import MetaTrader5 as mt5

        self.opened.append((symbol, volume, position_type))
        return SimpleNamespace(order=1, volume=volume, price=1.0, retcode=mt5.TRADE_RETCODE_DONE)

    def close_positions(self, robot_name: str, symbol=None, position_type=None):
        self.closed.append((symbol, position_type))

    def get_opened_positions(self, symbol=None, position_type=None):
        import MetaTrader5 as mt5

        if self.sells_open and position_type == mt5.ORDER_TYPE_SELL:
            return 1, pd.DataFrame()
        return 0, pd.DataFrame()

    def get_all_positions(self):
        return pd.DataFrame()

    def send_to_break_even(self, *args, **kwargs):
        return None

    def calculate_position_size(self, *args, **kwargs):
        return 0.0


def _make_robot(
    trader: RecordingTrader | None = None,
    paper_trading: bool = False,
    secondary_bars: pd.DataFrame | None = None,
):
    from mt5_trading.robot.rl_robot import RLRobot

    from quant_rl.live.rl_strategy import RLStrategyAdapter

    bars = _make_bars()
    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    secondary_data = (
        _StubData(secondary_bars, symbol="US500.cash") if secondary_bars is not None else None
    )
    robot = RLRobot(
        adapter=adapter,
        data_source=_StubData(bars),
        trader=cast(Trader, trader or RecordingTrader()),
        secondary_data_source=secondary_data,
        paper_trading=paper_trading,
    )
    return robot, adapter


def test_live_mode_places_buy_and_closes_opposite_sell() -> None:
    """BUY signal with an open sell → open the long and close the short."""
    trader = RecordingTrader(sells_open=True)
    robot, _ = _make_robot(trader, paper_trading=False)

    robot.trade()

    import MetaTrader5 as mt5

    assert len(trader.opened) == 1
    assert trader.opened[0][2] == mt5.ORDER_TYPE_BUY
    assert trader.closed == [("US100.cash", mt5.ORDER_TYPE_SELL)]


def test_paper_mode_never_places_orders() -> None:
    """PAPER_TRADING=true (dry-run default) must log, never call the broker."""
    trader = RecordingTrader()
    robot, _ = _make_robot(trader, paper_trading=True)

    robot.trade()

    assert trader.opened == []
    assert trader.closed == []


def test_robot_feed_secondary_bars_for_smt() -> None:
    """Stage 8 Task 8.4: a secondary data source must reach the adapter each
    cycle so SMT features are present in the live observation."""
    secondary = _make_bars(n=300, seed=99)
    robot, adapter = _make_robot(secondary_bars=secondary)

    robot.trade()  # signal() drives adapter.update_bars

    assert adapter._secondary_bars is not None  # noqa: SLF001
    assert len(adapter._secondary_bars) == len(secondary)  # noqa: SLF001


def test_too_few_bars_yields_no_signal() -> None:
    """With insufficient history the bridge returns NONE and the robot
    neither opens nor closes anything."""
    from mt5_trading.robot.rl_robot import RLRobot

    from quant_rl.live.rl_strategy import RLStrategyAdapter

    tiny = _make_bars(n=5)
    trader = RecordingTrader()
    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    robot = RLRobot(
        adapter=adapter,
        data_source=_StubData(tiny),
        trader=cast(Trader, trader),
        paper_trading=False,
    )
    robot.trade()
    assert trader.opened == []
    assert trader.closed == []


# --- live_trading_rl.py pure helpers ----------------------------------------


def _load_entrypoint_helpers():
    import live_trading_rl  # noqa: PLC0415

    return live_trading_rl


def test_live_risk_overrides_align_with_ftmo_block() -> None:
    """Stage 8 Task 8.2: the live (percentage) risk model is explicitly
    configured, and its $100k-equivalent per-trade risk matches the training
    ftmo.risk_per_trade_limit — no silent divergence."""
    import live_trading_rl  # noqa: PLC0415
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    risk = live_trading_rl.load_risk_overrides(cfg)

    assert risk["risk_per_symbol"] == 0.01
    assert risk["max_total_risk"] == 0.02
    # $100k balance * 1% == ftmo.risk_per_trade_limit ($1000)
    assert abs(100_000.0 * risk["risk_per_symbol"] - cfg.ftmo.risk_per_trade_limit) < 1e-6


def test_use_secondary_symbol_decision() -> None:
    """Task 8.4: SMT-capable models need secondary live data; primary_only /
    use_m1_only models do not."""
    import live_trading_rl  # noqa: PLC0415
    from omegaconf import OmegaConf

    smt_cfg = OmegaConf.create({"training": {"use_m1_only": False, "primary_only": False}})
    m1_cfg = OmegaConf.create({"training": {"use_m1_only": True, "primary_only": False}})
    primary_cfg = OmegaConf.create({"training": {"use_m1_only": False, "primary_only": True}})

    assert live_trading_rl._use_secondary_symbol(smt_cfg) is True
    assert live_trading_rl._use_secondary_symbol(m1_cfg) is False
    assert live_trading_rl._use_secondary_symbol(primary_cfg) is False


def test_resolve_run_config_prefers_saved_run_config(tmp_path) -> None:
    """The entrypoint must use the model run's saved config.yaml when
    available (train/live parity) and fall back to the repo default."""
    import live_trading_rl  # noqa: PLC0415

    run_dir = tmp_path / "rl_run" / "model"
    run_dir.mkdir(parents=True)
    (run_dir.parent / "config.yaml").write_text("training:\n  primary_only: true\n")
    model_path = run_dir / "ppo_final"

    resolved = live_trading_rl.resolve_run_config(model_path)

    assert resolved == run_dir.parent / "config.yaml"

    # No saved config → repo default
    resolved_default = live_trading_rl.resolve_run_config(tmp_path / "no_run" / "model" / "x")
    assert resolved_default.name == "default.yaml"
