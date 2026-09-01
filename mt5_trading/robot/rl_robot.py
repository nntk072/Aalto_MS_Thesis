"""RL-driven single-symbol trading robot (Master Roadmap Stage 8).

Mirrors the MT5 robot contract established by ``CrossOverRobot`` and
``MultiSymbolRobot``: the robot owns a duck-typed ``TradingStrategy``
(obtained from :meth:`RLStrategyAdapter.as_strategy`), a ``Trader`` and an
optional ``RiskManager``, and exposes a single ``trade()`` method that:

1. asks the strategy for a signal ``(symbol, Signal)``,
2. sizes the position via ``RiskManager`` (falling back to a default lot),
3. checks risk limits, and
4. opens/closes positions through the ``Trader`` — or only logs the intended
   order when ``paper_trading=True`` (dry run, the Stage 8 default).

Signal semantics match ``RLStrategyAdapter.as_strategy``: ``BUY`` closes any
open shorts and opens one long, ``SELL`` closes any open longs and opens one
short, ``HOLD``/``NONE`` leaves current exposure alone.
"""

from __future__ import annotations

import MetaTrader5 as mt5
from loguru import logger

from mt5_trading.adapters import Trader, TradingData
from mt5_trading.domain.risk_manager import RiskManager
from mt5_trading.domain.signal import Signal
from quant_rl.live.rl_strategy import RLStrategyAdapter


class RLRobot:
    """Trade a single symbol from a trained RL policy.

    Args:
        adapter: Built ``RLStrategyAdapter`` wrapping the trained PPO/SAC model.
        data_source: Primary-symbol M1 data source.
        trader: ``Trader`` implementation used to place/close orders.
        risk_manager: Optional ``RiskManager`` for position sizing and limits.
            Defaults to a fresh ``RiskManager()`` (see ``live_risk_overrides``
            in ``quant_rl/config/default.yaml`` for the recommended values).
        secondary_data_source: Optional secondary-symbol data source for SMT
            features — required when the deployed model was trained with
            SMT enabled (train/live feature parity, Stage 8 Task 8.4).
        default_lot_size: Fallback position size if risk-based sizing fails.
        stop_loss_pips: Stop-loss distance used by ``RiskManager`` sizing.
        magic_number: MT5 magic number tagging orders opened by this robot.
        paper_trading: When True (the safe default), every signal and intended
            order is logged but no orders are placed with the broker.
    """

    def __init__(
        self,
        adapter: RLStrategyAdapter,
        data_source: TradingData,
        trader: Trader,
        risk_manager: RiskManager | None = None,
        secondary_data_source: TradingData | None = None,
        default_lot_size: float = 0.1,
        stop_loss_pips: float = 50.0,
        magic_number: int = 20240102,
        paper_trading: bool = False,
    ) -> None:
        self.adapter = adapter
        self.data_source = data_source
        self.strategy = adapter.as_strategy(data_source, secondary_data_source)
        self.trader = trader
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()
        self.secondary_data_source = secondary_data_source
        self.default_lot_size = default_lot_size
        self.stop_loss_pips = stop_loss_pips
        self.magic_number = magic_number
        self.paper_trading = paper_trading
        self.name = "RL Robot"
        logger.info(
            "Starting %s (paper_trading=%s, symbol=%s, secondary=%s)",
            self.name,
            paper_trading,
            data_source.get_symbol(),
            secondary_data_source.get_symbol() if secondary_data_source is not None else None,
        )

    def calculate_position_size(self, symbol: str) -> float:
        """Risk-based position size, falling back to ``default_lot_size``."""
        try:
            size = self.risk_manager.calculate_position_size(symbol, self.stop_loss_pips)
            if size == 0.0:
                logger.warning(f"Using default lot size for {symbol}: {self.default_lot_size}")
                return self.default_lot_size
            return size
        except Exception as exc:  # noqa: BLE001  (live loop must not crash)
            logger.error(f"Error calculating position size for {symbol}: {exc}")
            return self.default_lot_size

    def check_risk_before_trade(self, symbol: str, position_size: float) -> tuple[bool, str]:
        """Check whether *position_size* is within live risk limits."""
        return self.risk_manager.check_risk_limits(symbol, position_size, self.stop_loss_pips)

    def _open(self, symbol: str, position_type: int) -> None:
        """Open a BUY/SELL position, or log it in paper mode."""
        position_size = self.calculate_position_size(symbol)
        is_allowed, reason = self.check_risk_before_trade(symbol, position_size)
        if not is_allowed:
            logger.warning(f"Trade not allowed for {symbol}: {reason}")
            return
        direction = "BUY" if position_type == mt5.ORDER_TYPE_BUY else "SELL"
        if self.paper_trading:
            logger.info(
                f"[PAPER] would open {direction} {symbol} vol={position_size:.2f} (risk: {reason})"
            )
            return
        result = self.trader.open_position(
            symbol,
            position_size,
            position_type,
            f"{self.name} buy position"
            if position_type == mt5.ORDER_TYPE_BUY
            else f"{self.name} sell position",
            self.magic_number,
            sl=None,
            tp=None,
        )
        if result is None:
            return  # AutoTrading disabled or error
        if getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"{direction} position opened for {symbol}: "
                f"Order #{result.order}, Volume: {result.volume}, Price: {result.price}"
            )
        else:
            logger.error(f"Failed to open {direction} position for {symbol}: {result.retcode}")

    def _close(self, symbol: str, position_type: int) -> None:
        """Close all positions of *position_type* for *symbol* (or log them)."""
        if self.paper_trading:
            logger.info(
                f"[PAPER] would close all "
                f"{'BUY' if position_type == mt5.ORDER_TYPE_BUY else 'SELL'} "
                f"positions for {symbol}"
            )
            return
        self.trader.close_positions(self.name, symbol, position_type)

    def trade(self) -> None:
        """Run one trading cycle: read the signal and act on it."""
        logger.info("Searching for trading signal")
        try:
            symbol, signal = self.strategy.signal()
        except Exception as exc:  # noqa: BLE001  (live loop must not crash)
            logger.exception(f"Error getting signal: {exc}")
            return

        if signal == Signal.BUY:
            total_buy, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_BUY)
            if total_buy == 0:
                logger.info(f"Buy signal detected for {symbol}")
                self._open(symbol, mt5.ORDER_TYPE_BUY)
            else:
                logger.info(f"Buy position already exists for {symbol}")
            total_sell, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_SELL)
            if total_sell > 0:
                logger.info(f"Closing existing sell positions for {symbol}")
                self._close(symbol, mt5.ORDER_TYPE_SELL)

        elif signal == Signal.SELL:
            total_sell, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_SELL)
            if total_sell == 0:
                logger.info(f"Sell signal detected for {symbol}")
                self._open(symbol, mt5.ORDER_TYPE_SELL)
            else:
                logger.info(f"Sell position already exists for {symbol}")
            total_buy, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_BUY)
            if total_buy > 0:
                logger.info(f"Closing existing buy positions for {symbol}")
                self._close(symbol, mt5.ORDER_TYPE_BUY)

        elif signal == Signal.NONE:
            logger.info("No trading signal found.")

        logger.info("Waiting for the next trading signal")
        logger.info("\n")
