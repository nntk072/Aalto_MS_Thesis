import math
from typing import Any

import MetaTrader5 as mt5
from loguru import logger


class RiskManager:
    """
    Manages risk across multiple trading positions.
    Calculates position sizes based on volatility and risk parameters.
    Ensures total exposure doesn't exceed risk limits.
    """

    def __init__(self, risk_per_symbol: float = 0.01, max_total_risk: float = 0.02):
        """
        Initialize the risk manager.

        Args:
            risk_per_symbol: Maximum risk per symbol as percentage of account balance (default: 1%)
            max_total_risk: Maximum total stop risk across positions (default: 2%)
        """
        self.risk_per_symbol = risk_per_symbol
        self.max_total_risk = max_total_risk

    def get_account_balance(self) -> float:
        """
        Get current account balance.

        Returns:
            Account balance in account currency
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.0
            return float(account_info.balance)
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0

    def get_account_equity(self) -> float:
        """
        Get current account equity (balance + floating P/L).

        Returns:
            Account equity in account currency
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.0
            return float(account_info.equity)
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            return 0.0

    @staticmethod
    def _position_value(position: Any, name: str, default: Any = None) -> Any:
        if isinstance(position, dict):
            return position.get(name, default)
        as_dict = getattr(position, "_asdict", None)
        if callable(as_dict):
            return as_dict().get(name, default)
        return getattr(position, name, default)

    @staticmethod
    def _symbol_risk_units(symbol_info: Any, stop_loss_pips: float) -> tuple[float, float, float]:
        """Return price distance, tick size and tick value for the configured stop."""
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0))
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0))
        digits = int(getattr(symbol_info, "digits", -1))
        pips = float(stop_loss_pips)
        if (
            not math.isfinite(tick_size)
            or not math.isfinite(tick_value)
            or not math.isfinite(pips)
            or tick_size <= 0.0
            or tick_value <= 0.0
            or pips <= 0.0
            or digits < 0
        ):
            raise ValueError("invalid symbol tick data or stop distance")
        pip_ticks = 10.0 if digits in (3, 5) else 1.0
        return pips * tick_size * pip_ticks, tick_size, tick_value

    @staticmethod
    def _position_stop_risk(position: Any) -> float | None:
        """Estimate an open position's dollar risk to its broker-side stop.

        A missing stop or incomplete broker metadata is unknown risk. Callers
        must fail closed instead of substituting floating P/L for stop risk.
        """
        symbol = RiskManager._position_value(position, "symbol")
        volume = RiskManager._position_value(position, "volume")
        open_price = RiskManager._position_value(position, "price_open")
        stop_price = RiskManager._position_value(position, "sl")
        if not symbol or volume is None or open_price is None or stop_price in (None, 0):
            return None
        info = mt5.symbol_info(str(symbol))
        if info is None:
            return None
        try:
            tick_size = float(info.trade_tick_size)
            tick_value = float(info.trade_tick_value)
            size = float(volume)
            distance = abs(float(open_price) - float(stop_price))
            risk = distance / tick_size * tick_value * size
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
        if (
            not all(math.isfinite(value) for value in (tick_size, tick_value, size, distance, risk))
            or tick_size <= 0.0
            or tick_value <= 0.0
            or size <= 0.0
            or distance <= 0.0
            or risk < 0.0
        ):
            return None
        return risk

    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        risk_amount: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate position size based on stop loss and risk amount.

        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss in pips (or points for non-forex)
            risk_amount: Amount to risk (if None, uses risk_per_symbol * balance)
            volatility_multiplier: Multiplier to adjust for volatility (higher volatility = smaller position)

        Returns:
            Position size in lots
        """
        try:
            # Select symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Symbol {symbol} not available")
                return 0.0

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0

            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick for {symbol}")
                return 0.0

            # Calculate risk amount if not provided
            if risk_amount is None:
                balance = self.get_account_balance()
                risk_amount = balance * self.risk_per_symbol
            if (
                risk_amount is None
                or not math.isfinite(float(risk_amount))
                or float(risk_amount) <= 0.0
                or not math.isfinite(float(volatility_multiplier))
                or float(volatility_multiplier) <= 0.0
            ):
                logger.warning(f"Invalid risk amount or volatility multiplier for {symbol}")
                return 0.0

            # Adjust risk amount by volatility multiplier and use the same
            # stop-distance conversion as order placement and risk checking.
            adjusted_risk = float(risk_amount) / float(volatility_multiplier)
            price_diff, tick_size, tick_value = self._symbol_risk_units(symbol_info, stop_loss_pips)

            # Calculate position size
            risk_per_lot = (price_diff / tick_size) * tick_value
            raw_size = adjusted_risk / risk_per_lot

            min_lot = float(symbol_info.volume_min)
            max_lot = float(symbol_info.volume_max)
            lot_step = float(symbol_info.volume_step)
            if (
                not all(math.isfinite(value) for value in (min_lot, max_lot, lot_step, raw_size))
                or min_lot <= 0.0
                or max_lot < min_lot
                or lot_step <= 0.0
                or raw_size < min_lot
            ):
                logger.warning(f"Risk budget cannot support minimum volume for {symbol}")
                return 0.0

            # Round down so broker volume quantization cannot increase planned
            # stop risk beyond the configured cash budget.
            position_size = min(max_lot, math.floor(raw_size / lot_step) * lot_step)
            if position_size < min_lot:
                return 0.0

            logger.info(
                f"Calculated position size for {symbol}: {position_size} lots "
                f"(risk: ${adjusted_risk:.2f}, SL: {stop_loss_pips} pips, "
                f"volatility_mult: {volatility_multiplier:.2f})"
            )

            return float(position_size)

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def calculate_stop_loss_price(
        self, symbol: str, position_type: int, stop_loss_pips: float
    ) -> float:
        """Build the broker-side SL using the same distance used for sizing."""
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not available")
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            raise ValueError(f"Missing symbol or quote data for {symbol}")
        distance, _, _ = self._symbol_risk_units(info, stop_loss_pips)
        if position_type == mt5.ORDER_TYPE_BUY:
            entry_price = float(tick.ask)
            stop_price = entry_price - distance
        elif position_type == mt5.ORDER_TYPE_SELL:
            entry_price = float(tick.bid)
            stop_price = entry_price + distance
        else:
            raise ValueError(f"Unsupported order type for stop calculation: {position_type}")
        digits = int(info.digits)
        if not math.isfinite(stop_price) or stop_price <= 0.0:
            raise ValueError(f"Invalid stop price for {symbol}: {stop_price}")
        return float(round(stop_price, digits))

    def get_total_exposure(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """
        Get total exposure across all open positions.

        Args:
            symbols: Optional list of symbols to filter by

        Returns:
            Dictionary with exposure metrics
        """
        try:
            positions = mt5.positions_get()
            if positions is None:
                logger.error("Failed to query open MT5 positions; aggregate risk is unknown")
                return {
                    "total_positions": 0,
                    "total_volume": 0.0,
                    "total_profit": 0.0,
                    "total_open_risk": 0.0,
                    "risk_unknown_positions": 1,
                    "query_failed": True,
                    "symbols": {},
                }
            if len(positions) == 0:
                return {
                    "total_positions": 0,
                    "total_volume": 0.0,
                    "total_profit": 0.0,
                    "total_open_risk": 0.0,
                    "risk_unknown_positions": 0,
                    "query_failed": False,
                    "symbols": {},
                }

            selected = [
                position
                for position in positions
                if not symbols or self._position_value(position, "symbol") in symbols
            ]
            symbol_exposure: dict[str, dict[str, float | int]] = {}
            total_volume = total_profit = total_open_risk = 0.0
            unknown = 0
            for position in selected:
                symbol = str(self._position_value(position, "symbol", ""))
                volume = float(self._position_value(position, "volume", 0.0) or 0.0)
                profit = float(self._position_value(position, "profit", 0.0) or 0.0)
                position_risk = self._position_stop_risk(position)
                total_volume += volume
                total_profit += profit
                if position_risk is None:
                    unknown += 1
                else:
                    total_open_risk += position_risk
                row = symbol_exposure.setdefault(
                    symbol,
                    {"positions": 0, "volume": 0.0, "profit": 0.0, "open_risk": 0.0},
                )
                row["positions"] = int(row["positions"]) + 1
                row["volume"] = float(row["volume"]) + volume
                row["profit"] = float(row["profit"]) + profit
                if position_risk is not None:
                    row["open_risk"] = float(row["open_risk"]) + position_risk

            return {
                "total_positions": len(selected),
                "total_volume": total_volume,
                "total_profit": total_profit,
                "total_open_risk": total_open_risk,
                "risk_unknown_positions": unknown,
                "query_failed": False,
                "symbols": symbol_exposure,
            }
        except Exception as e:
            logger.error(f"Error getting total exposure: {e}")
            return {
                "total_positions": 0,
                "total_volume": 0.0,
                "total_profit": 0.0,
                "total_open_risk": 0.0,
                "risk_unknown_positions": 1,
                "query_failed": True,
                "symbols": {},
            }

    def check_risk_limits(
        self, symbol: str, proposed_position_size: float, stop_loss_pips: float
    ) -> tuple[bool, str]:
        """
        Check if proposed position would exceed risk limits.

        Args:
            symbol: Trading symbol
            proposed_position_size: Proposed position size in lots
            stop_loss_pips: Stop loss in pips

        Returns:
            Tuple of (is_allowed, reason_message)
        """
        try:
            balance = self.get_account_balance()
            if not math.isfinite(balance) or balance <= 0.0:
                return False, "Account balance is unavailable or nonpositive"
            if not math.isfinite(proposed_position_size) or proposed_position_size <= 0.0:
                return False, "Proposed position size is unavailable or nonpositive"
            if (
                not math.isfinite(self.risk_per_symbol)
                or self.risk_per_symbol <= 0.0
                or not math.isfinite(self.max_total_risk)
                or self.max_total_risk <= 0.0
            ):
                return False, "Configured per-symbol or aggregate risk limit is invalid"

            # Calculate risk for this position
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not found"

            price_diff, tick_size, tick_value = self._symbol_risk_units(symbol_info, stop_loss_pips)
            position_risk = (price_diff / tick_size) * tick_value * proposed_position_size
            position_risk_percent = position_risk / balance

            # Check per-symbol risk limit
            if position_risk_percent > self.risk_per_symbol:
                return (
                    False,
                    f"Position risk ({position_risk_percent:.2%}) exceeds per-symbol limit ({self.risk_per_symbol:.2%})",
                )

            # Check total risk across all positions
            exposure = self.get_total_exposure()
            if exposure.get("query_failed") or exposure.get("risk_unknown_positions", 0):
                return False, "Existing position stop risk is unknown; refusing additional exposure"
            new_total_risk = float(exposure["total_open_risk"]) + position_risk
            new_total_risk_percent = new_total_risk / balance

            if new_total_risk_percent > self.max_total_risk:
                return (
                    False,
                    f"Total stop risk ({new_total_risk_percent:.2%}) would exceed maximum limit ({self.max_total_risk:.2%})",
                )

            return (
                True,
                f"Risk check passed (position risk: {position_risk_percent:.2%}, total stop risk: {new_total_risk_percent:.2%})",
            )

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, f"Error checking risk limits: {e}"

    def get_volatility_multiplier(self, atr_percentage: float, base_atr: float = 1.0) -> float:
        """
        Calculate volatility multiplier for position sizing.
        Higher volatility = smaller position size.

        Args:
            atr_percentage: ATR as percentage of price
            base_atr: Base ATR percentage for normalization (default: 1.0%)

        Returns:
            Multiplier (typically 0.5 to 2.0)
        """
        if base_atr == 0:
            base_atr = 1.0

        # If ATR is higher than base, reduce position size
        # If ATR is lower than base, increase position size (up to 2x)
        multiplier = base_atr / max(atr_percentage, 0.1)

        # Clamp between 0.5 and 2.0
        multiplier = max(0.5, min(2.0, multiplier))

        return multiplier
