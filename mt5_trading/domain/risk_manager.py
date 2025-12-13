import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class RiskManager:
    """
    Manages risk across multiple trading positions.
    Calculates position sizes based on volatility and risk parameters.
    Ensures total exposure doesn't exceed risk limits.
    """

    def __init__(self, risk_per_symbol: float = 0.02, max_total_risk: float = 0.05):
        """
        Initialize the risk manager.
        
        Args:
            risk_per_symbol: Maximum risk per symbol as percentage of account balance (default: 2%)
            max_total_risk: Maximum total risk across all positions (default: 5%)
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
            return account_info.balance
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
            return account_info.equity
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            return 0.0

    def calculate_position_size(self, symbol: str, stop_loss_pips: float, 
                                risk_amount: Optional[float] = None,
                                volatility_multiplier: float = 1.0) -> float:
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
            
            current_price = (tick.bid + tick.ask) / 2
            
            # Calculate risk amount if not provided
            if risk_amount is None:
                balance = self.get_account_balance()
                risk_amount = balance * self.risk_per_symbol
            
            # Adjust risk amount by volatility multiplier
            adjusted_risk = risk_amount / volatility_multiplier
            
            # Calculate tick size and value
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            contract_size = symbol_info.trade_contract_size
            
            # Convert stop loss pips to price difference
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                # 5-digit or 3-digit broker (pip = 10 points)
                price_diff = stop_loss_pips * tick_size * 10
            else:
                # Standard broker (pip = 1 point)
                price_diff = stop_loss_pips * tick_size
            
            if price_diff == 0:
                logger.warning(f"Invalid stop loss for {symbol}: {stop_loss_pips} pips")
                return 0.0
            
            # Calculate position size
            # Risk amount = (price_diff / tick_size) * tick_value * lot_size
            # lot_size = risk_amount / ((price_diff / tick_size) * tick_value)
            position_size = adjusted_risk / (price_diff * tick_value / tick_size)
            
            # Round to valid lot size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            # Round to nearest lot step
            position_size = round(position_size / lot_step) * lot_step
            
            # Clamp to valid range
            position_size = max(min_lot, min(position_size, max_lot))
            
            logger.info(f"Calculated position size for {symbol}: {position_size} lots "
                       f"(risk: ${adjusted_risk:.2f}, SL: {stop_loss_pips} pips, "
                       f"volatility_mult: {volatility_multiplier:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def get_total_exposure(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get total exposure across all open positions.
        
        Args:
            symbols: Optional list of symbols to filter by
            
        Returns:
            Dictionary with exposure metrics
        """
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return {
                    'total_positions': 0,
                    'total_volume': 0.0,
                    'total_profit': 0.0,
                    'symbols': {}
                }
            
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            
            # Filter by symbols if provided
            if symbols:
                df = df[df['symbol'].isin(symbols)]
            
            total_volume = df['volume'].sum()
            total_profit = df['profit'].sum()
            
            # Group by symbol
            symbol_exposure = {}
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                symbol_exposure[symbol] = {
                    'positions': len(symbol_df),
                    'volume': symbol_df['volume'].sum(),
                    'profit': symbol_df['profit'].sum()
                }
            
            return {
                'total_positions': len(df),
                'total_volume': total_volume,
                'total_profit': total_profit,
                'symbols': symbol_exposure
            }
        except Exception as e:
            logger.error(f"Error getting total exposure: {e}")
            return {
                'total_positions': 0,
                'total_volume': 0.0,
                'total_profit': 0.0,
                'symbols': {}
            }

    def check_risk_limits(self, symbol: str, proposed_position_size: float,
                         stop_loss_pips: float) -> Tuple[bool, str]:
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
            if balance == 0:
                return False, "Account balance is zero"
            
            # Calculate risk for this position
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not found"
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False, f"Failed to get tick for {symbol}"
            
            current_price = (tick.bid + tick.ask) / 2
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            
            # Calculate price difference for stop loss
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                price_diff = stop_loss_pips * tick_size * 10
            else:
                price_diff = stop_loss_pips * tick_size
            
            position_risk = abs(price_diff * tick_value * proposed_position_size / tick_size)
            position_risk_percent = (position_risk / balance) * 100 if balance > 0 else 0
            
            # Check per-symbol risk limit
            if position_risk_percent > (self.risk_per_symbol * 100):
                return False, f"Position risk ({position_risk_percent:.2f}%) exceeds per-symbol limit ({self.risk_per_symbol * 100:.2f}%)"
            
            # Check total risk across all positions
            exposure = self.get_total_exposure()
            current_total_risk = abs(exposure['total_profit']) / balance * 100 if balance > 0 else 0
            new_total_risk = current_total_risk + position_risk_percent
            
            if new_total_risk > (self.max_total_risk * 100):
                return False, f"Total risk ({new_total_risk:.2f}%) would exceed maximum limit ({self.max_total_risk * 100:.2f}%)"
            
            return True, f"Risk check passed (position risk: {position_risk_percent:.2f}%, total risk: {new_total_risk:.2f}%)"
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, f"Error checking risk limits: {e}"

    def get_volatility_multiplier(self, atr_percentage: float, 
                                 base_atr: float = 1.0) -> float:
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

