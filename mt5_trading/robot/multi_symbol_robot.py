import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
from loguru import logger

from mt5_trading.adapters import Trader, TradingStrategy
from mt5_trading.domain.multi_symbol_manager import MultiSymbolManager
from mt5_trading.domain.risk_manager import RiskManager
from mt5_trading.domain.signal import Signal


class MultiSymbolRobot:
    """
    Multi-Symbol Trading Robot that can trade multiple symbols simultaneously.
    Supports different strategies (Crossover, SMC, Trend Breakout, Combined) per symbol.
    """
    
    def __init__(self, symbol_manager: MultiSymbolManager,
                 trader: Trader,
                 strategies: Dict[str, TradingStrategy],
                 risk_manager: RiskManager,
                 default_lot_size: float = 0.1,
                 stop_loss_pips: float = 50.0):
        """
        Initialize Multi-Symbol Robot.
        
        Args:
            symbol_manager: MultiSymbolManager instance
            trader: Trader instance for executing trades
            strategies: Dictionary mapping symbol names to TradingStrategy instances
            risk_manager: RiskManager instance
            default_lot_size: Default lot size if risk-based sizing fails
            stop_loss_pips: Default stop loss in pips
        """
        self.symbol_manager = symbol_manager
        self.trader = trader
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.default_lot_size = default_lot_size
        self.stop_loss_pips = stop_loss_pips
        self.magic_number = 20240101
        self.name = 'Multi-Symbol Robot'
        
        logger.info(f"Initialized {self.name} with {len(strategies)} symbols")

    def calculate_position_size(self, symbol: str, volatility_multiplier: float = 1.0) -> float:
        """
        Calculate position size for a symbol based on risk management.
        
        Args:
            symbol: Trading symbol
            volatility_multiplier: Multiplier based on volatility
            
        Returns:
            Position size in lots
        """
        try:
            # Get volatility metrics
            volatility_metrics = self.symbol_manager.get_volatility_metrics(symbol)
            
            if volatility_metrics:
                atr_percentage = volatility_metrics.get('atr_percentage', 1.0)
                # Calculate volatility multiplier if not provided
                if volatility_multiplier == 1.0:
                    volatility_multiplier = self.risk_manager.get_volatility_multiplier(atr_percentage)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=self.stop_loss_pips,
                volatility_multiplier=volatility_multiplier
            )
            
            # Fallback to default if calculation fails
            if position_size == 0.0:
                position_size = self.default_lot_size
                logger.warning(f"Using default lot size for {symbol}: {position_size}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return self.default_lot_size

    def check_risk_before_trade(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """
        Check risk limits before opening a position.
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            
        Returns:
            Tuple of (is_allowed, reason_message)
        """
        return self.risk_manager.check_risk_limits(symbol, position_size, self.stop_loss_pips)

    def trade_symbol(self, symbol: str):
        """
        Execute trading logic for a single symbol.
        
        Args:
            symbol: Trading symbol to trade
        """
        try:
            # Get strategy for this symbol
            strategy = self.strategies.get(symbol)
            if strategy is None:
                logger.warning(f"No strategy found for {symbol}")
                return
            
            # Get signal from strategy
            logger.info(f"Checking signals for {symbol}")
            signal_symbol, signal = strategy.signal()
            
            if signal_symbol != symbol:
                logger.warning(f"Strategy returned different symbol: {signal_symbol} vs {symbol}")
            
            # Process buy signal
            if signal == Signal.BUY:
                total_buy, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_BUY)
                
                if total_buy == 0:
                    # Calculate position size
                    position_size = self.calculate_position_size(symbol)
                    
                    # Check risk limits
                    is_allowed, reason = self.check_risk_before_trade(symbol, position_size)
                    if not is_allowed:
                        logger.warning(f"Trade not allowed for {symbol}: {reason}")
                        return
                    
                    logger.info(f"Buy signal detected for {symbol}, opening position")
                    result = self.trader.open_position(
                        symbol,
                        position_size,
                        mt5.ORDER_TYPE_BUY,
                        f"{self.name} buy position",
                        self.magic_number,
                        sl=None,  # Stop loss can be calculated based on ATR
                        tp=None   # Take profit can be calculated based on ATR
                    )
                    
                    if result is None:
                        return  # AutoTrading disabled or error
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Buy position opened for {symbol}: "
                                   f"Order #{result.order}, Volume: {result.volume}, Price: {result.price}")
                    else:
                        logger.error(f"Failed to open buy position for {symbol}: {result.retcode}")
                else:
                    logger.info(f"Buy position already exists for {symbol}")
                
                # Close opposite positions
                total_sell, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_SELL)
                if total_sell > 0:
                    logger.info(f"Closing existing sell positions for {symbol}")
                    self.trader.close_positions(self.name, symbol, mt5.ORDER_TYPE_SELL)
            
            # Process sell signal
            elif signal == Signal.SELL:
                total_sell, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_SELL)
                
                if total_sell == 0:
                    # Calculate position size
                    position_size = self.calculate_position_size(symbol)
                    
                    # Check risk limits
                    is_allowed, reason = self.check_risk_before_trade(symbol, position_size)
                    if not is_allowed:
                        logger.warning(f"Trade not allowed for {symbol}: {reason}")
                        return
                    
                    logger.info(f"Sell signal detected for {symbol}, opening position")
                    result = self.trader.open_position(
                        symbol,
                        position_size,
                        mt5.ORDER_TYPE_SELL,
                        f"{self.name} sell position",
                        self.magic_number,
                        sl=None,
                        tp=None
                    )
                    
                    if result is None:
                        return  # AutoTrading disabled or error
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Sell position opened for {symbol}: "
                                   f"Order #{result.order}, Volume: {result.volume}, Price: {result.price}")
                    else:
                        logger.error(f"Failed to open sell position for {symbol}: {result.retcode}")
                else:
                    logger.info(f"Sell position already exists for {symbol}")
                
                # Close opposite positions
                total_buy, _ = self.trader.get_opened_positions(symbol, mt5.ORDER_TYPE_BUY)
                if total_buy > 0:
                    logger.info(f"Closing existing buy positions for {symbol}")
                    self.trader.close_positions(self.name, symbol, mt5.ORDER_TYPE_BUY)
            
            # No signal
            elif signal == Signal.NONE:
                logger.info(f"No trading signal for {symbol}")
            
        except Exception as e:
            logger.exception(f"Error trading {symbol}: {e}")

    def trade(self):
        """
        Execute trading logic for all symbols.
        """
        logger.info("=" * 80)
        logger.info(f"Starting trading cycle for {len(self.strategies)} symbols")
        logger.info("=" * 80)
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Account Balance: ${account_info.balance:.2f}, "
                       f"Equity: ${account_info.equity:.2f}, "
                       f"Margin: ${account_info.margin:.2f}")
        
        # Get current exposure
        exposure = self.risk_manager.get_total_exposure()
        logger.info(f"Current exposure: {exposure['total_positions']} positions, "
                   f"Total volume: {exposure['total_volume']:.2f} lots, "
                   f"Floating P/L: ${exposure['total_profit']:.2f}")
        
        # Trade each symbol
        for symbol in self.strategies.keys():
            try:
                self.trade_symbol(symbol)
            except Exception as e:
                logger.exception(f"Error in trading cycle for {symbol}: {e}")
        
        logger.info("=" * 80)
        logger.info("Trading cycle completed")
        logger.info("=" * 80)
        logger.info("")

    def get_status(self) -> Dict:
        """
        Get current status of the robot.
        
        Returns:
            Dictionary with robot status information
        """
        exposure = self.risk_manager.get_total_exposure()
        account_info = mt5.account_info()
        
        status = {
            'symbols': list(self.strategies.keys()),
            'total_positions': exposure['total_positions'],
            'total_volume': exposure['total_volume'],
            'total_profit': exposure['total_profit'],
            'symbol_exposure': exposure['symbols'],
            'account_balance': account_info.balance if account_info else 0.0,
            'account_equity': account_info.equity if account_info else 0.0
        }
        
        return status

