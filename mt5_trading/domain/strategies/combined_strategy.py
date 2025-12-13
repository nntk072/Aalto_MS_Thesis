from typing import Tuple, List

from mt5_trading.adapters import TradingStrategy, TradingData
from mt5_trading.domain.signal import Signal
from mt5_trading.domain.strategies.cross_over_strategy import CrossOverStrategy
from mt5_trading.domain.strategies.smc_strategy import SMCStrategy
from mt5_trading.domain.strategies.trend_breakout_strategy import TrendBreakoutStrategy


class CombinedStrategy(TradingStrategy):
    """
    Combined Strategy that uses multiple strategies and requires consensus.
    
    Can combine:
    - Crossover Strategy
    - SMC (Smart Money Concepts) Strategy
    - Trend Breakout Strategy
    """
    
    def __init__(self, trading_data: TradingData,
                 strategies: List[str] = None,
                 require_all: bool = False) -> None:
        """
        Initialize Combined Strategy.
        
        Args:
            trading_data: Data source for trading
            strategies: List of strategy names to use ['crossover', 'smc', 'trend_breakout']
                       If None, uses all strategies
            require_all: If True, all strategies must agree. If False, majority wins.
        """
        self.data = trading_data
        self.require_all = require_all
        
        # Initialize strategies
        self.strategies = {}
        
        if strategies is None:
            strategies = ['crossover', 'smc', 'trend_breakout']
        
        if 'crossover' in strategies:
            self.strategies['crossover'] = CrossOverStrategy(trading_data)
        
        if 'smc' in strategies:
            self.strategies['smc'] = SMCStrategy(trading_data)
        
        if 'trend_breakout' in strategies:
            self.strategies['trend_breakout'] = TrendBreakoutStrategy(trading_data)

    def signal(self) -> Tuple[str, Signal]:
        """
        Generate trading signal based on combined strategies.
        
        Returns:
            Tuple of (symbol, signal)
        """
        if len(self.strategies) == 0:
            symbol = self.data.get_symbol()
            return symbol, Signal.NONE
        
        # Get signals from all strategies
        signals = []
        symbol = self.data.get_symbol()
        
        for strategy_name, strategy in self.strategies.items():
            try:
                _, signal = strategy.signal()
                signals.append(signal)
            except Exception as e:
                # If strategy fails, skip it
                continue
        
        if len(signals) == 0:
            return symbol, Signal.NONE
        
        # Count signals
        buy_count = sum(1 for s in signals if s == Signal.BUY)
        sell_count = sum(1 for s in signals if s == Signal.SELL)
        none_count = sum(1 for s in signals if s == Signal.NONE)
        
        # Determine final signal
        if self.require_all:
            # All strategies must agree
            if buy_count == len(signals):
                return symbol, Signal.BUY
            elif sell_count == len(signals):
                return symbol, Signal.SELL
            else:
                return symbol, Signal.NONE
        else:
            # Majority wins (or at least 2 out of 3)
            total_strategies = len(signals)
            majority_threshold = (total_strategies + 1) // 2
            
            if buy_count >= majority_threshold:
                return symbol, Signal.BUY
            elif sell_count >= majority_threshold:
                return symbol, Signal.SELL
            else:
                return symbol, Signal.NONE

