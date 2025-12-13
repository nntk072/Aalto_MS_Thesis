import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict

from mt5_trading.adapters import TradingStrategy, TradingData
from mt5_trading.domain.signal import Signal


class SMCStrategy(TradingStrategy):
    """
    Smart Money Concepts (SMC) Strategy
    
    This strategy identifies:
    - Order Blocks (supply/demand zones)
    - Liquidity zones (stop hunts)
    - Market structure breaks
    - Fair Value Gaps (FVG)
    """
    
    def __init__(self, trading_data: TradingData, 
                 order_block_periods: int = 20,
                 fvg_lookback: int = 3) -> None:
        """
        Initialize SMC Strategy.
        
        Args:
            trading_data: Data source for trading
            order_block_periods: Number of periods to look back for order blocks
            fvg_lookback: Number of periods to look back for Fair Value Gaps
        """
        self.data = trading_data
        self.order_block_periods = order_block_periods
        self.fvg_lookback = fvg_lookback

    def identify_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Identify bullish and bearish order blocks.
        
        Order blocks are areas where institutional traders placed large orders.
        - Bullish order block: Strong bullish candle followed by consolidation
        - Bearish order block: Strong bearish candle followed by consolidation
        
        Returns:
            Tuple of (bullish_blocks, bearish_blocks) as boolean series
        """
        bullish_blocks = pd.Series([False] * len(df), index=df.index)
        bearish_blocks = pd.Series([False] * len(df), index=df.index)
        
        for i in range(self.order_block_periods, len(df)):
            # Look for strong bullish candle
            if df.iloc[i]['close'] > df.iloc[i]['open']:
                body_size = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
                candle_range = df.iloc[i]['high'] - df.iloc[i]['low']
                
                # Strong bullish candle (body > 60% of range)
                if body_size > 0.6 * candle_range and candle_range > 0:
                    # Check if followed by consolidation or small moves
                    if i < len(df) - 1:
                        next_candle_range = df.iloc[i+1]['high'] - df.iloc[i+1]['low']
                        if next_candle_range < 0.5 * candle_range:
                            bullish_blocks.iloc[i] = True
            
            # Look for strong bearish candle
            if df.iloc[i]['close'] < df.iloc[i]['open']:
                body_size = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
                candle_range = df.iloc[i]['high'] - df.iloc[i]['low']
                
                # Strong bearish candle (body > 60% of range)
                if body_size > 0.6 * candle_range and candle_range > 0:
                    # Check if followed by consolidation
                    if i < len(df) - 1:
                        next_candle_range = df.iloc[i+1]['high'] - df.iloc[i+1]['low']
                        if next_candle_range < 0.5 * candle_range:
                            bearish_blocks.iloc[i] = True
        
        return bullish_blocks, bearish_blocks

    def identify_fair_value_gaps(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Identify bullish and bearish Fair Value Gaps (FVG).
        
        FVG occurs when there's a gap between candles that gets filled later.
        - Bullish FVG: Gap between low of previous candle and high of next candle
        - Bearish FVG: Gap between high of previous candle and low of next candle
        
        Returns:
            Tuple of (bullish_fvg, bearish_fvg) as boolean series
        """
        bullish_fvg = pd.Series([False] * len(df), index=df.index)
        bearish_fvg = pd.Series([False] * len(df), index=df.index)
        
        for i in range(1, len(df) - 1):
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low']
            curr_high = df.iloc[i]['high']
            curr_low = df.iloc[i]['low']
            next_high = df.iloc[i+1]['high']
            next_low = df.iloc[i+1]['low']
            
            # Bullish FVG: gap between prev_low and next_high
            if next_low > prev_high and curr_low > prev_high:
                bullish_fvg.iloc[i] = True
            
            # Bearish FVG: gap between prev_high and next_low
            if next_high < prev_low and curr_high < prev_low:
                bearish_fvg.iloc[i] = True
        
        return bullish_fvg, bearish_fvg

    def identify_market_structure(self, df: pd.DataFrame) -> str:
        """
        Identify market structure (trend direction).
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(df) < 20:
            return 'neutral'
        
        # Use higher highs and higher lows for bullish
        # Use lower highs and lower lows for bearish
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()
        
        recent_highs = highs.iloc[-10:].values
        recent_lows = lows.iloc[-10:].values
        
        # Check for higher highs
        higher_highs = sum(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = sum(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        # Check for lower highs
        lower_highs = sum(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        lower_lows = sum(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if higher_highs >= 3 and higher_lows >= 3:
            return 'bullish'
        elif lower_highs >= 3 and lower_lows >= 3:
            return 'bearish'
        else:
            return 'neutral'

    def signal(self) -> Tuple[str, Signal]:
        """
        Generate trading signal based on SMC concepts.
        
        Returns:
            Tuple of (symbol, signal)
        """
        df: pd.DataFrame = self.data.get_data()
        
        if len(df) < self.order_block_periods + 5:
            symbol = self.data.get_symbol()
            return symbol, Signal.NONE
        
        # Identify order blocks
        bullish_blocks, bearish_blocks = self.identify_order_blocks(df)
        
        # Identify Fair Value Gaps
        bullish_fvg, bearish_fvg = self.identify_fair_value_gaps(df)
        
        # Identify market structure
        market_structure = self.identify_market_structure(df)
        
        # Calculate EMA for trend confirmation
        ema_fast = df['close'].ewm(span=9, adjust=False).mean()
        ema_slow = df['close'].ewm(span=21, adjust=False).mean()
        
        # Recent order blocks and FVGs
        recent_bullish_blocks = bullish_blocks.iloc[-self.order_block_periods:].any()
        recent_bearish_blocks = bearish_blocks.iloc[-self.order_block_periods:].any()
        recent_bullish_fvg = bullish_fvg.iloc[-self.fvg_lookback:].any()
        recent_bearish_fvg = bearish_fvg.iloc[-self.fvg_lookback:].any()
        
        # Current price position
        current_price = df['close'].iloc[-1]
        price_above_ema_fast = current_price > ema_fast.iloc[-1]
        price_above_ema_slow = current_price > ema_slow.iloc[-1]
        
        symbol = self.data.get_symbol()
        
        # Buy signal: Bullish structure + bullish order block/FVG + price above EMAs
        if market_structure == 'bullish':
            if (recent_bullish_blocks or recent_bullish_fvg) and price_above_ema_fast:
                return symbol, Signal.BUY
        
        # Sell signal: Bearish structure + bearish order block/FVG + price below EMAs
        if market_structure == 'bearish':
            if (recent_bearish_blocks or recent_bearish_fvg) and not price_above_ema_fast:
                return symbol, Signal.SELL
        
        # Additional confirmation: EMA crossover
        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
            if recent_bullish_blocks or recent_bullish_fvg:
                return symbol, Signal.BUY
        
        if ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
            if recent_bearish_blocks or recent_bearish_fvg:
                return symbol, Signal.SELL
        
        return symbol, Signal.NONE

