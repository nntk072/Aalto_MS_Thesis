import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict

from mt5_trading.adapters import TradingStrategy, TradingData
from mt5_trading.domain.signal import Signal


class TrendBreakoutStrategy(TradingStrategy):
    """
    Trend Breakout Strategy
    
    This strategy identifies:
    - Support and resistance levels
    - Breakouts from consolidation patterns
    - Trend continuation signals
    - Volume confirmation
    """
    
    def __init__(self, trading_data: TradingData,
                 lookback_period: int = 20,
                 breakout_threshold: float = 0.001) -> None:
        """
        Initialize Trend Breakout Strategy.
        
        Args:
            trading_data: Data source for trading
            lookback_period: Period to identify support/resistance levels
            breakout_threshold: Percentage threshold for breakout confirmation (default: 0.1%)
        """
        self.data = trading_data
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold

    def identify_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Identify support and resistance levels.
        
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if len(df) < self.lookback_period:
            return 0.0, 0.0
        
        recent_data = df.iloc[-self.lookback_period:]
        
        # Support: lowest low in lookback period
        support_level = recent_data['low'].min()
        
        # Resistance: highest high in lookback period
        resistance_level = recent_data['high'].max()
        
        return support_level, resistance_level

    def identify_consolidation(self, df: pd.DataFrame) -> bool:
        """
        Identify if market is in consolidation (ranging).
        
        Returns:
            True if market is consolidating, False otherwise
        """
        if len(df) < self.lookback_period:
            return False
        
        recent_data = df.iloc[-self.lookback_period:]
        
        # Calculate price range
        price_range = recent_data['high'].max() - recent_data['low'].min()
        avg_price = recent_data['close'].mean()
        
        # Consolidation: small price range relative to average price
        range_percentage = (price_range / avg_price) * 100 if avg_price > 0 else 0
        
        # Consider it consolidation if range is less than 1% of average price
        return range_percentage < 1.0

    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume-weighted average price (VWAP) for trend confirmation.
        
        Returns:
            VWAP series
        """
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
        elif 'real_volume' in df.columns:
            volume = df['real_volume']
        else:
            # Use typical price as proxy for volume
            volume = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap

    def detect_breakout(self, df: pd.DataFrame, support: float, resistance: float) -> Tuple[bool, str]:
        """
        Detect breakout from support or resistance.
        
        Returns:
            Tuple of (is_breakout, direction) where direction is 'bullish' or 'bearish'
        """
        if len(df) < 2:
            return False, ''
        
        current_close = df['close'].iloc[-1]
        previous_close = df['close'].iloc[-2]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # Bullish breakout: price breaks above resistance
        if current_high > resistance:
            # Confirm with threshold
            breakout_amount = (current_high - resistance) / resistance if resistance > 0 else 0
            if breakout_amount >= self.breakout_threshold:
                return True, 'bullish'
        
        # Bearish breakout: price breaks below support
        if current_low < support:
            # Confirm with threshold
            breakout_amount = (support - current_low) / support if support > 0 else 0
            if breakout_amount >= self.breakout_threshold:
                return True, 'bearish'
        
        return False, ''

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength using ADX (Average Directional Index).
        
        Returns:
            ADX value (0-100, higher = stronger trend)
        """
        if len(df) < 14:
            return 0.0
        
        try:
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            return adx[-1] if not np.isnan(adx[-1]) else 0.0
        except:
            return 0.0

    def signal(self) -> Tuple[str, Signal]:
        """
        Generate trading signal based on trend breakouts.
        
        Returns:
            Tuple of (symbol, signal)
        """
        df: pd.DataFrame = self.data.get_data()
        
        if len(df) < self.lookback_period + 5:
            symbol = self.data.get_symbol()
            return symbol, Signal.NONE
        
        # Identify support and resistance
        support, resistance = self.identify_support_resistance(df)
        
        if support == 0.0 or resistance == 0.0:
            symbol = self.data.get_symbol()
            return symbol, Signal.NONE
        
        # Detect breakout
        is_breakout, direction = self.detect_breakout(df, support, resistance)
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(df)
        
        # Calculate moving averages for confirmation
        sma_fast = df['close'].rolling(window=9).mean()
        sma_slow = df['close'].rolling(window=21).mean()
        
        current_price = df['close'].iloc[-1]
        price_above_sma_fast = current_price > sma_fast.iloc[-1]
        price_above_sma_slow = current_price > sma_slow.iloc[-1]
        
        # Calculate VWAP
        vwap = self.calculate_volume_profile(df)
        price_above_vwap = current_price > vwap.iloc[-1]
        
        symbol = self.data.get_symbol()
        
        # Bullish breakout signal
        if is_breakout and direction == 'bullish':
            # Confirm with trend strength and moving averages
            if trend_strength > 25:  # Strong trend
                if price_above_sma_fast and price_above_sma_slow:
                    return symbol, Signal.BUY
        
        # Bearish breakout signal
        if is_breakout and direction == 'bearish':
            # Confirm with trend strength and moving averages
            if trend_strength > 25:  # Strong trend
                if not price_above_sma_fast and not price_above_sma_slow:
                    return symbol, Signal.SELL
        
        # Trend continuation signals (if already in a trend)
        if trend_strength > 30:
            # Bullish trend continuation
            if price_above_sma_fast and price_above_sma_slow and price_above_vwap:
                if sma_fast.iloc[-1] > sma_slow.iloc[-1]:
                    # Check if price is near support (potential bounce)
                    distance_to_support = (current_price - support) / support if support > 0 else 0
                    if 0 < distance_to_support < 0.005:  # Within 0.5% of support
                        return symbol, Signal.BUY
            
            # Bearish trend continuation
            if not price_above_sma_fast and not price_above_sma_slow and not price_above_vwap:
                if sma_fast.iloc[-1] < sma_slow.iloc[-1]:
                    # Check if price is near resistance (potential rejection)
                    distance_to_resistance = (resistance - current_price) / current_price if current_price > 0 else 0
                    if 0 < distance_to_resistance < 0.005:  # Within 0.5% of resistance
                        return symbol, Signal.SELL
        
        return symbol, Signal.NONE

