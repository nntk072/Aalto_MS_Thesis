import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class VolatilityAnalyzer:
    """
    Analyzes symbol volatility using ATR (Average True Range) and other metrics.
    Ranks symbols by volatility to identify the most volatile trading opportunities.
    """

    def __init__(self, login: str, server: str, password: str, terminal_path: str, 
                 atr_period: int = 14, lookback_period: int = 20):
        """
        Initialize the volatility analyzer.
        
        Args:
            login: MT5 account login
            server: MT5 server name
            password: MT5 account password
            terminal_path: Path to MT5 terminal executable
            atr_period: Period for ATR calculation (default: 14)
            lookback_period: Number of periods to look back for volatility calculation (default: 20)
        """
        self.login = login
        self.server = server
        self.password = password
        self.terminal_path = terminal_path
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        
        # Initialize MT5 connection
        if not mt5.initialize(path=terminal_path):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            raise RuntimeError("Failed to initialize MT5")
        
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            raise RuntimeError("Failed to login to MT5")

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate Average True Range (ATR) for the given dataframe.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Average ATR value over the lookback period
        """
        if len(df) < self.atr_period + 1:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate True Range
        tr_list = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        if len(tr_list) < self.atr_period:
            return 0.0
        
        # Calculate ATR as simple moving average of TR
        atr_values = []
        for i in range(self.atr_period - 1, len(tr_list)):
            atr = np.mean(tr_list[i - self.atr_period + 1:i + 1])
            atr_values.append(atr)
        
        # Return average ATR over lookback period
        if len(atr_values) == 0:
            return 0.0
        
        return np.mean(atr_values[-self.lookback_period:]) if len(atr_values) >= self.lookback_period else np.mean(atr_values)

    def calculate_volatility_metrics(self, symbol: str, timeframe: int = mt5.TIMEFRAME_H1) -> Dict:
        """
        Calculate volatility metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            # Select symbol
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available")
                return None
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data available for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            if len(df) < self.atr_period + self.lookback_period:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate ATR
            atr = self.calculate_atr(df)
            
            # Calculate price range volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0  # Annualized
            
            # Calculate price range percentage
            price_range = (df['high'].max() - df['low'].min()) / df['close'].mean() * 100
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            current_price = (tick.bid + tick.ask) / 2 if tick else df['close'].iloc[-1]
            
            # Normalize ATR by price (ATR percentage)
            atr_percentage = (atr / current_price * 100) if current_price > 0 else 0.0
            
            return {
                'symbol': symbol,
                'atr': atr,
                'atr_percentage': atr_percentage,
                'volatility': volatility,
                'price_range': price_range,
                'current_price': current_price,
                'score': atr_percentage * 0.6 + volatility * 0.4  # Combined score
            }
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None

    def rank_symbols_by_volatility(self, symbols: List[str], 
                                   timeframe: int = mt5.TIMEFRAME_H1) -> List[Dict]:
        """
        Rank symbols by volatility (highest first).
        
        Args:
            symbols: List of symbols to analyze
            timeframe: MT5 timeframe for analysis
            
        Returns:
            List of symbol metrics sorted by volatility score (highest first)
        """
        results = []
        
        for symbol in symbols:
            metrics = self.calculate_volatility_metrics(symbol, timeframe)
            if metrics:
                results.append(metrics)
        
        # Sort by combined score (highest volatility first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results

    def select_top_volatile_symbols(self, symbols: List[str], 
                                    max_symbols: int = 3,
                                    min_volatility_threshold: float = 0.0,
                                    timeframe: int = mt5.TIMEFRAME_H1) -> List[str]:
        """
        Select top N most volatile symbols.
        
        Args:
            symbols: List of symbols to analyze
            max_symbols: Maximum number of symbols to select (default: 3)
            min_volatility_threshold: Minimum volatility score threshold
            timeframe: MT5 timeframe for analysis
            
        Returns:
            List of top volatile symbol names
        """
        ranked = self.rank_symbols_by_volatility(symbols, timeframe)
        
        # Filter by threshold and take top N
        filtered = [r['symbol'] for r in ranked if r['score'] >= min_volatility_threshold]
        selected = filtered[:max_symbols]
        
        logger.info(f"Selected {len(selected)} symbols: {selected}")
        for symbol_info in ranked[:max_symbols]:
            logger.info(f"  {symbol_info['symbol']}: ATR%={symbol_info['atr_percentage']:.2f}, "
                       f"Volatility={symbol_info['volatility']:.4f}, Score={symbol_info['score']:.4f}")
        
        return selected

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get basic information about a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            return {
                'symbol': symbol,
                'name': symbol_info.name,
                'description': symbol_info.description,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'trade_mode': symbol_info.trade_mode,
                'visible': symbol_info.visible,
                'select': symbol_info.select
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def __del__(self):
        """Cleanup MT5 connection."""
        mt5.shutdown()

