import MetaTrader5 as mt5
from typing import List, Dict, Optional
from loguru import logger

from mt5_trading.domain.data_sources.mt5_data import MT5Data
from mt5_trading.domain.volatility_analyzer import VolatilityAnalyzer


class MultiSymbolManager:
    """
    Manages multiple symbol data sources and their configurations.
    Handles symbol selection, data fetching, and volatility analysis.
    """
    
    def __init__(self, login: str, server: str, password: str, terminal_path: str,
                 symbols: List[str], timeframe: int = mt5.TIMEFRAME_H1):
        """
        Initialize Multi-Symbol Manager.
        
        Args:
            login: MT5 account login
            server: MT5 server name
            password: MT5 account password
            terminal_path: Path to MT5 terminal executable
            symbols: List of symbols to manage
            timeframe: MT5 timeframe for data
        """
        self.login = login
        self.server = server
        self.password = password
        self.terminal_path = terminal_path
        self.symbols = symbols
        self.timeframe = timeframe
        
        # Initialize MT5 connection
        if not mt5.initialize(path=terminal_path):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            raise RuntimeError("Failed to initialize MT5")
        
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            raise RuntimeError("Failed to login to MT5")
        
        # Initialize data sources for each symbol
        self.data_sources: Dict[str, MT5Data] = {}
        self.volatility_metrics: Dict[str, Dict] = {}
        
        self._initialize_data_sources()

    def _initialize_data_sources(self):
        """Initialize data sources for all symbols."""
        for symbol in self.symbols:
            try:
                # Select symbol in MT5
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Symbol {symbol} not available, skipping")
                    continue
                
                # Create data source
                data_source = MT5Data(
                    self.login,
                    self.server,
                    self.password,
                    self.terminal_path,
                    symbol,
                    self.timeframe
                )
                
                self.data_sources[symbol] = data_source
                logger.info(f"Initialized data source for {symbol}")
                
            except Exception as e:
                logger.error(f"Error initializing data source for {symbol}: {e}")

    def get_data_source(self, symbol: str) -> Optional[MT5Data]:
        """
        Get data source for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5Data instance or None if not found
        """
        return self.data_sources.get(symbol)

    def get_all_data_sources(self) -> Dict[str, MT5Data]:
        """
        Get all data sources.
        
        Returns:
            Dictionary mapping symbol names to MT5Data instances
        """
        return self.data_sources.copy()

    def update_volatility_metrics(self, analyzer: VolatilityAnalyzer):
        """
        Update volatility metrics for all symbols.
        
        Args:
            analyzer: VolatilityAnalyzer instance
        """
        for symbol in self.symbols:
            try:
                metrics = analyzer.calculate_volatility_metrics(symbol, self.timeframe)
                if metrics:
                    self.volatility_metrics[symbol] = metrics
            except Exception as e:
                logger.error(f"Error updating volatility metrics for {symbol}: {e}")

    def get_volatility_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Get volatility metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with volatility metrics or None
        """
        return self.volatility_metrics.get(symbol)

    def get_all_volatility_metrics(self) -> Dict[str, Dict]:
        """
        Get volatility metrics for all symbols.
        
        Returns:
            Dictionary mapping symbol names to volatility metrics
        """
        return self.volatility_metrics.copy()

    def add_symbol(self, symbol: str) -> bool:
        """
        Add a new symbol to the manager.
        
        Args:
            symbol: Trading symbol to add
            
        Returns:
            True if successfully added, False otherwise
        """
        if symbol in self.data_sources:
            logger.warning(f"Symbol {symbol} already exists")
            return True
        
        try:
            # Select symbol in MT5
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Symbol {symbol} not available")
                return False
            
            # Create data source
            data_source = MT5Data(
                self.login,
                self.server,
                self.password,
                self.terminal_path,
                symbol,
                self.timeframe
            )
            
            self.data_sources[symbol] = data_source
            self.symbols.append(symbol)
            logger.info(f"Added symbol {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the manager.
        
        Args:
            symbol: Trading symbol to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        if symbol not in self.data_sources:
            logger.warning(f"Symbol {symbol} not found")
            return False
        
        try:
            del self.data_sources[symbol]
            if symbol in self.volatility_metrics:
                del self.volatility_metrics[symbol]
            if symbol in self.symbols:
                self.symbols.remove(symbol)
            logger.info(f"Removed symbol {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    def refresh_data(self):
        """Refresh data for all symbols."""
        for symbol, data_source in self.data_sources.items():
            try:
                # Data will be refreshed when get_data() is called
                # This is just a placeholder for future optimization
                pass
            except Exception as e:
                logger.error(f"Error refreshing data for {symbol}: {e}")

    def get_active_symbols(self) -> List[str]:
        """
        Get list of active (available) symbols.
        
        Returns:
            List of symbol names that are available and initialized
        """
        return list(self.data_sources.keys())

    def __del__(self):
        """Cleanup MT5 connection."""
        mt5.shutdown()

