"""
Live Trading System for High Volatility Multi-Symbol Trading

This script implements live trading with 1-3 high volatility symbols using
multiple strategies: Crossover, SMC (Smart Money Concepts), and Trend Breakout.
"""

import os
import time
import yaml
from pathlib import Path
from typing import List, Dict
import sched
import threading
from dotenv import load_dotenv
import MetaTrader5 as mt5
from loguru import logger

from mt5_trading.domain import MT5Trader
from mt5_trading.domain.data_sources.mt5_data import MT5Data
from mt5_trading.domain.volatility_analyzer import VolatilityAnalyzer
from mt5_trading.domain.multi_symbol_manager import MultiSymbolManager
from mt5_trading.domain.risk_manager import RiskManager
from mt5_trading.domain.strategies.cross_over_strategy import CrossOverStrategy
from mt5_trading.domain.strategies.smc_strategy import SMCStrategy
from mt5_trading.domain.strategies.trend_breakout_strategy import TrendBreakoutStrategy
from mt5_trading.domain.strategies.combined_strategy import CombinedStrategy
from mt5_trading.robot.multi_symbol_robot import MultiSymbolRobot
from mt5_trading.logging_config import configure_logging

load_dotenv()
configure_logging()

# MT5 Configuration
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
LOGIN = os.getenv("LOGIN", "279260115")
PASSWORD = os.getenv("PASSWORD", "Leng3A69V@Una?")
SERVER = os.getenv("SERVER", "Exness-MT5Trial8")

# Trading Configuration
TRADING_INTERVAL_MINUTES = int(os.getenv("TRADING_INTERVAL_MINUTES", "60"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "2"))
STRATEGY_TYPE = os.getenv("STRATEGY_TYPE", "combined")  # crossover, smc, trend_breakout, combined


def load_config() -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config" / "symbols_config.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    return config


def get_symbol_candidates(config: Dict) -> List[str]:
    """Get symbol candidates from config."""
    candidates = []
    
    if 'symbol_categories' in config:
        categories = config['symbol_categories']
        
        # Add symbols from all categories
        for category, symbols in categories.items():
            if isinstance(symbols, list):
                candidates.extend(symbols)
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    return candidates


def select_volatile_symbols(candidates: List[str], max_symbols: int = 3) -> List[str]:
    """Select top volatile symbols."""
    logger.info(f"Analyzing volatility for {len(candidates)} symbol candidates...")
    
    try:
        analyzer = VolatilityAnalyzer(
            login=LOGIN,
            server=SERVER,
            password=PASSWORD,
            terminal_path=TERMINAL_PATH
        )
        
        # Get volatility config
        config = load_config()
        volatility_config = config.get('volatility_config', {})
        min_threshold = volatility_config.get('min_volatility_threshold', 0.5)
        
        # Select top volatile symbols
        selected = analyzer.select_top_volatile_symbols(
            symbols=candidates,
            max_symbols=max_symbols,
            min_volatility_threshold=min_threshold
        )
        
        return selected
        
    except Exception as e:
        logger.exception(f"Error selecting volatile symbols: {e}")
        # Fallback to first few candidates
        return candidates[:max_symbols]


def create_strategy(data_source: MT5Data, strategy_type: str):
    """Create strategy instance based on type."""
    if strategy_type == "crossover":
        return CrossOverStrategy(data_source)
    elif strategy_type == "smc":
        return SMCStrategy(data_source)
    elif strategy_type == "trend_breakout":
        return TrendBreakoutStrategy(data_source)
    elif strategy_type == "combined":
        return CombinedStrategy(data_source)
    else:
        logger.warning(f"Unknown strategy type: {strategy_type}, using combined")
        return CombinedStrategy(data_source)


def initialize_trading_system() -> MultiSymbolRobot:
    """Initialize the complete trading system."""
    logger.info("=" * 80)
    logger.info("INITIALIZING LIVE TRADING SYSTEM")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Get symbol candidates
    candidates = get_symbol_candidates(config)
    if not candidates:
        logger.error("No symbol candidates found in config")
        raise ValueError("No symbols configured")
    
    logger.info(f"Found {len(candidates)} symbol candidates")
    
    # Select volatile symbols
    selected_symbols = select_volatile_symbols(candidates, MAX_SYMBOLS)
    if not selected_symbols:
        logger.error("No symbols selected")
        raise ValueError("Failed to select symbols")
    
    logger.info(f"Selected {len(selected_symbols)} symbols: {selected_symbols}")
    
    # Get timeframe from config
    volatility_config = config.get('volatility_config', {})
    timeframe_str = volatility_config.get('timeframe', 'H1')
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    timeframe = timeframe_map.get(timeframe_str, mt5.TIMEFRAME_H1)
    
    # Initialize symbol manager
    symbol_manager = MultiSymbolManager(
        login=LOGIN,
        server=SERVER,
        password=PASSWORD,
        terminal_path=TERMINAL_PATH,
        symbols=selected_symbols,
        timeframe=timeframe
    )
    
    # Update volatility metrics
    analyzer = VolatilityAnalyzer(
        login=LOGIN,
        server=SERVER,
        password=PASSWORD,
        terminal_path=TERMINAL_PATH
    )
    symbol_manager.update_volatility_metrics(analyzer)
    
    # Create strategies for each symbol
    strategies = {}
    for symbol in selected_symbols:
        data_source = symbol_manager.get_data_source(symbol)
        if data_source:
            strategies[symbol] = create_strategy(data_source, STRATEGY_TYPE)
            logger.info(f"Created {STRATEGY_TYPE} strategy for {symbol}")
    
    # Initialize risk manager
    trading_config = config.get('trading_config', {})
    risk_per_symbol = trading_config.get('risk_per_symbol', 0.02)
    max_total_risk = trading_config.get('max_total_risk', 0.05)
    
    risk_manager = RiskManager(
        risk_per_symbol=risk_per_symbol,
        max_total_risk=max_total_risk
    )
    
    # Initialize trader
    trader = MT5Trader()
    
    # Initialize robot
    default_lot_size = trading_config.get('default_lot_size', 0.1)
    
    robot = MultiSymbolRobot(
        symbol_manager=symbol_manager,
        trader=trader,
        strategies=strategies,
        risk_manager=risk_manager,
        default_lot_size=default_lot_size
    )
    
    logger.info("=" * 80)
    logger.info("TRADING SYSTEM INITIALIZED")
    logger.info("=" * 80)
    logger.info(f"Symbols: {selected_symbols}")
    logger.info(f"Strategy: {STRATEGY_TYPE}")
    logger.info(f"Trading interval: {TRADING_INTERVAL_MINUTES} minutes")
    logger.info(f"Risk per symbol: {risk_per_symbol * 100}%")
    logger.info(f"Max total risk: {max_total_risk * 100}%")
    logger.info("=" * 80)
    
    return robot


# Scheduler setup
scheduler = sched.scheduler(time.time, time.sleep)
robot_instance = None


def run_trading_cycle():
    """Run a single trading cycle."""
    global robot_instance
    
    try:
        logger.info("=" * 80)
        logger.info(f"Starting trading cycle at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        if robot_instance is None:
            logger.error("Robot not initialized")
            return
        
        # Execute trading
        robot_instance.trade()
        
        # Log status
        status = robot_instance.get_status()
        logger.info(f"Status: {status['total_positions']} positions, "
                   f"Total P/L: ${status['total_profit']:.2f}")
        
        logger.info("Trading cycle completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"Error in trading cycle: {e}")


def schedule_trading():
    """Schedule the next trading cycle."""
    run_trading_cycle()
    scheduler.enter(TRADING_INTERVAL_MINUTES * 60, 1, schedule_trading)


def start_scheduler():
    """Start the trading scheduler."""
    scheduler.enter(0, 1, schedule_trading)
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()
    return t


def main():
    """Main function."""
    global robot_instance
    
    try:
        # Initialize trading system
        robot_instance = initialize_trading_system()
        
        # Run initial trading cycle
        logger.info("Running initial trading cycle...")
        run_trading_cycle()
        
        # Start scheduler
        logger.info(f"Starting scheduler (every {TRADING_INTERVAL_MINUTES} minutes)...")
        start_scheduler()
        
        # Keep main thread alive
        logger.info("Trading system is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
            
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

