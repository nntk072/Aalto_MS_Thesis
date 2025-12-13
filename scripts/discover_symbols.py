"""
Symbol Discovery Utility

This script discovers available symbols on the MT5 broker and categorizes them.
Helps identify correct symbol names for trading.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import MetaTrader5 as mt5
from dotenv import load_dotenv
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# MT5 Configuration
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
LOGIN = os.getenv("LOGIN")
PASSWORD = os.getenv("PASSWORD")
SERVER = os.getenv("SERVER")


def categorize_symbol(symbol: str, description: str = "") -> str:
    """
    Categorize a symbol based on its name and description.
    
    Args:
        symbol: Symbol name
        description: Symbol description
        
    Returns:
        Category name
    """
    symbol_upper = symbol.upper()
    desc_upper = description.upper()
    
    # Crypto
    if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'USDT', 'CRYPTO']):
        return 'crypto'
    
    # Gold and Precious Metals
    if any(metal in symbol_upper for metal in ['XAU', 'GOLD', 'XAG', 'SILVER', 'XPD', 'PALLADIUM', 'XPT', 'PLATINUM']):
        return 'commodities'
    
    # Oil and Energy
    if any(energy in symbol_upper for energy in ['OIL', 'WTI', 'BRENT', 'NGAS', 'GAS', 'CRUDE']):
        return 'commodities'
    
    # Other Commodities
    if any(commodity in symbol_upper for commodity in ['COPPER', 'CORN', 'WHEAT', 'SOY', 'SUGAR', 'COFFEE', 'COTTON']):
        return 'commodities'
    
    # Forex (major pairs)
    major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
    if symbol_upper in major_pairs:
        return 'forex'
    
    # Forex (other pairs - contains currency codes)
    currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'SGD', 'HKD', 'NOK', 'SEK', 'DKK', 'ZAR', 'MXN']
    if any(code in symbol_upper for code in currency_codes) and len(symbol_upper) == 6:
        return 'forex'
    
    # ETF
    if any(etf in symbol_upper for etf in ['ETF', 'SPY', 'QQQ', 'DIA', 'IWM', 'VTI']):
        return 'etf'
    
    # Stocks (if available)
    if '.' in symbol_upper or len(symbol_upper) <= 5:
        return 'stocks'
    
    return 'other'


def discover_symbols() -> Dict[str, List[Dict]]:
    """
    Discover all available symbols on the broker.
    
    Returns:
        Dictionary with categorized symbols
    """
    # Initialize MT5
    if not mt5.initialize(path=TERMINAL_PATH):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return {}
    
    if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return {}
    
    logger.info("Discovering symbols...")
    
    # Get all symbols
    symbols = mt5.symbols_get()
    if symbols is None:
        logger.error("Failed to get symbols")
        mt5.shutdown()
        return {}
    
    logger.info(f"Found {len(symbols)} total symbols")
    
    # Categorize symbols
    categorized = {
        'forex': [],
        'crypto': [],
        'commodities': [],
        'etf': [],
        'stocks': [],
        'other': []
    }
    
    for symbol_info in symbols:
        symbol = symbol_info.name
        description = symbol_info.description or ""
        
        # Only include visible and tradeable symbols
        if not symbol_info.visible:
            continue
        
        category = categorize_symbol(symbol, description)
        
        symbol_data = {
            'symbol': symbol,
            'description': description,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit,
            'trade_mode': symbol_info.trade_mode,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step
        }
        
        categorized[category].append(symbol_data)
    
    # Sort each category by symbol name
    for category in categorized:
        categorized[category].sort(key=lambda x: x['symbol'])
    
    mt5.shutdown()
    
    return categorized


def print_symbols(categorized: Dict[str, List[Dict]]):
    """Print discovered symbols in a formatted way."""
    print("\n" + "=" * 80)
    print("DISCOVERED SYMBOLS")
    print("=" * 80)
    
    for category, symbols in categorized.items():
        if len(symbols) == 0:
            continue
        
        print(f"\n{category.upper()} ({len(symbols)} symbols):")
        print("-" * 80)
        
        for symbol_data in symbols[:20]:  # Show first 20
            print(f"  {symbol_data['symbol']:15} - {symbol_data['description']}")
        
        if len(symbols) > 20:
            print(f"  ... and {len(symbols) - 20} more")


def save_symbols_to_config(categorized: Dict[str, List[Dict]], config_path: str):
    """
    Save discovered symbols to config file.
    
    Args:
        categorized: Categorized symbols dictionary
        config_path: Path to config file
    """
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Read existing config if it exists
    existing_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f) or {}
    
    # Update symbol categories
    if 'symbol_categories' not in existing_config:
        existing_config['symbol_categories'] = {}
    
    for category, symbols in categorized.items():
        if len(symbols) > 0:
            symbol_names = [s['symbol'] for s in symbols]
            existing_config['symbol_categories'][category] = symbol_names
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(existing_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved symbols to {config_path}")


def main():
    """Main function."""
    logger.info("Starting symbol discovery...")
    
    if not LOGIN or not PASSWORD or not SERVER:
        logger.error("Missing MT5 credentials in .env file")
        logger.error("Please set LOGIN, PASSWORD, and SERVER in .env")
        return
    
    # Discover symbols
    categorized = discover_symbols()
    
    if not categorized:
        logger.error("No symbols discovered")
        return
    
    # Print results
    print_symbols(categorized)
    
    # Save to config
    config_path = Path(__file__).parent.parent / "config" / "symbols_config.yaml"
    save_symbols_to_config(categorized, str(config_path))
    
    # Summary
    total_symbols = sum(len(symbols) for symbols in categorized.values())
    print("\n" + "=" * 80)
    print(f"SUMMARY: Discovered {total_symbols} tradeable symbols")
    print("=" * 80)
    
    for category, symbols in categorized.items():
        if len(symbols) > 0:
            print(f"  {category}: {len(symbols)} symbols")
    
    print(f"\nConfig saved to: {config_path}")


if __name__ == "__main__":
    main()

