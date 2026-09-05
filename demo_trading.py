"""Demo Trading System - Quickstart single-symbol demo.

This script provides the fastest path to see a first signal from the trading system.
It is designed for onboarding and quick testing with a single symbol.

Usage:
    # Paper trading mode (default) - logs signals only, no orders
    DEMO_SYMBOL=EURUSD PAPER_TRADING=true python demo_trading.py

    # With custom parameters
    DEMO_SYMBOL=US100.cash DEMO_LOT_SIZE=0.5 PAPER_TRADING=true python demo_trading.py
"""

import os
import sched
import threading
import time

import MetaTrader5 as mt5
from dotenv import load_dotenv
from loguru import logger
from mt5_trading.domain import CrossOverStrategy, MT5Data, MT5Trader
from mt5_trading.domain.mt5_connection import ensure_mt5_logged_in
from mt5_trading.logging_config import configure_logging
from mt5_trading.robot.cross_over_robot import CrossOverRobot

load_dotenv()
configure_logging()

# Configuration from environment variables with sensible defaults
terminal_path = os.getenv("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
login = os.getenv("LOGIN", "")
password = os.getenv("PASSWORD", "")
server = os.getenv("SERVER", "")
symbol = os.getenv("DEMO_SYMBOL", "EURUSD")
lot_size = float(os.getenv("DEMO_LOT_SIZE", "0.1"))
paper_trading = os.getenv("PAPER_TRADING", "true").strip().lower() != "false"

logger.info("=" * 60)
logger.info("DEMO TRADING SYSTEM - Quickstart")
logger.info("=" * 60)
logger.info(f"Symbol: {symbol}")
logger.info(f"Lot size: {lot_size}")
logger.info(f"Paper trading: {paper_trading}")
logger.info("=" * 60)

# Global robot instance - will be created lazily
_cross_over_robot = None


def get_robot():
    """Lazy initialization of the trading robot."""
    global _cross_over_robot

    if _cross_over_robot is None:
        ensure_mt5_logged_in(
            login=login,
            password=password,
            server=server,
            terminal_path=terminal_path,
        )

        # Initialize components
        eurusd_h1_data = MT5Data(symbol, mt5.TIMEFRAME_H1)
        cross_over_strategy = CrossOverStrategy(eurusd_h1_data)
        mt5_trader = MT5Trader()
        _cross_over_robot = CrossOverRobot(
            lot_size, mt5_trader, cross_over_strategy, paper_trading=paper_trading
        )

    return _cross_over_robot


# Scheduler setup: run every 60 minutes
scheduler = sched.scheduler(time.time, time.sleep)


def run_job():
    try:
        logger.info("Running scheduled trade cycle...")
        robot = get_robot()
        robot.trade()
        logger.info("Trade cycle completed.")
    except Exception as e:
        logger.exception(f"Scheduled job failed: {e}")


def schedule_hourly():
    run_job()
    scheduler.enter(60 * 60, 1, schedule_hourly)


def start_scheduler():
    scheduler.enter(0, 1, schedule_hourly)
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    logger.info("Starting scheduler (every 60 minutes)...")
    logger.info("Press Ctrl+C to stop.")

    # Run one immediate cycle to show it's working
    logger.info("Running initial trade cycle...")
    run_job()

    start_scheduler()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
