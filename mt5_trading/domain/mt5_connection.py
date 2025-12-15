# mt5_trading/domain/mt5_connection.py
from __future__ import annotations

import MetaTrader5 as mt5
from loguru import logger


def ensure_mt5_logged_in(*, login: str | int, password: str, server: str, terminal_path: str) -> None:
    """
    Initialize the MT5 terminal and login once per process.
    Safe to call multiple times; it won't re-login if already on the same account.
    """
    if mt5.terminal_info() is None:
        if not mt5.initialize(path=terminal_path):
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            raise RuntimeError("Failed to initialize MT5")

    try:
        login_int = int(login)
    except (TypeError, ValueError):
        raise ValueError(f"LOGIN must be an integer account number, got: {login!r}")

    acct = mt5.account_info()
    if acct is not None and getattr(acct, "login", None) == login_int:
        return  # already logged in to this account

    if not mt5.login(login=login_int, password=password, server=server):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        raise RuntimeError("Failed to login to MT5")