"""Monotonic management modes and price tolerance for position services."""

_PRICE_TOL = 1e-4
_SL_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}
_TP_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}

__all__ = ["_PRICE_TOL", "_SL_MODE_RANK", "_TP_MODE_RANK"]
