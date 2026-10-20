"""Typing bridge for mixins composed onto the final TradingEnv class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class RuntimeMixinBase:
    """Allow statically checked mixins to access sibling-composed attributes."""

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...


__all__ = ["RuntimeMixinBase"]
