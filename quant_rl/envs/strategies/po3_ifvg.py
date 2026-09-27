"""Idea 1 strategy: PO3 manipulation/distribution + IFVG entry zones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TradingStrategy


class PO3IFVGStrategy(TradingStrategy):
    """Direction follows distribution, then context, then the higher-timeframe bias.

    A sweep is not required. ``enforce_gate`` still only affects
    :meth:`validate_entry`. The overlay reads :meth:`context_direction`.
    """

    name = "po3_ifvg"
    required_features = (
        "asian_high",
        "asian_low",
        "sweep_high",
        "sweep_low",
        "po3_manipulation_low",
        "po3_manipulation_high",
        "po3_manipulation_end",
        "po3_distribution",
        "ifvg_bull_low",
        "ifvg_bull_high",
        "ifvg_bear_low",
        "ifvg_bear_high",
    )
    raw_columns = (
        "asian_high",
        "asian_low",
        "sweep_high_level",
        "sweep_low_level",
        "po3_manipulation_high",
        "po3_manipulation_low",
        "ifvg_bull_low",
        "ifvg_bull_high",
        "ifvg_bear_low",
        "ifvg_bear_high",
        "last_swing_high",
        "last_swing_low",
    )

    def __init__(self, *, enforce_gate: bool = True, require_asian_context: bool = True):
        self.enforce_gate = enforce_gate
        self.require_asian_context = require_asian_context

    def _asian_ok(self, row: pd.Series) -> bool:
        if not self.require_asian_context:
            return True
        return bool(
            np.isfinite(float(row.get("asian_high", np.nan)))
            and np.isfinite(float(row.get("asian_low", np.nan)))
        )

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Full PO3+IFVG chain when the gate is enforced (Agent.md §17)."""
        if not self.enforce_gate:
            return True
        if not self._asian_ok(row):
            return False
        if direction == 1:
            return bool(
                float(row.get("sweep_low", 0.0)) > 0
                and float(row.get("po3_distribution", 0.0)) > 0
                and float(row.get("po3_distribution_direction", 0.0)) == 1
                and float(row.get("ifvg_bull_active", 0.0)) > 0
                and float(row.get("price_in_ifvg_bull", 0.0)) > 0
            )
        if direction == -1:
            return bool(
                float(row.get("sweep_high", 0.0)) > 0
                and float(row.get("po3_distribution", 0.0)) > 0
                and float(row.get("po3_distribution_direction", 0.0)) == -1
                and float(row.get("ifvg_bear_active", 0.0)) > 0
                and float(row.get("price_in_ifvg_bear", 0.0)) > 0
            )
        return False

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Long -> manipulation low, short -> manipulation high (Agent.md §14).

        Returns the raw reference level; the environment validates geometry
        against the entry price and rejects invalid stops.
        """
        col = "po3_manipulation_low" if direction == 1 else "po3_manipulation_high"
        level = float(row.get(col, np.nan))
        return level if np.isfinite(level) else None

    def target_candidates(self, *, direction: int, row: pd.Series) -> dict[str, float]:
        """Named structural targets; the TP resolver validates each side."""

        def _get(col: str) -> float:
            value = float(row.get(col, np.nan))
            return value if np.isfinite(value) else np.nan

        if direction == 1:
            return {
                "buyside_liquidity": _get("sweep_high_level"),
                "previous_day_high_low": _get("prev_day_high"),
            }
        return {
            "sellside_liquidity": _get("sweep_low_level"),
            "previous_day_high_low": _get("prev_day_low"),
        }

    def entry_liquidity(self, row: pd.Series, direction: int) -> bool:
        """Swept, or within one ATR of the sweep or manipulation level."""
        if direction == 1:
            return _flag(row, "sweep_low") or _within_atr(row, "manipulation_low_distance_atr")
        if direction == -1:
            return _flag(row, "sweep_high") or _within_atr(row, "manipulation_high_distance_atr")
        return False

    def entry_gap(self, row: pd.Series, direction: int) -> bool:
        """Price is in, or within one ATR of, an IFVG or FVG on this side."""
        if direction == 1:
            return (
                _flag(row, "price_in_ifvg_bull")
                or _flag(row, "fvg_in_bull")
                or _gap_within_atr(row, "fvg_bull_dist")
            )
        if direction == -1:
            return (
                _flag(row, "price_in_ifvg_bear")
                or _flag(row, "fvg_in_bear")
                or _gap_within_atr(row, "fvg_bear_dist")
            )
        return False

    def context_direction(self, row: pd.Series) -> int:
        """Prefer distribution direction, then context, then the daily bias."""
        if float(row.get("po3_distribution", 0.0)) > 0:
            dist = float(row.get("po3_distribution_direction", 0.0))
            if dist != 0.0 and np.isfinite(dist):
                return int(np.sign(dist))
        for col in ("context_trade_direction", "htf_day_bias"):
            val = float(row.get(col, 0.0))
            if val != 0.0 and np.isfinite(val):
                return int(np.sign(val))
        return 0

    def sl_candidates(self, *, direction: int, row: pd.Series) -> list[tuple[str, float]]:
        """Structural SL ladder for trader ``sl_anchor`` selection."""
        if direction == 1:
            cols = (
                "last_swing_low",
                "po3_manipulation_low",
                "sweep_low_level",
                "asian_low",
                "london_low",
            )
        elif direction == -1:
            cols = (
                "last_swing_high",
                "po3_manipulation_high",
                "sweep_high_level",
                "asian_high",
                "london_high",
            )
        else:
            return []
        out: list[tuple[str, float]] = []
        for col in cols:
            if col not in row.index:
                continue
            level = float(row.get(col, np.nan))
            if np.isfinite(level):
                out.append((col, level))
        return out


def _flag(row: pd.Series, col: str) -> bool:
    return float(row.get(col, 0.0) or 0.0) > 0.0


def _within_atr(row: pd.Series, col: str) -> bool:
    value = float(row.get(col, np.nan))
    return bool(np.isfinite(value) and abs(value) <= 1.0)


def _gap_within_atr(row: pd.Series, col: str) -> bool:
    value = float(row.get(col, np.nan))
    return bool(np.isfinite(value) and value < 5.0 and value <= 1.0)
