"""Idea 2 strategy: distribution after an opposing gap fails."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import TradingStrategy, columns_matching

_LOWER_RANK = {"": 0, "M1": 1, "M5": 2, "M15": 3}
_StopGroup = tuple[str, str, str | None]


def _on(row: pd.Series, col: str) -> bool:
    return float(row.get(col, 0.0) or 0.0) > 0.0


def _ifvg_groups(
    columns: Iterable[object],
    *,
    active: str,
    area: str,
    origin: str,
) -> tuple[tuple[_StopGroup, ...], tuple[_StopGroup, ...]]:
    """Live-zone stop triples, lower timeframes before higher ones."""
    names = {str(col) for col in columns}
    lower: list[tuple[tuple[int, str], _StopGroup]] = []
    higher: list[tuple[str, _StopGroup]] = []
    for col in columns_matching(columns, active):
        prefix = "" if col == active else col[: -len(active)]
        area_col = prefix + area
        if area_col not in names:
            continue
        origin_col = prefix + origin
        item = (col, area_col, origin_col if origin_col in names else None)
        tf = prefix[:-1] if prefix.endswith("_") else ""
        if tf in _LOWER_RANK:
            lower.append(((_LOWER_RANK[tf], col), item))
        else:
            higher.append((tf, item))
    return (
        tuple(item for _, item in sorted(lower)),
        tuple(item for _, item in sorted(higher)),
    )


class DistributionStrategy(TradingStrategy):
    """Distribution is caught after an opposing fair-value gap fails.

    A long requires an up daily bias and a long distribution leg. The side
    is offered on every bar of that leg after a bullish inverse gap has
    printed, and not on the bar the gap fails. A break of structure is not
    required. The stop menu adds a respected lower-timeframe inverse gap
    and an aligned higher-timeframe one. The agent picks.
    """

    name = "distribution"
    required_features = (
        "sweep_high",
        "sweep_low",
        "sweep_high_level",
        "sweep_low_level",
        "sweep_high_reclaimed",
        "sweep_low_reclaimed",
        "bos_up",
        "bos_down",
        "po3_distribution_after_ifvg",
    )
    raw_columns = (
        "sweep_high_level",
        "sweep_low_level",
        "bos_up_level",
        "bos_down_level",
        "last_swing_high",
        "last_swing_low",
    )

    def __init__(self, *, enforce_gate: bool = True):
        self.enforce_gate = enforce_gate
        self._lower_bull: tuple[tuple[str, str, str | None], ...] | None = None
        self._higher_bull: tuple[tuple[str, str, str | None], ...] | None = None
        self._lower_bear: tuple[tuple[str, str, str | None], ...] | None = None
        self._higher_bear: tuple[tuple[str, str, str | None], ...] | None = None

    def bind_columns(self, columns: Iterable[object]) -> None:
        """Resolve inverse-gap stop columns once for the step loop."""
        self._lower_bull, self._higher_bull = _ifvg_groups(
            columns, active="ifvg_bull_active", area="ifvg_bull_low", origin="ifvg_bull_origin"
        )
        self._lower_bear, self._higher_bear = _ifvg_groups(
            columns, active="ifvg_bear_active", area="ifvg_bear_high", origin="ifvg_bear_origin"
        )

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Distribution after the opposing gap fails, when the gate is enforced."""
        if not self.enforce_gate:
            return True
        return self.distribution_ready(row, direction)

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Long -> swept low, short -> swept high (Agent.md §14).

        Returns the raw reference level; the environment validates geometry
        against the entry price and rejects invalid stops.
        """
        col = "sweep_low_level" if direction == 1 else "sweep_high_level"
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

    def context_direction(self, row: pd.Series) -> int:
        """Long when the up bias and the long distribution leg are both on.

        The leg must already be past the bar the opposing gap failed. A
        later fair-value-gap retest is not required.
        """
        return (
            1
            if self.distribution_ready(row, 1)
            else (-1 if self.distribution_ready(row, -1) else 0)
        )

    def distribution_ready(self, row: pd.Series, direction: int) -> bool:
        """True on a distribution bar after the matching inverse gap printed."""
        if direction not in (1, -1):
            return False
        if float(row.get("po3_distribution", 0.0) or 0.0) <= 0.0:
            return False
        dist = float(row.get("po3_distribution_direction", 0.0) or 0.0)
        bias = float(row.get("htf_day_bias", 0.0) or 0.0)
        if not np.isfinite(dist) or not np.isfinite(bias) or dist == 0.0 or bias == 0.0:
            return False
        if int(np.sign(dist)) != direction or int(np.sign(bias)) != direction:
            return False
        return float(row.get("po3_distribution_after_ifvg", 0.0) or 0.0) > 0.0

    def sl_candidates(self, *, direction: int, row: pd.Series) -> list[tuple[str, float]]:
        """Structural SL ladder (sweep extremes instead of manip)."""
        if direction == 1:
            cols = (
                "last_swing_low",
                "sweep_low_level",
                "asian_low",
                "london_low",
            )
        elif direction == -1:
            cols = (
                "last_swing_high",
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
        if direction == 1:
            lower = self._lower_bull
            higher = self._higher_bull
            active, area, origin = "ifvg_bull_active", "ifvg_bull_low", "ifvg_bull_origin"
        else:
            lower = self._lower_bear
            higher = self._higher_bear
            active, area, origin = "ifvg_bear_active", "ifvg_bear_high", "ifvg_bear_origin"
        if lower is None or higher is None:
            lower, higher = _ifvg_groups(row.index, active=active, area=area, origin=origin)
        for group in (*lower, *higher):
            if not _on(row, group[0]):
                continue
            for name in (group[1], group[2]):
                if name is None:
                    continue
                level = float(row.get(name, np.nan))
                if np.isfinite(level):
                    out.append((name, level))
        return out
