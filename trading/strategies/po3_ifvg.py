"""Idea 1 strategy: PO3 manipulation/distribution + IFVG entry zones."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import TradingStrategy, columns_matching
from .types import FeatureRow


class PO3IFVGStrategy(TradingStrategy):
    """A sell-side sweep or a confirmed swing low offers the long.

    The short is the mirror. Gaps, candles, and the other technical columns
    stay on the row for the network. ``enforce_gate`` still only affects
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
        "ifvg_bull_origin",
        "ifvg_bear_origin",
        "last_swing_high",
        "last_swing_low",
    )

    def __init__(
        self,
        *,
        enforce_gate: bool = True,
        require_asian_context: bool = True,
        require_price_retest: bool | None = None,
        entry_mode: str = "legacy",
        candidate_max_age_bars: int = 5,
        variant_id: str = "po3_ifvg_legacy",
    ):
        allowed_modes = {
            "legacy",
            "confirmed_first_eligible",
            "confirmed_retest",
            "preconfirmation_exploratory",
        }
        if entry_mode not in allowed_modes:
            raise ValueError(f"unsupported PO3/IFVG entry mode: {entry_mode!r}")
        if int(candidate_max_age_bars) < 1:
            raise ValueError("candidate_max_age_bars must be positive for PO3/IFVG modes")
        expected_retest = entry_mode == "confirmed_retest"
        if entry_mode != "legacy" and require_price_retest is not None:
            if bool(require_price_retest) != expected_retest:
                raise ValueError(
                    "strategy.ifvg.require_price_retest conflicts with "
                    f"strategy.entry.mode={entry_mode!r}"
                )
        self.enforce_gate = enforce_gate
        self.require_asian_context = require_asian_context
        self.entry_mode = entry_mode
        self.variant_id = str(variant_id)
        self.candidate_max_age_bars = int(candidate_max_age_bars)
        # IFVG may enter on a retest or, when this is off, inside a live zone.
        # An FVG entry is always a retest. Either gap may sit on any timeframe.
        self.require_price_retest = (
            (True if require_price_retest is None else bool(require_price_retest))
            if entry_mode == "legacy"
            else expected_retest
        )
        self._sweep_low: tuple[str, ...] | None = None
        self._sweep_high: tuple[str, ...] | None = None
        self._swing_low: tuple[str, ...] | None = None
        self._swing_high: tuple[str, ...] | None = None
        self._ifvg_bull_confirmed: tuple[str, ...] | None = None
        self._ifvg_bear_confirmed: tuple[str, ...] | None = None

    def bind_columns(self, columns: Iterable[object]) -> None:
        """Resolve sweep and swing columns once for the step loop."""
        self._sweep_low = columns_matching(columns, "sweep_low")
        self._sweep_high = columns_matching(columns, "sweep_high")
        self._swing_low = columns_matching(columns, "swing_low_event")
        self._swing_high = columns_matching(columns, "swing_high_event")
        self._ifvg_bull_confirmed = columns_matching(columns, "ifvg_bull_confirmed")
        self._ifvg_bear_confirmed = columns_matching(columns, "ifvg_bear_confirmed")

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Sweep or protected swing on this side when the gate is enforced."""
        if not self.enforce_gate:
            return True
        return self.entry_setup(row, direction)

    def sl_reference(self, *, direction: int, row: FeatureRow | pd.Series) -> float | None:
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
        """A live gap on any timeframe.

        An FVG counts only on the bar that retests it and holds. An IFVG
        counts on that same retest when ``require_price_retest`` is set, and
        on a close inside the still-live zone when it is not. A filled gap
        is not live, so it never counts.
        """
        if direction == 1:
            ifvg = (
                _any(row, "ifvg_retest_bull")
                if self.require_price_retest
                else _any(row, "price_in_ifvg_bull")
            )
            return ifvg or _any(row, "fvg_retest_bull")
        if direction == -1:
            ifvg = (
                _any(row, "ifvg_retest_bear")
                if self.require_price_retest
                else _any(row, "price_in_ifvg_bear")
            )
            return ifvg or _any(row, "fvg_retest_bear")
        return False

    def context_direction(self, row: pd.Series) -> int:
        """Long on a sell-side sweep or a confirmed swing low. Short mirrors it.

        Fair-value gaps and inverse gaps stay on the row for the network.
        They do not have to be true for the side to be offered. A sweep on
        both sides, or a swing confirmation on both sides, offers nothing.
        """
        if self.entry_mode == "confirmed_first_eligible":
            return self._confirmed_ifvg_direction(row)
        if self.entry_mode == "confirmed_retest":
            confirmed = self._confirmed_ifvg_direction(row)
            if confirmed != 0:
                return 0  # The retest, if any, must occur on a later candle.
            return self._retest_direction(row)
        if self.entry_mode == "preconfirmation_exploratory":
            return self._fvg_candidate_direction(row)

        swept_low = self._flag_any(row, self._sweep_low, "sweep_low")
        swept_high = self._flag_any(row, self._sweep_high, "sweep_high")
        if swept_low or swept_high:
            if swept_low and swept_high:
                return 0
            return 1 if swept_low else -1
        swing_low = self._flag_any(row, self._swing_low, "swing_low_event")
        swing_high = self._flag_any(row, self._swing_high, "swing_high_event")
        if swing_low and swing_high:
            return 0
        if swing_low:
            return 1
        if swing_high:
            return -1
        return 0

    def entry_setup(self, row: pd.Series, direction: int) -> bool:
        """True when this bar's sweep or protected swing matches ``direction``."""
        return self.context_direction(row) == direction and direction != 0

    def _confirmed_ifvg_direction(self, row: pd.Series) -> int:
        confirmed = _flag(row, "po3_ifvg_confirmed")
        direction = float(row.get("po3_distribution_direction", 0.0) or 0.0)
        if not confirmed or not np.isfinite(direction):
            return 0
        return int(np.sign(direction))

    def _retest_direction(self, row: pd.Series) -> int:
        if not _flag(row, "po3_distribution_after_ifvg"):
            return 0
        direction = float(row.get("po3_distribution_direction", 0.0) or 0.0)
        if not np.isfinite(direction) or direction == 0.0:
            return 0
        bull = _any(row, "ifvg_retest_bull") and self._candidate_age_ok(row, "bull")
        bear = _any(row, "ifvg_retest_bear") and self._candidate_age_ok(row, "bear")
        retest_direction = 0 if bull == bear else (1 if bull else -1)
        return retest_direction if retest_direction == int(np.sign(direction)) else 0

    def _candidate_age_ok(self, row: pd.Series, side: str) -> bool:
        names = columns_matching(row.index, f"ifvg_{side}_age")
        if not names:
            return False
        for name in names:
            try:
                age = float(row.get(name, -1.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(age) and 1.0 <= age <= self.candidate_max_age_bars:
                return True
        return False

    @staticmethod
    def _fvg_candidate_direction(row: pd.Series) -> int:
        bull = _any(row, "fvg_retest_bull")
        bear = _any(row, "fvg_retest_bear")
        return 0 if bull == bear else (1 if bull else -1)

    def _flag_any(self, row: pd.Series, bound: tuple[str, ...] | None, stem: str) -> bool:
        names = bound if bound is not None else columns_matching(row.index, stem)
        return any(_flag(row, name) for name in names)

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
        if direction == 1:
            out.extend(
                _gap_stops(
                    row, active="ifvg_bull_active", area="ifvg_bull_low", origin="ifvg_bull_origin"
                )
            )
            out.extend(
                _gap_stops(
                    row, active="fvg_bull_active", area="fvg_bull_low", origin="fvg_bull_origin"
                )
            )
        else:
            out.extend(
                _gap_stops(
                    row, active="ifvg_bear_active", area="ifvg_bear_high", origin="ifvg_bear_origin"
                )
            )
            out.extend(
                _gap_stops(
                    row, active="fvg_bear_active", area="fvg_bear_high", origin="fvg_bear_origin"
                )
            )
        if self.entry_mode != "legacy":
            edge = float(row.get("po3_ifvg_low" if direction == 1 else "po3_ifvg_high", np.nan))
            through = float(row.get("po3_ifvg_through_zone_extreme", np.nan))
            if np.isfinite(edge):
                out.append(("ifvg_far_edge", edge))
            if np.isfinite(through):
                out.append(("ifvg_through_zone_extreme", through))
        return out


def _flag(row: pd.Series, col: str) -> bool:
    return float(row.get(col, 0.0) or 0.0) > 0.0


def _columns(row: pd.Series, stem: str) -> list[str]:
    """``stem`` itself and the same stem on any timeframe prefix."""
    return [str(col) for col in row.index if str(col) == stem or str(col).endswith("_" + stem)]


def _any(row: pd.Series, stem: str) -> bool:
    return any(_flag(row, col) for col in _columns(row, stem))


def _prefixed(col: str, stem: str, other: str) -> str:
    if col == stem:
        return other
    return col[: -len(stem)] + other


def _gap_stops(row: pd.Series, *, active: str, area: str, origin: str) -> list[tuple[str, float]]:
    """Stop under a live gap, and under candle 1 of the 1-2-3 that left it.

    Shorts use the upper edge and candle 1's high. A zone that is not active
    on that timeframe is skipped, including a filled gap whose price was
    forward-filled.
    """
    found: list[tuple[str, float]] = []
    for col in sorted(_columns(row, area)):
        if not _flag(row, _prefixed(col, area, active)):
            continue
        level = float(row.get(col, np.nan))
        if np.isfinite(level):
            found.append((col, level))
        origin_col = _prefixed(col, area, origin)
        origin_level = float(row.get(origin_col, np.nan))
        if np.isfinite(origin_level):
            found.append((origin_col, origin_level))
    return found


def _within_atr(row: pd.Series, col: str) -> bool:
    value = float(row.get(col, np.nan))
    return bool(np.isfinite(value) and abs(value) <= 1.0)
