"""Idea 2 strategy: distribution after an opposing gap fails."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import TradingStrategy, columns_matching
from .types import FeatureRow

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
        "po3_distribution",
        "po3_distribution_direction",
        "htf_day_bias",
        "po3_distribution_after_ifvg",
    )
    raw_columns = (
        "sweep_high_level",
        "sweep_low_level",
        "bos_up_level",
        "bos_down_level",
        "last_swing_high",
        "last_swing_low",
        "previous_session_high",
        "previous_session_low",
        "previous_session_midpoint",
        "previous_session_completed_at_ns",
        "decision_close",
    )

    def __init__(
        self,
        *,
        enforce_gate: bool = True,
        entry_mode: str = "legacy",
        candidate_max_age_bars: int = 5,
        previous_session_pd_gate: bool = False,
        variant_id: str = "distribution_legacy",
    ):
        allowed = {"legacy", "confirmed_first_eligible", "confirmed_retest"}
        if entry_mode not in allowed:
            raise ValueError(f"unsupported distribution entry mode: {entry_mode!r}")
        if int(candidate_max_age_bars) < 1:
            raise ValueError("candidate_max_age_bars must be positive")
        self.enforce_gate = enforce_gate
        self.entry_mode = entry_mode
        self.candidate_max_age_bars = int(candidate_max_age_bars)
        self.previous_session_pd_gate = bool(previous_session_pd_gate)
        self.variant_id = str(variant_id)
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
        if self.enforce_gate and not self.distribution_ready(row, direction):
            return False
        return self.previous_session_pd_eligible(row, direction)

    def entry_rejection_reason(self, *, direction: int, row: pd.Series) -> str:
        """Return the explicit Idea 2 premium/discount rejection label."""
        if self.previous_session_pd_gate:
            midpoint = float(row.get("previous_session_midpoint", np.nan))
            close = float(row.get("decision_close", np.nan))
            if not np.isfinite(midpoint) or not np.isfinite(close):
                return "missing_previous_session"
            if direction == 1 and close >= midpoint:
                return "premium_block"
            if direction == -1 and close <= midpoint:
                return "discount_block"
        return "entry_gate"

    def previous_session_pd_eligible(self, row: pd.Series, direction: int) -> bool:
        """Per-entry strict discount/premium check; equality is ineligible."""
        if not self.previous_session_pd_gate:
            return True
        midpoint = float(row.get("previous_session_midpoint", np.nan))
        close = float(row.get("decision_close", np.nan))
        if not np.isfinite(midpoint) or not np.isfinite(close):
            return False
        return (
            close < midpoint if direction == 1 else close > midpoint if direction == -1 else False
        )

    def sl_reference(self, *, direction: int, row: FeatureRow | pd.Series) -> float | None:
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
        if self.entry_mode == "confirmed_first_eligible":
            confirmed = float(row.get("po3_ifvg_confirmed", 0.0) or 0.0) > 0.0
            if not confirmed:
                # The raw directional pulses are retained for individual IFVG sides.
                suffix = "bull" if direction == 1 else "bear"
                confirmed = float(row.get(f"ifvg_{suffix}_confirmed", 0.0) or 0.0) > 0.0
            return confirmed
        if self.entry_mode == "confirmed_retest":
            if float(row.get("po3_distribution_after_ifvg", 0.0) or 0.0) <= 0.0:
                return False
            suffix = "bull" if direction == 1 else "bear"
            if float(row.get(f"ifvg_retest_{suffix}", 0.0) or 0.0) <= 0.0:
                return False
            age = float(row.get(f"ifvg_{suffix}_age", -1.0) or -1.0)
            return 1.0 <= age <= self.candidate_max_age_bars
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
        if self.entry_mode != "legacy":
            edge = float(row.get("po3_ifvg_low" if direction == 1 else "po3_ifvg_high", np.nan))
            through = float(row.get("po3_ifvg_through_zone_extreme", np.nan))
            if np.isfinite(edge):
                out.append(("ifvg_far_edge", edge))
            if np.isfinite(through):
                out.append(("ifvg_through_zone_extreme", through))
        return out
