from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulation.environment.diagnostics import new_debug_window, summarize_window, wiring_from_env
from trading.strategies import BaselineStrategy

from .entry_diagnostics import _REJECTION_COUNTERS
from .entry_state import EntryCandidate, EntryEvidence, immutable_setup_levels
from .market_view import FeatureRow
from .mixin_base import RuntimeMixinBase


class DiagnosticsRuntimeMixin(RuntimeMixinBase):
    _debug_window: dict[str, Any] | None

    def enable_debug_trace(self) -> None:
        """Start per-rollout counters and step-failure snapshots."""
        self._debug_trace = True
        self._debug_window = new_debug_window()

    def debug_flush(self) -> dict[str, Any]:
        """Return the counters since the last flush and start a new window."""
        if not self._debug_trace or self._debug_window is None:
            return {}
        summary = summarize_window(self._debug_window)
        self._debug_window = new_debug_window()
        return summary

    def debug_wiring(self) -> dict[str, Any]:
        """Reward and risk settings this env was built with."""
        return wiring_from_env(self)

    def _note_session_open(self, session_id: int, feat_row: FeatureRow | pd.Series) -> None:
        """Remember the first bar of this session and a post-open manip end."""
        self._session_open_idx.setdefault(session_id, self.step_idx)
        if float(feat_row.get("po3_manipulation_end", 0.0) or 0.0) > 0.0:
            self._ny_manip_confirmed.add(session_id)

    def _arm_po3_confirm(self, session_id: int) -> None:
        """Let Idea 1 use a side only after this session's manipulation ends."""
        if getattr(self.strategy, "name", "") != "po3_ifvg":
            return
        setattr(
            self.strategy,
            "_session_manip_confirmed",
            session_id in self._ny_manip_confirmed,
        )

    def _entry_action_requested(self, discrete_action: int) -> bool:
        """Interpret whether the decoded policy action requests entry execution."""
        if not self.strategy_actions:
            return discrete_action in (-1, 1)
        intensity = self._decision_action.get("intensity_u", float("nan"))
        return bool(np.isfinite(intensity) and intensity >= self.entry_intensity_threshold)

    def entry_evidence_context(self, feat_row: FeatureRow | pd.Series) -> EntryEvidence:
        """Translate existing strategy predicates into one causal evidence record."""
        context_direction = int(self.strategy.context_direction(feat_row))
        names = tuple(str(name) for name in feat_row.index)

        def flagged(stem: str) -> bool:
            columns = (name for name in names if name == stem or name.endswith("_" + stem))
            for column in columns:
                try:
                    value = float(feat_row.get(column, 0.0))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return True
            return False

        bull_retest = flagged("ifvg_retest_bull") or flagged("fvg_retest_bull")
        bear_retest = flagged("ifvg_retest_bear") or flagged("fvg_retest_bear")
        sweep_low = flagged("sweep_low")
        sweep_high = flagged("sweep_high")
        setup_predicate = getattr(self.strategy, "entry_setup", None)
        setup_matches = (
            bool(setup_predicate(feat_row, context_direction))
            if context_direction in (-1, 1) and callable(setup_predicate)
            else context_direction in (-1, 1) and not callable(setup_predicate)
        )
        direction = context_direction
        if direction not in (-1, 1):
            direction = (
                1
                if bull_retest and not bear_retest
                else -1
                if bear_retest and not bull_retest
                else 0
            )
        baseline_sweep_setup = (
            isinstance(self.strategy, BaselineStrategy)
            and self.entry_state_machine is not None
            and sweep_low != sweep_high
            and direction not in (-1, 1)
        )
        if direction not in (-1, 1) and baseline_sweep_setup:
            direction = 1 if sweep_low else -1
        candidate = (
            self.entry_state_machine.candidate
            if isinstance(self.strategy, BaselineStrategy) and self.entry_state_machine is not None
            else None
        )
        if direction not in (-1, 1) and candidate is not None:
            direction = candidate.direction
        setup_matches = setup_matches or (baseline_sweep_setup and direction in (-1, 1))
        suffix = "bull" if direction == 1 else "bear"
        retest_confirmed = (direction == 1 and bull_retest) or (direction == -1 and bear_retest)
        distribution_ready = getattr(self.strategy, "distribution_ready", None)
        distribution_arm = (
            bool(distribution_ready(feat_row, direction))
            if direction in (-1, 1) and callable(distribution_ready)
            else False
        )

        setup_types: set[str] = set()
        if direction in (-1, 1):
            sweep_stem = "sweep_low" if direction == 1 else "sweep_high"
            swing_stem = "swing_low_event" if direction == 1 else "swing_high_event"
            if flagged(sweep_stem):
                setup_types.add("sweep")
            if flagged(swing_stem):
                setup_types.add("swing")
            if not isinstance(self.strategy, BaselineStrategy):
                for setup_type, stems in (
                    ("ifvg", (f"ifvg_{suffix}_active", f"ifvg_retest_{suffix}")),
                    ("fvg", (f"fvg_{suffix}_active", f"fvg_retest_{suffix}")),
                ):
                    if any(flagged(stem) for stem in stems):
                        setup_types.add(setup_type)
        if distribution_arm:
            setup_types.add("distribution")
        if setup_matches and not setup_types:
            setup_types.add(str(getattr(self.strategy, "name", "strategy")))

        snapshot: dict[str, float] = {}
        if direction in (-1, 1):
            geometry_stems = (
                f"ifvg_{suffix}_low",
                f"ifvg_{suffix}_high",
                f"fvg_{suffix}_low",
                f"fvg_{suffix}_high",
            )
            for geometry_stem in geometry_stems:
                for column in names:
                    if column != geometry_stem and not column.endswith("_" + geometry_stem):
                        continue
                    try:
                        value = float(feat_row.get(column, np.nan))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        snapshot[column] = value

        conflicting_sweeps = flagged("sweep_low") and flagged("sweep_high")
        conflicting_swings = flagged("swing_low_event") and flagged("swing_high_event")
        return EntryEvidence(
            bar=self.step_idx,
            evidence_detected=setup_matches,
            invalidated=conflicting_sweeps or conflicting_swings or (bull_retest and bear_retest),
            arm_condition=(
                setup_matches
                or retest_confirmed
                or distribution_arm
                or (candidate is not None and direction == candidate.direction)
            ),
            retest_confirmed=retest_confirmed,
            direction=direction,
            setup_types=tuple(sorted(setup_types)),
            setup_levels_snapshot=immutable_setup_levels(snapshot),
        )

    def _lock_unconfirmed_side(
        self,
        discrete_action: int,
        feat_row: FeatureRow | pd.Series,
        risk_frac: float,
        *,
        entry_candidate: EntryCandidate | None = None,
    ) -> tuple[int, float]:
        """During an open manipulation, keep only the side the strategy already offers.

        A direction override, or a side invented when the strategy has none,
        becomes no trade. The strategy's own sweep or swing entry stays.
        """
        if not self.strategy_actions or discrete_action == 0:
            return discrete_action, risk_frac
        from trading.orders.entries import manipulation_state

        if manipulation_state(feat_row) != "against":
            return discrete_action, risk_frac
        if (
            isinstance(self.strategy, BaselineStrategy)
            and entry_candidate is not None
            and discrete_action == entry_candidate.direction
        ):
            return discrete_action, risk_frac
        ctx = int(self.strategy.context_direction(feat_row))
        if discrete_action == ctx:
            return discrete_action, risk_frac
        self._reject("manipulation_unconfirmed")
        return 0, 0.0

    def _reject(self, reason: str) -> None:
        """Refuse an entry: bump its counter and stamp the reason, atomically.

        The mapping lives in :data:`_REJECTION_COUNTERS`. An unregistered
        reason raises ``KeyError`` here rather than silently producing a ratio
        or trade-log entry with no backing bucket.
        """
        self._entry_diag[_REJECTION_COUNTERS[reason]] += 1
        # A refusal that arrives after PPO requested an ARMED trigger counts as
        # a trigger refusal, so the arm->trigger conversion rate is measurable.
        if self._trigger_requested_this_step and self.entry_state_machine is not None:
            self.entry_state_machine.trigger_refused += 1
        self._entry_rejection_reason = reason

    def _in_open_blackout(self, session_id: int) -> bool:
        if self.open_manipulation_bars <= 0:
            return False
        start = self._session_open_idx.get(session_id, self.step_idx)
        return bool((int(self.step_idx) - int(start)) < int(self.open_manipulation_bars))

    def _matched_entry_level(self, session_id: int, discrete_action: int) -> tuple[str | None, Any]:
        """Match an entry to the most recently crossed liquidity level.

        Args:
            session_id: Current trading session identifier.
            discrete_action: Signed action (+1 long, -1 short).

        Returns:
            Tuple of (level name or None, crossing time or None). Longs
            match high levels, shorts match low levels.
        """
        candidates = {
            name
            for (sid, name) in self._level_crosses
            if sid == session_id
            and (name.endswith("high") if discrete_action > 0 else name.endswith("low"))
        }
        if not candidates:
            return None, None
        latest_cross_time = max(self._level_crosses[(session_id, name)] for name in candidates)
        latest_name = min(
            name
            for name in candidates
            if self._level_crosses[(session_id, name)] == latest_cross_time
        )
        return latest_name, self._level_crosses[(session_id, latest_name)]

    def entry_state_diagnostics(self) -> dict[str, Any]:
        """Aggregate entry-state lifecycle counters for reporting.

        Reads the FSM's own counters (never infers them from feature data), so
        the reporting surface is owned by the object that owns the lifecycle.
        Returns an empty dict when the FSM is off (A3), so callers can merge
        unconditionally.
        """
        machine = self.entry_state_machine
        if machine is None:
            return {}
        total_steps = max(1, int(self.episode_step_count))
        visits = dict(machine.visits)
        armed_bars = max(1, machine.armed_bars)
        return {
            "entry_state_machine": True,
            "entry_state_visits": visits,
            "entry_state_share": {
                state: round(count / total_steps, 6) for state, count in visits.items()
            },
            "candidate_created": int(machine.candidate_created),
            "candidate_expired": int(machine.candidate_expired),
            "candidate_invalidated": int(machine.candidate_invalidated),
            "trigger_requested": int(machine.trigger_requested),
            "trigger_refused": int(machine.trigger_refused),
            "entered": int(machine.entered),
            "closed": int(machine.closed),
            "armed_bars": int(machine.armed_bars),
            "arm_to_trigger_rate": round(machine.entered / armed_bars, 6),
            "trigger_refusal_rate": round(
                machine.trigger_refused / max(1, machine.trigger_requested), 6
            ),
        }
