from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from decisioning.actions import ActionSchema
from trading.orders.selection import decode_tp_mode

from .action_diagnostics import _decision_from_action, _empty_decision
from .market_view import FeatureRow
from .mixin_base import RuntimeMixinBase


class ActionDecodingMixin(RuntimeMixinBase):
    def _decode_strategy_action(
        self,
        action: int | float | np.ndarray[Any, Any],
        feat_row: FeatureRow | pd.Series | None,
    ) -> tuple[int, float, float, str]:
        assert isinstance(self.action_schema, ActionSchema)
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != self.action_schema.width:
            raise ValueError(
                f"action width {arr.size} disagrees with schema "
                f"{self.action_schema.version} {self.action_schema.names} "
                f"(expected {self.action_schema.width})"
            )
        tp_mode = "rr"
        risk_frac = self.risk_frac_range[0]
        rr_ratio = self.rr_ratio_range[0]
        assert isinstance(self.action_schema, ActionSchema)
        raw = np.clip(arr, -1.0, 1.0)
        u = 0.5 * (raw + 1.0)
        indices = self.action_schema.indices
        direction_offset = int(self.action_schema.has_direction)
        intensity = float(u[indices["intensity"]])
        entry_threshold = self.entry_intensity_threshold
        if feat_row is None:
            discrete_action = 0
        elif entry_threshold > 0.0 and intensity < entry_threshold:
            discrete_action = 0
            self._entry_diag["hold_low_intensity"] += 1
        else:
            ctx = int(self.strategy.context_direction(feat_row))
            discrete_action = (
                self._apply_direction_control(ctx, u)
                if direction_offset
                else (ctx if ctx in (-1, 1) else 0)
            )
            if discrete_action == 0 and ctx not in (-1, 1):
                self._entry_diag["hold_no_context"] += 1
        sl_idx = indices.get("sl_mode", -1)
        tp_idx = indices.get("tp_mode", -1)
        z_idx = indices.get("z1", -1)
        sl_mode_offset = int(sl_idx >= 0)
        tp_mode_offset = int(tp_idx >= 0)
        simplex_offset = 2 if z_idx >= 0 else 0
        exit_idx = indices["exit"]

        # Extract stop/risk (indices unchanged regardless of target_width)
        if u.size == self.action_schema.width:
            sl_raw = float(raw[indices["stop"]])
            self._selected_sl_anchor = float(np.clip(sl_raw + 1.0, 0.0, 2.0))
            r_lo, r_hi = self.risk_frac_range
            risk_frac = r_lo + float(u[indices["risk"]]) * (r_hi - r_lo)
            if self.allow_multi_tp:
                # Extract tp1_sel, tp2_sel, tp3_sel (raw unit-interval 0..1)
                self._selected_tp_selections = (
                    float(u[indices["tp1_sel"]]),
                    float(u[indices["tp2_sel"]]),
                    float(u[indices["tp3_sel"]]),
                )
                # rr_ratio from tp3 (primary TP slot) for legacy downstream use
                self._selected_tp_fraction = float(u[indices["tp3_sel"]])
                rr_ratio = self._selected_tp_fraction
            else:
                self._selected_tp_fraction = float(u[indices["target"]])
                rr_ratio = self._selected_tp_fraction
                self._selected_tp_selections = (-1.0, -1.0, -1.0)
        else:
            risk_frac = self.risk_frac_range[0]
            rr_ratio = 0.0

        # Extract exit dim
        if u.size > exit_idx and self.allow_ema_exit:
            self._selected_exit_mode = "ema_21" if float(u[exit_idx]) >= 0.5 else "structural"

        # Decode sl_mode
        self._selected_sl_mode = "structural"
        self._sl_mode_explicit = False
        if sl_mode_offset and u.size > sl_idx:
            u_sl = float(u[sl_idx])
            if abs(u_sl - 0.5) < self.sl_mode_defer_threshold:
                self._selected_sl_mode = self._default_sl_mode()
            else:
                sl_mode_idx = min(2, int(u_sl * 3))
                self._selected_sl_mode = ("fixed", "breakeven", "trailing")[sl_mode_idx]
                self._sl_mode_explicit = True
        else:
            self._selected_sl_mode = self.sl_mode

        # Decode tp_mode
        self._tp_mode_explicit = False
        if tp_mode_offset and u.size > tp_idx:
            u_tp = float(u[tp_idx])
            decoded, explicit = decode_tp_mode(
                u_tp, self.tp_mode, self.tp_mode_defer_threshold, is_unit=True
            )
            self._selected_tp_mode = decoded
            self._tp_mode_explicit = explicit
            tp_mode = decoded
        else:
            # No tp_mode dim: keep the legacy sentinel so flag-off decode stays "rr".
            tp_mode = self.tp_mode if self.allow_agent_tp_mode else "rr"
            self._selected_tp_mode = tp_mode

        # Decode z1/z2 for simplex
        if simplex_offset and u.size > z_idx + 1:
            self._selected_tp_z = (float(u[z_idx]), float(u[z_idx + 1]))
        else:
            self._selected_tp_z = (0.0, 0.0)

        self._decision_action = _decision_from_action(raw, u, self.action_schema)
        return discrete_action, risk_frac, rr_ratio, tp_mode

    def _decode_continuous_action(
        self, action: int | float | np.ndarray[Any, Any]
    ) -> tuple[int, float, float, str]:
        tp_mode = "rr"
        if isinstance(action, np.ndarray):
            action_value = float(action[0]) if action.size > 0 else 0.0
        else:
            action_value = float(action)
        if action_value > 0:
            discrete_action = 1
            risk_frac = self.max_risk_frac * action_value
        elif action_value < 0:
            discrete_action = -1
            risk_frac = self.max_risk_frac * abs(action_value)
        else:
            discrete_action = 0
            risk_frac = 0.0
        rr_ratio = self.rr_ratio_range[0]
        return discrete_action, risk_frac, rr_ratio, tp_mode

    def _decode_discrete_action(
        self, action: int | float | np.ndarray[Any, Any]
    ) -> tuple[int, float, float, str]:
        tp_mode = "rr"
        risk_frac = self.risk_frac_range[0]
        rr_ratio = self.rr_ratio_range[0]
        # Discrete actions
        if action == 0:
            discrete_action = 0
        elif 1 <= action <= 9:
            discrete_action = 1
            idx = int(action) - 1
            risk_variant = idx // 3
            rr_variant = idx % 3
            risk_levels = [
                self.risk_frac_range[0],
                (self.risk_frac_range[0] + self.risk_frac_range[1]) / 2,
                self.risk_frac_range[1],
            ]
            rr_levels = [
                self.rr_ratio_range[0],
                (self.rr_ratio_range[0] + self.rr_ratio_range[1]) / 2,
                self.rr_ratio_range[1],
            ]
            risk_frac = risk_levels[risk_variant]
            rr_ratio = rr_levels[rr_variant]
        elif 10 <= action <= 18:
            discrete_action = -1
            idx = int(action) - 10
            risk_variant = idx // 3
            rr_variant = idx % 3
            risk_levels = [
                self.risk_frac_range[0],
                (self.risk_frac_range[0] + self.risk_frac_range[1]) / 2,
                self.risk_frac_range[1],
            ]
            rr_levels = [
                self.rr_ratio_range[0],
                (self.rr_ratio_range[0] + self.rr_ratio_range[1]) / 2,
                self.rr_ratio_range[1],
            ]
            risk_frac = risk_levels[risk_variant]
            rr_ratio = rr_levels[rr_variant]
        else:  # action == 19
            discrete_action = 0
            risk_frac = self.risk_frac_range[0]
            rr_ratio = self.rr_ratio_range[0]
        return discrete_action, risk_frac, rr_ratio, tp_mode

    def _decode_action(
        self,
        action: int | float | np.ndarray[Any, Any],
        feat_row: FeatureRow | pd.Series | None = None,
    ) -> tuple[int, float, float, str]:
        """Decode strategy, continuous, or discrete actions through their stable schemas."""
        self._selected_sl_anchor = 0.0
        self._selected_tp_fraction = 0.0
        self._selected_exit_mode = "structural"
        self._decision_action = _empty_decision()
        self._entry_rejection_reason = ""
        if self.strategy_actions:
            discrete_action, risk_frac, rr_ratio, tp_mode = (
                ActionDecodingMixin._decode_strategy_action(self, action, feat_row)
            )
        elif self.continuous_actions:
            discrete_action, risk_frac, rr_ratio, tp_mode = (
                ActionDecodingMixin._decode_continuous_action(self, action)
            )
        else:
            discrete_action, risk_frac, rr_ratio, tp_mode = (
                ActionDecodingMixin._decode_discrete_action(self, action)
            )
        if (
            self.fixed_risk_usd <= 0.0
            and discrete_action in (1, -1)
            and risk_frac <= self.risk_floor
        ):
            self._entry_diag["hold_risk_floor"] += 1
            discrete_action = 0
        return discrete_action, risk_frac, rr_ratio, tp_mode
