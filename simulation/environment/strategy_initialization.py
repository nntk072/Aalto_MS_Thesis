"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.strategies import BaselineStrategy, TradingStrategy

from .action_diagnostics import _empty_decision
from .mixin_base import RuntimeMixinBase


class StrategyInitializationMixin(RuntimeMixinBase):
    def _initialize_strategy_runtime(
        self,
        strategy: TradingStrategy | None,
        strategy_actions: bool,
        sl_buffer_pts: float,
        bundle_data: Any | None,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
    ) -> None:
        # Strategy semantics (Idea 1/2) vs. legacy behaviour (Idea 3 baseline).
        self.strategy: TradingStrategy = strategy if strategy is not None else BaselineStrategy()
        self.strategy_actions = strategy_actions
        self.sl_buffer_pts = float(sl_buffer_pts)
        self._selected_sl_anchor = 0.0
        self._selected_tp_fraction = 0.0
        self._selected_exit_mode = "structural"
        self._selected_sl_mode = "fixed"
        self._sl_mode_explicit = False
        self._selected_tp_mode = self.tp_mode
        self._tp_mode_explicit = False
        self._selected_tp_selections = (-1.0, -1.0, -1.0)
        self._selected_tp_z = (0.0, 0.0)
        self._decision_action = _empty_decision()
        self._entry_rejection_reason = ""
        self._session_entry_counts: dict[int, int] = {}
        self._cooldown_until_bar: int = -1
        self._last_realized_close_pnl: float | None = None
        self._last_realized_close_r: float | None = None
        self._opened_this_step = False
        self._trigger_requested_this_step = False
        self._debug_trace = False
        self._debug_window: dict[str, Any] | None = None
        feature_names = (
            list(bundle_data.manifest["column_names"]["features_num"])
            if bundle_data is not None
            else list(features.columns)
            if features is not None
            else []
        )
        bind = getattr(self.strategy, "bind_columns", None)
        if callable(bind):
            bind(feature_names)
        if self.strategy_actions:
            missing = [c for c in self.strategy.required_features if c not in feature_names]
            if missing:
                raise ValueError(f"Missing strategy features: {missing}")
        self._max_tp_dist: np.ndarray[Any, Any] | None = None
        if self._bundle_arrays is not None and self.strategy_actions:
            max_tp = self._bundle_arrays["max_tp_dist"]
            self._max_tp_dist = max_tp if max_tp.size else None
        elif (
            self.strategy_actions
            and bars is not None
            and features is not None
            and "atr_5" in features.columns
        ):
            from features.context.reachability import max_tp_distance

            dist = max_tp_distance(bars, features["atr_5"]).reindex(bars.index)
            self._max_tp_dist = dist.to_numpy(dtype=float)


__all__ = ["StrategyInitializationMixin"]
