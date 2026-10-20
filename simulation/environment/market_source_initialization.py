"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from .mixin_base import RuntimeMixinBase


class MarketSourceInitializationMixin(RuntimeMixinBase):
    def _initialize_market_source(
        self,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
        bundle_data: Any | None,
        strict_research_eligibility: bool,
    ) -> None:
        self.bars: pd.DataFrame = cast(pd.DataFrame, bars)
        self.features: pd.DataFrame = cast(pd.DataFrame, features)
        self._bundle_arrays: dict[str, np.ndarray[Any, Any]] | None = (
            {name: np.asarray(array) for name, array in bundle_data.arrays.items()}
            if bundle_data is not None
            else None
        )
        if self._bundle_arrays is not None:
            self._feature_time_ms = (self._bundle_arrays["bar_time_ns"] // 1_000_000).astype(
                np.int64, copy=False
            )
        else:
            self._feature_time_ms = (
                pd.DatetimeIndex(self.features.index)
                .to_numpy(dtype="datetime64[ns]")
                .view(np.int64)
                // 1_000_000
            ).astype(np.int64, copy=False)
        bundle_labels = bundle_data.manifest.get("labels", {}) if bundle_data is not None else {}
        self._n_bars = (
            len(self._bundle_arrays["bar_time_ns"])
            if self._bundle_arrays is not None
            else len(bars)
            if bars is not None
            else 0
        )
        # Legacy permissive fallback remains available for old checkpoints.
        # Confirmatory runs can explicitly prohibit missing eligibility data.
        self.strict_research_eligibility = bool(strict_research_eligibility)
        if self._bundle_arrays is None:
            assert bars is not None and features is not None
            self._ny_indices = self._resolve_ny_indices(bars)
        else:
            times = pd.to_datetime(self._bundle_arrays["bar_time_ns"], utc=True)
            timezone = bundle_labels.get("bar_timezone")
            if timezone:
                times = times.tz_convert(str(timezone))
            if bundle_labels.get("has_session", False):
                sessions = list(bundle_labels.get("session", []))
                ny_codes = [idx for idx, label in enumerate(sessions) if label == "ny"]
                ny_indices = np.flatnonzero(np.isin(self._bundle_arrays["bar_session"], ny_codes))
                if not ny_indices.size and self.strict_research_eligibility:
                    raise ValueError("Strict research eligibility: no eligible NY session bars")
                self._ny_indices = ny_indices if ny_indices.size else np.arange(self._n_bars)
            else:
                self._ny_indices = self._resolve_ny_indices(
                    pd.DataFrame(index=pd.DatetimeIndex(times))
                )
        self._bundle_has_session = bool(bundle_labels.get("has_session", False))
        self._session_labels = list(bundle_labels.get("session", []))
        self._bundle_has_session_id = bool(bundle_labels.get("has_session_id", False))


__all__ = ["MarketSourceInitializationMixin"]
