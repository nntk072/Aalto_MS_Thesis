"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from features import attach_reachable_r, select_legacy_obs_columns, select_obs_columns

from .mixin_base import RuntimeMixinBase
from .observations import DEFAULT_HTF_WINDOWS, HTF_BRANCHES, run_starts, split_obs_columns


class ObservationArraysInitializationMixin(RuntimeMixinBase):
    def _initialize_observation_arrays(
        self,
        bundle_data: Any | None,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
        legacy_sequence_columns: bool | None,
        episodic: bool,
        obs_features_mmap: str | None,
        mtf: bool,
        mtf_windows: dict[str, int] | None,
    ) -> None:
        # Model-facing observation frame: the strategy's raw price-level
        # columns stay in ``features`` for execution/risk but must not enter
        # the normalised ``seq`` tensor (Agent.md §11). MTF blocks reuse the
        # same stems with a ``{TF}_`` prefix (e.g. ``M5_last_swing_high``),
        # matched via MTF_RAW_SUFFIXES so unnormalised HTF price magnitudes
        # never reach the encoder.
        # Drop non-numeric columns (e.g. string ``session`` labels) — they cannot
        # be cast to float32 and are not part of the model's normalized observation.
        if self._bundle_arrays is None:
            assert bars is not None and features is not None
            self._legacy_sequence_columns = bool(legacy_sequence_columns)
            select_columns = (
                select_legacy_obs_columns if self._legacy_sequence_columns else select_obs_columns
            )
            self._obs_features: pd.DataFrame = attach_reachable_r(
                select_columns(features, self.strategy.raw_columns),
                bars,
                features,
            )
            self._obs_columns = list(self._obs_features.columns)
        else:
            assert bundle_data is not None
            self._obs_features = cast(pd.DataFrame, None)
            self._obs_columns = list(bundle_data.manifest["column_names"]["obs_features"])
            saved_legacy_columns = bool(
                bundle_data.manifest.get("config", {}).get("legacy_sequence_columns", False)
            )
            if legacy_sequence_columns is not None and bool(legacy_sequence_columns) != (
                saved_legacy_columns
            ):
                raise ValueError(
                    "requested legacy sequence-column mode differs from the saved env bundle"
                )
            self._legacy_sequence_columns = saved_legacy_columns
        if self._legacy_sequence_columns and episodic:
            raise ValueError("legacy sequence columns are restricted to non-episodic replay")
        self._obs_features_mmap = obs_features_mmap

        # Cache numpy arrays for hot-path env stepping — avoids per-step pandas
        # iloc/Series-creation overhead that starves the GPU.
        #   - _obs_features / _features_arr: immutable post-construction
        #   - OHLC/spread bar caches: rebuilt via ``_sync_bar_arrays()`` (call
        #     again after in-place ``env.bars`` mutations in tests)
        # A read-only memmap is one shared copy for every SubprocVecEnv worker.
        if self._bundle_arrays is not None:
            self._obs_features_arr = self._bundle_arrays["obs_features"]
        elif obs_features_mmap:
            assert self._obs_features is not None
            shared = np.load(obs_features_mmap, mmap_mode="r")
            assert self._obs_features is not None
            if shared.shape != self._obs_features.shape:
                raise ValueError(
                    f"obs memmap shape {shared.shape} != obs columns {self._obs_features.shape}"
                )
            self._obs_features_arr = shared
        else:
            assert self._obs_features is not None
            self._obs_features_arr = self._obs_features.to_numpy(dtype=np.float32)
        self.mtf = bool(mtf)
        self._mtf_windows = dict(DEFAULT_HTF_WINDOWS)
        if mtf_windows:
            self._mtf_windows.update({str(k): int(v) for k, v in mtf_windows.items()})
        self._m1_idx, self._htf_idx = split_obs_columns(self._obs_columns)
        self._htf_starts: dict[str, np.ndarray[Any, Any]] = {}
        if self.mtf:
            for key, _, _ in HTF_BRANCHES:
                cols = self._htf_idx[key]
                block = (
                    self._obs_features_arr[:, cols]
                    if cols.size
                    else np.zeros((len(self._obs_features_arr), 0), dtype=np.float32)
                )
                self._htf_starts[key] = run_starts(np.asarray(block, dtype=np.float64))
        if self._bundle_arrays is not None:
            assert bundle_data is not None
            self._features_cols = list(bundle_data.manifest["column_names"]["features_num"])
            self._features_arr = self._bundle_arrays["features_num"]
            self._bar_times = self._bundle_arrays["bar_time_ns"].view("datetime64[ns]")
            self._session_ids_arr = (
                self._bundle_arrays["bar_session_id"] if self._bundle_has_session_id else None
            )
        else:
            assert bars is not None and features is not None
            _feat_numeric = features.select_dtypes(include="number")
            self._features_cols = list(_feat_numeric.columns)
            self._features_arr = _feat_numeric.to_numpy()
            self._bar_times = bars.index.to_numpy()
            self._session_ids_arr = (
                bars["session_id"].to_numpy(dtype=np.int64)
                if "session_id" in bars.columns
                else None
            )
        self._features_col_to_idx = {c: i for i, c in enumerate(self._features_cols)}
        self._sync_bar_arrays()


__all__ = ["ObservationArraysInitializationMixin"]
