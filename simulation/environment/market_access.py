import numpy as np
import pandas as pd

from trading.account.state import AccountState

from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase
from .timebase import BAR_TIME_TZ


class MarketAccessMixin(RuntimeMixinBase):
    def _bar_time(self, idx: int) -> pd.Timestamp:
        """Return bar ``idx`` as a tz-aware Timestamp in the broker timezone.

        Frames mode stores tz-aware Timestamps; bundle mode stores a naive
        ``datetime64[ns]`` array whose values are the UTC epoch of each bar, so
        the naive value is localised to UTC and converted to the broker
        timezone. This keeps the two data paths emitting identical (value +
        timezone) timestamps for the equity curve and trade log.
        """
        value = self._bar_times[idx]
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(BAR_TIME_TZ)

    def _create_account(self) -> AccountState:
        """Factory for fresh account state."""
        return AccountState(
            initial_balance=self.initial_balance, policy_mode=self.trading_policy.mode
        )

    def _sync_bar_arrays(self) -> None:
        """Rebuild OHLC/spread numpy caches from ``self.bars``.

        Call after construction and after any in-place mutation of ``env.bars``
        (tests that poke SL/TP levels via ``env.bars.loc[...] = ...``).
        """
        if self._bundle_arrays is not None:
            bars = self._bundle_arrays
            self._open_arr = bars["bar_ohlc"][:, 0]
            self._high_arr = bars["bar_ohlc"][:, 1]
            self._low_arr = bars["bar_ohlc"][:, 2]
            self._close_arr = bars["bar_ohlc"][:, 3]
            self._spread_arr = bars["bar_spread"]
            return
        assert self.bars is not None
        n = len(self.bars)
        self._open_arr = self.bars["open"].to_numpy(dtype=np.float64, copy=True)
        self._high_arr = self.bars["high"].to_numpy(dtype=np.float64, copy=True)
        self._low_arr = self.bars["low"].to_numpy(dtype=np.float64, copy=True)
        self._close_arr = self.bars["close"].to_numpy(dtype=np.float64, copy=True)
        if "spread" in self.bars.columns:
            self._spread_arr = self.bars["spread"].to_numpy(dtype=np.float64, copy=True)
        else:
            self._spread_arr = np.full(n, np.nan, dtype=np.float64)

    def _bar_at(self, idx: int) -> BarView:
        """OHLC (+ spread) view for bar ``idx`` without pandas."""
        return BarView(
            open=float(self._open_arr[idx]),
            high=float(self._high_arr[idx]),
            low=float(self._low_arr[idx]),
            close=float(self._close_arr[idx]),
            spread=float(self._spread_arr[idx]),
        )

    def _feature_row_at(self, idx: int) -> FeatureRow:
        """Feature-matrix row view for bar ``idx`` without ``pd.Series``."""
        return FeatureRow(
            self._features_arr[idx],
            self._features_col_to_idx,
            int(self._feature_time_ms[idx]),
        )

    def _current_session_id(self) -> int:
        """Return the session_id for the current bar."""
        if self._session_ids_arr is not None and 0 <= self.step_idx < len(self._session_ids_arr):
            return int(self._session_ids_arr[self.step_idx])
        return 0
