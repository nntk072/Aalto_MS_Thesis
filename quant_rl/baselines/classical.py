"""Classical trading baselines: buy-and-hold, EMA/MACD/RSI, multi-level breakout.

All strategies emit continuous actions in ``[-1, 1]`` and rely on the
``TradingEnv`` entry gate and SL/TP logic for risk enforcement, so the
comparison with RL agents is apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indicators import macd, rsi
from .base import BaseStrategy


def _ny_session_day(index: pd.DatetimeIndex, tz: str = "Etc/GMT-3") -> pd.Series:
    """Calendar day of each bar that falls in 16:30–23:00. Else missing."""
    ts = index.tz_convert(tz) if index.tz is not None else index.tz_localize(tz)
    minutes = ts.hour * 60 + ts.minute
    in_session = (minutes >= 16 * 60 + 30) & (minutes <= 23 * 60)
    return pd.Series(ts.normalize(), index=index).where(in_session)


class BuyAndHoldStrategy(BaseStrategy):
    """Long-only hold. ``mode`` selects how often a new long is requested.

    ``always`` asks for a long on every bar (the original constant signal).
    ``intraday`` asks once, on the first bar of each NY session, and is flat
    overnight because the environment closes the position at the session boundary.
    ``swing`` asks once, on the first NY bar of the frame, and must be scored
    with overnight holding left on so that single position can stay open.
    """

    def __init__(
        self,
        n_bars: int | None = None,
        size: float = 1.0,
        bars: pd.DataFrame | None = None,
        mode: str = "always",
        tz: str = "Etc/GMT-3",
        entry_from: int = 0,
    ) -> None:
        """Store a constant long, or a one-entry-per-session / one-entry signal.

        Args:
            n_bars: Length of a constant signal. Required when ``mode`` is ``always``.
            size: Long fraction in ``(0, 1]``.
            bars: OHLCV frame. Required for ``intraday`` and ``swing``.
            mode: ``always``, ``intraday``, or ``swing``.
            tz: Session clock. Broker time in this thesis is ``Etc/GMT-3``.
        """
        if mode not in {"always", "intraday", "swing"}:
            raise ValueError(f"unknown buy-and-hold mode: {mode}")
        self._size = float(min(max(size, 0.0), 1.0))
        self._mode = mode
        if mode == "always":
            if n_bars is None:
                raise ValueError("always buy-and-hold requires n_bars")
            super().__init__(n_bars)
            self._signals = None
            return
        if bars is None or not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError(f"{mode} buy-and-hold requires a DatetimeIndex")
        day = _ny_session_day(bars.index, tz)
        signal = pd.Series(0.0, index=bars.index)
        if mode == "intraday":
            first = day.dropna().groupby(day.dropna()).head(1).index
            signal.loc[first] = self._size
        else:
            positions = np.flatnonzero(day.notna().to_numpy())
            later = positions[positions >= int(entry_from)]
            if len(later):
                signal.iloc[int(later[0])] = self._size
        self._signals = signal.to_numpy(dtype=float)
        super().__init__(len(self._signals))

    def _signal(self, idx: int) -> float:
        if self._signals is None:
            return self._size
        return float(self._signals[idx])


class EMAMACDRSIStrategy(BaseStrategy):
    """Trend-following baseline combining EMA cross, MACD and RSI filter.

    Long when EMA(fast) > EMA(slow), MACD > signal and RSI is not
    overbought; short on the mirror condition; hold otherwise.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        ema_fast: int = 12,
        ema_slow: int = 26,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        """Pre-compute indicator signals from close prices.

        Args:
            bars: OHLCV DataFrame with a ``close`` column.
            ema_fast: Fast EMA span.
            ema_slow: Slow EMA span.
            rsi_period: RSI lookback.
            rsi_overbought: Block new longs above this RSI.
            rsi_oversold: Block new shorts below this RSI.
        """
        close = bars["close"].astype(float)
        ema_f = close.ewm(span=ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=ema_slow, adjust=False).mean()
        macd_df = macd(close, fast=ema_fast, slow=ema_slow)
        rsi_vals = rsi(close, period=rsi_period)

        trend_up = (ema_f > ema_s) & (macd_df["macd"] > macd_df["macd_signal"])
        trend_down = (ema_f < ema_s) & (macd_df["macd"] < macd_df["macd_signal"])
        not_overbought = rsi_vals < rsi_overbought
        not_oversold = rsi_vals > rsi_oversold

        signal = pd.Series(0.0, index=bars.index)
        signal[trend_up & not_overbought] = 1.0
        signal[trend_down & not_oversold] = -1.0
        self._signals = signal.fillna(0.0).to_numpy(dtype=float)
        super().__init__(len(self._signals))

    def _signal(self, idx: int) -> float:
        return float(self._signals[idx])


class SessionMomentumStrategy(BaseStrategy):
    """Hold the sign of the previous NY session's close-to-close return.

    The session window is 16:30–23:00 in ``tz``. The first session is flat.
    Bars outside that window stay flat. This is the Moskowitz, Ooi, and
    Pedersen sign rule, shortened from a year to one session so it can be
    scored on the same minute bars as the PPO policies.
    """

    def __init__(self, bars: pd.DataFrame, tz: str = "Etc/GMT-3") -> None:
        """Pre-compute one action per bar from the prior session return.

        Args:
            bars: OHLCV frame with a DatetimeIndex and a ``close`` column.
            tz: Session clock. Broker time in this thesis is ``Etc/GMT-3``.
        """
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError("session momentum requires a DatetimeIndex")
        close = bars["close"].astype(float)
        ts = bars.index
        ts = ts.tz_convert(tz) if ts.tz is not None else ts.tz_localize(tz)
        minutes = ts.hour * 60 + ts.minute
        in_session = (minutes >= 16 * 60 + 30) & (minutes <= 23 * 60)
        day = pd.Series(ts.normalize(), index=bars.index).where(in_session)

        session_return: dict[pd.Timestamp, float] = {}
        for session_day, prices in close.groupby(day):
            if pd.isna(session_day) or len(prices) < 2:
                continue
            session_return[session_day] = float(prices.iloc[-1] / prices.iloc[0] - 1.0)

        ordered = sorted(session_return)
        sign_by_day: dict[pd.Timestamp, float] = {}
        for i, session_day in enumerate(ordered):
            if i == 0:
                sign_by_day[session_day] = 0.0
            else:
                sign_by_day[session_day] = float(np.sign(session_return[ordered[i - 1]]))

        signal = day.map(lambda session_day: sign_by_day.get(session_day, 0.0)).fillna(0.0)
        self._signals = signal.to_numpy(dtype=float)
        super().__init__(len(self._signals))

    def _signal(self, idx: int) -> float:
        return float(self._signals[idx])
