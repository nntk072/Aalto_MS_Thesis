"""RL→MT5 live inference bridge (Chain C).

Adapts a trained SB3 RL model into the ``mt5_trading`` stack's
``TradingStrategy`` interface so live execution can consume the same
policy evaluated in backtests.

Contract:
- Input: an ``MT5Data``-like data source returning M1 OHLCV bars.
- Internally rebuilds the **exact** feature matrix used at training time
  via :func:`quant_rl.features.build.build_features` (same config file),
  so train/live feature parity holds by construction.
- Observations are built with the same dict contract as
  ``TradingEnv._get_observation``: ``{"seq": (T, F), "account": (5,)}``.
- Output: ``Signal.BUY/SELL/HOLD`` mapped from the policy's action
  (discrete PPO action ids or continuous SAC sizing fraction).

This module imports ``mt5_trading`` lazily so the RL stack remains
installable/usable without the MT5 runtime.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

if TYPE_CHECKING:  # pragma: no cover
    from mt5_trading.adapters import TradingData

DEFAULT_CONFIG = Path("quant_rl/config/default.yaml")


class RLStrategyAdapter:
    """Wraps a trained SB3 model as a live-trading strategy.

    Not a ``TradingStrategy`` subclass itself — use :meth:`as_strategy`
    to obtain the duck-typed ``signal()`` callable the MT5 robot loop
    expects, or call :meth:`predict_signal` directly.
    """

    def __init__(
        self,
        model: Any,
        config_path: str | Path = DEFAULT_CONFIG,
        features_config_path: str | Path | None = None,
        obs_window: int = 60,
    ) -> None:
        self.model = model
        cfg = OmegaConf.create(OmegaConf.load(config_path))
        if features_config_path is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(features_config_path))
        self.cfg = cfg
        env_cfg = getattr(self.cfg, "env", None)
        self.obs_window = int(getattr(env_cfg, "obs_window", obs_window))
        self._features: pd.DataFrame | None = None
        self._feature_lock = threading.Lock()
        self._secondary_bars: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Feature (re)build — mirrors training-time pipeline exactly.
    # ------------------------------------------------------------------
    def _rebuild_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the feature matrix from raw M1 bars (train-parity path)."""
        from quant_rl.features.build import build_features  # local: heavy import

        secondary: pd.DataFrame | None = None
        training = getattr(self.cfg, "training", None)
        use_secondary = not bool(getattr(training, "use_m1_only", False)) and not bool(
            getattr(training, "primary_only", False)
        )
        if use_secondary:
            # Live bridge consumes the primary symbol only; SMT vs the
            # secondary symbol requires wiring a second MT5 data source
            # via :meth:`update_bars(secondary_bars=...)`.
            secondary = self._secondary_bars

        with self._feature_lock:
            self._features = build_features(
                bars,
                secondary=secondary,
                cfg=self.cfg,  # type: ignore[arg-type]
                force=True,  # never read a stale cache in live mode
            )
        return self._features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_bars(self, bars: pd.DataFrame, secondary_bars: pd.DataFrame | None = None) -> None:
        """Feed the latest M1 bars; rebuilds features (call once per new bar)."""
        self._secondary_bars = secondary_bars
        self._rebuild_features(bars)

    def build_observation(
        self, account_state: dict[str, float] | None = None
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Construct the dict observation with the same contract as the env."""
        if self._features is None:
            raise RuntimeError("update_bars() must be called before build_observation()")

        feats = self._features
        seq = np.asarray(feats.iloc[-self.obs_window :].values, dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        if len(seq) < self.obs_window:
            pad = ((self.obs_window - len(seq), 0), (0, 0))
            seq = np.pad(seq, pad, mode="constant", constant_values=0.0)

        st = account_state or {}
        equity = float(st.get("equity", 1.0))
        pos_dir = float(st.get("position_direction", 0.0))
        open_pnl = float(st.get("open_pnl", 0.0))
        unrealised_r = (open_pnl / equity * 100) if equity > 0 else 0.0
        dist_to_sl = float(st.get("dist_to_sl", 0.0))
        account = np.array([equity, pos_dir, open_pnl, unrealised_r, dist_to_sl], dtype=np.float32)
        return {"seq": seq[np.newaxis, ...], "account": account[np.newaxis, ...]}

    def predict_signal(self, account_state: dict[str, float] | None = None) -> int:
        """Run the policy on the current observation; returns the raw action id."""
        obs = self.build_observation(account_state)
        action = self.model.predict(obs, deterministic=True)[0]
        return int(np.asarray(action).reshape(-1)[0])

    # ------------------------------------------------------------------
    # mt5_trading adapter
    # ------------------------------------------------------------------
    def as_strategy(self, data_source: TradingData) -> Any:
        """Return a duck-typed ``TradingStrategy`` wrapping this adapter."""
        from mt5_trading.domain.signal import Signal

        adapter = self

        class _RLStrategy:  # nested: needs closure over adapter
            """Duck-typed TradingStrategy: exposes .data and .signal()."""

            def __init__(self) -> None:
                self.data = data_source

            def signal(self) -> tuple[str, Signal]:
                symbol = self.data.get_symbol()  # type: ignore[no-untyped-call]
                bars = self.data.get_data()  # type: ignore[no-untyped-call]
                if bars is None or len(bars) < adapter.obs_window + 10:
                    return symbol, Signal.NONE
                adapter.update_bars(bars)
                try:
                    action = adapter.predict_signal()
                except Exception:  # noqa: BLE001  (live loop must not crash)
                    return symbol, Signal.NONE
                # Discrete contract: >0 → long, <0 → short, 0 → flat.
                if action > 0:
                    return symbol, Signal.BUY
                if action < 0:
                    return symbol, Signal.SELL
                return symbol, Signal.HOLD

        return _RLStrategy()


__all__ = ["RLStrategyAdapter"]
