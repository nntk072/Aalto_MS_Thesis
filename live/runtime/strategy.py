"""RL→MT5 live inference bridge (Chain C).

Adapts a trained SB3 RL model into the ``integrations.mt5`` stack's
``TradingStrategy`` interface so live execution can consume the same
policy evaluated in backtests.

Contract:
- Input: an ``MT5Data``-like data source returning M1 OHLCV bars.
- Internally rebuilds the **exact** feature matrix used at training time
  via :func:`features.pipeline.builder.build_features` (same config file),
  so train/live feature parity holds by construction.
- Observations are built with the same dict contract as
  ``TradingEnv._get_observation``:
  ``{"seq": (T, F), "seq_mask": (T,), "account": (6,)}``.
- Output: ``LiveSignal.BUY/SELL/HOLD`` mapped from the policy's action
  (discrete PPO action ids or continuous SAC sizing fraction).

This module imports ``integrations.mt5`` lazily so the RL stack remains
installable/usable without the MT5 runtime.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decisioning.adapters.sb3 import SB3DecisionProvider
from decisioning.contracts import (
    ContinuousAction,
    DecisionInput,
    DecisionProvider,
    DiscreteAction,
    StructuredAction,
)
from decisioning.observations import (
    OBSERVATION_SCHEMA_V2,
    ObservationSchema,
    account_state_v2,
    build_observation_schema,
    normalized_account_vector,
    pad_observation_window,
    validate_observation_schema,
)
from omegaconf import OmegaConf

from configuration.paths import repository_root
from features import attach_reachable_r, select_obs_columns
from live.runtime.contracts import LiveSignal, MarketDataSource
from trading.policy.config import resolve_trading_policy

DEFAULT_CONFIG = repository_root(__file__) / "config" / "base" / "default.yaml"


def build_live_observation_schema(cfg: Any) -> ObservationSchema:
    """Resolve the live observation schema using the training policy rules."""
    env_cfg = getattr(cfg, "env", None)
    version = (
        "account-state-v3-repeated-payout"
        if resolve_trading_policy(cfg).repeated_payout
        else str(getattr(env_cfg, "observation_schema_version", "account-state-v1"))
    )
    return build_observation_schema(
        version,
        entry_state_visibility=bool(
            getattr(env_cfg, "entry_state_machine", False)
            and getattr(env_cfg, "entry_state_observation", True)
        ),
    )


class RLStrategyAdapter:
    """Wraps a trained SB3 model as a live-trading strategy.

    Not a ``TradingStrategy`` subclass itself — use :meth:`as_strategy`
    to obtain the duck-typed ``signal()`` callable the MT5 robot loop
    expects, or call :meth:`predict_signal` directly.
    """

    def __init__(
        self,
        model: Any | None = None,
        config_path: str | Path = DEFAULT_CONFIG,
        features_config_path: str | Path | None = None,
        obs_window: int = 60,
        allow_legacy_observation_schema: bool = False,
        *,
        provider: DecisionProvider | None = None,
    ) -> None:
        if (model is None) == (provider is None):
            raise ValueError("provide exactly one of model or provider")
        cfg = OmegaConf.create(OmegaConf.load(config_path))
        if features_config_path is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(features_config_path))
        self.cfg = cfg
        env_cfg = getattr(self.cfg, "env", None)
        self.obs_window = int(getattr(env_cfg, "obs_window", obs_window))
        account_cfg = getattr(self.cfg, "account", None)
        self._initial_balance = float(getattr(account_cfg, "initial_balance", 100_000.0))
        self._observation_schema = build_live_observation_schema(self.cfg)
        if provider is not None:
            if provider.metadata.input_schema_version != self._observation_schema.version:
                raise ValueError(
                    "decision provider input schema differs from the live observation schema"
                )
            self.provider = provider
        else:
            validate_observation_schema(
                getattr(model, "_observation_schema", None),
                self._observation_schema,
                allow_legacy_v1=allow_legacy_observation_schema,
            )
            self.provider = SB3DecisionProvider(
                model,
                input_schema_version=self._observation_schema.version,
            )
        self._features: pd.DataFrame | None = None
        self._bars: pd.DataFrame | None = None
        self._last_close = 1.0
        self._feature_lock = threading.Lock()
        self._secondary_bars: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Feature (re)build — mirrors training-time pipeline exactly.
    # ------------------------------------------------------------------
    def _rebuild_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the feature matrix from raw M1 bars (train-parity path)."""
        from features import build_features  # local: heavy import

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
        self._bars = bars
        if "close" in bars.columns and len(bars):
            self._last_close = float(bars["close"].iloc[-1])
        self._rebuild_features(bars)

    def _raw_columns(self) -> tuple[str, ...]:
        env_cfg = getattr(self.cfg, "env", None)
        if not bool(getattr(env_cfg, "strategy_actions", False)):
            return ()
        name = str(getattr(getattr(self.cfg, "strategy", None), "name", "baseline"))
        if name == "po3_ifvg":
            from trading.strategies import PO3IFVGStrategy

            return tuple(PO3IFVGStrategy.raw_columns)
        if name == "distribution":
            from trading.strategies import DistributionStrategy

            return tuple(DistributionStrategy.raw_columns)
        return ()

    def build_observation(
        self, account_state: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Construct the dict observation with the same contract as the env."""
        if self._features is None:
            raise RuntimeError("update_bars() must be called before build_observation()")

        feats = select_obs_columns(self._features, self._raw_columns())
        if self._bars is not None:
            feats = attach_reachable_r(feats, self._bars, self._features)
        window = feats.iloc[-self.obs_window :]
        seq = np.nan_to_num(np.asarray(window.values, dtype=np.float32), nan=0.0)
        seq, seq_mask = pad_observation_window(seq, self.obs_window)

        st = account_state or {}
        equity = float(st.get("equity", self._initial_balance))
        pos_dir = float(st.get("position_direction", 0.0))
        open_pnl = float(st.get("open_pnl", 0.0))
        dist_to_sl = float(st.get("dist_to_sl", 0.0))
        trailing_dd = float(st.get("trailing_dd", 0.0))
        close = float(st.get("close", self._last_close))
        if self._observation_schema.version == OBSERVATION_SCHEMA_V2:
            required = {
                "planned_risk_usd",
                "trailing_drawdown_limit",
                "daily_loss",
                "daily_loss_limit",
                "loss_from_initial",
                "max_loss_limit",
                "risk_per_trade_limit",
                "entry_price",
                "initial_sl_price",
                "active_sl_price",
                "remaining_size",
                "initial_size",
                "position_age_bars",
                "tp_hit_mask",
                "tp1_price",
                "tp2_price",
                "tp3_price",
                "sl_mode",
                "tp_mode",
                "exit_mode",
            }
            missing = sorted(required.difference(st))
            if self._observation_schema.entry_state_visibility and "entry_state" not in st:
                missing.append("entry_state")
            if missing:
                raise ValueError(
                    "account-state-v2 live inference requires complete account state; "
                    f"missing {missing}"
                )
            account = account_state_v2(
                equity=equity,
                initial_balance=self._initial_balance,
                position_direction=pos_dir,
                open_pnl=open_pnl,
                planned_risk_usd=float(st["planned_risk_usd"]),
                trailing_drawdown_pct=trailing_dd,
                trailing_drawdown_limit=float(st["trailing_drawdown_limit"]),
                daily_loss=float(st["daily_loss"]),
                daily_loss_limit=float(st["daily_loss_limit"]),
                loss_from_initial=float(st["loss_from_initial"]),
                max_loss_limit=float(st["max_loss_limit"]),
                risk_per_trade_limit=float(st["risk_per_trade_limit"]),
                current_price=close,
                entry_price=float(st["entry_price"]),
                initial_sl_price=float(st["initial_sl_price"]),
                active_sl_price=float(st["active_sl_price"]),
                remaining_size=float(st["remaining_size"]),
                initial_size=float(st["initial_size"]),
                position_age_bars=float(st["position_age_bars"]),
                position_age_scale_bars=float(
                    getattr(self.cfg.env, "observation_age_scale_bars", 60.0)
                ),
                tp_hit_mask=int(st["tp_hit_mask"]),
                tp_prices=(
                    float(st["tp1_price"]),
                    float(st["tp2_price"]),
                    float(st["tp3_price"]),
                ),
                sl_mode=str(st["sl_mode"]),
                tp_mode=str(st["tp_mode"]),
                exit_mode=str(st["exit_mode"]),
                entry_state=str(st["entry_state"]) if "entry_state" in st else None,
            )
        else:
            account = normalized_account_vector(
                equity=equity,
                initial_balance=self._initial_balance,
                pos_dir=pos_dir,
                open_pnl=open_pnl,
                dist_to_sl=dist_to_sl,
                trailing_dd=trailing_dd,
                close=close,
            )
        return {"seq": seq, "seq_mask": seq_mask, "account": account}

    def predict_signal(self, account_state: dict[str, float] | None = None) -> int | float:
        """Run the provider and return its discrete id or scalar direction."""
        obs = self.build_observation(account_state)
        decision = self.provider.predict(
            DecisionInput(
                observation=obs,
                schema_version=self._observation_schema.version,
            ),
            deterministic=True,
        )
        action = decision.action
        if isinstance(action, DiscreteAction):
            return action.index
        if isinstance(action, ContinuousAction):
            return action.value
        if isinstance(action, StructuredAction) and "direction" in action.coordinates:
            return float(action.coordinates["direction"])
        raise ValueError("live decisions must be discrete, scalar or include direction")

    # ------------------------------------------------------------------
    # integrations.mt5 adapter
    # ------------------------------------------------------------------
    def as_strategy(
        self,
        data_source: MarketDataSource,
        secondary_data_source: MarketDataSource | None = None,
    ) -> Any:
        """Return a duck-typed ``TradingStrategy`` wrapping this adapter.

        Parameters
        ----------
        data_source:
            Primary-symbol MT5 data source (M1 bars).
        secondary_data_source:
            Optional secondary-symbol data source used for SMT divergence
            features. When the deployed model was trained with SMT enabled
            (``training.use_m1_only``/``primary_only`` both false), the live
            feature vector silently misses those features unless the
            secondary bars are fed each tick via ``update_bars(...,
            secondary_bars=...)`` — this wired path exists precisely to avoid
            that train/live mismatch.
        """
        adapter = self

        class _RLStrategy:  # nested: needs closure over adapter
            """Duck-typed TradingStrategy: exposes .data and .signal()."""

            def __init__(self) -> None:
                self.data = data_source

            def signal(self) -> tuple[str, LiveSignal]:
                symbol = self.data.get_symbol()
                bars = self.data.get_data()
                if bars is None or len(bars) < adapter.obs_window + 10:
                    return symbol, LiveSignal.NONE
                secondary_bars: pd.DataFrame | None = None
                if secondary_data_source is not None:
                    secondary_bars = secondary_data_source.get_data()
                adapter.update_bars(bars, secondary_bars=secondary_bars)
                try:
                    action = adapter.predict_signal()
                except Exception:  # noqa: BLE001  (live loop must not crash)
                    return symbol, LiveSignal.NONE
                # Discrete contract: >0 → long, <0 → short, 0 → flat.
                if action > 0:
                    return symbol, LiveSignal.BUY
                if action < 0:
                    return symbol, LiveSignal.SELL
                return symbol, LiveSignal.HOLD

        return _RLStrategy()


__all__ = ["RLStrategyAdapter"]
