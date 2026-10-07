# mypy: ignore-errors
"""Score rule baselines in TradingEnv on the thesis train and test splits.

Does not train a network. Writes one summary per rule under outputs/rule_baselines/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_rl.baselines import EMAMACDRSIStrategy, SessionMomentumStrategy  # noqa: E402
from quant_rl.baselines.cash_hold import fit_hold_lots, simulate_hold  # noqa: E402
from quant_rl.config import load_config  # noqa: E402
from quant_rl.data.pipeline import run_pipeline  # noqa: E402
from quant_rl.data.split import get_split_config, split_bars  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation.runner import run_episode  # noqa: E402

RULES = (
    "buy_and_hold",
    "ema_macd_rsi",
    "session_momentum",
)


def _strategy(name: str, bars: pd.DataFrame):
    if name == "ema_macd_rsi":
        strategy = EMAMACDRSIStrategy(bars)
    elif name == "session_momentum":
        strategy = SessionMomentumStrategy(bars)
    else:
        raise ValueError(name)
    strategy.fast_forward(60)
    return strategy


def _index_return_pct(bars: pd.DataFrame) -> float:
    """Unlevered move from the first traded bar to the last close."""
    if len(bars) <= 60:
        return 0.0
    start = float(bars["close"].iloc[60])
    end = float(bars["close"].iloc[-1])
    if start == 0.0:
        return 0.0
    return (end / start - 1.0) * 100.0


def _score(name: str, bars: pd.DataFrame) -> dict:
    strategy = _strategy(name, bars)
    features = bars.select_dtypes(include="number").copy()
    features["last_swing_low"] = bars["low"].astype(float).rolling(5, min_periods=1).min().shift(1)
    features["last_swing_high"] = (
        bars["high"].astype(float).rolling(5, min_periods=1).max().shift(1)
    )

    def action_fn(obs, _strategy=strategy):
        return np.array([_strategy.act(obs)], dtype=np.float32)

    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=60,
        continuous_actions=True,
        episodic=False,
        reward_mode="dsr",
        block_overnight=True,
        rr_ratio_range=(1.0, 3.0),
    )
    metrics = run_episode(env, action_fn=action_fn, max_steps=len(bars) + 5)
    closes = [t for t in env.trade_log if "close" in str(t.get("type", ""))]
    return {
        "sharpe": float(metrics.sharpe),
        "total_pnl": float(metrics.total_pnl),
        "total_return_pct": float(metrics.total_return_pct),
        "max_drawdown": float(metrics.max_drawdown),
        "n_trades": int(metrics.n_trades),
        "n_closes": len(closes),
        "breach_count": int(metrics.breach_count),
        "index_return_pct": _index_return_pct(bars),
    }


def main() -> None:
    cfg = load_config([])
    data = run_pipeline(cfg, force=False)
    bars = data[cfg.data.primary]["M1"]
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars = split_bars(bars, train_end, test_start)
    out = ROOT / "outputs" / "rule_baselines"
    out.mkdir(parents=True, exist_ok=True)
    fitted = fit_hold_lots(train_bars)
    print("fitted lots", fitted, flush=True)
    report: dict[str, dict] = {
        "buy_and_hold": {
            "lots": fitted,
            "train": simulate_hold(train_bars, fitted["lots"]),
            "test": simulate_hold(test_bars, fitted["lots"]),
        }
    }
    print("buy_and_hold", report["buy_and_hold"]["test"], flush=True)
    for name in ("ema_macd_rsi", "session_momentum"):
        print(f"scoring {name}", flush=True)
        report[name] = {
            "train": _score(name, train_bars),
            "test": _score(name, test_bars),
        }
        print(name, report[name]["test"], flush=True)
    flat = {
        "sharpe": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "n_closes": 0,
        "breach_count": 0,
        "index_return_pct": 0.0,
    }
    report["flat"] = {"train": flat, "test": dict(flat)}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
