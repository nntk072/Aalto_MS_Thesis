"""RL live/paper trading entrypoint (Master Roadmap Stage 8).

Loads a trained SB3 PPO/SAC checkpoint and drives ``RLRobot`` against a live
MT5 M1 feed.

Safeguards
----------
- ``PAPER_TRADING`` defaults to ``true``: every signal and intended order is
  logged, no orders are placed. Set ``PAPER_TRADING=false`` explicitly to trade
  for real.
- When the deployed model was trained with SMT/secondary-symbol features
  (``training.use_m1_only=false`` and ``training.primary_only=false``), a
  secondary ``MT5Data`` source is wired and fed each tick via
  ``update_bars(secondary_bars=...)`` so the live feature vector matches
  training (Stage 8 Task 8.4). If the run was trained ``primary_only=True`` the
  secondary source is skipped on purpose — see ``_use_secondary_symbol``.

Usage
-----
    # One dry-run cycle (logs signals only; default PAPER_TRADING=true)
    RL_MODEL_PATH=outputs/.../model/ppo_final python live_trading_rl.py --once

    # Continuous paper run on a demo feed
    PAPER_TRADING=true RL_MODEL_PATH=outputs/.../model/ppo_final \\
        python live_trading_rl.py

    # REAL orders (only once the Stage-10 paper trial period has passed)
    PAPER_TRADING=false RL_MODEL_PATH=outputs/.../model/ppo_final \\
        python live_trading_rl.py

See DEPLOYMENT.md for the paper->live promotion protocol and model versioning.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mt5_trading.robot.rl_robot import RLRobot

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
from mt5_trading.logging_config import configure_logging  # noqa: E402

DEFAULT_CONFIG = Path("quant_rl/config/default.yaml")


def load_model(model_path: Path) -> Any:
    """Load an SB3 PPO/SAC checkpoint, inferring the algorithm class from the
    accompanying training log when possible (fallback: try PPO, then SAC)."""
    from stable_baselines3 import PPO, SAC

    run_dir = model_path.resolve().parent.parent
    algo_hint: str | None = None
    for name in ("training_log.json", "metrics.json"):
        report = run_dir / name
        if report.is_file():
            try:
                algo_hint = json.loads(report.read_text()).get("algo")
            except (ValueError, OSError):
                pass
            if algo_hint:
                break

    if algo_hint == "sac":
        return SAC.load(str(model_path))
    if algo_hint == "ppo":
        return PPO.load(str(model_path))
    # Unknown algo: try PPO first, then SAC.
    try:
        return PPO.load(str(model_path))
    except Exception:
        return SAC.load(str(model_path))


def resolve_run_config(model_path: Path) -> Path:
    """Prefer the model run's saved ``config.yaml`` (best train/live parity),
    falling back to the repo default."""
    run_dir = model_path.resolve().parent.parent
    saved = run_dir / "config.yaml"
    return saved if saved.is_file() else DEFAULT_CONFIG


def _use_secondary_symbol(cfg: Any) -> bool:
    """Whether the deployed model needs secondary-symbol (SMT) live data.

    Mirror of ``RLStrategyAdapter._rebuild_features``: SMT is used unless the
    training run was configured ``use_m1_only=True`` or ``primary_only=True``.
    """
    training = getattr(cfg, "training", None)
    use_m1_only = bool(getattr(training, "use_m1_only", False))
    primary_only = bool(getattr(training, "primary_only", False))
    return not use_m1_only and not primary_only


_LIVE_RISK_DEFAULTS: dict[str, float] = {
    "risk_per_symbol": 0.01,
    "max_total_risk": 0.02,
    "default_lot_size": 0.1,
    "stop_loss_pips": 50.0,
    "interval_minutes": 1.0,
}


def load_risk_overrides(cfg: Any) -> dict[str, float]:
    """Extract the ``live_risk_overrides`` block for ``RiskManager``/cadence.

    These are the *live sizing* parameters (percentage-of-balance model), not
    the FTMO dollar kill-switches used in training/eval — see the comment on
    ``live_risk_overrides:`` in ``quant_rl/config/default.yaml`` for why the
    two models differ and how they are kept aligned. Any key missing from the
    run's config falls back to the repo default *with an explicit warning*, so
    a config that silently drops the block can't quietly change live risk.
    """
    block = getattr(cfg, "live_risk_overrides", None)
    out = dict(_LIVE_RISK_DEFAULTS)
    if block is None:
        logger.warning(
            "no live_risk_overrides block in run config — using repo defaults {}",
            out,
        )
        return out
    for key, default in _LIVE_RISK_DEFAULTS.items():
        value = block.get(key, None) if hasattr(block, "get") else getattr(block, key, None)
        if value is None:
            logger.warning("live_risk_overrides.{} missing — using default {}", key, default)
            continue
        out[key] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    """CLI/env arguments for the RL live/paper entrypoint."""
    parser = argparse.ArgumentParser(description="RL live/paper trading on MT5 (Stage 8)")
    parser.add_argument(
        "--model",
        default=os.environ.get("RL_MODEL_PATH"),
        help="Path to a trained SB3 checkpoint (e.g. outputs/<run>/model/ppo_final). "
        "Defaults to $RL_MODEL_PATH.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("RL_CONFIG_PATH"),
        help="Run config.yaml to drive features/risk (default: the run's saved config.yaml, "
        "falling back to quant_rl/config/default.yaml).",
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("RL_SYMBOL", "US100"),
        help="Primary MT5 symbol the policy was trained on (default: US100).",
    )
    parser.add_argument(
        "--secondary-symbol",
        default=os.environ.get("RL_SECONDARY_SYMBOL", "US500"),
        help="Secondary symbol for SMT features, only wired when the model was trained "
        "with them (default: US500).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single trade cycle and exit (default: loop forever).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=None,
        help="Override the M1 cadence between cycles (default: live_risk_overrides.interval_minutes).",
    )
    return parser.parse_args()


def build_robot(model_path: Path, run_config: Path, args: argparse.Namespace) -> RLRobot:
    """Construct the fully-wired ``RLRobot`` from env/CLI config."""
    try:
        import MetaTrader5 as mt5  # noqa: F401  (used via mt5.* below)
        from mt5_trading.domain.data_sources.mt5_data import MT5Data
        from mt5_trading.domain.mt5_connection import ensure_mt5_logged_in
        from mt5_trading.domain.risk_manager import RiskManager
        from mt5_trading.domain.trader import MT5Trader
        from mt5_trading.robot.rl_robot import RLRobot

        from quant_rl.live.rl_strategy import RLStrategyAdapter
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "MetaTrader5":
            raise SystemExit(
                "MetaTrader5 is not available on this machine — run "
                "live_trading_rl.py on the Windows MT5 host (see DEPLOYMENT.md)."
            ) from None
        raise
    from omegaconf import OmegaConf

    model = load_model(model_path)
    cfg = OmegaConf.load(run_config)
    risk = load_risk_overrides(cfg)
    paper_trading = os.environ.get("PAPER_TRADING", "true").strip().lower() != "false"

    login = os.environ.get("LOGIN")
    password = os.environ.get("PASSWORD")
    server = os.environ.get("SERVER")
    terminal_path = os.environ.get("TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if not login or not password or not server:
        raise SystemExit("LOGIN / PASSWORD / SERVER must be set in the environment (see .env)")

    ensure_mt5_logged_in(login=login, password=password, server=server, terminal_path=terminal_path)

    data_source = MT5Data(args.symbol, mt5.TIMEFRAME_M1)

    secondary_data_source = None
    if _use_secondary_symbol(cfg):
        logger.info(f"Wiring secondary symbol {args.secondary_symbol} for SMT features")
        secondary_data_source = MT5Data(args.secondary_symbol, mt5.TIMEFRAME_M1)
    else:
        logger.info(
            "Skipping secondary symbol: model trained primary_only/M1-only "
            "(no SMT features to feed live)"
        )

    adapter = RLStrategyAdapter(model=model, config_path=str(run_config))
    return RLRobot(
        adapter=adapter,
        data_source=data_source,
        trader=MT5Trader(),
        risk_manager=RiskManager(
            risk_per_symbol=risk["risk_per_symbol"], max_total_risk=risk["max_total_risk"]
        ),
        secondary_data_source=secondary_data_source,
        default_lot_size=risk["default_lot_size"],
        stop_loss_pips=risk["stop_loss_pips"],
        paper_trading=paper_trading,
    )


def main() -> int:
    """Entry point: run one cycle (``--once``) or a continuous loop."""
    from omegaconf import OmegaConf

    args = parse_args()
    if not args.model:
        raise SystemExit(
            "RL_MODEL_PATH is required (or pass --model) — point it at a trained "
            "outputs/<run>/model/ppo_final checkpoint"
        )
    configure_logging()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model checkpoint not found: {model_path}")

    run_config = Path(args.config) if args.config else resolve_run_config(model_path)
    risk = load_risk_overrides(OmegaConf.load(run_config))
    interval_minutes = (
        float(args.interval_minutes) if args.interval_minutes else risk["interval_minutes"]
    )

    robot = build_robot(model_path, run_config, args)
    logger.info("Starting RL live/paper loop (interval=%.0f min)", interval_minutes)
    while True:
        robot.trade()
        if args.once:
            break
        time.sleep(int(interval_minutes * 60))
    logger.info("RL live/paper run finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
