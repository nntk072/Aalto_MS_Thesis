"""Evaluate a trained PPO checkpoint and rewrite ``training/`` + ``testing/``.

Loads an existing RL run's ``config.yaml`` snapshot and a saved model
checkpoint, rolls it through ``TradingEnv`` via
``quant_rl.eval.rollout.evaluate_model``, and overwrites both split
trees — no retraining. Eval uses FTMO daily $5k / max $10k from initial
(the PPO $7k trailing cap is training-only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import logging
import re
import shutil
from datetime import datetime
from typing import cast

from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, split_bars, split_train_test
from quant_rl.eval.export import save_run
from quant_rl.eval.rollout import evaluate_model
from quant_rl.evaluation import calculate_metrics
from quant_rl.features.build import FEATURE_CACHE_VERSION, build_features
from quant_rl.models.ppo_policy import ClampedStdMultiInputPolicy
from quant_rl.train.train_rl import (
    _eval_guardrail_kwargs,
    _max_loss_per_trade,
    _strategy_from_cfg,
    _strategy_risk_ranges,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def _load_run_config(run_dir: Path) -> DictConfig:
    """Prefer the run's own config snapshot so split boundaries/account
    settings match exactly what produced its checkpoints."""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        log.info("Loading config snapshot: %s", cfg_path)
        return cast(DictConfig, OmegaConf.load(cfg_path))
    log.warning("No config.yaml in %s; falling back to default config", run_dir)
    return load_config([])


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    """Return the model checkpoint to evaluate.

    Prefers an explicit ``--checkpoint``, then ``model/ppo_final.zip``
    (written on successful training completion), then the highest-step
    ``ppo_ckpt_*_steps.zip`` (for a run whose training hasn't finished, or
    was interrupted, but has intermediate checkpoints).
    """
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path

    model_dir = run_dir / "model"
    final = model_dir / "ppo_final.zip"
    if final.exists():
        return final

    ckpts = list(model_dir.glob("ppo_ckpt_*_steps.zip"))
    if not ckpts:
        raise SystemExit(f"No model checkpoints found under {model_dir}")

    def _steps(p: Path) -> int:
        m = _CKPT_STEPS_RE.search(p.name)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_steps)
    log.warning("No ppo_final.zip in %s; using latest checkpoint: %s", model_dir, latest.name)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint and rewrite training/ + testing/ artifacts."
    )
    parser.add_argument("--run", required=True, help="Path to an existing RL run directory")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (relative to --run or absolute); defaults to "
        "model/ppo_final.zip, else the highest-step model/ppo_ckpt_*_steps.zip",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force the data pipeline to rebuild caches"
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    cfg = _load_run_config(run_dir)
    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    log.info("Loading model: %s", ckpt_path)
    _ = ClampedStdMultiInputPolicy
    model = PPO.load(ckpt_path)

    data = run_pipeline(cfg, force=args.force)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{cfg.data.primary}_features_{FEATURE_CACHE_VERSION}.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start)
    log.info(
        "Evaluating train=%d bars (≤%s)  test=%d bars (≥%s) under FTMO eval guardrails",
        len(train_bars),
        train_end,
        len(test_bars),
        test_start,
    )

    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    eval_common = dict(
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        guardrail_kwargs=_eval_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        continuous_actions=False,
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        max_episode_steps=None,
    )

    test_result = evaluate_model(model, bars=test_bars, features=test_feat, **eval_common)
    test_result["initial_balance"] = cfg.account.initial_balance
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    log.info(
        "[test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  fail_time=%s",
        test_m.sharpe,
        test_m.max_drawdown * 100,
        test_m.n_trades,
        test_m.total_return_pct,
        test_result.get("fail_time"),
    )

    train_result = evaluate_model(model, bars=train_bars, features=train_feat, **eval_common)
    train_result["initial_balance"] = cfg.account.initial_balance
    train_m = calculate_metrics(
        train_result["equity"],
        trades=train_result["trades"],
        n_sessions=train_result.get("n_sessions", 1),
        n_breach_sessions=train_result.get("n_breach_sessions", 0),
    )
    log.info(
        "[train] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  fail_time=%s",
        train_m.sharpe,
        train_m.max_drawdown * 100,
        train_m.n_trades,
        train_m.total_return_pct,
        train_result.get("fail_time"),
    )
    if test_m.n_trades == 0:
        log.error(
            "Experiment failure: evaluation produced zero trades; "
            "metrics are not meaningful until action behavior is diagnosed."
        )

    for name in ("training", "testing"):
        split_dir = run_dir / name
        if split_dir.exists():
            log.info("Clearing stale %s/ artifacts: %s", name, split_dir)
            shutil.rmtree(split_dir)

    save_run(
        run_dir=run_dir,
        train_result=train_result,
        train_metrics=train_m,
        train_bars=train_bars,
        train_secondary=train_sec,
        test_result=test_result,
        test_metrics=test_m,
        test_bars=test_bars,
        test_secondary=test_sec,
        cfg=cfg,
        save_plots=getattr(cfg.output, "save_plots", True),
        save_html=getattr(cfg.output, "save_html", True),
        save_csv=getattr(cfg.output, "save_csv", True),
        dpi=getattr(cfg.output, "dpi", 150),
    )

    training_log_path = run_dir / "training_log.json"
    training_log = json.loads(training_log_path.read_text()) if training_log_path.exists() else {}
    training_log.update(
        {
            "test_sharpe": float(test_m.sharpe),
            "test_max_dd": float(test_m.max_drawdown),
            "test_trades": test_m.n_trades,
            "test_return": float(test_m.total_return_pct),
            "test_breaches": test_result.get("n_breach_sessions", 0),
            "test_survived_full_year": bool(test_result.get("survived_full_year", False)),
            "test_fail_time": (
                str(test_result["fail_time"]) if test_result.get("fail_time") is not None else None
            ),
            "eval_checkpoint": ckpt_path.name,
            "eval_timestamp": datetime.now().isoformat(),
            "eval_ftmo": True,
        }
    )
    training_log_path.write_text(json.dumps(training_log, indent=2))

    log.info("Done. Evaluation artifacts written to: %s", run_dir)


if __name__ == "__main__":
    main()
