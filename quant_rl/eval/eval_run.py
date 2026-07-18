"""Evaluate a trained PPO checkpoint and (re)write a run's ``testing/`` artifacts.

Loads an existing RL run's ``config.yaml`` snapshot and a saved model
checkpoint, rolls it through ``TradingEnv`` via
``quant_rl.eval.rollout.evaluate_model``, and overwrites just the
``testing/`` split (equity/trades/metrics/charts) — no retraining, no
re-running the data pipeline's PPO loop.

Use this to:
  * Finish evaluation for a run whose training completed (checkpoints
    exist) but which never produced ``testing/`` artifacts — e.g. because
    evaluation previously used a hardcoded hold-forever stub instead of the
    trained model.
  * Re-evaluate an existing run after a fix to ``TradingEnv``/
    ``evaluate_model`` using an already-trained checkpoint.

Usage
-----
    cd Aalto_MS_Thesis
    uv run python -m quant_rl.eval.eval_run --run outputs/20260719_014725_rl_train_seed42
    uv run python -m quant_rl.eval.eval_run --run outputs/... \\
        --checkpoint model/ppo_ckpt_328350_steps.zip
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
from quant_rl.data.split import get_split_config, split_train_test
from quant_rl.eval.export import save_run
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.rollout import evaluate_model
from quant_rl.features.build import build_features

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
        description="Evaluate a trained PPO checkpoint and rewrite a run's testing/ artifacts."
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
    model = PPO.load(ckpt_path)

    data = run_pipeline(cfg, force=args.force)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{cfg.data.primary}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    train_end, test_start = get_split_config(cfg)
    _, test_bars, _, test_feat = split_train_test(primary_m1, features, train_end, test_start)
    log.info("Evaluating on %d test bars (≥%s)…", len(test_bars), test_start)

    test_result = evaluate_model(
        model,
        bars=test_bars,
        features=test_feat,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        risk_frac_range=(cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        rr_ratio_range=(cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
        dsr_eta=cfg.env.reward_dsr_eta,
    )
    test_result["initial_balance"] = cfg.account.initial_balance
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    log.info(
        "[test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%",
        test_m.sharpe,
        test_m.max_drawdown * 100,
        test_m.total_trades,
        test_m.total_return * 100,
    )

    testing_dir = run_dir / "testing"
    if testing_dir.exists():
        log.info("Clearing stale testing/ artifacts: %s", testing_dir)
        shutil.rmtree(testing_dir)

    save_run(
        run_dir=run_dir,
        test_result=test_result,
        test_metrics=test_m,
        test_bars=test_bars,
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
            "test_trades": test_m.total_trades,
            "test_return": float(test_m.total_return),
            "eval_checkpoint": ckpt_path.name,
            "eval_timestamp": datetime.now().isoformat(),
        }
    )
    training_log_path.write_text(json.dumps(training_log, indent=2))

    log.info("Done. Evaluation artifacts written to: %s", testing_dir)


if __name__ == "__main__":
    main()
