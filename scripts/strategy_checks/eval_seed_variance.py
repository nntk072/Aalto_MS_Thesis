"""Measure how much a run's metrics move when only the RNG seed changes.

``train_rl.py`` evaluates with ``deterministic=False``, so the rollout samples
actions from the policy distribution and every evaluation of the same
checkpoint yields a different trade set. A single reported Sharpe is therefore
one draw, not a property of the policy.

This re-evaluates one checkpoint across N seeds on each split and reports the
spread. It writes no charts and no artifacts: it exists to quantify how much of
a headline number is sampling noise.

Usage (inside a GPU allocation, or anywhere the venv matches the arch):
    python -m scripts.strategy_checks.eval_seed_variance --run outputs/<run> --seeds 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.split import get_split_config, split_bars, split_train_test
from quant_rl.data.ticks import ticks_covering
from quant_rl.eval.rollout import evaluate_model
from quant_rl.evaluation import calculate_metrics
from quant_rl.features.build import FEATURE_CACHE_VERSION, build_features
from quant_rl.models.ppo_policy import ClampedStdMultiInputPolicy
from quant_rl.train.train_rl import (
    _eval_guardrail_kwargs,
    _fill_delay_ms,
    _max_loss_per_trade,
    _strategy_from_cfg,
    _strategy_risk_ranges,
)


def _summarise(values: list[float]) -> dict[str, float]:
    """Mean, spread, and range for one metric across seeds."""
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _eval_kwargs(cfg: Any) -> dict[str, Any]:
    """Env kwargs mirroring train_rl.py's post-training evaluation."""
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    return dict(
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
        # Matches train_rl.py: actions are sampled, so the seed is what varies.
        deterministic=False,
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
        open_manipulation_bars=int(cfg.env.get("open_manipulation_bars", 10)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        peak_trailing_dd_limit=float(cfg.env.get("peak_trailing_dd_limit", 0.0)),
        agent_direction_control=bool(cfg.env.get("agent_direction_control", False)),
        direction_override_threshold=float(cfg.env.get("direction_override_threshold", 0.0)),
        fill_delay_ms=_fill_delay_ms(cfg),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed-variance of run metrics.")
    parser.add_argument("--run", required=True, help="Run directory with model/ppo_final.zip")
    parser.add_argument("--seeds", type=int, default=8, help="Number of seeds to evaluate")
    parser.add_argument("--base-seed", type=int, default=1000, help="First seed, then increments")
    parser.add_argument("--splits", default="testing", help="Comma-separated: testing,training")
    parser.add_argument("--out", default=None, help="Optional JSON path for the full table")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = cast("DictConfig", OmegaConf.load(run_dir / "config.yaml"))
    _ = ClampedStdMultiInputPolicy
    model = PPO.load(run_dir / "model" / "ppo_final.zip")

    data = run_pipeline(cfg)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")
    feat_cache = (
        Path(cfg.data.cache_dir) / f"{cfg.data.primary}_features_{FEATURE_CACHE_VERSION}.parquet"
    )
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start)

    ticks = build_tick_books(cfg).get(str(cfg.data.primary))
    eval_common = _eval_kwargs(cfg)
    results: dict[str, Any] = {}

    for split in args.splits.split(","):
        split = split.strip()
        bars = test_bars if split == "testing" else train_bars
        feat = test_feat if split == "testing" else train_feat
        rows: list[dict[str, float]] = []
        for i in range(args.seeds):
            seed = args.base_seed + i
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = evaluate_model(
                model,
                bars=bars,
                features=feat,
                tickbook=ticks_covering(ticks, bars),
                **eval_common,
            )
            m = calculate_metrics(
                result["equity"],
                trades=result["trades"],
                n_sessions=result.get("n_sessions", 1),
                n_breach_sessions=result.get("n_breach_sessions", 0),
            )
            rows.append(
                {
                    "seed": seed,
                    "sharpe": float(m.sharpe),
                    "total_return_pct": float(m.total_return_pct),
                    "max_drawdown": float(m.max_drawdown),
                    "n_trades": float(m.n_trades),
                }
            )
            print(
                f"[{split}] seed={seed} sharpe={m.sharpe:+.3f} "
                f"return={m.total_return_pct:+.2f}% dd={m.max_drawdown * 100:.2f}% "
                f"trades={m.n_trades}",
                flush=True,
            )
        results[split] = {"rows": rows}

        print(f"\n=== {split}: spread across {len(rows)} seeds ===", flush=True)
        for key, label, scale in (
            ("sharpe", "Sharpe", 1.0),
            ("total_return_pct", "Return %", 1.0),
            ("max_drawdown", "MaxDD %", 100.0),
            ("n_trades", "Trades", 1.0),
        ):
            s = _summarise([r[key] for r in rows])
            print(
                f"  {label:<9} mean={s['mean'] * scale:+.3f} sd={s['stdev'] * scale:.3f} "
                f"min={s['min'] * scale:+.3f} max={s['max'] * scale:+.3f}",
                flush=True,
            )
        sharpes = [r["sharpe"] for r in rows]
        print(
            f"  -> Sharpe spans {min(sharpes):+.3f} to {max(sharpes):+.3f} across "
            f"{len(sharpes)} seeds of the SAME checkpoint.\n",
            flush=True,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
