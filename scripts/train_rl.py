"""DEPRECATED — Train a PPO/SAC agent with an optional sequence encoder and VAE context.

Use ``python -m quant_rl.train.train_rl`` instead (the canonical entrypoint;
same features plus ``--mvp``/``--force``/``--out`` and typo-safe config
overrides). This script is kept for the wandb sweep preview / deprecation
transition and will be deleted once those consumers move.

Training happens on a chronological in-sample slice (≤ --train-end) and all
reported metrics are split into an in-sample block (for diagnosing
overfitting) and an out-of-sample block (the numbers that matter) computed
on the held-out slice (≥ --test-start).  Use --walk-forward for a
purged/embargoed walk-forward estimate instead of a single split.

Smoke-test example (50k steps):
    python scripts/train_rl.py --algo ppo --arch gru --use-vae 0 \
        --bars-csv data/us100_2025.csv --steps 50000 --seed 42

Walk-forward example:
    python scripts/train_rl.py --bars-csv data/us100_2025.csv \
        --steps 50000 --walk-forward --wf-splits 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from quant_rl.data.split import split_train_test  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.eval.rollout import make_action_fn  # noqa: E402
from quant_rl.evaluation import (  # noqa: E402
    build_run_report,
    purged_walk_forward,
    run_episode,
)
from quant_rl.models.agent import build_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--features-csv", help="Feature CSV; defaults to numeric bars columns")
    parser.add_argument("--config", default="quant_rl/config/default.yaml")
    parser.add_argument(
        "--features-config",
        default=None,
        help="Variant features YAML (e.g. config/features_po3_mtf.yaml); its 'features' block is merged over the main config",
    )
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--arch", default="gru", choices=["tcn", "gru", "transformer"])
    parser.add_argument("--use-vae", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default="models/rl_runs")
    parser.add_argument("--train-end", default="2025-12-31", help="Last in-sample date")
    parser.add_argument("--test-start", default="2026-01-01", help="First out-of-sample date")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run purged/embargoed walk-forward over the full sample",
    )
    parser.add_argument("--wf-splits", type=int, default=3, help="Walk-forward folds")
    parser.add_argument("--purge-bars", type=int, default=60, help="Purge gap at fold boundary")
    parser.add_argument("--embargo-bars", type=int, default=20, help="Embargo gap at fold boundary")
    parser.add_argument(
        "--wf-steps", type=int, default=None, help="Steps per walk-forward fold (default: --steps)"
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load the YAML config and apply CLI overrides."""
    loaded = OmegaConf.load(args.config)
    # Chain B variant support: merge a variant features block over the base.
    if getattr(args, "features_config", None):
        variant = OmegaConf.load(args.features_config)
        loaded = OmegaConf.merge(loaded, variant)
    overrides = {
        # NOTE: default.yaml spells the key ``learning_rate``, not ``lr``.
        # Omegaconf.from_dotlist silently *creates* a stray ``ppo.lr`` key
        # otherwise and the sweep would train at the default LR forever.
        f"{args.algo}.learning_rate": args.lr,
        f"{args.algo}.ent_coef": args.entropy_coef,
        f"{args.algo}.batch_size": args.batch_size,
    }
    merged = OmegaConf.merge(
        loaded, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items() if v is not None])
    )
    return OmegaConf.create(merged)


def evaluate_split(
    model: Any,
    algo: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict[str, Any], TradingEnv]:
    """Run one deterministic episode of *model* over *bars* and score it."""
    env = TradingEnv(
        bars=bars,
        features=features,
        use_sweep_reward=True,
        continuous_actions=algo == "sac",
        # Eval mode: a guardrail breach blocks trading for the rest of the
        # session instead of truncating the episode, mirroring run_backtest.
        episodic=False,
    )
    metrics = run_episode(env, action_fn=make_action_fn(model, continuous_actions=algo == "sac"))
    return build_run_report(metrics, env.trade_log), env


def run_walk_forward(
    args: argparse.Namespace,
    cfg: DictConfig,
    bars: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Train + evaluate per purged/embargoed fold and aggregate the results."""
    wf_steps = args.wf_steps if args.wf_steps is not None else args.steps
    folds: list[dict[str, Any]] = []
    for split in purged_walk_forward(
        len(bars),
        n_splits=args.wf_splits,
        test_size=0.2,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    ):
        print(f"=== walk-forward fold {split.fold} ===")
        fold_model = build_agent(
            TradingEnv(
                bars=bars.iloc[split.train_idx],
                features=features.iloc[split.train_idx],
                use_sweep_reward=True,
                continuous_actions=args.algo == "sac",
            ),
            cfg,
            arch=args.arch,
            algo=args.algo,
            use_vae=bool(args.use_vae),
        )
        fold_model.set_random_seed(args.seed + split.fold)
        fold_model.learn(total_timesteps=wf_steps, progress_bar=False)
        report, _ = evaluate_split(
            fold_model,
            args.algo,
            bars.iloc[split.test_idx],
            features.iloc[split.test_idx],
        )
        folds.append(
            {
                "fold": split.fold,
                "train_bars": int(len(split.train_idx)),
                "test_bars": int(len(split.test_idx)),
                **report,
            }
        )

    sharpes = [f["sharpe"] for f in folds]
    drawdowns = [f["max_drawdown"] for f in folds]
    return {
        "folds": folds,
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "max_drawdown_mean": float(np.mean(drawdowns)),
    }


def main() -> None:
    """Build the agent, train for --steps, evaluate and persist artifacts."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    cfg = load_config(args)

    # Seed torch's global RNG *before* model construction — network init /
    # dropout draw from it and SB3 only seeds its own generator otherwise.
    torch.manual_seed(args.seed)

    train_bars, test_bars, train_features, test_features = split_train_test(
        bars, features, args.train_end, args.test_start
    )
    if train_bars.empty or test_bars.empty:
        raise SystemExit(
            f"empty split with --train-end {args.train_end} / --test-start {args.test_start}: "
            f"train={len(train_bars)} bars, test={len(test_bars)} bars"
        )
    print(f"split: train={len(train_bars)} bars, test={len(test_bars)} bars (held-out)")

    run_name = args.run_name or f"{args.algo}_{args.arch}_vae{args.use_vae}_seed{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        import wandb  # type: ignore[import-not-found]

        wandb.init(
            project="aalto-liquidity-sweep",
            name=run_name,
            config={
                "algo": args.algo,
                "arch": args.arch,
                "use_vae": args.use_vae,
                "steps": args.steps,
                "seed": args.seed,
                "train_end": args.train_end,
                "test_start": args.test_start,
            },
        )

    train_env = TradingEnv(
        bars=train_bars,
        features=train_features,
        use_sweep_reward=True,
        continuous_actions=args.algo == "sac",
    )
    model = build_agent(
        train_env,
        cfg,
        arch=args.arch,
        algo=args.algo,
        use_vae=bool(args.use_vae),
    )
    model.set_random_seed(args.seed)
    model.learn(total_timesteps=args.steps, progress_bar=False)

    model_path = out_dir / "model"
    model.save(model_path)

    in_sample, _ = evaluate_split(model, args.algo, train_bars, train_features)
    out_of_sample, _ = evaluate_split(model, args.algo, test_bars, test_features)

    report: dict[str, Any] = {
        "run_name": run_name,
        "split": {
            "train_end": args.train_end,
            "test_start": args.test_start,
            "train_bars": int(len(train_bars)),
            "test_bars": int(len(test_bars)),
        },
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
    }

    if args.walk_forward:
        report["walk_forward"] = run_walk_forward(args, cfg, bars, features)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    if args.wandb:
        wandb.log({"ins_sharpe": in_sample["sharpe"], "oos_sharpe": out_of_sample["sharpe"]})
        wandb.finish()

    print(json.dumps(report, indent=2))
    print(f"artifacts saved under {out_dir}")


if __name__ == "__main__":
    main()
