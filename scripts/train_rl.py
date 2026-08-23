"""Train a PPO/SAC agent with an optional sequence encoder and VAE context.

Smoke-test example (50k steps):
    python scripts/train_rl.py --algo ppo --arch gru --use-vae 0 \
        --bars-csv data/us100_2025.csv --steps 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import run_episode, sweep_delay_breakdown  # noqa: E402
from quant_rl.models.agent import build_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--features-csv", help="Feature CSV; defaults to numeric bars columns")
    parser.add_argument("--config", default="config/env.yaml")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--arch", default="gru", choices=["tcn", "gru", "transformer"])
    parser.add_argument("--use-vae", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default="models/rl_runs")
    return parser.parse_args()


def main() -> None:
    """Build the agent, train for --steps, evaluate and persist artifacts."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    loaded = OmegaConf.load(args.config)
    cfg = OmegaConf.create(loaded) if not isinstance(loaded, OmegaConf) else loaded
    from omegaconf import DictConfig

    assert isinstance(cfg, DictConfig)

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
            },
        )

    train_env = TradingEnv(bars=bars, features=features, use_sweep_reward=True)
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

    eval_env = TradingEnv(bars=bars, features=features, use_sweep_reward=True)

    def policy(obs: dict[str, Any]) -> tuple[Any, Any]:
        prediction: tuple[Any, Any] = model.predict(obs, deterministic=True)
        return prediction

    def action_fn(obs: dict[str, Any]) -> float:
        return float(policy(obs)[0])

    metrics = run_episode(eval_env, action_fn=action_fn)
    delays = sweep_delay_breakdown(eval_env.trade_log)
    report = {
        "run_name": run_name,
        "sharpe": round(metrics.sharpe, 3),
        "sortino": round(metrics.sortino, 3),
        "calmar": round(metrics.calmar, 3),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "total_return_pct": round(metrics.total_return_pct, 3),
        "breach_count": metrics.breach_count,
        **{k: round(v, 3) for k, v in delays.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    if args.wandb:
        wandb.log(report)
        wandb.finish()

    print(json.dumps(report, indent=2))
    print(f"artifacts saved under {out_dir}")


if __name__ == "__main__":
    main()
