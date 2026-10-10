"""Run the ablation experiment matrix from config/experiments.yaml.

Trains each variant on the in-sample slice and scores the held-out OOS slice
through the shared evaluation pipeline (``build_run_report``), including any
``pnl_dist_*`` extras already produced by the main metrics stack.

Example:
    python scripts/matrix/ablation_runner.py --bars-csv data/us100_2025.csv \\
        --features-csv data/us100_feat.csv --steps 8192 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_rl.config import load_config  # noqa: E402
from quant_rl.data.split import split_train_test  # noqa: E402
from quant_rl.eval.rollout import make_action_fn  # noqa: E402
from quant_rl.evaluation import build_run_report, run_episode  # noqa: E402
from quant_rl.models.agent import build_agent  # noqa: E402
from quant_rl.train.ablation_utils import merge_variant_cfg  # noqa: E402
from quant_rl.train.train_rl import make_env  # noqa: E402

_AVG_KEYS = (
    "sharpe",
    "sortino",
    "max_drawdown",
    "n_trades",
    "win_rate",
    "total_return_pct",
    "profit_factor",
    "breach_count",
    "breach_rate",
    # Tier 1 entry-state funnel: averaged so a collapsing FSM shows up as
    # zero/near-zero across seeds rather than being hidden in extras.
    "entry_opened",
    "entry_attempts",
    "entry_rejection_rate",
    "candidate_created",
    "candidate_expired",
    "candidate_invalidated",
    "trigger_requested",
    "trigger_refused",
    "entered",
    "closed",
    "armed_bars",
    "arm_to_trigger_rate",
    "trigger_refusal_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True, help="OHLCV CSV with DatetimeIndex")
    parser.add_argument(
        "--features-csv",
        help="Feature CSV aligned to bars; defaults to numeric columns of bars",
    )
    parser.add_argument("--rl-config", default=None, help="Base RL config (default.yaml)")
    parser.add_argument("--experiments", default="config/experiments.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Override per-variant steps")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seed list (default: experiments.yaml defaults.seeds)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset of variant names to run",
    )
    parser.add_argument("--reward", choices=["dsr", "sweep"], default="dsr")
    parser.add_argument("--out-dir", default="results/ablations")
    parser.add_argument(
        "--allow-locked-oos-for-selection",
        action="store_true",
        help=(
            "Acknowledge that locked OOS scores may be used for ranking variants. "
            "Default treats OOS as final_report_only (T-03.2)."
        ),
    )
    return parser.parse_args()


def evaluate_oos(
    model: Any,
    *,
    algo: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: DictConfig,
    reward: str,
) -> dict[str, Any]:
    """Deterministic OOS episode scored via ``build_run_report``.

    Returns the report plus the raw env artifacts (trade log, entry
    diagnostics, FSM lifecycle) so the caller can write per-seed full
    artifacts for debugging without re-running the episode.
    """
    env = make_env(bars, features, cfg, algo=algo, reward=reward, episodic=False)
    metrics = run_episode(
        env,
        action_fn=make_action_fn(model, continuous_actions=algo == "sac"),
    )
    report = build_run_report(metrics, env.trade_log)
    report["_artifacts"] = {
        "trade_log": list(getattr(env, "trade_log", [])),
        "entry_diag": dict(getattr(env, "_entry_diag", {})),
        "entry_state_diagnostics": (
            env.entry_state_diagnostics()
            if callable(getattr(env, "entry_state_diagnostics", None))
            else {}
        ),
    }
    return report


def train_and_score_variant(
    *,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    base_cfg: DictConfig,
    variant: dict[str, Any],
    defaults: dict[str, Any],
    steps: int,
    seed: int,
    reward: str,
    train_end: str,
    test_start: str,
    out_dir: str,
    name: str,
) -> dict[str, Any]:
    """Train one seed of a variant and return the OOS report (+ metadata)."""
    algo = str(variant.get("algo", defaults.get("algo", "ppo")))
    arch = str(variant.get("arch", defaults.get("arch", "gru")))
    use_vae = bool(int(variant.get("use_vae", defaults.get("use_vae", 0))))
    strategy = str(variant.get("strategy", defaults.get("strategy", "baseline")))
    include_pd = bool(variant.get("include_pd_context", defaults.get("include_pd_context", False)))

    if use_vae:
        return {
            "status": "skipped",
            "reason": "use_vae=1 requires a loaded VAE; pass a trained VAE path in a future revision",
            "algo": algo,
            "arch": arch,
            "seed": seed,
        }

    cfg = merge_variant_cfg(
        base_cfg,
        strategy=strategy,
        include_pd_context=include_pd,
        variant=variant,
    )
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        bars, features, train_end, test_start
    )
    if train_bars.empty or test_bars.empty:
        return {
            "status": "skipped",
            "reason": f"empty split train={len(train_bars)} test={len(test_bars)}",
            "seed": seed,
        }

    try:
        env = make_env(train_bars, train_feat, cfg, algo=algo, reward=reward, episodic=True)
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc), "seed": seed}

    model = build_agent(env, cfg, arch=arch, algo=algo, use_vae=False)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    oos = evaluate_oos(
        model,
        algo=algo,
        bars=test_bars,
        features=test_feat,
        cfg=cfg,
        reward=reward,
    )
    artifacts = oos.pop("_artifacts", {})
    action_space = getattr(env, "action_space", None)
    action_space_width = (
        int(action_space.shape[0])
        if action_space is not None and hasattr(action_space, "shape")
        else None
    )
    oos.update(
        {
            "status": "ok",
            "algo": algo,
            "arch": arch,
            "strategy": strategy,
            "include_pd_context": include_pd,
            "seed": seed,
            "steps": steps,
            "action_space_width": action_space_width,
            "observation_account_dim": int(cfg.env.get("account_dim", 0))
            if "account_dim" in cfg.env
            else None,
            "env_flags": {
                key: bool(cfg.env.get(key, False))
                for key in (
                    "strategy_actions",
                    "allow_agent_sl_mode",
                    "allow_agent_tp_mode",
                    "allow_multi_tp",
                    "entry_state_machine",
                    "entry_state_observation",
                    "arm_requires_retest",
                )
            },
        }
    )
    if artifacts:
        art_dir = Path(out_dir) / name / f"seed{seed}"
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "trade_log.json").write_text(
            json.dumps(artifacts["trade_log"], indent=2, default=str)
        )
        (art_dir / "entry_diag.json").write_text(json.dumps(artifacts["entry_diag"], indent=2))
        (art_dir / "entry_state.json").write_text(
            json.dumps(artifacts["entry_state_diagnostics"], indent=2, default=str)
        )
        (art_dir / "config.json").write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, default=str)
        )
    return oos


def _flatten_report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Lift ``extras`` scalar keys (e.g. ``pnl_dist_*``) to the top level."""
    flat = dict(report)
    extras = report.get("extras")
    if isinstance(extras, dict):
        for key, val in extras.items():
            if key not in flat:
                flat[key] = val
    return flat


def _average_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean numeric metrics over successful seed reports."""
    ok = [_flatten_report_metrics(r) for r in seed_reports if r.get("status") == "ok"]
    if not ok:
        return {"status": "all_skipped", "n_ok": 0}
    averaged: dict[str, Any] = {"status": "ok", "n_ok": len(ok)}
    for key in _AVG_KEYS:
        vals = [float(r[key]) for r in ok if key in r and r[key] is not None]
        if vals:
            averaged[key] = round(sum(vals) / len(vals), 4)
    # Preserve distributional extras when present on every ok seed.
    dist_keys = sorted(
        {k for r in ok for k in r if isinstance(k, str) and k.startswith("pnl_dist_")}
    )
    for key in dist_keys:
        vals = [float(r[key]) for r in ok if key in r and isinstance(r[key], (int, float))]
        if len(vals) == len(ok):
            averaged[key] = round(sum(vals) / len(vals), 4)
    return averaged


def load_experiments(path: Path) -> dict[str, Any]:
    """Load the experiments YAML as a plain dict."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {path}")
    return raw


def main() -> None:
    import sys as _sys

    print(
        "WARNING: ablation_runner.py does not use quant_rl.train.variant_resolver and runs "
        "one shared features CSV for every variant. Prefer scripts/train/train_one_variant.sh "
        "for comparable results (see CHANGES.md).",
        file=_sys.stderr,
    )
    """Train and score every selected variant, writing one JSON report each."""
    args = parse_args()
    spec = load_experiments(Path(args.experiments))
    defaults = dict(spec.get("defaults", {}))
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    base_cfg = load_config(config_path=args.rl_config) if args.rl_config else load_config()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    steps = args.steps if args.steps is not None else int(defaults.get("steps", 50_000))
    seeds = list(args.seeds) if args.seeds is not None else list(defaults.get("seeds", [42]))
    train_end = str(defaults.get("train_end", "2025-12-31"))
    test_start = str(defaults.get("test_start", "2026-01-01"))
    oos_role = (
        "selection_acknowledged" if args.allow_locked_oos_for_selection else "final_report_only"
    )
    if oos_role == "final_report_only":
        print(
            "NOTE (T-03.2): locked OOS is final_report_only. "
            "Use walk-forward / validation for architecture selection; "
            "pass --allow-locked-oos-for-selection only if you knowingly rank on OOS."
        )

    variants = list(spec.get("variants", []))
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if str(v.get("name")) in wanted]
        missing = wanted - {str(v.get("name")) for v in variants}
        if missing:
            raise SystemExit(f"unknown variants: {sorted(missing)}")

    for variant in variants:
        name = str(variant["name"])
        print(f"=== {name}: steps={steps} seeds={seeds} ===")
        seed_reports: list[dict[str, Any]] = []
        for seed in seeds:
            report = train_and_score_variant(
                bars=bars,
                features=features,
                base_cfg=base_cfg,
                variant=variant,
                defaults=defaults,
                steps=steps,
                seed=int(seed),
                reward=args.reward,
                train_end=train_end,
                test_start=test_start,
                out_dir=str(out_root),
                name=name,
            )
            seed_reports.append(report)

        aggregated = _average_seed_reports(seed_reports)
        aggregated.update(
            {
                "name": name,
                "n_seeds": len(seeds),
                "steps": steps,
                "oos_role": oos_role,
                "train_end": train_end,
                "test_start": test_start,
                "per_seed": seed_reports,
            }
        )
        out_path = out_root / f"{name}.json"
        out_path.write_text(json.dumps(aggregated, indent=2))
        summary = {k: v for k, v in aggregated.items() if k != "per_seed"}
        print(json.dumps(summary, indent=2))

    print(f"ablation reports saved under {out_root}")


if __name__ == "__main__":
    main()
