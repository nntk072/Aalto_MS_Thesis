"""Training CLI parsing and resolved configuration loading."""

from __future__ import annotations

import argparse
import logging
from typing import cast

from omegaconf import DictConfig, OmegaConf

from configuration import apply_overrides, load_config
from configuration.paths import repository_root
from training.configuration.variants import load_variant_config, merge_variant_cfg

log = logging.getLogger(__name__)

_CONFIG_DIR = repository_root(__file__) / "config"
_STRATEGY_CONFIGS = {
    "po3_ifvg": _CONFIG_DIR / "2_po3_ifvg.yaml",
    "distribution": _CONFIG_DIR / "3_distribution.yaml",
}


def parse_train_args() -> argparse.Namespace:
    """Parse CLI arguments for the RL training entrypoint."""
    parser = argparse.ArgumentParser(description="Train PPO/SAC on structure-aware trading.")
    parser.add_argument(
        "--config",
        default=None,
        help="Base config YAML (default: config/base/default.yaml). "
        "Lets this entrypoint load config/features_*_mtf.yaml variant configs.",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seed list (default: single --seed)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant name from config/experiments.yaml (overrides --strategy if set in variant)",
    )
    parser.add_argument(
        "--quality-variant",
        default=None,
        help="New staged variant from config/quality_experiments.yaml",
    )
    parser.add_argument("--mvp", action="store_true", help="MVP mode: first 30 days only")
    parser.add_argument("--force", action="store_true", help="Force data pipeline rerun")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Continue PPO optimizer/policy state from an intact periodic checkpoint",
    )
    parser.add_argument(
        "--allow-legacy-observation-schema",
        action="store_true",
        help="Explicitly resume an unversioned account-state-v1 checkpoint",
    )
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="RL algorithm")
    parser.add_argument(
        "--arch",
        choices=["tcn", "lite_tcn", "gru", "transformer", "mtf"],
        default=None,
        help="Encoder architecture",
    )
    parser.add_argument("--reward", choices=["dsr", "sweep"], default="dsr", help="Reward function")
    parser.add_argument(
        "--strategy",
        choices=["baseline", "po3_ifvg", "distribution"],
        default="baseline",
        help="RL strategy variant (Agent.md: Idea 1 = po3_ifvg, Idea 2 = distribution, Idea 3 = baseline).",
    )
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help="Condition the policy on a frozen VAE narrative latent (requires --vae-path)",
    )
    parser.add_argument(
        "--vae-path",
        default=None,
        help="Path to a trained VAE state_dict (.pth from scripts/train_vae.py)",
    )
    parser.add_argument(
        "--vae-config",
        default="config/vae.yaml",
        help="VAE architecture YAML (must match the checkpoint)",
    )
    parser.add_argument(
        "--allow-locked-oos-for-selection",
        action="store_true",
        help="Rank on locked-OOS Sharpe. Only use if you knowingly tune on the holdout.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log run metrics to Weights & Biases")
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run purged walk-forward validation"
    )
    parser.add_argument("--wf-splits", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--purge-bars", type=int, default=60, help="Purge bars for walk-forward")
    parser.add_argument(
        "--embargo-bars", type=int, default=20, help="Embargo bars for walk-forward"
    )
    parser.add_argument(
        "--wf-steps",
        type=int,
        default=None,
        help="Timesteps per WF fold (default: reuse main timesteps)",
    )
    return parser.parse_args()


def _load_merged_config(args: argparse.Namespace) -> DictConfig:
    """Load base config and merge strategy variant if requested.

    CLI ``overrides`` apply *after* the strategy overlay merge: the Idea 1
    variant pins ``env.reward_mode: pnl``, which used to clobber an explicit
    ``env.reward_mode=rr`` override and silently ran every RR run as PnL.
    """
    cfg = load_config(None, config_path=args.config)
    # Merge the strategy variant config (Idea 1/2) on top of the base config.
    # Baseline (Idea 3) leaves cfg untouched.
    if args.strategy in _STRATEGY_CONFIGS:
        variant_path = _STRATEGY_CONFIGS[args.strategy]
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(variant_path)))
        log.info("Merged strategy config: %s", variant_path)
    elif args.strategy != "baseline":
        raise ValueError(f"unknown strategy: {args.strategy}")

    # Merge variant flags from experiments.yaml (--variant).
    if getattr(args, "variant", None):
        variant_dict = load_variant_config(args.variant)
        variant_strategy = str(variant_dict.get("strategy", args.strategy))
        include_pd = bool(variant_dict.get("include_pd_context", False))
        cfg = merge_variant_cfg(
            cfg,
            strategy=variant_strategy,
            include_pd_context=include_pd,
            variant=variant_dict,
        )
        log.info(
            "Merged variant: %s (strategy=%s include_pd=%s)",
            args.variant,
            variant_strategy,
            include_pd,
        )

    quality_variant = getattr(args, "quality_variant", None)
    if quality_variant:
        if getattr(args, "variant", None) or args.strategy != "baseline":
            raise ValueError(
                "--quality-variant cannot be combined with historical --variant or a "
                "non-baseline --strategy"
            )
        from training.configuration.quality import apply_quality_variant

        cfg = apply_quality_variant(cfg, str(quality_variant))
        log.info("Applied staged RL quality variant: %s", quality_variant)

    return apply_overrides(cfg, args.overrides)
