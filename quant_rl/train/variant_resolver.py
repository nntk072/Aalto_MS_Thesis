"""Single source of truth for ablation variants (config/experiments.yaml).

The launcher used to hard-code architecture, algorithm, feature profile and
strategy, and passed ``--variant`` to ``train_rl`` so the variant was merged a
second time with different semantics. This module resolves a variant once into
the exact ``train_rl`` arguments and config overrides, and writes a manifest
that records them.

It is deliberately plain Python + PyYAML (no OmegaConf, no torch) so it can run
in the launcher before the training environment is activated.

Override semantics mirror ``quant_rl.train.ablation_utils.merge_variant_cfg``:
  * ladder flags -> ``env.<flag>``
  * ``include_pd_context`` -> ``features.include_pd_context`` + structure enabled
  * strategy (po3_ifvg / distribution) -> ``--strategy`` (train_rl merges the
    strategy YAML); ``baseline`` -> ``strategy.name=baseline`` so that the
    default ``po3_ifvg`` block in default.yaml cannot leak into a baseline rung.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.yaml"
DEFAULT_BASE = "quant_rl/config/default.yaml"
FULL_PO3_BASE = "config/features/full_po3_mtf.yaml"

STRATEGIES = ("baseline", "po3_ifvg", "distribution")
ALGOS = ("ppo", "sac")
ARCHS = ("tcn", "gru", "transformer", "mtf")
REWARD_MODES = ("dsr", "pnl", "rr")

LADDER_FLAGS: tuple[str, ...] = (
    "entry_state_machine",
    "entry_state_observation",
    "allow_agent_sl_mode",
    "allow_agent_tp_mode",
    "allow_multi_tp",
    "agent_direction_control",
    "strategy_actions",
    "arm_requires_retest",
    "candidate_max_age_bars",
)

# Feature/risk regime applied to every variant that uses strategy actions or a
# strategy (the full PO3 feature stack). Same keys the old launcher applied,
# minus the ftmo.trailing_dd_limit=0.07 line, which equals the default.
FULL_PO3_OVERRIDES: tuple[str, ...] = (
    "features.include_session_ohlc=true",
    "features.liquidity.enabled=true",
    "features.po3_state_mtf.enabled=true",
    "features.ifvg_mtf.enabled=true",
    "features.include_strategy_state=true",
    "env.open_manipulation_bars=0",
    "env.peak_trailing_dd_limit=0",
)

PD_CONTEXT_OVERRIDES: tuple[str, ...] = (
    "features.include_pd_context=true",
    "features.structure.enabled=true",
)


class VariantError(ValueError):
    """Raised when a variant cannot be resolved into a runnable configuration."""


def load_experiments(path: Path | str | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path or EXPERIMENTS_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("variants"), list):
        raise VariantError("experiments.yaml must contain a 'variants' list")
    return raw


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_variant(
    name: str,
    *,
    experiments_path: Path | str | None = None,
    steps: int | None = None,
    seeds: list[int] | None = None,
    vae_path: str | None = None,
) -> dict[str, Any]:
    """Resolve one variant into a fully explicit, validated configuration."""
    exp = load_experiments(experiments_path)
    defaults = dict(exp.get("defaults") or {})
    matches = [v for v in exp["variants"] if str(v.get("name")) == name]
    if len(matches) != 1:
        raise VariantError(f"variant {name!r} must be defined exactly once (found {len(matches)})")
    declared = {**defaults, **matches[0]}

    if "reward_mode" not in matches[0]:
        raise VariantError(f"{name}: reward_mode must be declared explicitly on the variant")
    reward_mode = str(matches[0]["reward_mode"])
    if reward_mode not in REWARD_MODES:
        raise VariantError(f"{name}: reward_mode {reward_mode!r} not in {REWARD_MODES}")

    strategy = str(declared.get("strategy", "baseline"))
    if strategy not in STRATEGIES:
        raise VariantError(f"{name}: strategy {strategy!r} not in {STRATEGIES}")

    algo = str(declared.get("algo", "ppo"))
    arch = str(declared.get("arch", "gru"))
    if algo not in ALGOS:
        raise VariantError(f"{name}: algo {algo!r} not in {ALGOS}")
    if arch not in ARCHS:
        raise VariantError(f"{name}: arch {arch!r} not in {ARCHS}")

    use_vae = bool(int(declared.get("use_vae", 0)))
    if use_vae and not vae_path:
        raise VariantError(
            f"{name}: use_vae=1 needs a trained VAE; set VAE_PATH to a .pth file "
            "(the old launcher silently ran this variant without a VAE)"
        )

    strategy_actions = bool(
        matches[0].get("strategy_actions", declared.get("strategy_actions", False))
    )
    include_pd = bool(declared.get("include_pd_context", False))
    full_po3 = strategy != "baseline" or strategy_actions
    base_config = FULL_PO3_BASE if full_po3 else DEFAULT_BASE
    if not (PROJECT_ROOT / base_config).is_file():
        raise VariantError(f"{name}: base config missing: {base_config}")

    resolved_steps = int(steps if steps is not None else declared.get("steps", 100_000))
    resolved_seeds = list(seeds if seeds is not None else declared.get("seeds", [42]))
    if resolved_steps <= 0:
        raise VariantError(f"{name}: steps must be positive, got {resolved_steps}")
    if not resolved_seeds:
        raise VariantError(f"{name}: at least one seed is required")

    overrides: list[str] = []
    if full_po3:
        overrides.extend(FULL_PO3_OVERRIDES)
    if include_pd:
        overrides.extend(PD_CONTEXT_OVERRIDES)
    for key in LADDER_FLAGS:
        if key in matches[0]:
            overrides.append(f"env.{key}={_fmt(matches[0][key])}")
    # strategy_actions implies full_po3, which already sets include_strategy_state.
    if strategy == "baseline":
        overrides.append("strategy.name=baseline")
    overrides.append(f"env.reward_mode={reward_mode}")
    overrides.append(f"ppo.total_timesteps={resolved_steps}")  # train_rl uses this for SAC too

    return {
        "variant": name,
        "algo": algo,
        "arch": arch,
        "use_vae": use_vae,
        "vae_path": vae_path if use_vae else None,
        "strategy": strategy,
        "strategy_actions": strategy_actions,
        "reward_mode": reward_mode,
        "include_pd_context": include_pd,
        "feature_profile": "full_po3" if full_po3 else "default",
        "base_config": base_config,
        "steps": resolved_steps,
        "seeds": [int(s) for s in resolved_seeds],
        "overrides": overrides,
    }


def train_args(resolved: dict[str, Any], seed: int) -> list[str]:
    """Arguments for ``python -m quant_rl.train.train_rl`` for one seed.

    ``--out`` is added by the launcher. Overrides come last because train_rl
    applies positional ``key=value`` overrides after every config merge.
    """
    args = [
        "--config",
        resolved["base_config"],
        "--strategy",
        resolved["strategy"],
        "--algo",
        resolved["algo"],
        "--arch",
        resolved["arch"],
        "--seed",
        str(seed),
        "--seeds",
        str(seed),
    ]
    if resolved["use_vae"]:
        args += ["--use-vae", "--vae-path", str(resolved["vae_path"])]
    return args + list(resolved["overrides"])


def consistency_table(names: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Per-variant summary of the settings that drive cross-variant comparisons."""
    rows = []
    for name in names:
        kw = dict(kwargs)
        if kw.get("vae_path") is None:
            kw["vae_path"] = "UNSET"  # table is a dry view; VAE presence is not checked here
        r = resolve_variant(name, **kw)
        rows.append(
            {
                k: r[k]
                for k in (
                    "variant",
                    "strategy",
                    "strategy_actions",
                    "reward_mode",
                    "include_pd_context",
                    "feature_profile",
                    "algo",
                    "arch",
                    "use_vae",
                    "steps",
                )
            }
        )
    return rows


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("meta", "args", "manifest", "table"):
        p = sub.add_parser(cmd)
        if cmd != "table":
            p.add_argument("variant")
        p.add_argument("--steps", type=int, default=None)
        p.add_argument("--seeds", default=None, help="comma or space separated")
        p.add_argument("--vae-path", default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--out", type=Path, default=None)
    ns = parser.parse_args(argv)

    seeds = None
    if ns.seeds:
        seeds = [int(s) for s in ns.seeds.replace(",", " ").split()]
    kwargs = {"steps": ns.steps, "seeds": seeds, "vae_path": ns.vae_path}

    if ns.cmd == "table":
        names = [str(v["name"]) for v in load_experiments()["variants"]]
        rows = consistency_table(names, steps=ns.steps, vae_path=ns.vae_path)
        keys = list(rows[0].keys())
        print(" | ".join(keys))
        print("|".join("---" for _ in keys))
        for row in rows:
            print(" | ".join(str(row[k]) for k in keys))
        return 0

    resolved = resolve_variant(ns.variant, **kwargs)
    if ns.cmd == "meta":
        print(f"ALGO={resolved['algo']}")
        print(f"ARCH={resolved['arch']}")
        print(f"USE_VAE={int(resolved['use_vae'])}")
        print(f"STEPS={resolved['steps']}")
        print(f"SEEDS={' '.join(str(s) for s in resolved['seeds'])}")
        print(f"STRATEGY={resolved['strategy']}")
        print(f"FEATURE_PROFILE={resolved['feature_profile']}")
        print(f"REWARD_MODE={resolved['reward_mode']}")
        return 0
    if ns.cmd == "args":
        if ns.seed is None:
            raise SystemExit("args requires --seed")
        for item in train_args(resolved, ns.seed):
            print(item)
        return 0
    if ns.cmd == "manifest":
        if ns.seed is None or ns.out is None:
            raise SystemExit("manifest requires --seed and --out")
        manifest = dict(resolved, seed=ns.seed, train_args=train_args(resolved, ns.seed))
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_main())
