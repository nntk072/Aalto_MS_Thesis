"""The resolver's overrides must reproduce the legacy merge_variant_cfg config.

Requires OmegaConf (repo environment). Expected differences are listed in
EXPECTED_DIFF_KEYS and are the intended fixes, nothing else.
"""

from __future__ import annotations

import pytest

omegaconf = pytest.importorskip("omegaconf")
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from quant_rl.config import apply_overrides, load_config  # noqa: E402
from quant_rl.train.ablation_utils import load_variant_config, merge_variant_cfg  # noqa: E402
from quant_rl.train.variant_resolver import (  # noqa: E402
    DEFAULT_BASE,
    FULL_PO3_BASE,
    PROJECT_ROOT,
    load_experiments,
    resolve_variant,
)

STEPS = 100_000
# Launcher overrides the old script applied for full-PO3 variants (ftmo trailing
# line removed in the fix: it equals the default, so it changes nothing).
LEGACY_FULL_PO3 = [
    "features.include_session_ohlc=true",
    "features.liquidity.enabled=true",
    "features.po3_state_mtf.enabled=true",
    "features.ifvg_mtf.enabled=true",
    "features.include_strategy_state=true",
    "env.open_manipulation_bars=0",
    "env.peak_trailing_dd_limit=0",
    "ftmo.trailing_dd_limit=0.07",
]
STRAT_YAML = {
    "po3_ifvg": "config/2_po3_ifvg.yaml",
    "distribution": "config/3_distribution.yaml",
}
# Intentional: baseline-named rungs now run BaselineStrategy, not the default po3 block.
EXPECTED_DIFF_KEYS = {"strategy.name"}


def _flatten(cfg) -> dict[str, object]:
    container = OmegaConf.to_container(cfg, resolve=True)
    out: dict[str, object] = {}

    def walk(node: object, prefix: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        else:
            out[prefix[:-1]] = node

    walk(container)
    return out


def _legacy(name: str):
    variant = load_variant_config(name)
    strategy = str(variant.get("strategy", "baseline"))
    sa = bool(variant.get("strategy_actions", False))
    base = FULL_PO3_BASE if (strategy != "baseline" or sa) else DEFAULT_BASE
    cfg = load_config(None, config_path=PROJECT_ROOT / base)
    cfg = merge_variant_cfg(
        cfg,
        strategy=strategy,
        include_pd_context=bool(variant.get("include_pd_context", False)),
        variant=variant,
    )
    overrides = (LEGACY_FULL_PO3 if (strategy != "baseline" or sa) else []) + [
        f"ppo.total_timesteps={STEPS}"
    ]
    return apply_overrides(cfg, overrides)


def _new(name: str):
    resolved = resolve_variant(name, steps=STEPS, vae_path="placeholder.pth")
    cfg: DictConfig = load_config(None, config_path=PROJECT_ROOT / resolved["base_config"])
    if resolved["strategy"] in STRAT_YAML:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(PROJECT_ROOT / STRAT_YAML[resolved["strategy"]]))  # type: ignore[assignment]
    return apply_overrides(cfg, resolved["overrides"])


@pytest.mark.parametrize("name", [str(v["name"]) for v in load_experiments()["variants"]])
def test_resolver_matches_legacy_config(name: str) -> None:
    legacy = _flatten(_legacy(name))
    new = _flatten(_new(name))
    diff = {k for k in set(legacy) | set(new) if legacy.get(k) != new.get(k)}
    unexpected = diff - EXPECTED_DIFF_KEYS
    assert not unexpected, f"{name}: unexpected config drift: " + ", ".join(
        f"{k} legacy={legacy.get(k)!r} new={new.get(k)!r}" for k in sorted(unexpected)
    )
    if diff:
        # Only the baseline-named rungs and controls may differ, and only in strategy.name.
        assert diff == EXPECTED_DIFF_KEYS
        assert resolve_variant(name, vae_path="placeholder.pth")["strategy"] == "baseline"
