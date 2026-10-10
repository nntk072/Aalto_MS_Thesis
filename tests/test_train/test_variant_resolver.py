"""Tests for quant_rl.train.variant_resolver and the variant launcher.

Plain Python (PyYAML only). Runs under pytest, or directly: python -m tests.test_train.test_variant_resolver
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from quant_rl.train.variant_resolver import (
    PROJECT_ROOT,
    VariantError,
    consistency_table,
    load_experiments,
    resolve_variant,
    train_args,
)

LAUNCHER = PROJECT_ROOT / "scripts" / "train" / "train_one_variant.sh"


def _all_names() -> list[str]:
    return [str(v["name"]) for v in load_experiments()["variants"]]


def _write_experiments(data: dict[str, object]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "experiments.yaml"
    tmp.write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp


def _overrides_by_key(resolved: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in resolved["overrides"]:
        key, value = item.split("=", 1)
        assert key not in out, f"duplicate override {key}"
        out[key] = value
    return out


def test_every_variant_resolves_and_declares_reward_mode() -> None:
    names = _all_names()
    assert len(names) == len(set(names)) == 29
    for name in names:
        resolved = resolve_variant(
            name, vae_path="placeholder.pth"
        )  # only the VAE variant needs it
        assert resolved["reward_mode"] in ("dsr", "pnl", "rr")
        assert resolved["steps"] == 100_000
        assert resolved["seeds"] == [42, 43, 44, 45, 46]


def test_baseline_ladder_rungs_pin_baseline_strategy_name() -> None:
    # Before the fix these rungs silently ran the default po3_ifvg strategy.
    for name in ("ladder_a0_flat", "ladder_a1_sl", "ladder_a4b_entry_fsm"):
        resolved = resolve_variant(name)
        assert resolved["strategy"] == "baseline"
        assert _overrides_by_key(resolved)["strategy.name"] == "baseline"


def test_strategy_rungs_use_strategy_flag_not_name_override() -> None:
    resolved = resolve_variant("ladder_a0_flat_dist")
    assert resolved["strategy"] == "distribution"
    assert "strategy.name" not in _overrides_by_key(resolved)
    args = train_args(resolved, 42)
    assert args[args.index("--strategy") + 1] == "distribution"


def test_controls_run_declared_algo_and_arch() -> None:
    sac = resolve_variant("ablation_sac_agent")
    assert (sac["algo"], sac["arch"]) == ("sac", "gru")
    transformer = resolve_variant("ablation_transformer_encoder")
    assert transformer["arch"] == "transformer"
    args = train_args(sac, 42)
    assert args[args.index("--algo") + 1] == "sac"
    assert args[args.index("--arch") + 1] == "gru"


def test_vae_variant_requires_vae_path() -> None:
    try:
        resolve_variant("ablation_conditional_narrative")
    except VariantError as exc:
        assert "VAE" in str(exc)
    else:
        raise AssertionError("use_vae=1 without a VAE path must fail")
    resolved = resolve_variant("ablation_conditional_narrative", vae_path="/x/vae.pth")
    args = train_args(resolved, 42)
    assert "--use-vae" in args and args[args.index("--vae-path") + 1] == "/x/vae.pth"


def test_reward_mode_must_be_declared() -> None:
    data = load_experiments()
    data["variants"] = [{"name": "no_reward", "arch": "tcn"}]
    path = _write_experiments(data)
    try:
        resolve_variant("no_reward", experiments_path=path)
    except VariantError as exc:
        assert "reward_mode" in str(exc)
    else:
        raise AssertionError("missing reward_mode must fail")


def test_steps_and_seeds_overrides_flow_into_args() -> None:
    resolved = resolve_variant("ladder_a2_tp", steps=8192, seeds=[7])
    assert _overrides_by_key(resolved)["ppo.total_timesteps"] == "8192"
    assert resolved["seeds"] == [7]
    args = train_args(resolved, 7)
    assert args[args.index("--seed") + 1] == "7"
    assert "--seeds" not in args


def test_overrides_come_after_strategy_and_have_no_duplicates() -> None:
    resolved = resolve_variant("ladder_a4b_po3_ifvg")
    keys = _overrides_by_key(resolved)  # asserts uniqueness
    assert keys["env.reward_mode"] == "pnl"
    assert keys["features.include_pd_context"] == "true"
    assert "ftmo.trailing_dd_limit" not in keys  # was a no-op at default 0.07


def test_consistency_table_covers_all_variants() -> None:
    rows = consistency_table(_all_names())
    assert len(rows) == 29
    by_name = {r["variant"]: r for r in rows}
    # The confound this module makes visible: baseline-named rungs and PO3 rungs
    # differ in reward mode and PD context.
    assert by_name["ladder_a0_flat"]["reward_mode"] == "dsr"
    assert by_name["ladder_a0_flat_po3"]["reward_mode"] == "pnl"
    assert by_name["ladder_a0_flat"]["include_pd_context"] is False
    assert by_name["ladder_a0_flat_po3"]["include_pd_context"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher requires Linux/WSL")
def test_launcher_dry_run_prints_every_seed() -> None:
    out_dir = tempfile.mkdtemp()
    env = {
        "PATH": "/usr/bin:/bin",
        "VARIANT": "ablation_sac_agent",
        "DRY_RUN": "1",
        "OUT_DIR": out_dir,
        "PYTHON": sys.executable,
    }
    proc = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "algo=sac arch=gru" in proc.stdout
    for seed in (42, 43, 44, 45, 46):
        assert f"ablation_sac_agent__seed{seed}" in proc.stdout
    assert "--algo sac" in proc.stdout and "--arch gru" in proc.stdout
    assert not any(Path(out_dir).glob("*.done"))  # dry run must not write markers


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher requires Linux/WSL")
def test_launcher_refuses_vae_variant_without_vae_path() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "VARIANT": "ablation_conditional_narrative",
        "DRY_RUN": "1",
        "OUT_DIR": tempfile.mkdtemp(),
        "PYTHON": sys.executable,
    }
    proc = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    assert proc.returncode != 0
    assert "VAE" in proc.stderr


def test_manifest_cli_writes_resolved_config() -> None:
    import json
    import os

    out = Path(tempfile.mkdtemp()) / "resolved_config.json"
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_rl.train.variant_resolver",
            "manifest",
            "ladder_a1_sl_dist",
            "--seed",
            "43",
            "--out",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        timeout=120,
    )
    manifest = json.loads(out.read_text())
    assert manifest["variant"] == "ladder_a1_sl_dist"
    assert manifest["seed"] == 43
    assert manifest["strategy"] == "distribution"
    assert "--seed" in manifest["train_args"] and "43" in manifest["train_args"]


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
