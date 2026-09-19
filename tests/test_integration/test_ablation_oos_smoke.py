"""Fast smoke tests for ablation YAML, report aggregator, and OOS cost helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.ablation_runner import _average_seed_reports, load_experiments, merge_variant_cfg
from scripts.report_ablations import format_table, load_ablation_reports
from scripts.test_oos import make_cost_model

from quant_rl.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "config" / "experiments.yaml"


@pytest.mark.integration
class TestExperimentsYaml:
    def test_experiments_yaml_loads_and_has_pd_variants(self) -> None:
        spec = load_experiments(EXPERIMENTS)
        assert "defaults" in spec and "variants" in spec
        names = {str(v["name"]) for v in spec["variants"]}
        assert "ablation_baseline_unconditional" in names
        assert "ablation_pd_context" in names
        assert "ablation_po3_ifvg" in names
        # Defaults must not resurrect the skipped parallel validation package.
        raw = yaml.safe_load(EXPERIMENTS.read_text())
        assert "quant_rl.validation" not in json.dumps(raw)


@pytest.mark.integration
class TestAblationHelpers:
    def test_merge_variant_cfg_pd_and_strategy(self) -> None:
        base = load_config()
        cfg = merge_variant_cfg(base, strategy="po3_ifvg", include_pd_context=True)
        assert bool(cfg.features.include_pd_context) is True
        assert bool(cfg.env.strategy_actions) is True
        assert int(cfg.env.n_envs) == 1

    def test_average_seed_reports_means_and_skips(self) -> None:
        seeds: list[dict[str, Any]] = [
            {
                "status": "ok",
                "sharpe": 1.0,
                "max_drawdown": 0.1,
                "n_trades": 2,
                "pnl_dist_mean": 1.0,
            },
            {
                "status": "ok",
                "sharpe": 3.0,
                "max_drawdown": 0.3,
                "n_trades": 4,
                "pnl_dist_mean": 3.0,
            },
            {"status": "skipped", "reason": "vae"},
        ]
        avg = _average_seed_reports(seeds)
        assert avg["status"] == "ok"
        assert avg["n_ok"] == 2
        assert avg["sharpe"] == 2.0
        assert avg["pnl_dist_mean"] == 2.0


@pytest.mark.integration
class TestReportAblations:
    def test_load_and_format_table(self, tmp_path: Path) -> None:
        sample = {
            "name": "ablation_baseline_unconditional",
            "status": "ok",
            "n_seeds": 1,
            "n_ok": 1,
            "steps": 64,
            "sharpe": 0.5,
            "max_drawdown": 0.01,
            "n_trades": 2,
            "win_rate": 0.5,
            "pnl_dist_cvar_95": -0.1,
        }
        (tmp_path / "ablation_baseline_unconditional.json").write_text(json.dumps(sample))
        rows = load_ablation_reports(tmp_path)
        assert len(rows) == 1
        table = format_table(rows)
        assert "ablation_baseline_unconditional" in table
        assert "pnl_dist_cvar_95" in table


@pytest.mark.integration
class TestOosCostHelpers:
    def test_make_cost_model_widens_spread(self) -> None:
        cm = make_cost_model(0.6, 0.2)
        assert cm.spread_points == pytest.approx(0.8)
        assert cm.slippage_points == pytest.approx(0.2)
