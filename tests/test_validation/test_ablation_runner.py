"""Unit tests for the ablation experiment runner (PLAN 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import scripts.ablation_runner as ablation


def _fake_scorer(bars, features, rl_cfg, **kwargs):  # noqa: ANN001, ANN003
    """Deterministic stand-in for train_and_score."""
    seed = kwargs["seed"]
    sharpe = 0.5 + 0.1 * seed
    return (
        {
            "sharpe": sharpe,
            "sortino": sharpe * 2,
            "max_drawdown": 0.05,
            "breach_count": 0,
        },
        [],
        None,
    )


@pytest.fixture()
def bars() -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=50, freq="15min")
    return pd.DataFrame({"close": range(50)}, index=idx)


@pytest.mark.unit
class TestRunExperiments:
    def test_writes_one_report_per_variant(self, bars: pd.DataFrame, tmp_path: Path) -> None:
        # Arrange
        spec = {
            "defaults": {"steps": 1000, "seeds": [42, 43], "algo": "ppo"},
            "variants": [{"name": "v_base"}, {"name": "v_vae", "use_vae": 1}],
        }

        # Act
        reports = ablation.run_experiments(
            bars,
            bars,
            rl_cfg={},
            spec=spec,
            steps=10,
            seeds=[42],
            out_dir=str(tmp_path),
            scorer=_fake_scorer,
        )

        # Assert
        assert len(reports) == 2
        assert (tmp_path / "v_base.json").exists()
        assert (tmp_path / "v_vae.json").exists()

    def test_aggregates_across_seeds(self, bars: pd.DataFrame, tmp_path: Path) -> None:
        # Arrange
        spec = {"variants": [{"name": "v", "algo": "sac"}]}

        # Act
        reports = ablation.run_experiments(
            bars,
            bars,
            rl_cfg={},
            spec=spec,
            steps=10,
            seeds=[42, 43],
            out_dir=str(tmp_path),
            scorer=_fake_scorer,
        )

        # Assert — mean of sharpe(42)=0.5+4.2 and sharpe(43)=0.5+4.3
        assert reports[0]["sharpe"] == pytest.approx((0.5 + 4.2 + 0.5 + 4.3) / 2, abs=1e-3)
        assert reports[0]["n_seeds"] == 2
        assert len(reports[0]["per_seed"]) == 2

    def test_report_file_contents_match_returned(self, bars: pd.DataFrame, tmp_path: Path) -> None:
        # Arrange
        spec = {"variants": [{"name": "v"}]}

        # Act
        reports = ablation.run_experiments(
            bars,
            bars,
            rl_cfg={},
            spec=spec,
            steps=10,
            seeds=[1],
            out_dir=str(tmp_path),
            scorer=_fake_scorer,
        )

        # Assert
        saved = json.loads((tmp_path / "v.json").read_text())
        assert saved == reports[0]
