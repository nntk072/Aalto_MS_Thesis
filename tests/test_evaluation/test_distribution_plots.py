"""Export tests for the distributional plots (PLAN 9, WP-E)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from quant_rl.evaluation.plots import plot_pnl_box_by_regime, plot_pnl_histogram


@pytest.fixture()
def out_png(tmp_path: Path) -> Path:
    """Destination PNG path inside a temp directory."""
    return tmp_path / "pnl.png"


@pytest.mark.unit
class TestPnlHistogram:
    def test_exports_file(self, out_png: Path) -> None:
        # Arrange
        pnls = [-120.0, -50.0, -20.0, 10.0, 15.0, 18.0, 25.0, 40.0, 250.0]

        # Act
        result = plot_pnl_histogram(pnls, out_png)

        # Assert
        assert result == str(out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0

    def test_empty_input_still_writes_figure(self, out_png: Path) -> None:
        # Act
        plot_pnl_histogram([], out_png)

        # Assert
        assert out_png.exists()


@pytest.mark.unit
class TestBoxByRegime:
    def test_exports_file_with_groups(self, out_png: Path) -> None:
        # Arrange
        groups = {
            "overall": [100.0, -50.0, 25.0],
            "long": [100.0, 25.0],
            "short": [-50.0],
        }

        # Act
        plot_pnl_box_by_regime(groups, out_png)

        # Assert
        assert out_png.exists()
        assert out_png.stat().st_size > 0

    def test_all_empty_groups_still_writes(self, out_png: Path) -> None:
        # Act
        plot_pnl_box_by_regime({"overall": []}, out_png)

        # Assert
        assert out_png.exists()
