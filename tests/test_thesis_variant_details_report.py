from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts.research.thesis.evaluation.report_variant_details import (
    _annotate_trade_outcome,
    _flatten_markdown_rows,
    _report_value,
    _trade_outcome_caption,
    _validate_markdown_images,
    _validate_report_figures,
)


@pytest.mark.parametrize(
    ("value", "field", "expected"),
    [
        (0.101815, "max_drawdown", "10.18%"),
        (0.1, "peak_trailing_dd_limit", "10.00%"),
        (12.571, "total_return_pct", "12.57%"),
        (0.0003, "learning_rate", "0.0003"),
        (250.0, "net_pnl", "250.000"),
    ],
)
def test_report_value_formats_percentages_with_units(
    value: float, field: str, expected: str
) -> None:
    assert _report_value(value, field) == expected


def test_nested_parameters_render_as_readable_rows_without_raw_objects() -> None:
    rows = _flatten_markdown_rows(
        {
            "policy": {
                "learning_rate": 0.0003,
                "risk_frac_range": [0.0, 0.01],
                "layers": [64, 32],
                "enabled": True,
            }
        }
    )

    assert rows == [
        ("policy / learning rate", "0.0003"),
        ("policy / risk frac range", "0.00%, 1.00%"),
        ("policy / layers", "64, 32"),
        ("policy / enabled", "Yes"),
    ]
    assert "{" not in str(rows)
    assert "}" not in str(rows)


@pytest.mark.parametrize(
    ("category", "split", "rank", "trade", "expected"),
    [
        (
            "winners",
            "training",
            2,
            {"net_pnl": 125.5},
            "TYPICAL WINNER | TRAINING | SAMPLE 2 | NET PNL +125.50 account units",
        ),
        (
            "losers",
            "testing",
            1,
            {"net_pnl": -42.0},
            "TYPICAL LOSER | TESTING | SAMPLE 1 | NET PNL -42.00 account units",
        ),
        (
            "breakeven",
            "training",
            1,
            {},
            "TYPICAL BREAK-EVEN | TRAINING | SAMPLE 1 | NET PNL not recorded",
        ),
    ],
)
def test_trade_outcome_caption_is_explicit(
    category: str, split: str, rank: int, trade: dict[str, Any], expected: str
) -> None:
    assert _trade_outcome_caption(category, split, rank, trade) == expected


def test_markdown_image_validation_rejects_missing_plot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing or empty figures"):
        _validate_markdown_images("![required plot](figures/missing.png)", tmp_path)


def test_report_figure_contract_rejects_missing_required_plots(tmp_path: Path) -> None:
    report = {
        "variants": [
            {
                "variant": "ladder_a0",
                "training": {"generated_figures": {"learner_diagnostics": "learner.png"}},
                "splits": {"training": {"generated_figures": {"equity_curve": "equity.png"}}},
            }
        ]
    }

    with pytest.raises(RuntimeError, match="required training figures were not generated"):
        _validate_report_figures(report, tmp_path)


def test_selected_trade_chart_gets_an_outcome_banner(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    chart_path = tmp_path / "trade.png"
    figure, axes = plt.subplots(figsize=(4, 2))
    axes.plot([0, 1], [0, 1])
    figure.savefig(chart_path)
    plt.close(figure)
    original_height = mpimg.imread(chart_path).shape[0]

    _annotate_trade_outcome(chart_path, "TYPICAL WINNER | TRAINING | SAMPLE 1", "winners")

    annotated = mpimg.imread(chart_path)
    assert annotated.shape[0] > original_height
