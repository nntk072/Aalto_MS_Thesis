from __future__ import annotations

import json
from datetime import UTC, datetime

from decisioning.contracts import ActionSchemaRef, PredictorMetadata

from evaluation.contracts import EvaluationCase, MetricResult, ReportDataset
from evaluation.reporting.dataset_renderers import render_json, render_markdown_table
from market_data.contracts import DatasetRef, MarketEventSchema


def test_two_renderers_consume_the_same_metric_dataset() -> None:
    dataset = DatasetRef(
        "logical-bars",
        "US100.cash",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        "UTC",
        MarketEventSchema("bar", "ohlc-v1", ("time", "close"), "time"),
        "test-source:sha256:fixture",
    )
    provider = PredictorMetadata(
        "test-provider",
        "rule",
        "rule_based",
        "input-v1",
        ActionSchemaRef("discrete-v1", "d2", ("hold", "enter"), "discrete"),
    )
    report = ReportDataset(
        EvaluationCase("case-1", dataset, "input-v1", ("execution",), provider),
        (MetricResult("net_pnl", 125.0, "USD"), MetricResult("max_drawdown", 0.08, "fraction")),
    )

    json_result = json.loads(render_json(report))
    table_result = render_markdown_table(report)

    assert json_result["metrics"][0] == {"name": "net_pnl", "unit": "USD", "value": 125.0}
    assert "| max_drawdown | 0.08 | fraction |" in table_result
    assert report.metrics[0].value == 125.0
