"""Typed inputs and metric outputs for evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass

from decisioning.contracts import PredictorMetadata
from market_data.contracts import DatasetRef


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    dataset: DatasetRef
    input_schema_version: str
    metric_suites: tuple[str, ...]
    provider: PredictorMetadata | None = None


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True, slots=True)
class ReportDataset:
    case: EvaluationCase
    metrics: tuple[MetricResult, ...]


__all__ = ["EvaluationCase", "MetricResult", "ReportDataset"]
