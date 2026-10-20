"""Versioned model-input views produced from market datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

InputKind = Literal["handcrafted_features", "raw_market", "learned_representation"]


@dataclass(frozen=True, slots=True)
class InputSchema:
    schema_id: str
    version: str
    kind: InputKind
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.schema_id or not self.version or not self.columns:
            raise ValueError("input schema identity, version, and columns are required")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("input schema columns must be unique")


@dataclass(frozen=True, slots=True)
class InputView:
    schema: InputSchema
    dataset_id: str
    values: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("input view must identify its source dataset")
        width = len(self.schema.columns)
        if any(len(row) != width for row in self.values):
            raise ValueError("input values do not match the declared schema width")


__all__ = ["InputKind", "InputSchema", "InputView"]
