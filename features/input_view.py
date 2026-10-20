"""Adapters that expose tabular inputs through the feature-neutral view contract."""

from __future__ import annotations

import pandas as pd

from features.contracts import InputSchema, InputView


def input_view_from_frame(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    schema: InputSchema,
) -> InputView:
    """Select declared columns from a tabular source without adding features."""
    missing = [column for column in schema.columns if column not in frame.columns]
    if missing:
        raise ValueError(f"input frame is missing schema columns: {missing}")
    values: tuple[tuple[float, ...], ...] = tuple(
        tuple(float(value) for value in row)
        for row in frame.loc[:, list(schema.columns)].itertuples(index=False, name=None)
    )
    return InputView(schema=schema, dataset_id=dataset_id, values=values)


__all__ = ["input_view_from_frame"]
