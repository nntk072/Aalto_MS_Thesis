from __future__ import annotations

from typing import Any

from artifacts.identity import artifact_id


def market_data_cache_id(
    *, dataset_id: str, transform: str, config: Any, schema_version: int = 1
) -> str:
    return str(
        artifact_id(
            "market-data-cache",
            {
                "dataset_id": dataset_id,
                "transform": transform,
                "config": config,
                "schema_version": schema_version,
            },
        )
    )
