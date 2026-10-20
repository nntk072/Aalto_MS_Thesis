from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from artifacts.atomic import write_json_atomic
from artifacts.identity import artifact_id


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    schema_version: int
    symbol: str
    timeframe: str
    source_identities: tuple[str, ...]
    start: str | None
    end: str | None
    row_count: int
    timezone: str
    partition: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def build(
        cls,
        *,
        symbol: str,
        timeframe: str,
        source_identities: tuple[str, ...],
        start: str | None,
        end: str | None,
        row_count: int,
        timezone: str,
        partition: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetManifest:
        semantic = {
            "symbol": symbol,
            "timeframe": timeframe,
            "sources": source_identities,
            "start": start,
            "end": end,
            "row_count": row_count,
            "timezone": timezone,
            "partition": partition,
            "metadata": metadata or {},
        }
        return cls(
            str(artifact_id("dataset", semantic)),
            1,
            symbol,
            timeframe,
            source_identities,
            start,
            end,
            row_count,
            timezone,
            partition,
            metadata,
        )

    def write(self, path: str | Path) -> Path:
        return write_json_atomic(path, asdict(self))
