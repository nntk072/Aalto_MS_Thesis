"""Logical market dataset identity and source interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MarketEventSchema:
    """Declared event semantics, independent of the physical file format."""

    event_type: str
    version: str
    fields: tuple[str, ...]
    time_field: str

    def __post_init__(self) -> None:
        if not self.event_type or not self.version or not self.time_field:
            raise ValueError("event type, schema version, and time field are required")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("event schema fields must be nonempty and unique")
        if self.time_field not in self.fields:
            raise ValueError("event schema time field must be declared")


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Stable logical dataset request that contains no storage path semantics."""

    dataset_id: str
    instrument_id: str
    start: datetime
    end: datetime
    timezone: str
    schema: MarketEventSchema
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.instrument_id or not self.provenance_id:
            raise ValueError("dataset, instrument, and provenance identities are required")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("dataset bounds must be timezone-aware")
        if self.start >= self.end:
            raise ValueError("dataset start must precede end")
        if not self.timezone:
            raise ValueError("dataset timezone is required")


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Rows returned for a logical dataset request."""

    reference: DatasetRef
    events: tuple[Mapping[str, object], ...]


class DatasetSource(Protocol):
    """Resolve a logical dataset reference through one source or storage backend."""

    def load(self, reference: DatasetRef) -> MarketDataset: ...


__all__ = ["DatasetRef", "DatasetSource", "MarketDataset", "MarketEventSchema"]
