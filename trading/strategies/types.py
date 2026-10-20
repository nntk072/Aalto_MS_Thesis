from __future__ import annotations

from typing import Any, Protocol


class FeatureRow(Protocol):
    @property
    def index(self) -> Any: ...

    def get(self, key: str, default: Any = ...) -> Any: ...

    def __getitem__(self, key: str) -> float: ...
