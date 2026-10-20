"""Typing bridge for phase mixins composed on the orchestra pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class PipelineMixinBase:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...
