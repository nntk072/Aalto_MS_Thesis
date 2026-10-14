"""Versioned, flag-derived coordinate contract for strategy-overlay actions.

An action width is not an action schema: several different flag combinations
produce the same number of coordinates. Use names when decoding, logging or
constructing policy priors. Existing coordinate ordering is intentionally kept.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "overlay-action-v2"


@dataclass(frozen=True)
class ActionSchema:
    names: tuple[str, ...]
    has_direction: bool
    has_sl_mode: bool
    has_tp_mode: bool
    has_multi_tp: bool
    has_simplex: bool
    version: str = SCHEMA_VERSION

    @property
    def width(self) -> int:
        return len(self.names)

    @property
    def indices(self) -> dict[str, int]:
        result = {name: idx for idx, name in enumerate(self.names)}
        if self.has_multi_tp:
            result["target"] = result["tp1_sel"]
        return result

    @property
    def key(self) -> str:
        canonical: dict[tuple[bool, bool, bool, bool, bool], str] = {
            (False, False, False, False, False): "5",
            (True, False, False, False, False): "6_dir",
            (False, True, False, False, False): "6_sl",
            (False, False, True, False, False): "6_tp",
            (True, True, False, False, False): "7",
            (True, False, True, False, False): "7_tp",
            (True, True, True, False, False): "8",
            (True, True, True, True, False): "10",
            (False, True, True, True, True): "11",
            (True, True, True, True, True): "12",
        }
        flags = (
            self.has_direction,
            self.has_sl_mode,
            self.has_tp_mode,
            self.has_multi_tp,
            self.has_simplex,
        )
        if flags in canonical:
            return canonical[flags]
        return "v2:" + "".join(str(int(flag)) for flag in flags)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe coordinate contract."""
        return {
            "version": self.version,
            "key": self.key,
            "width": self.width,
            "coordinates": list(self.names),
            "has_direction": self.has_direction,
            "has_sl_mode": self.has_sl_mode,
            "has_tp_mode": self.has_tp_mode,
            "has_multi_tp": self.has_multi_tp,
            "has_simplex": self.has_simplex,
        }


def build_action_schema(
    *,
    has_direction: bool = False,
    has_sl_mode: bool = False,
    has_tp_mode: bool = False,
    has_multi_tp: bool = False,
    has_simplex: bool = False,
) -> ActionSchema:
    if has_simplex and not has_multi_tp:
        raise ValueError("simplex fractions require multi-TP targets")
    coordinates: list[str] = []
    if has_direction:
        coordinates.append("direction")
    coordinates.extend(("intensity", "stop", "risk"))
    coordinates.extend(("tp1_sel", "tp2_sel", "tp3_sel") if has_multi_tp else ("target",))
    if has_sl_mode:
        coordinates.append("sl_mode")
    if has_tp_mode:
        coordinates.append("tp_mode")
    if has_simplex:
        coordinates.extend(("z1", "z2"))
    coordinates.append("exit")
    return ActionSchema(
        names=tuple(coordinates),
        has_direction=has_direction,
        has_sl_mode=has_sl_mode,
        has_tp_mode=has_tp_mode,
        has_multi_tp=has_multi_tp,
        has_simplex=has_simplex,
    )
