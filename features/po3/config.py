"""Compatibility exports for the original PO3 rules module.

New code should import from the focused detection, trigger, and zone modules.
"""

from .fvg_detection import FVGConfig, detect_fvg
from .ifvg_detection import IFVGConfig, detect_ifvg_confirmation
from .trigger_rules import (
    EntryConfig,
    EntryTriggerType,
    detect_entry_trigger,
    detect_htf_fvg,
    detect_ltf_ifvg,
    detect_po3_entries,
)
from .zones import FVGZone, build_fvg_zones, build_fvg_zones_for_plot

__all__ = [
    "EntryConfig",
    "EntryTriggerType",
    "FVGConfig",
    "FVGZone",
    "IFVGConfig",
    "build_fvg_zones",
    "build_fvg_zones_for_plot",
    "detect_entry_trigger",
    "detect_fvg",
    "detect_htf_fvg",
    "detect_ifvg_confirmation",
    "detect_ltf_ifvg",
    "detect_po3_entries",
]
