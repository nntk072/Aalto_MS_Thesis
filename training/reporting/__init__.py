"""Training-facing terminal and diagnostic rendering helpers."""

from .box_table import kv_lines, kv_widths, render_box_table

__all__ = ["kv_lines", "kv_widths", "render_box_table"]
