"""Repository paths shared by campaign workflow services."""

from configuration.paths import repository_root

ROOT = repository_root(__file__)
DEFAULT_VARIANTS = ROOT / "scripts" / "variant_list.txt"
