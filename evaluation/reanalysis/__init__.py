def main() -> None:
    """Run the reanalysis entry point without importing it during package loading."""
    from .runner import main as reanalysis_main

    reanalysis_main()


__all__ = [
    "main",
]
