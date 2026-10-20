def main() -> None:
    """Run the training entry point without importing it during package loading."""
    from .training import main as training_main

    training_main()


__all__ = [
    "main",
]
