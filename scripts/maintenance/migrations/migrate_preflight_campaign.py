#!/usr/bin/env python3
"""Apply a reviewed campaign preflight migration through the package API."""

from experiments.campaigns.preflight_migration import main

if __name__ == "__main__":
    raise SystemExit(main())
