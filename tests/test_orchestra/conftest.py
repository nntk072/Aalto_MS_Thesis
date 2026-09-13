"""Shared fixtures for orchestra tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def stub_clis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend all model CLIs are installed (CI runners have no agent binaries)."""
    monkeypatch.setattr("shutil.which", lambda cli: f"/usr/bin/{cli}")
