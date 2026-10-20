from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifacts.atomic import _atomic_replace, write_json_atomic, write_text_atomic
from artifacts.identity import ArtifactId, artifact_id


def test_artifact_id_is_stable_for_mapping_order() -> None:
    first = artifact_id("dataset", {"symbol": "US100", "timeframe": "M1"})
    second = artifact_id("dataset", {"timeframe": "M1", "symbol": "US100"})

    assert first == second
    assert isinstance(first, ArtifactId)
    assert str(first) == f"dataset:{first.digest}"


def test_atomic_writers_create_parent_and_replace_content(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "report.txt"
    json_path = tmp_path / "nested" / "manifest.json"

    assert write_text_atomic(text_path, "complete\n") == text_path
    assert write_json_atomic(json_path, {"b": 2, "a": 1}) == json_path
    assert text_path.read_text(encoding="utf-8") == "complete\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_atomic_replace_preserves_destination_and_cleans_temp_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"previous\n")

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("artifacts.atomic.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        _atomic_replace(destination, b"replacement\n")

    assert destination.read_bytes() == b"previous\n"
    assert list(tmp_path.iterdir()) == [destination]
