from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.matrix import audit_real_market_bundle_parity as audit


def test_raw_input_hashes_use_configured_files_without_campaign_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = {
        "primary.csv": b"primary M1 data",
        "secondary.csv": b"secondary M1 data",
        "primary_ticks.csv": b"primary tick data",
    }
    for filename, contents in files.items():
        (data_dir / filename).write_bytes(contents)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            raw_dir="data",
            m1_files={"US100.cash": "primary.csv", "US500.cash": "secondary.csv"},
            tick_files={"US100.cash": "primary_ticks.csv", "US500.cash": ""},
        )
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    result = audit._raw_input_hashes(cfg)

    assert result == {
        f"data/{filename}": hashlib.sha256(contents).hexdigest()
        for filename, contents in files.items()
    }


def test_raw_input_hashes_fail_closed_when_configured_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").mkdir()
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            raw_dir="data",
            m1_files={"US100.cash": "missing.csv"},
            tick_files={},
        )
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="Configured market input not found"):
        audit._raw_input_hashes(cfg)


def test_audit_source_inventory_includes_project_code_and_configs(tmp_path: Path) -> None:
    expected = (
        "config/strategies/po3.yaml",
        "quant_rl/envs/strategies/po3.py",
        "quant_rl/features/build.py",
        "scripts/matrix/audit_real_market_bundle_parity.py",
        "scripts/variant_list.txt",
    )
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked input\n", encoding="utf-8")
    (tmp_path / "quant_rl" / "ignored.txt").write_text("not code/config\n", encoding="utf-8")

    result = audit._audit_source_paths(tmp_path)

    assert tuple(path.relative_to(tmp_path).as_posix() for path in result) == expected
