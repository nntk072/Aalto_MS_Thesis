from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/audit_replacement_fold_fit.py"
SPEC = importlib.util.spec_from_file_location("replacement_fold_fit_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_profile_key_tracks_source_clock_model() -> None:
    cfg = OmegaConf.create(
        {
            "features": {},
            "data": {"tz": "Etc/GMT-3", "timestamp_clock": "native"},
            "session": {"tz": "Etc/GMT-3"},
            "strategy": {"session": {"timezone": "Etc/GMT-3"}},
        }
    )

    native_key = AUDIT.profile_key(cfg)
    cfg.data.timestamp_clock = "ftmo_mt5_us_dst"

    assert AUDIT.profile_key(cfg) != native_key


def test_report_path_resolves_under_repository(tmp_path: Path) -> None:
    report_path, relative_path = AUDIT.resolve_report_path(Path("reports/fold-fit.json"), tmp_path)

    assert report_path == tmp_path / "reports/fold-fit.json"
    assert relative_path == Path("reports/fold-fit.json")


def test_report_path_rejects_repository_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="under repository root"):
        AUDIT.resolve_report_path(Path("../outside.json"), tmp_path)
