"""Transactional publication behavior for the named feature-cache builder."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from scripts.matrix import rebuild_named_prebuilt_feature_caches as cache_builder


@pytest.mark.unit
def test_htf_confirmed_cache_requires_sweep_audit_parquet(tmp_path: Path) -> None:
    cache_path = tmp_path / "idea1.parquet"
    htf_cfg = OmegaConf.create({"features": {"po3_reference_mode": "htf_confirmed"}})

    with pytest.raises(RuntimeError, match="required sweep-audit parquet"):
        cache_builder._validate_staged_sweep_audit(cache_path, htf_cfg)

    audit_path = Path(f"{cache_path}.htf_sweeps.parquet")
    audit_path.write_bytes(b"audited sweep references")
    assert cache_builder._validate_staged_sweep_audit(cache_path, htf_cfg) == audit_path


@pytest.mark.unit
def test_legacy_cache_does_not_require_sweep_audit_parquet(tmp_path: Path) -> None:
    cache_path = tmp_path / "baseline.parquet"
    legacy_cfg = OmegaConf.create({"features": {}})

    assert cache_builder._validate_staged_sweep_audit(cache_path, legacy_cfg) is None


@pytest.mark.unit
def test_resolved_config_record_hashes_effective_merged_configuration() -> None:
    cfg = OmegaConf.create(
        {
            "features": {"sessions": {"ny": {"start": "08:30", "end": "15:00"}}},
            "data": {"primary": "US100.cash"},
        }
    )
    record = cache_builder._resolved_config_record(cfg)
    assert record["value"] == OmegaConf.to_container(cfg, resolve=True)
    assert len(record["sha256"]) == 64

    cfg.features.sessions.ny.end = "14:00"
    changed = cache_builder._resolved_config_record(cfg)
    assert changed["sha256"] != record["sha256"]


@pytest.mark.unit
def test_loaded_source_hashes_include_cache_builder() -> None:
    sources = cache_builder._loaded_project_source_hashes()
    relative_path = Path(cache_builder.__file__).resolve().relative_to(cache_builder.ROOT.resolve())
    assert relative_path.as_posix() in sources
    assert sources[relative_path.as_posix()] == cache_builder.sha256_file(
        Path(cache_builder.__file__).resolve()
    )


@pytest.mark.unit
def test_fresh_market_forces_raw_rebuild_in_temporary_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create({"data": {"raw_dir": "data", "cache_dir": "shared-cache"}})
    captured: dict[str, object] = {}

    def fake_run_pipeline(
        config: DictConfig, *, force: bool = False
    ) -> dict[str, dict[str, pd.DataFrame]]:
        captured["raw_dir"] = config.data.raw_dir
        captured["cache_dir"] = config.data.cache_dir
        captured["force"] = force
        assert Path(str(config.data.cache_dir)).is_dir()
        return {"US100.cash": {"M1": pd.DataFrame()}}

    monkeypatch.setattr(cache_builder, "run_pipeline", fake_run_pipeline)
    market = cache_builder._load_fresh_market(cfg)

    assert captured["raw_dir"] == str((cache_builder.ROOT / "data").resolve())
    assert captured["force"] is True
    assert captured["cache_dir"] != "shared-cache"
    assert not Path(str(captured["cache_dir"])).exists()
    assert cfg.data.cache_dir == "shared-cache"
    assert "US100.cash" in market


def _transaction_files(tmp_path: Path) -> tuple[dict[Path, Path], Path, Path, Path]:
    backup = tmp_path / "backup"
    backup.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    staged: dict[Path, Path] = {}
    for name in ("baseline.parquet", "idea1.parquet"):
        target = tmp_path / name
        target.write_text(f"old:{name}", encoding="utf-8")
        (backup / name).write_text(f"old:{name}", encoding="utf-8")
        staged_path = stage / name
        staged_path.write_text(f"new:{name}", encoding="utf-8")
        staged[target] = staged_path

    report_tmp = stage / "report.json.tmp"
    report_tmp.write_text('{"status":"passed"}\n', encoding="utf-8")
    report = tmp_path / "report.json"
    return staged, backup, report_tmp, report


@pytest.mark.unit
def test_publish_staged_files_replaces_caches_and_report(tmp_path: Path) -> None:
    staged, backup, report_tmp, report = _transaction_files(tmp_path)

    def validate() -> None:
        assert all(target.read_text(encoding="utf-8").startswith("new:") for target in staged)

    cache_builder._publish_staged_files(staged, backup, report_tmp, report, validate)

    assert all(target.read_text(encoding="utf-8").startswith("new:") for target in staged)
    assert report.read_text(encoding="utf-8") == '{"status":"passed"}\n'
    assert not report_tmp.exists()


@pytest.mark.unit
def test_publish_staged_files_rolls_back_on_validation_failure(tmp_path: Path) -> None:
    staged, backup, report_tmp, report = _transaction_files(tmp_path)

    def fail_validation() -> None:
        raise RuntimeError("simulated post-publish verification failure")

    with pytest.raises(RuntimeError, match="simulated post-publish"):
        cache_builder._publish_staged_files(staged, backup, report_tmp, report, fail_validation)

    assert all(target.read_text(encoding="utf-8").startswith("old:") for target in staged)
    assert not report.exists()


@pytest.mark.unit
def test_publish_staged_files_rolls_back_partial_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, backup, report_tmp, report = _transaction_files(tmp_path)
    target_a, target_b = tuple(staged)
    original_replace = os.replace

    def fail_on_second_target(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        if Path(target) == target_b:
            raise OSError("simulated replacement failure")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_on_second_target)
    with pytest.raises(OSError, match="simulated replacement failure"):
        cache_builder._publish_staged_files(staged, backup, report_tmp, report, lambda: None)

    assert target_a.read_text(encoding="utf-8") == "old:baseline.parquet"
    assert target_b.read_text(encoding="utf-8") == "old:idea1.parquet"
    assert not report.exists()
