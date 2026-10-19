"""Tests that campaign bar-cache prebuild fails closed on symlink paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from scripts.triton import prebuild_campaign_bar_cache

from quant_rl.data.pipeline import BAR_CACHE_VERSION
from quant_rl.data.ticks import TICK_CACHE_VERSION


def test_prebuild_campaign_bar_cache_builds_all_configured_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": str(cache_dir),
                "symbols": ["US100.cash", "US500.cash"],
                "timeframes": ["M1", "M5"],
                "datasets": {"training": {}, "testing": {}},
            }
        }
    )
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda: cfg)

    def fake_run_pipeline(
        _cfg: object, *, force: bool = False
    ) -> dict[str, dict[str, pd.DataFrame]]:
        assert force
        data = {
            symbol: {timeframe: pd.DataFrame(index=range(3)) for timeframe in cfg.data.timeframes}
            for symbol in cfg.data.symbols
        }
        for symbol in cfg.data.symbols:
            for timeframe in cfg.data.timeframes:
                for dataset_name in cfg.data.datasets:
                    path = (
                        cache_dir
                        / dataset_name
                        / f"{symbol}_{timeframe}_{BAR_CACHE_VERSION}.parquet"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"prepared")
        return data

    def fake_build_tick_books(_cfg: object, *, force: bool = False) -> dict[str, None]:
        assert force
        for symbol in cfg.data.symbols:
            for dataset_name in cfg.data.datasets:
                path = cache_dir / dataset_name / f"{symbol}_ticks_{TICK_CACHE_VERSION}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"prepared")
        return {symbol: None for symbol in cfg.data.symbols}

    monkeypatch.setattr(prebuild_campaign_bar_cache, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(prebuild_campaign_bar_cache, "build_tick_books", fake_build_tick_books)
    prebuild_campaign_bar_cache.main()

    output = capsys.readouterr().out
    assert '"cache_files": 8' in output
    assert '"tick_cache_files": 4' in output


def test_prebuild_campaign_bar_cache_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "cache"
    link.symlink_to(target, target_is_directory=True)
    cfg = OmegaConf.create(
        {"data": {"cache_dir": str(link), "symbols": ["US100.cash"], "timeframes": ["M1"]}}
    )
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda: cfg)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        prebuild_campaign_bar_cache.main()
