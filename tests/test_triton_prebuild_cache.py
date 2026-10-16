"""Tests that campaign bar-cache prebuild fails closed on symlink paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.triton import prebuild_campaign_bar_cache

from quant_rl.data.pipeline import BAR_CACHE_VERSION


def test_prebuild_campaign_bar_cache_builds_all_configured_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            cache_dir=str(cache_dir),
            symbols=["US100.cash", "US500.cash"],
            timeframes=["M1", "M5"],
        )
    )
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda: cfg)

    def fake_run_pipeline(_cfg: object) -> dict[str, dict[str, pd.DataFrame]]:
        data = {
            symbol: {timeframe: pd.DataFrame(index=range(3)) for timeframe in cfg.data.timeframes}
            for symbol in cfg.data.symbols
        }
        for symbol in cfg.data.symbols:
            for timeframe in cfg.data.timeframes:
                path = cache_dir / f"{symbol}_{timeframe}_{BAR_CACHE_VERSION}.parquet"
                path.write_bytes(b"prepared")
        return data

    monkeypatch.setattr(prebuild_campaign_bar_cache, "run_pipeline", fake_run_pipeline)
    prebuild_campaign_bar_cache.main()

    assert '"cache_files": 4' in capsys.readouterr().out


def test_prebuild_campaign_bar_cache_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "cache"
    link.symlink_to(target, target_is_directory=True)
    cfg = SimpleNamespace(
        data=SimpleNamespace(cache_dir=str(link), symbols=["US100.cash"], timeframes=["M1"])
    )
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda: cfg)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        prebuild_campaign_bar_cache.main()
