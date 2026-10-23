"""Tests that campaign bar-cache prebuild fails closed on symlink paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from market_data.commands import prebuild_campaign_cache as prebuild_campaign_bar_cache
from market_data.loading.pipeline import BAR_CACHE_VERSION
from market_data.loading.ticks import TICK_CACHE_VERSION


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
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda **_kwargs: cfg)
    monkeypatch.delenv("CAMPAIGN_SPEC", raising=False)

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
    monkeypatch.setattr(prebuild_campaign_bar_cache, "load_config", lambda **_kwargs: cfg)
    monkeypatch.delenv("CAMPAIGN_SPEC", raising=False)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        prebuild_campaign_bar_cache.main()


def test_campaign_prebuild_uses_shared_variant_price_grid(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "campaign.yaml"
    spec_path.write_text(
        "campaign_id: price-grid-test\n"
        "timesteps: 2000000\n"
        "seeds: [50]\n"
        "variants: [ladder_a4b_entry_fsm_qgrid025, ladder_a4a_entry_fsm_hidden_qgrid025]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CAMPAIGN_SPEC", str(spec_path))

    overrides = prebuild_campaign_bar_cache._campaign_data_overrides()

    assert "data.price_discretization.enabled=true" in overrides
    assert "data.price_discretization.increment=0.25" in overrides
    assert "data.price_discretization.rounding=nearest_half_away_from_zero" in overrides
    assert "data.cache_dir=cache/train_test_2025_2026_qgrid025_v1" in overrides
