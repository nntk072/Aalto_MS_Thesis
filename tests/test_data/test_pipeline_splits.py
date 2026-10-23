from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from market_data.loading.pipeline import run_pipeline


def _write_m1(path: Path, date: str, time: str, open_price: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        f"{date}\t{time}\t{open_price}\t{open_price + 1}\t{open_price - 1}\t"
        f"{open_price}\t10\t0\t2\n",
        encoding="utf-8",
    )


def test_pipeline_builds_independent_train_test_bar_caches(tmp_path: Path) -> None:
    symbols = ["US100.cash", "US500.cash"]
    train_names = {symbol: f"{symbol}_M1.csv" for symbol in symbols}
    test_names = {symbol: f"{symbol}_M1.csv" for symbol in symbols}
    for symbol in symbols:
        for name, date, first_time, price in (
            ("training", "2025.12.31", "23:57:00", 100),
            ("testing", "2026.01.01", "00:00:00", 200),
        ):
            path = tmp_path / name / f"{symbol}_M1.csv"
            _write_m1(path, date, first_time, price)
            second_time = "23:59:00" if name == "training" else "00:01:00"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{date}\t{second_time}\t{price + 1}\t{price + 2}\t{price}\t{price + 1}\t10\t0\t2\n"
                )

    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "tz": "Etc/GMT-3",
                "symbols": symbols,
                "timeframes": ["M1"],
                "datasets": {
                    "training": {
                        "raw_dir": str(tmp_path / "training"),
                        "m1_files": train_names,
                    },
                    "testing": {
                        "raw_dir": str(tmp_path / "testing"),
                        "m1_files": test_names,
                    },
                },
            },
            "session": {"end": "23:00"},
        }
    )

    frames = run_pipeline(cfg)

    for symbol in symbols:
        combined = frames[symbol]["M1"]
        assert len(combined) == 4
        assert combined.index.is_monotonic_increasing
        assert combined.index[0].strftime("%Y-%m-%d") == "2025-12-31"
        assert combined.index[1].strftime("%Y-%m-%d") == "2025-12-31"
        assert combined.index[2].strftime("%Y-%m-%d") == "2026-01-01"
        assert combined.index[3].strftime("%Y-%m-%d") == "2026-01-01"
        assert (
            tmp_path / "cache" / "training" / f"{symbol}_M1_v3-native-timeframes.parquet"
        ).is_file()
        assert (
            tmp_path / "cache" / "testing" / f"{symbol}_M1_v3-native-timeframes.parquet"
        ).is_file()


def test_pipeline_can_load_only_training_dataset(tmp_path: Path) -> None:
    symbol = "US100.cash"
    names = {symbol: f"{symbol}_M1.csv"}
    for split, date, price in (
        ("training", "2025.12.31", 100.0),
        ("testing", "2026.01.01", 200.0),
    ):
        path = tmp_path / split / names[symbol]
        _write_m1(path, date, "12:00:00", price)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{date}\t12:01:00\t{price + 1}\t{price + 2}\t{price}\t{price + 1}\t10\t0\t2\n"
            )
    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "tz": "Etc/GMT-3",
                "symbols": [symbol],
                "timeframes": ["M1"],
                "datasets": {
                    split: {"raw_dir": str(tmp_path / split), "m1_files": names}
                    for split in ("training", "testing")
                },
            },
            "session": {"end": "23:00"},
        }
    )

    frames = run_pipeline(cfg, datasets=["training"])

    assert len(frames[symbol]["M1"]) == 2
    assert frames[symbol]["M1"].iloc[0]["close"] == 100.0
    assert (tmp_path / "cache" / "training" / f"{symbol}_M1_v3-native-timeframes.parquet").is_file()
    assert not (
        tmp_path / "cache" / "testing" / f"{symbol}_M1_v3-native-timeframes.parquet"
    ).exists()


def test_pipeline_applies_same_price_grid_to_training_and_testing(tmp_path: Path) -> None:
    symbol = "US100.cash"
    names = {symbol: f"{symbol}_M1.csv"}
    for split, date in (("training", "2025.12.31"), ("testing", "2026.01.01")):
        path = tmp_path / split / names[symbol]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
            f"{date}\t12:00:00\t25633.68\t25633.99\t25633.61\t25633.68\t10\t0\t60\n",
            encoding="utf-8",
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{date}\t12:01:00\t25633.68\t25633.99\t25633.61\t25633.68\t10\t0\t60\n")

    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "tz": "Etc/GMT-3",
                "symbols": [symbol],
                "timeframes": ["M1"],
                "price_discretization": {
                    "enabled": True,
                    "increment": 0.25,
                    "rounding": "nearest_half_away_from_zero",
                },
                "datasets": {
                    split: {"raw_dir": str(tmp_path / split), "m1_files": names}
                    for split in ("training", "testing")
                },
            },
            "session": {"end": "23:00"},
        }
    )

    frames = run_pipeline(cfg)

    combined = frames[symbol]["M1"]
    assert combined["close"].tolist() == pytest.approx([25633.75] * 4)
    assert combined["high"].tolist() == pytest.approx([25634.0] * 4)
    assert combined["low"].tolist() == pytest.approx([25633.5] * 4)
    for split in ("training", "testing"):
        identity = (
            tmp_path / "cache" / split / f"{symbol}_M1_v3-native-timeframes.parquet.source.sha256"
        ).read_text(encoding="utf-8")
        assert "price-grid:nearest_half_away_from_zero:0.25" in identity
        raw_text = (tmp_path / split / names[symbol]).read_text(encoding="utf-8")
        assert "25633.68" in raw_text


def test_pipeline_loads_native_timeframe_files_into_separate_split_caches(
    tmp_path: Path,
) -> None:
    symbol = "US100.cash"
    timeframe_files: dict[str, dict[str, str]] = {"M1": {}, "M3": {}}
    first_m3_close = {"training": 1000.0, "testing": 2000.0}

    for split, first_close in first_m3_close.items():
        raw_dir = tmp_path / split
        m1_name = f"{symbol}_M1.csv"
        m3_name = f"{symbol}_M3.csv"
        _write_m1(
            raw_dir / m1_name,
            "2025.12.31" if split == "training" else "2026.01.01",
            "23:54:00" if split == "training" else "00:00:00",
            100.0 if split == "training" else 200.0,
        )
        m1_date = "2025.12.31" if split == "training" else "2026.01.01"
        m1_second_time = "23:55:00" if split == "training" else "00:01:00"
        m1_price = 100.0 if split == "training" else 200.0
        with (raw_dir / m1_name).open("a", encoding="utf-8") as stream:
            stream.write(
                f"{m1_date}\t{m1_second_time}\t{m1_price + 1}\t{m1_price + 2}"
                f"\t{m1_price}\t{m1_price + 1}\t10\t0\t2\n"
            )
        date = "2025.12.31" if split == "training" else "2026.01.01"
        first_time = "23:54:00" if split == "training" else "00:00:00"
        second_time = "23:57:00" if split == "training" else "00:03:00"
        (raw_dir / m3_name).write_text(
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
            f"{date}\t{first_time}\t{first_close}\t{first_close + 1}\t{first_close - 1}\t{first_close}\t10\t0\t2\n"
            f"{date}\t{second_time}\t{first_close + 1}\t{first_close + 2}\t{first_close}\t{first_close + 1}\t10\t0\t2\n",
            encoding="utf-8",
        )
        timeframe_files["M1"][split] = m1_name
        timeframe_files["M3"][split] = m3_name

    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "tz": "Etc/GMT-3",
                "symbols": [symbol],
                "timeframes": ["M1", "M3"],
                "datasets": {
                    split: {
                        "raw_dir": str(tmp_path / split),
                        "m1_files": {symbol: timeframe_files["M1"][split]},
                        "timeframe_files": {
                            tf: {symbol: files[split]} for tf, files in timeframe_files.items()
                        },
                    }
                    for split in first_m3_close
                },
            },
            "session": {"end": "23:00"},
        }
    )

    frames = run_pipeline(cfg)

    assert frames[symbol]["M3"]["close"].tolist() == [
        first_m3_close["training"],
        first_m3_close["training"] + 1.0,
        first_m3_close["testing"],
        first_m3_close["testing"] + 1.0,
    ]
    assert (tmp_path / "cache" / "training" / f"{symbol}_M3_v3-native-timeframes.parquet").is_file()
    assert (tmp_path / "cache" / "testing" / f"{symbol}_M3_v3-native-timeframes.parquet").is_file()
