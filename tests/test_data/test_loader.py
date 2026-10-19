from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_rl.data.loader import load_bars


def test_load_bars_accepts_date_only_daily_export(tmp_path: Path) -> None:
    path = tmp_path / "daily.tsv"
    path.write_text(
        "<DATE>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2025.01.02\t10\t12\t9\t11\t100\t0\t2\n",
        encoding="utf-8",
    )

    bars = load_bars(path)

    assert bars.index.tolist() == [pd.Timestamp("2025-01-02 00:00:00")]
    assert bars.iloc[0]["close"] == 11
    assert bars.iloc[0]["spread"] == 2


def test_load_bars_keeps_intraday_time_from_export(tmp_path: Path) -> None:
    path = tmp_path / "minute.tsv"
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2025.01.02\t09:31:00\t10\t12\t9\t11\t100\t0\t2\n",
        encoding="utf-8",
    )

    bars = load_bars(path)

    assert bars.index.tolist() == [pd.Timestamp("2025-01-02 09:31:00")]
