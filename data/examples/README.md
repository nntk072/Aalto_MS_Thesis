# `data/` — What Lives Here and How It Flows

The real CSVs in this directory are **gitignored** (3.8 GB total). This
`examples/` subfolder contains one small, tracked sample so a developer can
understand the format without the full dataset. **`US100.cash_M1_*.csv` here
is the first 2,880 rows (≈2 trading days) extracted verbatim from the real
file** — same columns, same encoding, same naming convention.

---

## Table of Contents

- [Timeframe CSVs Warning](#-timeframe-csvs)
- [File naming convention](#file-naming-convention)
- [File format (bar files)](#file-format-bar-files)
- [How the pipeline consumes it](#how-the-pipeline-consumes-it)
- [Using the sample](#using-the-sample)

---

## File naming convention

```
<SYMBOL>_<TIMEFRAME>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>.csv
   │          │              │                │
   │          │              └ first bar time  └ last bar time (broker tz)
   │          └ M1, M5, M15, M30, H1, H4, Daily …
   └ e.g. US100.cash, US500.cash
```

## ⚠️ Timeframe CSVs

Any higher-timeframe CSV files in this directory other than the M1 source
(`US100.cash_M1_*`, `US500.cash_M1_*`) are mislabeled duplicates of finer
timeframes and are NOT used by the pipeline. `quant_rl/data/` resamples
every timeframe (M5/M15/M30/H1/H4/D1) from the true M1 source. Do not load
the other CSVs directly.

Files present locally (all ignored by git):

| File group | Role |
|---|---|
| `US100.cash_M1_*`, `US500.cash_M1_*` | **The only bar files the pipeline reads** (`data.m1_files` in `quant_rl/config/default.yaml`). Every other timeframe is *resampled from M1 in code*. |
| `*_M5/M15/M30/H1/H4/Daily_*` | MT5 exports of the same data at higher timeframes — kept for reference/spot-checks, **not read by the pipeline** (they are known duplicates of what resampling produces). |
| `US100.cash_2025…_2026….csv` (no TF tag) | Real tick data (`<BID> <ASK> <LAST> <VOLUME> <FLAGS>`) used by `data.tick_files` for accurate fill pricing (spread/slippage) in backtests. |

## File format (bar files)

MT5 tab-separated export, **no quoting**, header row included:

```
<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
2024.12.30\t01:05:00\t21515.15\t21515.15\t21501.85\t21502.65\t101\t0\t290
```

| Column | Meaning |
|---|---|
| `<DATE>` `<TIME>` | Bar open time in **broker server timezone** — config assumes `Etc/GMT-3` (UTC+3), see `data.tz`. |
| `<OPEN> <HIGH> <LOW> <CLOSE>` | Prices in index points, 2 d.p. |
| `<TICKVOL>` | Number of ticks in the bar (real activity measure). |
| `<VOL>` | Always `0` — CFDs have no real volume. |
| `<SPREAD>` | Spread in **integer points**. Convert to price with `data.point_size` (0.01) → 290 pts = $2.90. |

## How the pipeline consumes it

1. `quant_rl/data/pipeline.py` reads **only** the `data.m1_files` CSVs from
   `data.raw_dir` (`data:` block in `quant_rl/config/default.yaml`).
2. Bars are parsed in broker tz (`Etc/GMT-3`), resampled to all
   `data.timeframes`, features computed, and cached to parquet in
   `data.cache_dir` (`cache/`, also gitignored).
3. The in-sample / out-of-sample split is date-based:
   `train_end: "2025-12-31"` / `test_start: "2026-01-01"`.

## Using the sample

The sample keeps the **exact real filename**, so you can drop it into `data/`
(or point `data.raw_dir` at `data/examples`) and exercise the pipeline,
features, and dataloaders without the full 3.8 GB. Note the sample only spans
2024-12-30 → 2025-01-02, so any experiment on it must use config overrides
that keep that window (e.g. `--mvp`-style short runs); the default split dates
fall outside it.

What the sample demonstrates beyond a plain slice:
- **Weekend/holiday gap** — it crosses New Year, so you can see the session
  break between 2024-12-31 and 2025-01-02 in the timestamps.
- **Constant `SPREAD=290` periods** — overnight/low-liquidity spreads differ
  from intraday ones; the backtester uses this column for fill pricing.
- **Varying `<TICKVOL>`** — activity proxy used as a feature input.
