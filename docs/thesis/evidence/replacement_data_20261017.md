# Replacement dataset and bounded causal checks — 2026-10-17

## Disposition

The supplied replacement data, rebuilt caches, tick execution schedule, and
the two bounded feature-causality audits passed their stated technical checks.
This does **not** admit the new data for thesis selection or publication:
provider/feed identity, data-use/publication terms, and authoritative broker
calendar/timezone/DST interpretation remain undocumented. No training campaign
or new-data checkpoint exists. Overall readiness remains **NOT READY**.

## Dataset and cache identity

The requested chronological split is training through 2025-12-31 inclusive and
testing from 2026-01-01 through the last available bar on 2026-10-16. Source M1
files have 630,787 US100 rows and 628,929 US500 rows; both start at
2024-12-30 01:05 and end at 2026-10-16 23:49 in the configured broker clock.
The inventory found no unparseable/duplicate/out-of-order timestamps, missing
or nonfinite numeric values, invalid OHLC ordering, negative spreads, or
negative tick-volume values. Every source row has zero real volume; tick volume
must not be described as centralized traded volume. Large time gaps include
market closures and were not classified as missing sessions without an
authoritative calendar.

M1 train/test rows: US100 351,432/279,355; US500 349,973/278,956. The current
resolver schedule is fully covered by the rebuilt tick data for the configured
`[0, 30 ms)` randomized latency and 60-second quote wait:

| Symbol | Train schedule | Test schedule | Coverage |
|---|---:|---:|---:|
| US100.cash | 99,492 / 99,492 | 79,069 / 79,069 | 100% |
| US500.cash | 99,482 / 99,482 | 79,068 / 79,068 | 100% |

The historical campaign contract counted 99,504 US100 train decisions, 12
more than the current resolver schedule. The schedule-shape audit found shorter
NY-labeled sessions on 2025-12-02 (382 bars) and 2025-12-04 (388 bars), versus
the common 391-bar count. These are schedule exceptions, not established data
holes: holiday or broker-specific reduced-hours treatment could explain them.
The tick cache has no quotes in the corresponding minute bins, so ticks cannot
reconstruct bars there, but that fact does not distinguish a scheduled closure
from a feed/export omission. The official NYSE Group 2025 cash-equity calendar
lists its holidays and early closes; Dec 2 and Dec 4 are not listed among them
([NYSE Group calendar](https://www.nasdaq.com/press-release/nyse-group-announces-2025-2026-and-2027-holiday-and-early-closings-calendar-2024-11)).
That rules out those dates as standard NYSE Group cash-equity holiday/early-close
dates, but it does not establish the schedule of the unidentified US100.cash
CFD provider or the export semantics. US500 also has 22 minute-level
differences across observed NY intervals that require schedule-aware interpretation. Adjacent
source bars match bid-tick OHLC at cent precision (9/9 US100 and 29/29 US500
comparisons), which supports the sampled candle interval/price-side convention
only. Obtain the provider's holiday/special-hours calendar and export semantics,
then reconcile the frozen campaign schedule; do not classify these intervals
as missing data or impute prices without that evidence.

Rebuilt TickBook caches contain 135,043,551 US100 and 37,246,832 US500 rows.
All cached quotes are finite, positive, and uncrossed; there are no descending
timestamp pairs. Millisecond timestamp ties remain: 300,381 adjacent US100
ties and 29,090 US500 ties have different quotes. Stable source-row order is
preserved and the lookup selects the first source-ordered quote at/after
arrival. Since source timestamps have only millisecond precision, actual
sub-millisecond order inside a tie cannot be recovered. This is a documented
precision limitation, not evidence of decreasing time.

Three full-range named feature caches were rebuilt and verified against the
replacement M1 inputs: baseline (630,787 rows, 194 columns), Idea 1 PO3/IFVG
(630,787 rows, 532 columns), and Idea 2 distribution (630,787 rows, 538
columns). The presence of an M10 bar cache does not mean M10 is visible to each
profile; each profile follows its resolved feature-timeframe config.

## Causal checks

- Prefix invariance: **16/16** sampled cutoff/profile cases passed across four
  representative feature profiles. Every comparison had zero changed feature
  values. This is a finite sample of cutoffs, not an exhaustive proof for all
  timestamps or execution/label/model-selection paths.
- Fold-local fitting: **20/20** checks passed across five expanding purged
  folds and four representative feature profiles. After a 20% OHLC poison was
  applied to each validation interval in both instruments, training features
  stayed identical and validation features changed in model-visible columns.
  This supports train/validation separation for these tested paths; it does not
  certify every feature implementation or the full experiment pipeline.

The exact machine reports are stored in
[`evidence/replacement_data_20261017/`](evidence/replacement_data_20261017/).
Reproducible scripts are
[`audit_replacement_fold_fit.py`](../../scripts/research/audit_replacement_fold_fit.py),
[`audit_replacement_schedule_shape.py`](../../scripts/research/audit_replacement_schedule_shape.py),
and
[`audit_missing_m1_tick_reconstruction.py`](../../scripts/research/audit_missing_m1_tick_reconstruction.py);
the cutoff audit is
[`audit_real_market_feature_prefix_causality.py`](../../scripts/matrix/audit_real_market_feature_prefix_causality.py).
The full current-tree GH200 gate for the source snapshot is logged at
[`full_ci_gate_final.log`](../../outputs/data_refresh_20261017/full_ci_gate_final.log):
Ruff format, Ruff lint, full-tree Mypy, and 1,731 tests passed. The gate result
does not turn these bounded data checks into strategy evidence.

## Validity boundary and next action

The Jan 1–Jul 3, 2026 test interval was previously inspected. Jul 4–Oct 16 is
only a candidate fresh segment if the exposure record confirms it was not used
for model or human selection and the evaluation protocol is frozen in advance.
No new-data campaign should be used for thesis selection/publication until the
data-use source is accepted, the broker calendar/timezone interpretation is
reviewed, the exact campaign execution schedule is reconciled, and the test
exposure plan is frozen. Any later training result remains exploratory until
its training and evaluation gates pass; this report contains no return,
survival, or profitability result.
