# Replacement dataset and bounded causal checks — 2026-10-17

## Disposition

The supplied replacement data, rebuilt caches, and observed tick schedule have
passed their recorded bounded checks. The user identifies the source as the
FTMO Demo server and explicitly authorizes internal model training, considering
the data adequate for that purpose. The underlying liquidity provider and
account-specific export/publication terms are unknown; FTMO's current
[public Terms and Conditions](https://ftmo.com/en/terms-and-conditions/)
describe a liquidity-provider-powered feed but do not identify this account's
provider or establish thesis-publication rights. Treat this as user-attested
training use with unresolved source/publication disclosure, not independent
verification of those terms. The v23 cache rebuild passed for all 18 variant
bindings; fresh v23 fold-fit and real-market prefix audits are running. No
replacement-data training campaign or checkpoint exists, and P1/P2 are not
closed. Overall readiness remains **NOT READY**.
The provenance disposition is recorded in
[`provider_provenance_status_20261018.json`](replacement_data_20261017/provider_provenance_status_20261018.json).

## Dataset and cache identity

### FTMO MetaTrader timestamp basis — 2026-10-18

The user identifies the source as the FTMO Demo server. FTMO's public account
specification documents MetaTrader server time as GMT+2 plus daylight saving;
its 2025 and 2026 notices document the switch to GMT+3 with the US DST change
and back to GMT+2 in November. The exact account login/server and export path
for these CSVs remain unverified. The repository previously assigned every
naive timestamp a fixed `Etc/GMT-3` offset. For winter dates this made a server
wall time of 16:30 appear as 07:30 CT, while FTMO's GMT+2 clock implies 08:30
CT. Three of the four named feature profiles (17 of 18 variants) enable the
CT-based session feature block. The versioned correction and source references
are recorded in
[`ftmo_mt5_time_mapping_audit_20261018.json`](replacement_data_20261017/ftmo_mt5_time_mapping_audit_20261018.json).
The correction passed 58 focused tests plus full-tree format, lint, and Mypy;
the v23 feature caches have not yet been rebuilt. Do not use v22 session-feature
caches for a new screen.

The requested chronological split is training through 2025-12-31 inclusive and
one continuous test/report interval from 2026-01-01 through the last available
bar on 2026-10-16. The full interval is exposed exploratory/report-only evidence:
January–July outcomes were previously inspected, so no later subperiod is
classified as fresh or confirmatory. Source M1 files have 630,787 US100 rows and
628,929 US500 rows; both start at 2024-12-30 01:05 and end at 2026-10-16 23:49
in the configured broker clock. The full interval is present in the rebuilt M1,
M5, M10, M15, M30, H1, H4, and D1 caches and in all four resolved variant
feature caches. The test schedule ends on October 16 for both symbols (79,069
US100 and 79,068 US500 NY-session decisions); no 2026 test rows enter the 2025
training mask. The exact observed test timestamps are frozen in
[`decision_schedule_fingerprint_final.json`](replacement_data_20261017/decision_schedule_fingerprint_final.json).
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
from a feed/export omission. Nasdaq's official 2025 calendar lists early closes
on July 3, November 28, and December 24; December 2 and 4 are not scheduled
Nasdaq early-close days. Its official 2026 calendar likewise lists December 24,
not December 2 or 4, as the December early close ([Nasdaq 2025 calendar](https://www.nasdaq.com/docs/2025/01/06/2025holidayandtradinghours.pdf),
[Nasdaq market holiday schedule](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule)).
This rules out a scheduled Nasdaq holiday/early close as the explanation for
these dates. It does not establish the schedule of the unidentified US100.cash
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

Four full-range feature profiles were transactionally rebuilt and verified
against the replacement M1 inputs, then bound across all 18 campaign variants:
the requested baseline cache (1 variant; 630,787 rows, 194 columns), a
supplemental baseline-with-strategy-actions cache (5 variants; 630,787 rows,
438 columns), Idea 1 PO3/IFVG (6 variants; 630,787 rows, 532 columns), and
Idea 2 distribution (6 variants; 630,787 rows, 538 columns). The durable
[18-variant rebuild report](replacement_data_20261017/named_feature_cache_rebuild_18variant_20261017_job21020179.json)
records cache hashes, source M1 fingerprints, and each variant-to-cache
binding. The three requested named caches remain present; the supplemental
profile prevents five baseline strategy-action variants from falling back to
an untracked runtime build. The fresh rerun report
[`named_feature_cache_rebuild_18variant_20261017_job21020179_rerun1.json`](replacement_data_20261017/named_feature_cache_rebuild_18variant_20261017_job21020179_rerun1.json)
records all 18 variant bindings, the four rebuilt profiles, source hashes, and
cache identities at HEAD `132da3a`. The corresponding full-period bar/tick
rebuild report is
[`full_period_data_cache_rebuild_20261017_job21020179_rerun1.json`](replacement_data_20261017/full_period_data_cache_rebuild_20261017_job21020179_rerun1.json).
The presence of an M10 bar cache does not mean M10 is visible to each profile;
each profile follows its resolved feature-timeframe config.

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
The repository-level test-tail exposure inventory is generated by
[`audit_candidate_holdout_repository_exposure.py`](../../scripts/research/audit_candidate_holdout_repository_exposure.py).
The full current-tree GH200 gate for the source snapshot is logged at
[`full_ci_gate_final.log`](../../outputs/data_refresh_20261017/full_ci_gate_final.log):
Ruff format, Ruff lint, full-tree Mypy, and 1,731 tests passed. The gate result
does not turn these bounded data checks into strategy evidence.

## Validity boundary and next action

January–July 2026 outcomes were previously inspected. Treat the single
2026-01-01–2026-10-16 test/report interval as exposed exploratory/report-only
evidence; do not designate any later subperiod as fresh holdout or use it for
confirmatory claims. Internal training use is user-authorized. Before a screen
result can support selection, the current FTMO clock/session assumptions, exact
execution schedule, and evaluation protocol must pass their technical gates.
Unknown provider and publication terms must remain explicit in public reporting
and raw-data packaging. Any later training result remains exploratory until its
training and evaluation gates pass; this report contains no return, survival,
or profitability result.

The repository exposure inventory found that the pre-intake US100 and US500 M1
caches both ended at 2026-07-03 19:59 broker time; the full-range replacement
source files have filesystem modification times on 2026-10-17. It searched
pre-intake text artifacts in `outputs/`, `cache/`, and
`data/prebuilt_feature_cache/` for references to dates in the later portion of
the report interval. Some files matched date strings, but reviewed examples
are UTC artifact/log timestamps or unrelated date text, not evidence of policy
results on those market bars. The
raw match inventory is preserved in
[`candidate_holdout_exposure.json`](replacement_data_20261017/candidate_holdout_exposure.json).
An additional pre-intake archive screen covered 612 ZIP and 3 GZIP archives
with zero read errors. It found 397 date references from the later part of the
report interval across 15 archives. Path triage classified the recorded samples as source/configuration,
source manifests/checkpoint-migration maps, or two resource-telemetry snapshots;
no matched member path was a direct market-data, evaluation-report, or strategy-
metric artifact. This narrows what the archive text scan found, but does not
prove the tail was not used: dates inside source/config files do not establish
how their outputs were used, and binary/oversized members, external systems,
and human exposure remain outside the audit. See the machine report
[`candidate_holdout_archive_exposure_20261017_triaged.json`](replacement_data_20261017/candidate_holdout_archive_exposure_20261017_triaged.json)
and the full gate log
[`candidate_holdout_archive_audit_gate_triaged.log`](../../../outputs/data_refresh_20261017/candidate_holdout_archive_audit_gate_triaged.log).
These bounded repository scans cannot establish external or undocumented
human exposure. They do not change the assigned role of the full 2026 interval:
exposed exploratory/report-only evidence. No replacement-data training or
policy evaluation has run.
