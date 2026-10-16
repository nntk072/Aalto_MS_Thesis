# G1 data and feature evidence archive — 2026-10-16

These JSON reports and their audit utilities were copied byte-for-byte from
`outputs/agent_state/` into the tracked thesis evidence tree. Historical audit
source snapshots use the `.py.txt` suffix so repository formatters do not
rewrite them; the suffix can be removed to execute a copy. The raw M1 and tick
inputs were not copied. Report hashes are unchanged by the archive copy.

## Evidence and scope

| Artifact | Result | Evidence boundary |
|---|---|---|
| `g1_raw_m1_quality_audit_20261016.json` | US100: 528,760 rows; US500: 527,139; both run 2024-12-30–2026-07-03. No invalid timestamps, ordering breaks, duplicate timestamps, invalid numeric/OHLC rows, negative tick volume, volume, or spread. Hashes match the campaign manifest. Train/test row counts are 351,550/177,210 for US100 and 350,098/177,041 for US500. | Data-level file and calendar scan. A gap count above twice the modal one-minute interval is not a missing-bar claim; market sessions and holidays were not checked against an authoritative schedule. |
| `g1_tick_quality_scan_20261016.json` | US100: 49,160,484 rows, 50,146 invalid/missing bid/ask rows, 62,266 adjacent duplicate timestamps. US500: 31,134,588 rows, 136,559 invalid/missing bid/ask rows, 27,814 adjacent duplicate timestamps. Neither file had an ordering break, crossed quote, or locked quote. Both recorded file hashes match the campaign manifest. | Streamed scan of bid/ask and timestamps. It did not validate feed identity, broker clock/DST truth, every optional tick column, or tick-cache equality. Missing bid/ask values are forward-filled by `build_tick_book`; leading rows without a prior quote are dropped. Duplicate timestamps are retained. Stale-quote age and trade-level impact still need review. |
| `g1_feature_cache_raw_parity_20261016.json` | Four profiles matched fresh raw recomputation over all 528,760 rows; index and columns matched, maximum delta 0, no value or NaN-mask mismatches, and feature sidecars matched their expected hashes. | This parity run used source snapshot `8a98f7792f83ce4bbe5cb84f4a2137657d791c02`. `quant_rl/features/build.py` has changed since that snapshot, so rerun against the final frozen source before G1 closes. |
| `g1_feature_cache_raw_parity_current_source_2026-10-16.json` | **Fresh current-source run:** all four representative feature profiles matched their existing caches across 528,760 rows. Index/columns matched; value and NaN-mask mismatches were 0; maximum delta was 0; all four content-hash sidecars matched. The four cache-file SHA-256 values, 32 source/config hashes, two M1 input hashes, and script hash were rechecked after the run with no mismatches. | The audit read cache parquet/sidecars and built raw features without a cache output path; it did not modify training caches. It ran on job `20990395`, source head `2fbe0a59f4aef88690214520ec26cdc0c2fc1447`, campaign fingerprint `91d2f05c4c13e67753a47b88a1c900f4887ced66dad51980a018eb09465dd10e`. It covers four representative configs, not an independent invocation for each of the 18 variants, and is not strategy-performance or complete G1 evidence. |
| `../../g1_campaign_feature_cache_resolution_2026-10-16.json` | **Fresh current-source run:** all 18 saved active-campaign configs resolved to one of the four raw-to-cache parity profiles. Every resolved cache exists, its sidecar matches the expected feature hash, and its file SHA-256 matches the corresponding parity-audited cache. Counts: baseline 1, management 5, Idea 1 PO3/IFVG 6, Idea 2 distribution 6. | Confirms the active 18-run config-to-cache mapping for the recorded campaign fingerprint and source inputs. It does not add 18 independent raw-feature builds; it connects all 18 actual cache resolutions to the four full raw/cache comparisons above. Repeat if the final source, configs, inputs, or caches change. |
| `g1_feature_prefix_current_source_2026-10-16.json` | **Fresh current-source run:** 12/12 cutoff/profile comparisons passed with zero changed features and zero maximum delta. Four feature groups cover all 18 resolved variants. All 32 source/config hashes, both M1 input hashes, and the audit-script hash matched the files at verification. | Raw feature builds (`force=True`, no feature cache). This establishes sampled prefix invariance at the three listed cutoffs, not every leakage path, cache parity, fold-local fitting, or strategy validity. It ran on job `20990395`; the source snapshot is identified by the recorded hashes rather than a recorded Git SHA. Rerun after any relevant feature/config/data change. |
| `g1_real_market_feature_prefix_20261016_main.json` | Historical report: 12/12 cutoffs passed across four profiles. | Used source snapshot `8a98f7792f83ce4bbe5cb84f4a2137657d791c02`; retain for provenance, but use the fresh current-source report above for current prefix evidence. |
| `g1_full_split_budget_math_20261016.json` | At 20M aggregate steps / 64 environments, arithmetic allows 312,500 transitions per environment, or 3.14 full-split episode lengths if no early resets occur. | Budget arithmetic from a previous campaign snapshot, not observed full-year episode coverage. Final evaluation and episode/reset telemetry remain required. |

The four raw input SHA-256 values recorded by the M1/tick reports are present
unchanged in the active recovery campaign manifest. The fresh prefix report's
two M1 hashes also match that manifest. Thus the M1/tick scans and prefix
comparison used the same raw inputs as the active campaign. The active recovery
campaign has a different run fingerprint; none of these reports is an outcome
for that campaign.

## Remaining G1 blockers

- Vendor/feed identity, export provenance, and data-use/publication terms are
  not established by the repository or file headers. The user confirmed on
  2026-10-16 that no provider/terms record is available; obtain authoritative
  documentation or leave the provenance/publication limitation explicit.
- The configured `Etc/GMT-3` offset is an assumption. The source broker clock
  and its daylight-saving schedule are unverified.
- No authoritative trading calendar has adjudicated whether long M1 gaps are
  expected closures or missing observations.
- Tick missing-quote forward-fill behavior has not been assessed for quote age
  or its effect on actual fills; exact duplicate timestamp selection needs
  trade-level impact review.
- Current-source raw feature-prefix invariance passed at the listed cutoffs;
  repeat it if feature/config/data sources change. Current-source raw-to-cache
  parity passed for four representative profiles, and all 18 active saved
  configs now map to those profiles with matching cache bytes and sidecars.
  Repeat both checks if final source/config/data/cache identities change.
  Existing synthetic tests do not replace these real-data checks.
- Purged-fold/fold-local normalization and full-split episode coverage require
  separate evidence; training is still active.
- Focused current-source unit/contract test results are recorded in
  `../../g1_contract_tests_2026-10-16.json`. They passed, but do not replace the
  real-data prefix/cache rerun.

The companion bounded M1 scan for the current recovery fingerprint is
`../../data_quality_audit_2026-10-16.json`, generated by
`../../audit_m1_data_quality_2026-10-16.py`. It confirms current-manifest M1 hashes
and basic OHLC integrity but intentionally leaves the blockers above open.
