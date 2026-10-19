# PO3 / IFVG / Idea 2 decisions and provenance

**Status:** decision baseline recorded before source changes for Idea 1_2.<br>
**Captured:** 2026-10-13, Europe/Helsinki.<br>
**Checkout:** `main`, HEAD `b00b5e85622ec1410c8a11cc063cd52c87d3976d`.<br>
**Plan:** `/scratch/work/nguyenl37/Idea 1_2.md`, SHA-256
`dbc81dc28f63630e065a30fbf27c75d28562ac878728715b5fc48a588f2826ac`.<br>
**Companion thesis alignment:** `/scratch/work/nguyenl37/Thesis_Alignment_PO3_IFVG.md`,
SHA-256 `30eab7378e929633bb7e18effe3c6ff224e38be9b7e717dd8df1a18616f1debb`.

The thesis author's approved behavioral decisions are D1-D10 in the linked
plan. D11-D12 are implementation rules required to make those decisions
causal and auditable. This record freezes their current interpretation for
implementation; it does not claim they have already been implemented.

## Frozen behavior contracts

| ID | Decision | Executable interpretation |
| --- | --- | --- |
| D1 | Reclaim ends manipulation. | First closed candle reclaiming the exact swept reference; the sweep candle may also reclaim if its close qualifies. Long: `close > swept_low`; short: `close < swept_high`. |
| D2 | Distribution starts at confirmed IFVG. | Reclaim and IFVG confirmation are distinct events; a waiting state between them is valid. |
| D3 | IFVG closes through the far zone edge. | Strict close beyond the existing opposing FVG far edge; zero additional overshoot. Validate FVG orientation and side inequalities. |
| D4 | Sweep reference may be M5/M15/H1. | Reference pivot and its confirmation must be observable before the sweep. M1 may trigger a sweep but an M1-only pivot cannot qualify. |
| D5 | Main entry mode is `confirmed_first_eligible`. | IFVG signal is known at confirmation close; submit after that close and fill at the first later executable event. Retest and preconfirmation modes are separately named variants. |
| D6 | Idea 2 uses the most recent completed labeled session. | Midpoint is `(previous_session_high + previous_session_low)/2`, selecting the most recent completed session among the existing labels and excluding the active unfinished session. Use the configured label calendar/timezone and record its boundaries and DST behavior. |
| D7 | Retest is optional. | The optional retest variant accepts a later touch anywhere in the confirmed IFVG zone; no mandatory 50% depth. |
| D8 | Stop floor remains 10 US100 points and 0.5 short-window ATR. | Effective minimum is `max(10 points, 0.5 * ATR_short)` with symbol scaling and units explicit. |
| D9 | Structural TP preference remains 1.5R–5R. | Log nearest admissible fallback if preferred references are unavailable; fallback must be at least 1R. Preserve multi-TP fractions. |
| D10 | Do not invent distribution termination. | Preserve current re-arm/reset/session behavior; characterize and document it. |
| D11 | Idea 2 bias applies to each proposed new entry. | On the last closed decision candle, long requires `close < previous_session_midpoint`, short requires `close > midpoint`; equality/missing session blocks with a reason. Existing positions are not retroactively canceled. |
| D12 | Confirmation and execution are distinct. | Confirmation at close `t` may produce a signal at `t`; no fill can be backdated into bar `t`. Persist signal bar, submission time, and fill bar/time separately. |

The approved state sequence is `IDLE → SWEEP/MANIPULATION →
RECLAIMED_WAITING_FOR_IFVG → DISTRIBUTION_CONFIRMED`. Setup IDs and timestamps
must tie the swept reference, reclaim, FVG and IFVG confirmation together.
Legacy phase behavior must remain selectable and separately identified for
reproduction; semantic changes require feature/cache versioning.

## Config and runtime provenance at the decision baseline

The hashes below are the pre-source-change decision baseline. The two strategy
overlays have since opted into the revised HTF reference mode; current hashes
are recorded in the WP9 update. `default.yaml` and `experiments.yaml` already
had unrelated/in-progress working-tree edits at baseline, which are preserved.
The frozen 18-variant campaign YAML is not modified by this Idea work.

### Variant and config migration map

| Identity | Reference / phase / entry semantics | Idea 2 gate | Run identity |
| --- | --- | --- | --- |
| Existing legacy results | `m1_legacy` + `legacy`; preserve original effective retest/gate settings | Off | Historical namespaces listed below stay unchanged. |
| Revised Idea 1 | `htf_confirmed` + `reclaim_then_ifvg` + `confirmed_first_eligible`; M1 PO3 may confirm from configured M1/HTF IFVGs | Off | `strategy.variant_id=po3_confirmed_ifvg_htf_mtf_v1`; overlay `config/2_po3_ifvg.yaml`. |
| Revised Idea 2 | `htf_confirmed` + `reclaim_then_ifvg` + `confirmed_first_eligible`; M1 PO3 may confirm from configured M1/HTF IFVGs | Latest completed labeled session midpoint | `strategy.variant_id=distribution_prev_session_pd_mtf_v1`; overlay `config/3_distribution.yaml`. |
| Optional ablations | `confirmed_retest` or `preconfirmation_exploratory` as separately selected modes | Per predeclared protocol; off for Idea 1 | Require a new distinct config/hash/label before any run; none is added to the frozen 18-run registry here. |

Migration is overlay-based: the default/legacy behavior remains the baseline,
the two Idea overlays select the revised semantics, and the config content
hash plus `strategy.variant_id` distinguishes them. Existing result folders
are never relabeled or reused as evidence for these variants.

| File | Current SHA-256 | HEAD SHA-256 / note |
| --- | --- | --- |
| `config/2_po3_ifvg.yaml` | `f9d937e5363c22cfd222d03deca8808cab1b6ea8b0c22d2ff706b0cdd7ea4252` | same |
| `config/3_distribution.yaml` | `0c338fa4b2964a09b9059f8de66d7853cec52a4970fec6d7466cedfb6826fed6` | same |
| `quant_rl/config/default.yaml` | `af731e0aaf08efc979338ebf3e89bfc74362f699a5116dfc79d9a62566386fd0` | `4cc2369a32a1696d0a70b15bed74d8d3f476ea7e4d021b6bd716b467eb8d86bd` |
| `config/experiments.yaml` | `ae5d70e9aa3231f42ae43c1dcf8962f99741cf60f9fc5c03def287b060288b6b` | `332ac0a3c22da374eb086679df17e4982d5844ffdce8b06d9f1fdf6dedbcfec6` |
| `config/campaigns/18_variant_seed42_20m.yaml` | `d5f11af1b2ab604dbac0c1020bf2c544ddb16ed9632ffad3bb704aa410101a83` | absent at HEAD |

Current configured data is M1 US100.cash/US500.cash in `Etc/GMT-3`, split at
train end 2025-12-31 / test start 2026-01-01; the active replacement market
files end 2026-10-16. The full 2026 report interval is exposed development
data and is not a fresh final holdout. Historical decisions below remain tied
to the exact source/cache fingerprints recorded when they were made.

Current effective/pre-change settings and known contradictions to resolve:

- PO3 and distribution overlays use strategy actions, PnL reward, `Etc/GMT-3`,
  and `swing_period: 5`; the 18-variant registry uses PPO/TCN and full PO3
  features for strategy/action variants. Do not silently add Idea variants to
  the frozen 18-key campaign.
- `IFVGConfig.close_through_threshold` currently defaults to `0.5`; the
  overlays set `strategy.ifvg.require_price_retest: true`, while
  `env.arm_requires_retest` defaults false and `strategy.entry.enforce_gate`
  defaults false. Effective wiring must be traced, not inferred from one key.
- PO3 state currently ends manipulation at a break of its running extreme and
  starts distribution there. `build_distribution_after_ifvg` makes the
  confirmation bar unavailable to its post-confirmation-ready flag. These
  legacy semantics need a versioned revised path.
- Default `risk.min_sl_points` is 10.0 and `risk.min_sl_atr_mult` is 0.5;
  `env.sl_buffer_pts` is 0.0 while strategy risk overlays set
  `strategy.risk.sl_buffer_pts` to 1.0. The effective value and any revised
  per-variant override must be logged.
- Session timezone is `Etc/GMT-3`; existing session labels include Asia,
  London, and NY. The author selected the **most recent completed labeled
  session** for D6 on 2026-10-13. Confirm exact existing session-boundary and
  timezone/DST behavior before WP5; do not substitute “previous trading day.”
- The exact ATR timeframe/period for `ATR_short` and same-candle
  sweep/reclaim event ordering still require implementation-level records and
  tests before producing results. Preserve candidate provenance.

WP9 implementation rule: revised overlays use confirmed M5/M15/H1 fractal
pivots from the existing timeframe-specific left/right windows. The pivot's
reference price is its wick extreme; its availability time is the end of the
HTF confirmation bar. An M1 bar cannot use a reference before that time. Keep
at most the eight most recently confirmed, unswept references per side and
timeframe, matching the existing retained swing capacity. For a candle that
crosses multiple references, select in this order: M5, M15, H1; smallest wick
overshoot; latest pivot time; lower reference price. Consume every crossed
reference, including too-deep breakouts. The selected reference timeframe,
pivot time, confirmation time and candidate count are emitted with the
feature row; the detector can return an audit table listing every crossed
alternative. Legacy M1 reference mode remains selectable. Revised overlays
set `features.po3_reference_mode: htf_confirmed`; the default remains
`m1_legacy`.

## Existing result identities to preserve

Do not relabel or pool these output namespaces as results of the revised
specification: `core-18x5`, `pilot_18_seed42_100k`,
`dev_a0_a1_seed42_100k_exposed_2026`,
`dev_a0_a1_seed42_100k_exportfix_exposed_2026`,
`resource_profile_a0_seed42_2m_exposed_20261013`,
`benchmark_18x1m_20261011`, and prior `preprod_*`, `ram-*`, `concurrency-*`,
and `gpu_matrix*` outputs. Their configuration/data provenance varies; use the
historical OOS exposure audit and archive disposition record before citing
them. Revised strategy variants need distinct names, config/feature hashes,
manifests and result labels.

## Implementation and verification update — 2026-10-13

WP2, WP1, WP3, WP4 and WP5 are implemented in the working tree. The revised
strategy modes preserve the `legacy` path; Idea overlays have explicit
`variant_id` values and do not alter the frozen 18-run campaign. Revised phase
features retain the confirmation-row pulse and distinguish later-ready rows.
Zero-latency strategy entries submit using the decision close and fill at the
next bar open, with separate signal, submission and fill bar/time fields.

The Idea 2 session range uses the already configured labels in
`quant_rl/config/default.yaml`: Asia 19:00–23:00, London 02:00–05:00, and NY
08:30–15:00, all in DST-aware `America/Chicago`. The most recently ended
well-formed session wins. A session is complete only when its expected M1
timestamps are contiguous through the final bar; an interior hole or truncated
tail is skipped. Its reference becomes visible at its close. Numeric labels
are `1=asian`, `2=london`, `3=ny`, and `0=no completed session`. Features
include high, low, midpoint, label code, and completion time. Only
`config/3_distribution.yaml` enables this block and the per-entry gate. Strict
inequalities make equality ineligible; `premium_block`, `discount_block`, and
`missing_previous_session` are separately logged. Open positions do not pass
through the new-entry gate. Accepted and rejected entries retain the session
range and decision-close audit fields.

WP4 keeps `_target_menu` authoritative and retains its preferred 1.5R–5R
window with nearest admissible ≥1R fallback and `no_structural_reference_within_rr_window`
reason. Open rows now expose each active target leg's executed price,
structural reference, and lot fraction; non-strategy execution continues to
distinguish the executed RR price from its nearest structural anchor.

WP6 adds `docs/decisions/fixtures/idea12_synthetic_fixtures.json`, a renderer
at `scripts/research/render_idea12_synthetic_figures.py`, and three standalone
SVGs in `docs/thesis/figures/`. These are clearly synthetic illustrations,
not replacements for unavailable original Figure 17/18 sources and not
historical or performance evidence. The separate thesis alignment note was
updated; the submitted manuscript and results remain untouched.

WP7 established feature-cache schema `v18-previous-session-pd`. The MTF IFVG
continuation below advances it to `v19-causal-mtf-ifvg`. The revised overlay
identities are `po3_confirmed_ifvg_htf_mtf_v1` and
`distribution_prev_session_pd_mtf_v1`. The environment stop buffer is the
effective source; the two revised overlays set the legacy strategy buffer key
to the same value for review clarity. The 10-point and 0.5×ATR floors remain
the configured effective minimum.

Scoped verification on x86_64 Python 3.12.14:

- `171 passed` across `test_htf_sweeps.py`, `test_po3_config.py`,
  `test_po3_state.py`, `test_build_pipeline.py`, `test_session_ohlc.py`,
  `test_strategy_env.py`, `test_distribution_reward.py`,
  `test_entry_state_machine.py`, `test_bundle_worker.py`,
  `test_entry_levels.py`, `test_idea12_fixtures.py`, and
  `test_train_rl_decomposition.py`.
- Repo-wide `ruff check .`, `ruff format --check .`, and `mypy .` pass; mypy
  checked 395 source files. The scoped mypy run over the Idea implementation
  and new session/fixture tests also passed.
- The configured `.venv` is ARM64 and cannot run here; `.venv-x86` was used.
  No training, Slurm job, campaign run, commit, or result generation occurred.

### E1–E21 disposition

| Evidence | Status | Evidence / scope |
| --- | --- | --- |
| E1 | VERIFIED_EXISTING | Candidate expiry stays at five bars; entry state tests cover expiry and causal arming (`tests/test_envs/test_entry_state_machine.py`). |
| E2 | FIXED | Revised PO3 uses confirmed M5/M15/H1 references; `tests/test_features/test_htf_sweeps.py`. |
| E3–E4 | FIXED | Reclaim and IFVG confirmation/distribution are separate events; `tests/test_features/test_po3_state.py`. |
| E5 | FIXED | Revised overlays use strict far-edge close and zero additional overshoot; `tests/test_features/test_po3_config.py`. |
| E6–E7 | FIXED | Explicit modes, retest alias validation, confirmation-close/next-open timing; strategy and entry-state tests. |
| E8–E9 | FIXED | Idea 2 adds per-entry latest-completed-session midpoint gate; `tests/test_features/test_session_ohlc.py`, `tests/test_envs/test_distribution_reward.py`. |
| E10–E11 | FIXED | IFVG far edge and observed through-zone extreme join auditable SL candidates; `tests/test_backtest/test_entry_levels.py`. |
| E12 | VERIFIED_EXISTING | Defaults remain 10 points and 0.5 ATR; revised overlays inherit them, confirmed by config inspection and risk tests. |
| E13 | FIXED | Stop-anchor decode is clipped to `[0,2]`; strategy action tests. |
| E14–E16 | VERIFIED_EXISTING | Structural target menu retained; preferred RR/fallback selection tested in `tests/test_backtest/test_entry_levels.py`. |
| E17 | FIXED | Executed TP leg prices and anchor reference are separately recorded; entry-level tests. |
| E18 | VERIFIED_EXISTING | Legacy RR helpers remain in place; strategy-action path still uses `entry_levels.py`. |
| E19 | FIXED | Signal, submission and fill bar/time are distinct; `tests/test_envs/test_entry_state_machine.py`. |
| E20 | FIXED | Revised overlays align both stop-buffer keys; environment `sl_buffer_pts` is effective and candidate rejection reasons are audited. |
| E21 | DEFERRED | Separate methodology/figure note and synthetic diagrams are ready; manuscript edits and recovery of original figure source remain deferred by plan boundary. |

The pre-training implementation work packages are complete, subject to review
of the synthetic diagrams. This does **not** mark the strategy or thesis READY
for final training or results.
The Idea plan excludes that campaign; the old thesis readiness status is
unchanged.

Current provenance after WP5/WP7 (SHA-256):

| File / identity | Current value |
| --- | --- |
| `config/2_po3_ifvg.yaml` | `69e3383e47725d765078e23ab8e8f71769579ac01564de0afe111d0eb1e33dd3` |
| `config/3_distribution.yaml` | `81841a20c02f6a5b45e7ef39b74e4e42a8e2e9acb3518865280fca7b4395f3bf` |
| `quant_rl/config/default.yaml` | `dfa0a5b0d195b40156eed52ad4db7ea20d174bb63753017929bfe17655a0d7e1` |
| `Thesis_Alignment_PO3_IFVG.md` | `656affd4e0a49f828b3b9af152cb021b9eecd257c93c712f04f9cbf261c608cb` |
| Feature schema after MTF IFVG continuation | `v19-causal-mtf-ifvg` |

The plan itself remains unchanged at SHA-256
`dbc81dc28f63630e065a30fbf27c75d28562ac878728715b5fc48a588f2826ac`.

## WP9 implementation update — 2026-10-13

`quant_rl/features/htf_sweeps.py` now detects M1 sweeps against confirmed
M5/M15/H1 fractal wick extremes. It excludes partial trailing HTF bars, makes
a pivot available at the end of its right-hand confirmation bar, and retains
up to eight unswept references per side/timeframe. The fixed selection order
and candidate audit fields above are frozen here. If requested, the detector
returns all crossed candidates; strategy feature builds persist that table at
`<feature-cache>.htf_sweeps.parquet`. Selected timeframe code, pivot time,
confirmation time and eligible candidate count are also in the feature rows.

`features.po3_reference_mode` preserves `m1_legacy` as the default and selects
`htf_confirmed` in `config/2_po3_ifvg.yaml` and `config/3_distribution.yaml`.
The feature cache schema is now `v17-htf-po3-ref`. Current config hashes after
this WP are:

| File | Current SHA-256 |
| --- | --- |
| `quant_rl/config/default.yaml` | `7ac03adf2a621839ac611b78f57c8787b3fb7ad7feb54c048d2a247f6112ea6f` |
| `config/2_po3_ifvg.yaml` | `91cc9495b709fc47ac338d624e173775f38acd764639ce6ce82f189b3ae9f4d6` |
| `config/3_distribution.yaml` | `da1d391be860049e1aab46b7bb14beb7b1e21e7f043b44c6a64c17d203fc73e4` |

Verification on the available x86_64 Python 3.12.14 environment: 35 scoped
tests passed across `test_htf_sweeps.py`, `test_po3_state.py`, and
`test_build_pipeline.py`; scoped Ruff check and format check passed. Tests
cover each reference timeframe and side, pending-reference rejection, stable
multi-reference choice with alternative logging, partial HTF bar exclusion,
prefix causality, a real resampled M5 confirmation, DST-aware resampling, and
feature-cache/audit-sidecar round-trip. The configured ARM64 `.venv` cannot run
on this x86_64 shell; `.venv-x86` was used. No training or Slurm work ran.

## MTF swing and IFVG alignment continuation — 2026-10-13

The feature builder computes local swing/structure features from resampled
M5/M15/H1 candles and aligns confirmed levels and BOS flags to the M1 timeline.
Pivot events appear only on their right-side confirmation candle; the alignment
shift exposes them at the next HTF boundary. Revised Idea overlays also use
confirmed M5/M15/H1 pivot wick references for M1 sweep detection. These are
local swings in each source timeframe, not M1 swings relabeled as HTF.

The revised M1 PO3 state now accepts IFVG confirmation pulses from M1 and every
timeframe listed under `features.ifvg_mtf.timeframes`. Idea overlays enable
M5/M15/H1; the builder also supports the resampler's M30/H4/D1 options when
explicitly configured. HTF IFVG pulses and active zones are shifted to the first
M1 row at/after source-bar completion. The detector receives the same strict
IFVG settings on each timeframe. The HTF FVG origin index is mapped back to
the M1 index before setup-age and sweep-origin checks. HTF retest flags remain
one-bar pulses at source-bar completion. M1 wins same-side,
same-row collisions; otherwise configured timeframe list order is the
tie-break. Direction-specific confirmation timeframe codes use minutes
(M1=1, M5=5, M15=15, M30=30, H1=60, H4=240, D1=1440); the selected code is
also emitted as `po3_ifvg_confirmation_tf_code` on a confirmed setup.

This does not combine independent MTF PO3 state machines into one state. It
keeps M1 as the PO3 setup clock and lets a causally completed IFVG from a
configured source timeframe complete that setup. Legacy defaults remain
unchanged unless `ifvg_mtf.enabled` is enabled. The feature cache version was
advanced to `v19-causal-mtf-ifvg`, and the Idea overlay IDs changed so new runs
cannot be confused with the previous M1-only identities.

Focused verification on x86_64 Python 3.12.14: 92 tests passed across
`test_po3_state.py`, `test_po3_ifvg_state.py`, `test_variants.py`,
`test_build_pipeline.py`, `test_htf_alignment.py`, `test_structure.py`,
`test_pivots_swings.py`, and `test_htf_sweeps.py`; scoped Ruff and Mypy passed.
Tests cover M5/M15/H1 swing alignment and causality, IFVG pulse completion,
collision priority, source-zone metadata, mapped M1 origin, and M1 PO3
confirmation from M5. This is implementation verification only; no training,
Slurm campaign, or strategy-performance claim was produced. The 18-run
campaign and thesis readiness status remain unchanged.
