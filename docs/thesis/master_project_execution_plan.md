# Aalto Thesis Trading System — Master Project Execution Plan

**Status:** ACTIVE — empirical results package is not yet frozen
**Revision:** 2026-10-16 — single-agent horizontal-execution profile
**Authority:** This file is the single operational roadmap for thesis-critical experiment, correctness, infrastructure, evaluation, evidence, review, archival, manuscript, and submission work.
**Primary objective:** Complete the P0–P9 evidence package, then use its frozen artifacts to finish and submit a defensible thesis. The active user goal includes P10/P11 manuscript and submission work; those stages remain downstream of the evidence gates.
**Deadline context:** Freeze results early enough to support the supervisor-PDF target of **2026-11-09** and faculty submission deadline of **2026-11-23**. Dates remain targets; evidence validity takes precedence.
**Execution principle:** Keep one long-running Codex agent productive horizontally across independent lanes; parallelize independent work aggressively and serialize only unresolved scientific dependencies, shared-control mutations, or resource contention that would invalidate a measurement.

**Latest live state (2026-10-17 03:36 Helsinki):** the contaminated source
campaign remains preserved and no training campaign is running. Job `20990395`
is terminal `FAILED`; MI210 job `21007645` completed its earlier bounded P1
audits and transactional cache rebuild. H200 job `21004765` on `gpu59` (1 H200,
48 CPUs, 512 GiB) completed at 03:35 with Slurm state `COMPLETED`, exit `0:0`,
after 36:56. Its final CUDA smoke, Ruff format/lint, full Mypy, and 1,594-test
gate passed (`outputs/gpu_ci_gates/p2_postrepair_20261017_job21004765/gpu_ci_final3.log`).
Stopping the previously authorized recovery supervisor also ended its child
H200 `salloc` earlier than requested; this was an unintended loss of the H200
reservation. No `scancel` was run. Following fresh per-node inventory
and an immediate forecast, approved MI210 job `21012389` started on `gpuamd1`
(MI210 x1, 8 CPUs, 128 GiB; through 09:26). ROCm 6.3.1/PyTorch 2.7.1+rocm6.3
GPU smoke and all 1,594 tests passed
(`outputs/gpu_ci_gates/p1_mi210_20261017_job21012389/mi210_full_validation.log`).
Only the MI210 allocation remains RUNNING, with its validation step ended and
no trainer active. Q5 now has a tested, opt-in primal-dual PPO implementation;
its Q4 parent remains unresolved, so no Q-stage run is launchable. No new
18-variant campaign was launched. See
[`live_status_2026-10-17_0336.md`](live_status_2026-10-17_0336.md).

**Earlier scheduler snapshot (2026-10-17 00:20 Helsinki):** job `20990395` was
terminal `FAILED`; job `21004765` was pending on `gpu61` because the requested
node was down, drained, rebooting, or reserved, with a forecast at Oct 18
19:30. The `thesis-causal-repair` tmux session was preserved. See
[`live_status_2026-10-17_0020.md`](live_status_2026-10-17_0020.md).

**P1 source repair update (2026-10-17 00:10 Helsinki; gap-policy measurement
still pending):** the causal repair exposes only a clock-completed session. A
follow-up source review found that the old default window `16:30–23:00 CT` was
inconsistent with the configured NY window `08:30–15:00 CT`; in fixed UTC+3
data time these identify different sessions. The feature builder now passes
the configured NY window and cache identity is
`v22-causal-configured-ny-session`. The M1 audit derives its session window
from the default feature config and records its hash. Missing-bar handling
remains provisional: observed rows are used, while the audit reports row counts
and the nominal final bar without asserting a market calendar.

**P1 post-repair validation update (2026-10-17 02:21 Helsinki):** the MI210
node ran the normalized M1 data audit (`data_quality_audit_2026-10-16_v4.json`),
the fold-poison audit (`g1_real_market_fold_fit_scope_post_array_opt_2026-10-17.json`),
and the full four-profile/12-cutoff prefix audit
(`g1_feature_prefix_post_repair_2026-10-17.json`). The poison check passed for
four representative profiles on one early purged fold (41,920 train and 6,000
validation rows each): validation poisoning left all training features exactly
unchanged and changed validation features. Prefix invariance passed all 12
sampled comparisons across the four profiles. All three named caches were
transactionally rebuilt with report
[`prebuilt_cache_rebuild_2026-10-17_mi210.json`](prebuilt_cache_rebuild_2026-10-17_mi210.json);
previous files/sidecars were copied to
`data/prebuilt_feature_cache/backups/pre-repair_2026-10-17_mi210_21007645/`.
These bounded checks do not establish exhaustive causality, market-calendar/DST
validity, current strategy quality, or confirmatory evidence. The fresh screen
remains gated on the remaining reviews, post-boundary CI, and an available
approved allocation.

**P1 audit-preservation update (2026-10-17 00:05 Helsinki):** the fold-poison
rerun now writes a new post-repair report instead of overwriting the immutable
failed report, and its source diagnosis no longer asserts the old defect as a
current result. The M1 completeness audit also refuses to overwrite its v2
report. Both audits are prepared but unrun; the causal tests and cache rebuild
remain open. See [`live_status_2026-10-17_0005.md`](live_status_2026-10-17_0005.md).

**P1 configured-session correction (2026-10-17 00:10 Helsinki):**
`prev_ny_high_low()` previously defaulted to `16:30–23:00 CT`, unlike the
configured CT NY window `08:30–15:00`. The feature builder now passes the
configured boundaries; the direct function defaults match them; the feature
cache version is `v22-causal-configured-ny-session`; and the M1 audit reads its
window from the feature config. Existing named caches remain preserved but are
stale and must not be used until P1 validation and the transactional rebuild.

**P1 campaign-config provenance check (2026-10-17 00:14 Helsinki):** all 18
saved worker configs in the failed campaign record the configured NY window
`08:30–15:00` CT. The M1 report builder now checks each worker's saved config
against its manifest hash and the default feature-session config, and records
the per-run config hashes. This check is authored but has not been executed.

---

**P1 cache-rebuild provenance hardening (2026-10-17 00:20 Helsinki):** the
transactional cache builder now records each full resolved merged config and
its digest, hashes the selected YAML inputs and loaded in-repository `quant_rl`
modules, and records runtime/platform/host/Slurm and dependency-lock metadata.
It also records dirty-worktree status and a hash of the tracked working diff.
Focused provenance regressions were added. Neither tests nor the cache rebuild
have run; all three named caches remain stale until GPU-node validation and the
transactional rebuild succeed. The M1 audit report builder's open-item text was
also corrected: the recovery campaign is terminal, but the current attempt
lacks full-split episode telemetry for `ladder_a0_flat`.

**P1 raw-input cache isolation (2026-10-17 00:23 Helsinki):** cache rebuilding
now forces raw-M1 ingestion into a temporary intermediate-bar cache, avoiding
stale shared parquet inputs and preserving the project's reusable bar caches.
A focused regression was added; it has not run.

## 1. How to use this roadmap

This roadmap is self-contained. Every active task must map to a work package,
entry condition, acceptance criterion, or explicit extension below. Supporting
logs, manifests, tests, checkpoints, data hashes, reports, review records, and
results artifacts are evidence; they do not create independent work queues.

For every task:

1. identify the work package and its dependency state;
2. state whether the task is scientific, correctness, infrastructure, training,
   evaluation, evidence, review, or optional extension work;
3. preserve immutable run evidence and active experiment identities;
4. make the smallest change needed to answer the registered question;
5. measure the result before broadening scope;
6. record failures and inactive runs rather than erasing them;
7. advance only when the package exit criterion is met.

Do not create another project-wide roadmap. Update this file when an author
decision materially changes dependencies, scope, or acceptance criteria.

---

## 2. Project completion states

### 2.1 `CORE_SYSTEM_EVIDENCE_VALID`

The software and research pipeline are valid at the scope used by the thesis
when all of the following hold:

- the reported data path is causal and fit only on authorized training data;
- observation/action schemas are versioned and checkpoint-compatible;
- entry, SL, TP, partial-fill, cost, reward, guardrail, and termination
  semantics are verified on the actual reported path;
- the reported policy checkpoints load and evaluate deterministically;
- position-level metrics reconstruct from raw fills/trades/equity;
- every reported run has source/config/data/cache/checkpoint provenance;
- exploratory, selection, robustness, and final-report roles are labelled
  correctly.

### 2.2 `THESIS_EVIDENCE_COMPLETE`

`CORE_SYSTEM_EVIDENCE_VALID` plus enough empirical evidence to answer the
claims actually retained in the thesis:

- one valid matched experiment addresses the structural-overlay question;
- the relevant mechanism comparisons are either validly estimated or explicitly
  described as system-level/confounded comparisons;
- paired seed robustness and transaction-cost sensitivity are reported for the
  thesis-level comparison, unless an unavoidable failure is disclosed as a
  limitation rather than hidden;
- account survival, economic outcomes, activity, behavior, uncertainty, and
  failures are reported together;
- the exposed 2026 interval is never represented as pristine confirmation;
- independent review findings are resolved, bounded, or explicitly disclosed.

A negative or inconclusive result can satisfy this state. Profitability is not
a definition of thesis completion.

### 2.3 `RESULTS_PACKAGE_COMPLETE`

This closes the P0–P9 results-evidence phase; the active goal continues through
P10/P11. It requires
`THESIS_EVIDENCE_COMPLETE` plus a complete, machine-indexed handoff package:

- every thesis-relevant run has its resolved config, command, source/data/cache identity,
  stdout/stderr/Slurm/controller logs, training traces, checkpoint(s), hashes, and load checks;
- every valid run has deterministic raw evaluation outputs, including equity, positions, fills,
  breaches, actions/behavior summaries, reward components where available, and cost-stress outputs
  where applicable;
- every comparison has seed-level tables plus aggregate statistics/uncertainty appropriate to its
  evidence role;
- all final figures and table-source files are generated from frozen machine-readable artifacts;
- every failed, incomplete, inactive, or zero-trade attempt is retained with an explicit disposition;
- every empirical finding has a result card linking the claim boundary to exact evidence paths;
- a global result manifest maps RQ/question -> run IDs -> checkpoints -> raw outputs -> statistics ->
  figure/table artifacts -> allowed interpretation;
- the archive can be understood without tmux history, shell scrollback, or reconstructing human memory.

**P9 closes the results-evidence phase, not the active user goal.** The agent then continues through
P10/P11 to assemble the evidence-backed manuscript and submission artifacts.

### 2.4 `THESIS_PACKAGE_READY`

`RESULTS_PACKAGE_COMPLETE` plus downstream thesis-authoring work:

- Chapter 8 contains real, traceable results rather than placeholders;
- Discussion, Conclusions, Introduction, and Abstract use exactly the same
  evidence-backed answers;
- methods, data, configs, costs, account limits, model-selection rules, and
  experimental roles are internally consistent;
- all figures and tables are generated from frozen evidence;
- administrative metadata, acknowledgements, licence, data availability, AI
  disclosure, references, appendices, and reproducibility notes are complete;
- the final PDF builds cleanly and has been visually inspected.

### 2.5 `SUBMISSION_COMPLETE`

The supervisor-review cycle is closed, the final submission package is frozen,
and the faculty version has been submitted with its evidence archive retained.

### 2.6 `RESEARCH_EXTENSION_COMPLETE`

The complete Q0–Q7 program has reached a declared stopping point with
reproducible control/treatment decisions for every stage. By user direction on
2026-10-16, Q0–Q7 are compulsory parts of the overall Codex mission. They remain
downstream of thesis submission by default and are not prerequisites for
`RESULTS_PACKAGE_COMPLETE` or `SUBMISSION_COMPLETE`; a Q-stage result enters the
thesis only if it independently passes its evidence gates and is deliberately
promoted before the thesis evidence freeze.

---

## 3. Thesis question hierarchy and scope control

The manuscript currently contains four empirical questions. They do not all
have equal priority for the November deadline.

### RQ1 — structural restriction: thesis-critical

**Question:** Does the structural overlay improve net risk-adjusted performance
and account survival relative to a control with matched reward, action space,
architecture, costs, data, training budget, and evaluation?

This is the minimum empirical intervention required for the central causal
claim. A complete-system comparison with different rewards or action spaces is
useful context but cannot substitute for this matched comparison.

### RQ2 — encoder dependence: extension

A multi-encoder comparison is valuable only if the same matched comparison can
be repeated under comparable encoder families. It is **not allowed to delay the
core thesis**. If it is not completed with valid evidence, the thesis should say
`not tested` rather than manufacture an answer.

For the thesis-critical line, freeze one encoder. The current TCN path is the
natural default unless a pre-experiment author decision changes it.

### RQ3 — management mechanisms: important but attribution-sensitive

The SL, TP, multi-target, and lifecycle ladder may support mechanism claims only
where compared runs share the required interface and controls. The 18-variant
screen is valuable system-level evidence and a selection tool, but a rung whose
reward, policy interface, eligibility, or action meaning changes cannot be
presented as an isolated causal effect.

The thesis may answer RQ3 partially: report valid paired mechanism effects and
label the remaining ladder outcomes as complete-system comparisons.

### RQ4 — robustness and generalization: split into achievable components

Required now:

- paired seed variability;
- cost sensitivity;
- dependence-aware uncertainty;
- complete model-search/trial accounting;
- honest reporting of failures and inactive runs.

A genuinely unseen period can support confirmatory generalization only if new,
uninspected data are acquired and frozen before evaluation. If that data do not
exist before submission, do **not** block the whole thesis; state that the 2026
interval is exposed exploratory evidence and that pristine temporal
confirmation remains future work.

### Scope discipline

For the thesis-critical path:

- primary RL algorithm: **PPO**;
- primary encoder: **one frozen encoder** (TCN by default);
- primary structural novelty: **PO3/IFVG-based structural eligibility/context**;
- canonical scientific simulator: **TradingEnv deterministic evaluation**;
- one matched RQ1 treatment/control is more important than adding model families.

Do not put SAC, VAE/narrative embeddings, broad GRU/Transformer tournaments,
online replay/adaptation, PPO-Lagrangian/CPO, live-money deployment, or a new
reward family onto the thesis critical path. Some appear later as controlled
extensions.

---

## 4. Frozen technical and scientific contracts

These are invariants for all thesis evidence unless the author changes a rule
**before** the affected experiment starts and records a new lineage.

### 4.1 Strategy and entry semantics

- PO3/IFVG events are causal; a reference, pivot, FVG/IFVG, or session value is
  usable only after its confirmation/completion time.
- Reclaim and IFVG confirmation are distinct events; the waiting interval is a
  valid state.
- The main entry mode is post-confirmation first eligible execution. A signal
  confirmed on bar `t` cannot be filled retroactively inside bar `t`.
- Idea 2/distribution applies its directional/session bias to each proposed new
  entry, not to an already-open position.
- The inversion/confirmation bar itself is not an entry fill bar under the
  post-confirmation mode.
- Retest is optional unless a separately named treatment explicitly requires it.
- M1 remains the setup clock; causally completed configured higher-timeframe
  references/IFVG events may complete or contextualize the M1 setup rather than
  creating undocumented independent state machines.
- Missing mandatory eligibility inputs fail closed in strict research mode.

### 4.2 Stop-loss and stop-candidate contract

- Long structural trailing references confirmed swing **lows**.
- Short structural trailing references confirmed swing **highs**.
- Reference confirmation must be observable before use.
- Stop updates take effect no earlier than the next executable event.
- Stops may tighten but must not loosen.
- Initial risk remains immutable for RR/accounting quantities that depend on it.
- Gap-through and ambiguous-bar behavior must follow the declared execution
  convention and remain the same between compared systems.
- For non-legacy PO3/IFVG and distribution entries, the entry-time stop menu must
  preserve both IFVG-zone protective candidates when causally available:
  **(A)** the confirmed IFVG far edge and **(B)** the most adverse observed candle
  excursion through the confirmed IFVG zone before the eligible entry. The
  through-zone extreme is setup-scoped; it must never use a candle that occurs
  after the entry decision/fill, and it must be mirrored correctly for long/short.
- The current implementation already exposes the through-zone candidate as
  `ifvg_through_zone_extreme`; treat this as a **preserve-and-verify contract**, not
  as missing functionality. Final validation must cover long and short geometry,
  minimum-distance rejection, provenance, and candidate selection.
- Higher-timeframe swing stops must be a **set of recent confirmed swings**, not
  only one `last_swing_low/high` value per timeframe. For every configured HTF
  used by the final experiment (currently M5/M15/H1), expose distinct recent
  confirmed swing lows for longs and swing highs for shorts. Freeze the exact
  lookback/maximum candidate count before P7. The frozen implementation uses the
  three most recent distinct accepted swings per side and timeframe that remain
  unswept. A level is stale once swept or displaced from that three-event window
  by newer distinct confirmed levels; age is recorded in completed source-timeframe
  bars, with no arbitrary wall-clock expiry. The opt-in final profile is
  `config/features/pd_context_mtf_swing_candidates.yaml`; the default count remains
  one to preserve the current screening campaign semantics. Each candidate records
  timeframe, pivot/reference time, completed-bar confirmation time, price, age,
  and rejection/selection reason.
- Candidate construction is causal and fail-closed: filter wrong-side levels,
  minimum-distance violations, stale/unconfirmed references and duplicates at
  instrument precision (`costs.point_size`). HTF confirmation time is the close of
  the source bar, and the existing one-bar HTF alignment must not expose it earlier.
  Do not synthesize a missing swing merely to fill the menu.

### 4.3 Target and partial-exit contract

- Long active targets must remain above entry/current executable geometry and in
  strictly increasing slot order; short targets use the mirrored order.
- Disabled targets remain disabled; hit targets never reactivate.
- Structural target references carry provenance: identity, timeframe, original
  price, adjusted/executed price, confirmation time, and selection/fallback reason.
- If only one or two valid structural references exist, do not invent a third.
- Duplicate levels are removed/handled at instrument precision.
- Partial-exit fractions are anchored to the initial position size; the final
  close absorbs broker-rounding residual.
- Quantity, commissions, realized PnL, reward attribution, and remaining size
  must conserve exactly across partial exits.
- A partial close does not trigger a full-position cooldown or flat lifecycle.

### 4.4 Management-mode contract

- `fixed`: target remains fixed.
- `breakeven`: the existing target-mode behavior is a one-time pullback of the
  nearest active target to `entry ± alpha × initial_risk`; it does **not** mean
  moving the stop to breakeven.
- `trailing`: only the nearest remaining unhit target may extend to a newly
  confirmed directional structural reference, subject to ordering/geometry.
- EMA/structural exits are named separately from the three target modes.
- Agent mode override/defer behavior and thresholds are fixed in the resolved
  config and must be recorded in the run manifest.

### 4.5 Action-schema contract

The action layout is declarative and versioned. When all currently intended
continuous management controls are active, the semantic layout is:

```text
[direction, intensity, stop, risk,
 tp1_sel, tp2_sel, tp3_sel,
 sl_mode, tp_mode,
 z1, z2,
 exit]
```

This is the 12-coordinate multi-TP/simplex form. Reduced configurations must be
constructed through the declared schema rather than inferred from width alone.
Checkpoint loading must validate schema version/key/coordinates. Legacy
unversioned checkpoints require explicit migration/opt-in; ambiguous layouts
are not guessed.

### 4.6 Fill, timing, and cost contract

- Decision time, order-submission time, fill time, and valuation time remain
  distinguishable.
- An entry at bar close is not exposed to the earlier high/low of that same bar.
- No new position is opened when the configured session/terminal rules make it
  unexecutable.
- Tick/quote execution uses the declared bid/ask, spread, slippage, latency, and
  contract units consistently.
- When OHLC alone cannot determine same-bar order, use the declared conservative
  convention and report the limitation.
- Every fill is counted once; position-level metrics aggregate all legs of one
  completed position rather than treating each partial exit as an independent
  trade.

### 4.7 Risk and survival contract

- Hard account constraints remain authoritative regardless of shaped reward.
- Training and evaluation use the frozen hard-breach penalty semantics for the
  selected lineage.
- Max-loss and peak trailing-drawdown violations remain explicit survival
  failures according to the frozen gate.
- Net PnL, equity slope, and trade count are reported outcomes, not hidden hard
  pass gates.
- Zero-trade/inactive behavior is reported as inactive/zero activity, not as a
  successful trading strategy merely because it avoided a breach.

### 4.8 Data, fitting, causality, and cache contract

- Primary development/training calendar ends at **2025-12-31** unless a new
  protocol is authorized before a new lineage.
- Purged chronological walk-forward folds inside the training calendar are used
  for model/config selection when selection is required.
- Fit-dependent transformations are fitted inside each training fold only and
  frozen for the corresponding validation fold.
- Multi-timeframe features use completed source bars and causal confirmation
  times. A current-source real-market prefix audit has passed at three listed
  cutoffs across four feature groups representing the 18 resolved variants;
  repeat it after relevant source/input changes. This sampled check does not
  replace the remaining fold, execution, or label-causality reviews. Current-
  source raw-to-cache parity also passed for four representative feature
  profiles; confirm each resolved variant's cache mapping at the final freeze.
- Data/cache identities include effective feature config, source data, split,
  and fitting scope. Named prebuilt caches fail closed on missing/mismatched
  hashes rather than silently rebuilding into the same identity.
- Full-data audit records date coverage, symbols, timezone, duplicates, missing
  bars, invalid OHLCV, resampling counts, feature NaNs, and session labels.
- The fixed-offset/session-time convention and broker DST behavior must be
  checked and described explicitly; a one-hour label shift cannot be ignored.

### 4.9 Validation-calendar contract

- `<= 2025-12-31`: train/in-sample and walk-forward selection domain.
- `2026-01-01` through the available `2026-07-03` data: **exposed development / exploratory report-only evidence** because outcomes have already been inspected.
- A later rerun on those dates does not restore pristine holdout status.
- A truly fresh final OOS period must be newly acquired after the exposed
  boundary, frozen before inspection, and never used to select the model.
- Changing calendar roles/dates is human-gated.
- Absence of fresh data does not justify relabelling exposed data; it changes the
  strength of the conclusion.

### 4.10 Evaluation and statistics contract

Canonical thesis metrics come from deterministic `TradingEnv` episode
reconstruction unless a table explicitly names another engine.

Report at least where meaningful:

- net PnL / return and final equity;
- Sharpe, Sortino, and Calmar with one documented annualization convention;
- maximum drawdown;
- profit factor and expectancy;
- completed-position win rate and position count;
- fill count and incomplete-position count separately;
- turnover, exposure, holding duration, and transaction costs;
- breach counts/types and survival/coverage;
- action/activity distributions and entry funnel diagnostics;
- reward-component attribution versus economic outcomes;
- seed-level results and paired treatment-control differences;
- transaction-cost sensitivity;
- behavioral timing/structural diagnostics where causally reconstructible,
  including entry delay from structural confirmation and level/session breakdown.

For uncertainty:

- preserve temporal dependence using block/dependence-aware procedures;
- compute drawdown uncertainty from valid return/equity paths rather than
  independently shuffling equity points;
- emphasize paired effect sizes and intervals for matched experiments;
- retain the complete search/trial ledger and apply the selected multiple-testing
  adjustment where appropriate;
- no statistical adjustment can repair selection on an already-inspected test
  period.

### 4.11 Run identity and provenance contract

Every run/attempt must have a unique immutable identity containing or linking:

- role: `smoke`, `development`, `selection`, `robustness`, `final_report_only`,
  or another explicit registered role;
- source commit/snapshot identity and dirty-file record;
- full resolved config and config hash/diff;
- raw data and feature/cache hashes;
- train/validation/test calendar roles;
- observation/action/feature/metric schema versions;
- algorithm, encoder, reward, risk, costs, execution settings;
- seed, requested and completed budget, `n_envs`/rollout geometry;
- hardware/software/runtime environment;
- exact command;
- checkpoint path/hash;
- logs, raw fills/trades/equity, metrics, figures, and gate outcomes.

A retry receives a new attempt identity and retains the failed attempt. Do not
overwrite a failure, infer missing config from current defaults, or backfill
provenance that never existed.

### 4.12 Compute, resource, and controller contract

- No GPU training on login nodes.
- Reuse an authorized allocation when available; do not implicitly create or
  cancel Slurm work.
- Active fingerprinted campaigns are immutable until a safe boundary.
- Controller, allocation, scheduler, worker, failure, and resource logs are
  durable on scratch/storage, not only tmux scrollback.
- Resume validates checkpoint and run identity; it does not double-schedule a
  completed key.
- Resource decisions use job cgroup measurements and process PSS/USS where
  appropriate, not summed shared RSS or node-wide free memory.
- A production memory profile should retain at least 64 GiB reserve and enough
  measured headroom (target at least 1.35× predicted peak) within the actual
  cgroup limit.
- Throughput is judged by reliable aggregate completed timesteps/hour and
  end-to-end experiment makespan, not individual worker FPS alone.
- For the active 18-worker full-split screen, the approved shape is two nodes,
  one GH200 per node, 48 CPUs and 512 GiB per node, with 18 workers at five
  CPUs each (`n_envs=64`, shared bundle). This is a campaign-specific setting,
  not a universal allocation request; keep the two GPUs distributed across the
  two nodes and re-check the actual queue and cgroup limits before future jobs.


### 4.13 Mandatory per-run results evidence bundle

Every training/evaluation attempt that can influence the final results must leave a durable
artifact bundle. This is mandatory for successful, failed, incomplete, inactive, and
zero-trade runs. A result is not results-package-usable if the number survives but its run
identity, logs, config, checkpoint, or raw records do not.

Recommended structure (names may vary, contents may not):

```text
outputs/results_evidence/<experiment_id>/<attempt_or_seed>/
  manifest.json
  artifact_index.json
  resolved_config.yaml
  command.txt
  source/
    commit.txt
    dirty.patch                 # when applicable
    environment.txt             # Python/package/CUDA/hardware identity
  logs/
    stdout.log
    stderr.log
    slurm.out                   # when Slurm is used
    slurm.err                   # when separate
    controller.log              # campaign/controller events when applicable
    resource_usage.csv          # wall time, CPU/GPU/memory where measured
  train/
    progress.csv                # or equivalent scalar training history
    monitor.csv                 # if produced
    tensorboard/                # if used
    checkpoint_final.*
    checkpoint_selected.*       # only when a declared selection rule chooses one
    checkpoint_hashes.sha256
    checkpoint_metadata.json    # step, schema, config hash, source/data identity
  eval/
    metrics.json
    metrics.csv
    equity.csv
    positions.csv
    fills.csv
    breaches.csv
    actions.csv                 # or action summary + raw reference
    reward_components.csv       # where available
    entry_funnel.csv            # where applicable
    cost_stress.csv             # when this run enters cost sensitivity
  figures/
    learning_curve.*
    equity_curve.*
    drawdown_curve.*
    activity_or_action_plot.*
    additional_result_figures.*
  results/
    finding.md
    table_rows.csv
```

Minimum checkpoint rule:

- preserve the final checkpoint for every completed RL run;
- preserve the selected checkpoint only when a **predeclared development-only**
  selection rule chooses it; do not call an arbitrary checkpoint `best`;
- record exact completed timesteps, checkpoint step, file size, hash, action/observation
  schema versions, and successful load validation;
- periodic checkpoints are optional except where needed for crash recovery or a
  registered checkpoint-selection analysis;
- evaluation/export failure does **not** justify retraining when the checkpoint and
  training lineage are valid. Repair evaluation and recover from the saved checkpoint.

For a failed run with no checkpoint, retain the manifest, resolved config, command,
logs, failure reason, completed step count, and any partial artifacts. Failure evidence is
part of the experiment ledger, not disposable noise.

### 4.14 Result finding card: the thesis-writing handoff interface

Every completed work package or empirical run that produces a result-relevant
finding must also create/update a short machine-linkable `finding.md`. Its purpose is
to make the final results package self-explanatory so thesis writing can happen afterward without
rereading raw logs. The agent produces the evidence summary, not chapter prose. Each finding card
must contain:

1. **Finding ID and status:** `verified`, `negative`, `inconclusive`, `inactive`,
   `failed`, `engineering-only`, or `not-tested`.
2. **Question/hypothesis:** what was actually being tested or verified.
3. **Evidence role:** correctness, one-seed screening, matched final evidence,
   robustness, runtime, or documentation.
4. **Run IDs / seeds / completed steps / `n_envs`.**
5. **Control and treatment definitions** where a comparison is made.
6. **Headline measurements** with uncertainty only when statistically justified.
7. **Artifacts:** direct relative paths/IDs for config, logs, checkpoint(s), raw eval,
   table rows, and figures.
8. **Allowed claim:** the strongest wording the evidence supports.
9. **Forbidden overclaim / limitations:** e.g. single-seed, exposed OOS, confounded
   action space, incomplete run, or system-level comparison only.
10. **Likely thesis destination:** candidate chapter/section/table/figure where the finding belongs.
11. **One-paragraph evidence interpretation:** concise factual interpretation of what the
    result supports and does not support; this is a handoff note, not thesis prose.

Single-seed seed-50 findings are therefore preserved as clearly labelled
**screening/system-behaviour evidence** while the final >=5 paired-seed conclusion is
still pending. When P7 completes, the final paired finding card supersedes only the
headline robustness/causal conclusion; it does not erase valid implementation,
behavioural, failure, or mechanism observations from earlier stages.

---

## 5. Current state that determines the first actions

Treat this as a planning snapshot, not live cluster telemetry.

1. The bounded correctness/implementation packages for action schemas, fill
   accounting, causal SL/TP behavior, costs/timing, metrics, fold fitting,
   strict eligibility, structural targets, matched-comparison tooling,
   provenance, parity, and documentation have focused regression evidence.
   They do **not** by themselves prove end-to-end strategy quality.
2. The current thesis screening lineage is the 18-variant full-split campaign:
   six mechanism rungs across baseline, PO3/IFVG, and distribution families,
   one discovery seed, 20M PPO timesteps, `n_envs=64`, shared-bundle backend.
   Treat it as in progress until all 18 identities are reconciled to terminal
   artifacts.
3. The evaluator observation-schema forwarding defect was fixed in main commit
   `f65e6ff` and its focused regression passed. P2 implementation is in place;
   end-to-end deterministic recovery from a real saved checkpoint remains open.
   Repair evaluation and recover valid checkpoints instead of retraining them.
4. The 2026 report interval has been inspected and is permanently exposed.
   No pristine confirmatory holdout is currently available in the retained
   dataset.
5. Independent research sign-off remains open.
6. A previous planning snapshot described a coherent full manuscript draft, but
   its source was not found in the canonical checkout during the 2026-10-16
   artifact inventory. The only nearby PDF found was
   `/scratch/work/nguyenl37/Thesis_Defense_PO3_IFVG.pdf`, a 27-page defense
   slide deck rather than a manuscript. Locate and assess the manuscript source
   before P10; if no source exists, create the thesis from the frozen evidence
   package. A follow-up filename inventory under `/scratch/work/nguyenl37`
   and `/home/nguyenl37` found no thesis `.tex`, `.qmd`, `.docx`, `.odt`, or
   `.typ` file; the only matches outside the project were bundled compiler
   library manuals. This bounds the search to those local roots and file types;
   an external/cloud copy has not been ruled out.
7. Device-resident PPO rollout staging has measured exact minibatch parity and a
   strong PPO-core throughput improvement. Higher trainer concurrency and larger
   minibatches have promising throughput measurements, but production adoption
   still requires end-to-end quality/resource acceptance. Closed/no-gain probes
   should not be repeated without a changed bottleneck.
8. Q1 state-v2, an equity-delta reward foundation, semantic action heads,
   diagnostics, constraint telemetry, and reproducible seed/schema wiring exist
   as implementation foundations. Hybrid policy distribution and learned
   Lagrangian constraint optimization are not yet established research paths.
9. Stop-candidate review has two different dispositions. The IFVG through-zone
   extreme is already generated, exposed to the strategy stop menu, and covered by
   a focused regression; preserve it. HTF swing-stop code now supports an opt-in
   three-candidate set with source-timeframe provenance, causal confirmation time,
   age, instrument-tick deduplication, and entry-level eligibility/rejection audit.
   The default remains latest-only, preserving the active screening campaign.
   Before final matched P7 runs, validate the enabled profile end to end (including
   IFVG coexistence, selected/rejected provenance, and independent review), then
   freeze its exact source/config/cache identity.
10. Runtime/protocol truth must override stale manuscript examples. `n_envs=64`
   is the accepted thesis runtime. For the comparable one-seed discovery/development
   stage, use **seed 50 as the canonical first seed** for newly launched or relaunched
   variants. Preserve any already-completed seed-42 attempts as historical/exploratory
   evidence; do not rerun them solely to erase a seed mismatch, and do not mix seed 42
   with seed 50 as though they were the same one-seed comparison. The final >=5-seed
   paired confirmation is deliberately postponed until all implementation, one-seed
   findings, candidate selection, evaluation, and runtime decisions are frozen.

Therefore the immediate objective is **not to design more model families**. It
is to finish/reconcile the current screen, make final evaluation trustworthy,
freeze a valid matched experiment, and obtain the complete evidence package needed
for thesis writing afterward.

### Live execution update — 2026-10-16 12:04 Helsinki

Campaign `ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016` retains its
original fingerprint `3474621efd6a1481bd2a20bce0cfd2c0b69969df1849c6cd738d19dc5bb4ba34`.
All 18 attempt-4 workers failed before training at `cache_dir.mkdir` because
`cache` was a self-referential symlink. Their logs and provenance remain intact.
The cache path is now a real directory with the 14 configured nonempty bar
caches. Resume pointers were corrected to the hash-verified attempt-3 checkpoints
(11M–14M steps); the correction and immutable backups are recorded in
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/cache_recovery.json`
and adjacent recovery reports. The campaign manifest remains 18/18
`interrupted`; no resumed attempt has started.

Slurm job `20985591` is RUNNING on `gpuarm1` and `gpuarm2`, with one GH200,
48 CPUs and 512 GiB RAM per node, scheduled to end at 17:32 Helsinki. It has
only the allocation step and `.extern`; no worker steps or trainers are active.
Its training release marker remains absent. This allocation began before the
persistent worker-dispatch changes on disk, so it stays held until natural
expiry. The updated wrapper, dispatcher, recovery checks, and evidence-index
writer passed a fresh gpuarm1 gate at 12:11 Helsinki: full-tree Ruff
format/check, full-tree Mypy (436 source files), and 44 focused tests. The
evidence index was built on gpuarm1; it verifies all 228 recorded source hashes
and all four recorded market-data input hashes, while retaining limitations
for complete split exposure and checkpoint deserialization. These changes are
still uncommitted. The live allocation's preflight process predates the
dispatcher and its start marker remains absent; do not release that older
controller into worker-step fan-out. Do not infer training progress, strategy
evidence, or a READY state from the allocation or saved partial checkpoints.

The point-in-time machine-readable record is
[`live_campaign_snapshot_2026-10-16_1158.json`](live_campaign_snapshot_2026-10-16_1158.json).
The current checkout is `/scratch/work/nguyenl37/Aalto_MS_Thesis`, branch
`main`, HEAD `45ef1a0`; 228 recorded campaign source hashes were verified with
no mismatches before the dispatcher work, which changes only paths outside the
campaign source-hash manifest.

### Live execution update — 2026-10-16 13:00 Helsinki

Current snapshot: [`live_campaign_snapshot_2026-10-16_1300.json`](live_campaign_snapshot_2026-10-16_1300.json).
The canonical checkout is still `main`; pushed HEAD is `56fc32a`, with the new
checkpoint-backend audit and this status update being verified for commit.
Slurm job `20985591` remains RUNNING on two GH200 nodes, with 48 CPUs and 512
GiB RAM per node; it ends at 17:32:47 Helsinki. It has no worker steps or
trainers, and both GPUs are idle after P2 replays. Leave its release marker
absent because its already-running controller predates the persistent worker
dispatcher and would use the old worker-step fan-out. The live supervisor can
pick up the committed wrapper in its next allocation.

The campaign remains 18/18 `interrupted`, all resume pointers target
hash-verified attempt-3 checkpoints at 11M–14M steps, and the campaign
fingerprint remains unchanged. The refreshed evidence index verifies all
228/228 campaign source hashes without mismatch. For P2, a partial 14M
baseline checkpoint replayed identically for equity and metrics but exposed
cross-process `level_type` label variation; a separate active-policy
frame/bundle parity replay matched every compared result field for 177,210
test bars. Both are historical exploratory audits, not confirmatory evidence.
P1 and P2 remain open, and readiness remains NOT READY.

### Live execution update — 2026-10-16 13:08 Helsinki

On `gpuarm1`, all 18 attempt-3 recovery checkpoints selected for attempt 4
passed SHA-256, provenance, campaign identity/fingerprint, saved-step, and
`PPO.load(device="cpu")` checks. The report is
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/checkpoint_load_audit_gpuarm1_20261016_rerun.json`.
The generic audit CLI did not follow these pointers because it selects the
failed attempt-4 counter; this limitation and the bounded result are recorded
under P2. No trainer was started.

Allocation `20985591` remains RUNNING on `gpuarm1` and `gpuarm2`, two GH200s,
48 CPUs and 512 GiB RAM per node, scheduled to end at 17:32:47 Helsinki.
At 13:08 both GH200s reported 0% utilization and 0 MiB used; Slurm shows only
the allocation, `.extern`, and `.0` steps. The old controller's release marker
remains absent because its worker fan-out predates the persistent dispatcher.
The guarded supervisor remains available to request the next allocation after
this one ends; no duplicate allocation or extra Slurm step was submitted.

The documentation update passed full-tree Ruff format/check, full-tree Mypy
(438 source files), and 15 focused recovery/evidence tests on `gpuarm1` at
13:06. Readiness remains NOT READY; P1 and P2 remain open.

### Live execution update — 2026-10-16 20:24 Helsinki

Read-only scheduler verification shows allocation `20990395` still RUNNING on
`gpuarm1` and `gpuarm2`, with exactly two GH200s, 96 CPUs and 1 TiB RAM total.
Its scheduled window is 17:34:29–23:34:29 Helsinki. The primary supervisor
(`run_campaign_18way_cpumask_supervisor.sh`) and the authorized boundary watcher
remain alive. The user authorized this job to finish naturally, then authorized
stopping only its automatic reacquisition supervisor; do not `scancel` the job.

The recovery allocation supervisor is sleeping while the current
`thesis18cpumask` job remains active; it reports 18 campaign records active,
zero failed, and zero succeeded. Independent live artifacts now confirm all 18
workers: all 18 attempt logs were refreshed after 20:20, and the two node
resource samplers report nine worker root PIDs each with `campaign_active=1`.
At 20:27, logged cumulative progress ranged from 15.23M to 19.60M of 20M per
variant (median 15.62M); reported per-worker FPS ranged from 419 to 551
(median 459). The sum of the 18 latest FPS fields is about 8,345 steps/s, a
log-derived estimate rather than an independently measured throughput window.
The latest sample, timestamped 20:26:45 Helsinki, reports GPU utilization
99%/100%, VRAM use 82,735/84,093 MiB, cgroup current RAM 175.5/154.5 GiB, and
cgroup peaks 177.0/156.5 GiB; OOM counters are zero. These are point samples,
not final run-wide resource summaries. The current recovery attempt still has
no `episode_equity.csv` files, but the resumed source lineage has earlier
attempt-2 episode telemetry; its coverage is audited below. Preserve the
current job and its fingerprints until the safe boundary.

A checkpoint-lineage budget audit is now recorded at
[`p1_checkpoint_full_split_budget_2026-10-16.json`](p1_checkpoint_full_split_budget_2026-10-16.json), with its reproducible generator at
[`audit_checkpoint_full_split_budget_current_campaign_2026-10-16.py`](audit_checkpoint_full_split_budget_current_campaign_2026-10-16.py).
One full 99,504-step train split across 64 fresh environments requires
6,368,256 aggregate steps. The audit traces all 18 current checkpoints through
attempt-3 provenance to attempt-2 training records and verifies 64 unique
environment indices per variant, each with exactly 99,504 steps,
`completed_data_split=true`, and `episode_end_reason=data_end` (1,152 records).
The current resumed baseline starts at 14M and has only 6M steps left in its
20M cap, enough for at most 93,750 additional steps/environment; it cannot
complete a second full split during this continuation. Thus one chronological
train-split traversal per environment is evidenced before the current
checkpoints, while current-attempt episode telemetry is still absent. This is
exposure evidence, not proof of active trading throughout the year, a valid
feature path, or strategy quality.

The same per-environment records show why episode completion must be separated
from survival and activity: 621/1,152 training episodes recorded a
`trailing_dd` breach before continuing to `data_end`; two variants
(`ladder_a4a_entry_fsm_hidden` and `ladder_a4b_entry_fsm`) recorded zero trades
in all 64 episodes each. All 16 variants with trades had negative mean
episode-equity deltas. These are in-sample training diagnostics from a
feature-contaminated lineage, not deterministic evaluation PnL, and cannot be
used to select a strategy. Per-variant counts and episode-level hashes are in
the same audit JSON.

The real-market fold-fit poison audit remains the controlling validity result:
current NY-session high/low features leak future within-session values in
profiles mapped to 17 of 18 variants. The current campaign is therefore
exploratory and cannot support selection or confirmation. At the boundary,
finish the authorized supervisor stop, preserve the run artifacts, repair the
feature and regression, bump/rebuild the three versioned feature caches, rerun
the real-market prefix/fold and cache audits, and prepare a fresh matched
lineage; request allocation approval before launching it. Do not resume the
contaminated checkpoints as thesis evidence. The full-split training
implementation is already present on main (`46a8ce7`) and in the active source
snapshot (`413729e`). Attempt-2 telemetry verifies one complete chronological
split per environment in each current checkpoint lineage; this does not remove
the causal-feature failure or prove active trading after a breach. P1–P4
remain open and readiness remains NOT READY.

### Live checkpoint-lineage and trainer update — 2026-10-16 20:39 Helsinki

Allocation `20990395` remains RUNNING on `gpuarm[1-2]` at 3:04:59/6:00:00.
All 18 attempt logs are live; they show 15.42M–19.997M of 20M total steps per
variant, median 15.92M, and latest per-worker FPS 419–549 (median 458). The
reacquisition supervisor and the user-authorized terminal-state watcher remain
alive. The current attempt still has no `episode_equity.csv`; keep waiting for
this allocation to finish naturally and preserve its source/checkpoints.

The 20:38 version of the checkpoint-lineage audit now verifies that the 18
attempt-3 checkpoints descend from attempt-2 runs in which every environment
recorded 99,504 steps and `data_end`. It also summarizes the training episodes:
621/1,152 recorded trailing-drawdown breaches; two variants recorded 64/64
zero-trade episodes; all 16 variants with trades had negative mean episode
equity deltas. These figures are exploratory training diagnostics from a
feature-contaminated lineage, not final evaluation metrics. The report and
generator are linked above; P1/P2 and G1 remain open.

---

## 6. Dependency graph: Codex agent horizontal execution to results freeze

```text
                          ┌─────────────────────────────────────┐
                          │ P0 Safe boundary + current snapshot │
                          └────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┼────────────────────────┐
                    │                      │                        │
                    ▼                      ▼                        ▼
          P1 Finish/reconcile      P2 Evaluation/report      E0 Evidence/result
          18-way screening         fix in isolation          pipeline + indexes
                    │                      │                        │
                    └──────────────┬───────┴──────────────┬─────────┘
                                   ▼                      │
                        P3 Canonical screen reports       │
                                   │                      │
                                   ▼                      │
                     P4 G0-G3 validity/evidence audit ◄──┘
                                   │
                                   ▼
                       P5 Freeze final experiment
                    RQ1 matched pair + claim boundaries
                                   │
                         ┌─────────┴──────────┐
                         ▼                    ▼
                 P6 Runtime/resource     R0 Independent
                 production freeze       review/design pass
                         │                    │
                         └─────────┬──────────┘
                                   ▼
                 P7 Final paired seeds + cost robustness
                                   │
                         ┌─────────┴──────────┐
                         ▼                    ▼
                    S0 Statistics       E1 Result aggregation
                    + final figures     + finding cards/index
                         └─────────┬──────────┘
                                   ▼
                       P8 Evidence/review freeze
                                   │
                                   ▼
                   P9 RESULTS_PACKAGE_COMPLETE
                                   │
                                   ▼
           P10 manuscript writing / PDF assembly in active user goal
                                   │
                                   ▼
                    Supervisor PDF — 2026-11-09
                                   │
                                   ▼
                    Faculty submission — 2026-11-23
```

The P0–P9 results-evidence phase ends at **`RESULTS_PACKAGE_COMPLETE`**. The
active user goal continues through P10/P11 manuscript, PDF assembly, feedback,
and faculty submission. Q0–Q7 are also compulsory for the overall mission by
user direction on 2026-10-16. They remain a separate post-thesis research
program by default so they cannot contaminate or delay the frozen thesis
evidence. A Q stage may run earlier only after its entry criteria pass and when
it does not consume resources or attention needed by thesis-critical gates.

## 7. Parallelization and resource-lock rules

A task is parallel-safe only if all are true:

1. it does not mutate source/config/data/cache files in an active run identity;
2. it does not require the winner or conclusion of an unfinished experiment;
3. it uses a separate worktree/output namespace where code changes are involved;
4. it does not create resource contention that would confound a performance
   benchmark or starve thesis-critical training;
5. its result can be accepted/rejected independently;
6. it does not expose a fresh holdout before model selection is frozen.

### Parallel work that is encouraged

- evaluator/schema fix while PPO training continues;
- numerical-report/postprocess tooling while training continues;
- evidence-bundle tooling, raw-output validators, result tables, and figure-generation code
  while experiments continue;
- independent reviewer package preparation while G0–G3 evidence is assembled;
- deterministic statistics/figure code against already-finished seed artifacts
  while other seeds run;
- implementation-only work for later Q stages in a separate worktree when it
  consumes no thesis-critical compute or integration bandwidth.

### Work that must remain serial

- selecting a treatment before the screening/evidence needed for that decision
  is terminal;
- Q2 before Q1, Q3 before Q2, and so on;
- final matched robustness before RQ1 control/treatment semantics are frozen;
- throughput benchmarking of competing configurations on hardware where they
  interfere with one another;
- final result aggregation/claim disposition before the corresponding result is frozen;
- fresh-holdout evaluation before selection and trial accounting are locked.

---

## 7A. Codex agent long-running horizontal operating protocol

This protocol is for **one long-running Codex agent** handling work across long-running
training and evaluation jobs. The agent coordinates and works across independent lanes rather than
executing one serial task at a time.

### 7A.1 Never-idle invariant

If a training job is running and any independent task is `READY`, the agent works on that task. Waiting
for compute is valid only when every remaining task is genuinely blocked by that compute result.
Training time should overlap with evaluator work, correctness tests, recovery tooling, evidence
packaging, statistics code, result figures, provenance checks, and review preparation.

### 7A.2 Persistent control state

Maintain one compact machine-readable/Markdown control record alongside outputs, for example
`outputs/agent_state/current_status.md` plus `outputs/agent_state/state.json`. This is **not another
roadmap**. It records only:

- current master-plan revision/hash;
- active package and lane;
- READY / RUNNING / BLOCKED / DONE tasks;
- active training/evaluation job identities and immutable fingerprints;
- last verified checkpoint/result per run;
- blockers and exact unblock condition;
- next three highest-priority actions;
- last full CI/evaluator/evidence-pipeline validation state.

Update this control state after every meaningful terminal event, launch, failure, merge, experiment
freeze, or evidence seal. This makes context compaction/restart recoverable without rereading the
repository or guessing what happened.

### 7A.3 Atomic execution units

Break work into units that can be verified independently: one bug/fix, one contract test set, one
run reconciliation, one evaluator recovery, one evidence bundle, one statistics output, one launch
matrix, one review disposition. Each unit ends with:

`change/result -> validation -> artifact paths -> status update -> next dependency`.

Do not keep large unvalidated working changes open merely because the agent can run continuously.

### 7A.4 Source and resource isolation

- active training source/config/data/cache fingerprints are immutable;
- correctness/evaluator/runtime changes use isolated worktrees/branches or otherwise proven-safe
  boundaries until merged;
- a benchmark never competes with critical training if contention would invalidate its result;
- evaluation/post-processing may use spare resources only when it cannot materially slow the active
  scientific campaign;
- the agent never assumes that an interactive allocation will persist indefinitely: checkpoint/resume and
  durable logs must make scheduler interruption recoverable.

### 7A.5 Training-aware event loop

While long jobs run, the agent cycles through event-driven work rather than polling constantly:

1. verify active jobs have durable identity/log/checkpoint paths;
2. work the highest-priority independent READY unit;
3. when a run becomes terminal, immediately classify and seal its evidence bundle;
4. evaluate/recover that checkpoint if the evaluator path is ready;
5. update aggregate screen/final-pair status and dependency gates;
6. launch the next authorized run only if its scientific/runtime inputs are frozen;
7. return to independent READY work;
8. if **no independent READY unit exists**, enter `WAITING_COMPUTE` instead of spending model tokens on repetitive status polling.

### 7A.6 Quiet-wait and adaptive polling protocol

`Never-idle` does **not** mean `never-sleep`. When every useful lane is blocked only by running
training/evaluation jobs, the agent must deliberately pause model activity. A long-running campaign must
not cause one-minute conversational/model turns such as “training is still running at 1.xxx”.

#### Event-first rule

Prefer external/job-state events over model-driven polling whenever the execution environment permits
it. A watcher or scheduler-side process may record job state, timestamps, checkpoint creation, and
terminal status into a compact machine-readable file. The model should read that state only when it
wakes; the watcher itself must not require repeated model inference.

#### Polling cadence when event notification is unavailable

Use coarse adaptive polling based on expected run duration and observed progress:

- **launch health check:** once after roughly 2–5 minutes, only to catch immediate startup/config/data failures;
- **early stabilization check:** once around 15–30 minutes if the job is still expected to run for hours;
- **steady-state training:** normally no more than once every **45–90 minutes** for multi-hour jobs;
- **very long stable runs (>8 h remaining):** polling may stretch to **90–120 minutes**;
- **near expected completion:** shorten to roughly **15–30 minutes** only when the latest measured rate places completion within about one hour;
- **known checkpoint boundary / scheduler wall-time boundary:** wake shortly after that boundary to validate checkpoint/resume state;
- **failure signal or terminal transition:** wake immediately when such an event is externally available.

These are upper-level defaults, not reasons to wake the model unnecessarily. If a robust external
watcher can detect terminal state, prefer sleeping until that event.

#### Token-saving status discipline

During `WAITING_COMPUTE`:

- do **not** emit repetitive user-facing or internal prose updates;
- do **not** reread full logs; query only compact state (`squeue`/`sacct` summary, one-line progress,
  file existence/mtime/size, or a small `tail`) needed to decide whether an event occurred;
- do **not** rewrite `current_status.md` merely because elapsed time changed;
- update persistent control state only on a meaningful event: launch health confirmed, checkpoint
  sealed, rate materially changed, failure, resume, terminal completion, evaluator completion,
  evidence seal, or dependency-gate transition;
- avoid recalculating ETA every poll unless throughput materially changed;
- avoid re-summarizing unchanged findings into model context.

A quiet poll that finds no meaningful state change should end immediately without generating a new
analysis narrative.

#### Sleep-state record

Before pausing, the agent writes only the minimal durable fields needed for recovery:

```text
state = WAITING_COMPUTE
blocking_jobs = [...]
last_verified_event = ...
last_progress_sample = ...
expected_next_event = ...
next_check_not_before = ...
wake_conditions = [terminal, failure, checkpoint_boundary, scheduled_coarse_check]
next_ready_action_after_wake = ...
```

This state is enough for context compaction or model restart. No minute-by-minute history belongs in
the control file or conversational context.

#### Example for an approximately 14-hour 20M run

A healthy long run might require model attention approximately at:

```text
launch
  -> +3 min startup health
  -> +20 min stabilization
  -> +60–90 min coarse check
  -> then every ~60–120 min while stable
  -> ~30 min cadence only near estimated completion
  -> terminal event -> evaluate/package immediately
```

If other independent tasks are READY, the agent performs them instead of sleeping. If none are READY,
quiet waiting is the correct and token-efficient behavior.

### 7A.7 Reasoning-effort escalation policy

Use the least reasoning effort that can resolve the current unit safely. Treat reasoning effort as a
limited resource: spend it on decisions whose consequences or uncertainty justify the cost, not on
routine waiting or repeated status checks.

- **Low effort:** routine status checks, log extraction, inventories, straightforward document edits,
  deterministic command/result interpretation, and other bounded tasks with clear acceptance criteria.
- **Medium effort:** ordinary implementation/debugging, multi-step dependency analysis, experiment
  planning within a frozen protocol, or diagnosis that needs evidence from several artifacts.
- **High effort:** scientific-validity and causal-claim decisions, ambiguous or conflicting evidence,
  data leakage/provenance concerns, checkpoint or schema compatibility, active-campaign source-boundary
  decisions, major architecture changes, and resource choices that could invalidate a run.
- **Escalate once more only when** the current effort level leaves a material correctness or decision
  uncertainty unresolved. State the uncertainty and evidence needed; do not repeatedly increase effort
  to compensate for missing data or an external wait.
- **Do not escalate** for elapsed time, long-running jobs, large log volume that can be summarized with
  bounded queries, or routine polling. Read the smallest relevant evidence and enter `WAITING_COMPUTE`
  when no independent READY work remains.
- After a high-stakes decision, record the decision, evidence, and remaining uncertainty in the relevant
  plan/status/evidence artifact so a later agent does not repeat the same reasoning from scratch.

### 7A.8 Retry and failure discipline

- infrastructure interruption with a valid matching checkpoint -> resume under a new attempt identity;
- deterministic code/config/data defect -> stop that affected lineage, fix/test once, then relaunch;
- evaluator/export defect after successful training -> repair/recover, do not retrain;
- repeated identical failure -> investigate root cause rather than blind retry;
- every retry keeps the original attempt and traceback in the ledger.

### 7A.9 Context-budget discipline

The agent should prefer repository artifacts over conversational memory. After completing a substantial
unit, write the result to the appropriate manifest/status/evidence file and use those files as the
source of truth. Long-running success means **state survives context compression**, not that every
prior detail remains in the model context.

### 7A.10 Stop condition

The continuous agent does not stop merely because training or P9 is finished. P9 closes the
results-evidence phase after final logs, configs, checkpoints, raw outputs, statistics, figures,
finding cards, failures, hashes, and global indexes are present and cross-validated. The active user
goal ends only at P11 `SUBMISSION_COMPLETE`.

---

# PART I — RESULTS-CRITICAL EXECUTION

## 8. P0 — Establish the safe execution boundary

**Priority:** immediate
**Dependencies:** none
**Parallel with:** evaluator preparation, evidence tooling, result-index preparation

### Actions

1. Snapshot the current source state, unresolved local changes, active campaign
   identity, output directories, controller/supervisor state, and available
   checkpoints without changing them.
2. Reconcile which branch/source snapshot produced each active checkpoint.
3. Treat campaign source/config/data/cache fingerprints as immutable.
4. Keep evaluator fixes, benchmark changes, manuscript edits, and extension
   implementations isolated from the live campaign source boundary.
5. Verify the latest full CI/format/lint/type status available for the exact
   source that will be used after the campaign; rerun only when a merge/change
   makes the evidence stale.
6. Create one short dated status entry containing only transient operational
   facts. Do not hard-code job IDs/FPS/allocation countdowns into this roadmap.

### Exit criterion

The current experiment and every parallel engineering/evidence lane have clear,
non-overlapping provenance boundaries.

---

## 9. P1 — Finish and reconcile the 18-way full-split screen

**Priority:** critical path
**Dependencies:** P0
**Parallel with:** P2, E0, report-tool preparation

For each of the 18 variant identities:

1. classify training state as completed budget, valid terminal, infrastructure
   interrupted/recoverable, infrastructure failed, or invalid;
2. confirm full chronological split exposure from episode/reset telemetry rather
   than inferring it from aggregate timesteps;
3. preserve the latest valid checkpoint and hash;
4. reconcile worker log, controller events, run manifest, checkpoint lineage,
   and artifact timestamps;
5. preserve every failed attempt and original traceback/exit status;
6. distinguish training failure from final-evaluation/export/controller failure;
7. never use a stale dashboard/campaign summary when primary logs/artifacts
   disagree.

### Screen interpretation

This is a **one-seed selection/system-behavior screen**. It can:

- identify broken/inactive/unsafe configurations;
- identify promising candidates for a matched experiment;
- reveal mechanism/strategy behavior worth investigating;
- provide exploratory complete-system comparisons.

It cannot by itself establish robust treatment effects or generalization.
Cross-family differences that include reward/action/strategy changes are not
isolated effects of the structural overlay.

### Current seed-50 recovery disposition

The terminal campaign summary records 11 succeeded, 4 train-survival gate
failures, and 3 interrupted variants. Current-attempt full-split episode
telemetry is present for 17 variants (1,088 environments, one 99,504-step
episode per environment); `ladder_a0_flat` has no completed-split telemetry in
this attempt. Keep this distinct from the earlier checkpoint-lineage audit,
which verified a prior full-split traversal for all 18 source lineages. Eight
variants have zero trades in both train and report-only test evaluation. The
fold-poison audit finds lookahead in 17/18 configurations; the sole passing
profile, `ladder_a0_flat`, failed its train-survival gate. Thus the screen has
no variant that is both causally admissible and passes the declared training
survival gate. See the evidence-backed
[`P1 screen disposition`](evidence/p1_18_variant_screen_disposition_2026-10-16.md)
and campaign JSON report; do not use the screen for selection or confirmatory
claims.

### Exit criterion

Exactly 18 terminal identities are reconciled. Every successful training run
has a loadable checkpoint and complete source/config/data lineage; every other
identity has an evidence-backed reason and retained artifacts.

---

## 10. P2 — Repair and harden deterministic evaluation

**Priority:** immediate parallel blocker removal
**Dependencies:** P0
**Parallel with:** P1
**Use against active checkpoints only after:** the source boundary is safe

### Required correction

Make final evaluation reconstruct the observation/action/environment contract
used during training. Resolve the current mismatch for at least:

- `observation_schema_version`;
- `observation_age_scale_bars`;
- all action-schema/management flags required by the saved checkpoint.

Do **not** fix compatibility by silently dropping schema information.

### Regression requirements

Verify:

- account-state v1 and v2 where supported;
- frame/shared-bundle paths used by the relevant checkpoints;
- saved checkpoint loading and action-schema validation;
- deterministic train and report-only test evaluation;
- recovered evaluation from a completed checkpoint without retraining;
- legacy/default behavior when new schema arguments are absent;
- observation coordinate count/order and action coordinate count/order;
- no change to trade/fill/equity results caused solely by the compatibility fix.

### Audit notes — open until exercised on a real checkpoint

- `eval_run.py` now forwards the observation schema and age scale and validates
  observation/action schemas before replay. The current `ladder18_seed50` saved
  config uses the defaults `risk.sl_mode=fixed`, SL/TP defer thresholds `0.1`,
  TP breakeven alpha `0.5`, PPO, and no VAE, so these values match the current
  evaluator defaults. Generic management-value forwarding is also covered:
  `_build_eval_common` maps the saved SL/TP mode, TP reference, defer thresholds,
  breakeven alpha, and EOD-risk settings into report evaluation, and
  `tests/test_envs/test_eligibility_wiring.py::test_management_settings_match_training_and_evaluation_environment_kwargs`
  compares non-default values with the training environment. On 2026-10-16,
  the focused evaluation/config suite passed 22 tests across that file,
  `tests/test_rollout.py`, and `tests/test_eval/test_eval_run_provenance.py`.
  This closes generic management-value forwarding coverage; it does not replace
  exact saved-checkpoint recovery/parity or completed-selected-checkpoint
  acceptance.
- Existing frame/shared-bundle evaluation parity coverage uses a hold action.
  It does not prove equality of action-dependent fills, equity, breaches, and
  trade summaries for the campaign policy. Add or run that stronger comparison
  against an authorized checkpoint before treating P2 as closed.
- An older review checklist reports an all-18-variant fixed-trace frames/bundle
  comparison, but its cited report directories are absent from this checkout
  and the result cannot be hash-rechecked. Do not count that claim toward P2;
  the currently available selected-checkpoint parity audits do not establish
  all-18 parity or completed-checkpoint acceptance.
- A real 14M-step, interrupted `ladder_a0_flat__seed50` attempt-3 checkpoint was
  loaded and evaluated twice on gpuarm1, over 351,550 training bars and 177,210
  test bars per pass. Equity CSVs and aggregate metrics were byte-identical.
  Trade event rows were identical except `level_type`: 53 training rows and 23
  test rows changed from `london_low` to `asian_low`. Source inspection finds
  `_matched_entry_level` chooses the maximum crossing time over a set, without
  a tie-break for equal timestamps. A 12-process characterization of this
  method with equal `london_low`/`asian_low` crossing times and different
  `PYTHONHASHSEED` values returned each label for six seeds, confirming the
  tie-break is process-order dependent. Because `level_type` feeds temporal
  subgroup reports, full report determinism is not established.
  The detailed artifact is
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/p2_replay_determinism_audit_20261016.json`.
  The stable lexical tie-break was committed at 14:12 on 2026-10-16 in
  `6827b3822f75feab46821747f1ba3a24f3c72723` and has a regression test in
  `tests/test_envs/test_trading_env_decomposition.py`. The two replay passes
  ran at 14:01 and 14:11, both before that correction (committed at 14:12:37),
  so they do not verify the fixed implementation. A source-boundary audit on
  2026-10-16 confirmed that the
  active recovery campaign's `source_snapshot.zip` contains the corrected
  `quant_rl/envs/trading_env.py` (SHA-256
  `4056dd2608420ac232da2b12a5f727a5cfe648c8256438c04a81eef7ce8f1a81`, equal
  to the current checkout). Repeat the cross-`PYTHONHASHSEED` deterministic
  evaluation after the active campaign boundary; do not edit fingerprinted
  source while the campaign is live.
- The same checkpoint then completed a full 177,210-bar deterministic test
  replay through both the frame and shared-bundle environments. All compared
  fields matched exactly: equity, trade rows, account, breach state/events,
  actions, rewards, step counts, and aggregate metrics (107 trades in each
  backend). The machine-readable report is
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/p2_checkpoint_frame_bundle_a0_14m_20261016/audit.json`.
  This closes backend parity for this partial baseline checkpoint only; the
  repeat-run label mismatch, generic management-parameter forwarding, and
  parity for a completed selected checkpoint remain open.
- Extending parity to the complete 351,550-bar in-sample training split first
  returned a false backend mismatch (231 frame trades vs 202 bundle trades).
  A lockstep trace isolated the first semantic divergence at step 394 / bar
  index 2,294 (`2024-12-31 16:34:00+03:00`): the frame evaluator had the
  evaluation trailing guardrail disabled (`trailing_ratio=0`), while the
  bundle worker was using the training 7% trailing threshold
  (`trailing_ratio=0.0175562`). The audit had built its bundle environment as
  a training worker, unlike `_eval_guardrail_kwargs`; this was an evaluation
  spec mismatch. `EnvSpec` now explicitly marks evaluation workers, and the
  worker disables both trailing thresholds only for those specs. Training
  workers retain their configured survival gate. Focused bundle/parity tests
  pass (11 tests). Replaying the same saved checkpoint and entire training
  split with the corrected evaluation spec matched all result fields and
  aggregate metrics exactly: 99,504 steps and 231 trades per backend. Evidence:
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/p2_checkpoint_frame_bundle_train_a0_14m_20261016/audit.json`,
  its `source_snapshot.json`, and `initial_failure_diagnosis.json`. This is
  in-sample parity for one interrupted checkpoint, not strategy or
  confirmatory evidence; P2 remains open for the other regression requirements
  and a completed selected checkpoint.
- A second action-dependent in-sample parity audit used the interrupted
  12M-step `ladder_a0_flat_dist__seed50` attempt-3 checkpoint. The report records
  the original campaign fingerprint, attempt provenance, saved config and
  feature-cache hashes, the `main` HEAD and clean working tree, and the audit
  script hash. Frame and shared-bundle evaluation matched every recorded field
  (equity, trades, account, breaches/events, actions, rewards, diagnostics,
  step/session counts, and aggregate metrics) over 99,504 steps on the 351,550
  training bars; both paths recorded 771 trade-event rows / 59 completed trades.
  The result is negative (about -$10,630 net PnL, 15.8% maximum drawdown, and
  one breach), so it is useful only as backend-parity evidence for this
  interrupted in-sample checkpoint. It does not show a successful or completed
  policy and is not confirmatory evidence. Report:
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/p2_checkpoint_frame_bundle_train_a0_dist_12m_20261016/audit.json`.
  P2 remains open for completed-selected-checkpoint acceptance, repeated
  deterministic report validation, and the other schema/compatibility
  requirements above.
- The same 12M `ladder_a0_flat_dist__seed50` attempt-3 training-split parity
  audit was repeated on `main` at `15eef986974e45d162537cf5a086b4de4484dfb4`.
  Again all frame/shared-bundle comparisons passed over 99,504 steps, with 59
  completed trades and the same report-level metrics and provenance hashes as
  the first audit. The repeated report is
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/p2_checkpoint_frame_bundle_train_a0_dist_12m_repeat1_20261016/audit.json`.
  This confirms repeatability of the reported parity outcome and summary values
  for this checkpoint; the audit does not retain serialized cross-run trade
  event rows, so it does not resolve the separate `level_type` determinism
  finding or close P2.
- The report-only evaluator reconstructs the run from frames even when the
  training snapshot says `data_source=bundle`. The backend parity contract must
  be demonstrated for each exact reported checkpoint and source/config
  lineage; a generic unit test or checkpoint hash is insufficient.
- All 18 exact attempt-3 resume checkpoints selected for attempt 4 were
  checksum-checked and sequentially deserialized with `PPO.load(...,
  device="cpu")` on `gpuarm1`. Each loaded timestep matched the recovery record
  (11M–14M); the report also records attempt-3 provenance hashes and the active
  campaign fingerprint:
  `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/checkpoint_load_audit_gpuarm1_20261016_rerun.json`.
  This closes checkpoint deserialization for the current resume inputs only;
  it does not establish that workers can resume successfully or that later
  checkpoints/configs evaluate correctly. The generic
  `audit_campaign_checkpoint_loads.py` did not resolve these inputs because it
  selects the manifest's latest attempt (failed attempt 4), rather than the
  preserved attempt-3 recovery pointers. Fix or replace that audit path after
  the active campaign boundary.

Do not edit fingerprinted training/evaluation source while the active campaign
is live. Complete any required source correction after the campaign boundary,
or prove that the selected evaluation lineage uses only the audited default
values, then run the exact saved-checkpoint recovery comparison.

### Exit criterion

A saved policy can be evaluated deterministically with exactly the observation,
action, risk, cost, and execution semantics recorded for its training lineage.

---

## 11. P3 — Separate numerical evaluation from visualization and produce canonical screen reports

**Priority:** critical path
**Dependencies:** P1 + P2
**Parallelism:** independent checkpoint evaluations may run concurrently if
resources and filesystem load remain controlled

### 11.1 Numerical report first

For each valid checkpoint:

1. deterministically evaluate the complete training split;
2. deterministically evaluate the exposed report-only test split;
3. save raw equity, fills/trades, positions, breaches, actions/diagnostics, and
   machine-readable metric summaries;
4. reconstruct reported metrics from raw position/equity records;
5. verify costs, data/cache hashes, split role, observation/action schemas, and
   checkpoint lineage;
6. record run status independently from visualization status.

A trained policy must not be retrained merely because chart rendering or final
report export failed.

### 11.2 Output profiles

Support three explicit modes if needed for throughput:

- `full`: all mandatory numerical artifacts plus full requested figures;
- `campaign_fast`: all mandatory science/provenance artifacts, with expensive
  optional per-trade graphics bounded/deferred;
- `plots_only`: idempotent post-processing from frozen raw artifacts, with no
  policy retraining or metric reinterpretation.

Numerical parity between `full` and `campaign_fast` is mandatory for actions,
trades, fills, equity, risk events, reward summaries, and metrics.

### 11.3 Canonical per-run artifact pack

At minimum retain:

- resolved config and effective semantic versions;
- source/data/cache hashes;
- run/attempt manifest and command;
- checkpoint/hash;
- deterministic train report;
- deterministic report-only test report;
- raw fills/trades/positions/equity and timestamps;
- gate/breach records;
- activity/action/reward diagnostics when available;
- runtime/resource summary;
- visualization manifest with completed/deferred/failed artifacts.

### Report-generator audit — 2026-10-16

Static review found that `scripts/matrix/summarize_campaign_results.py` sets
the summary-level `seed` to `42` only when every job uses seed 42; otherwise it
sets the field to `null`. This makes a homogeneous seed-50 campaign summary
misstate its seed even though the per-run rows retain it. The live
`campaign_controller.py` invokes this helper when the campaign terminates, so
keep both files unchanged until job `20990395` is terminal. After that source
boundary, update the schema to report the actual seed set (and a scalar seed
only when homogeneous), add seed-50 and mixed-seed regressions, and regenerate
the descriptive report without altering campaign results or fingerprints.

### Exit criterion

Every successfully trained screen run has a canonical numerical report pack, or
an explicit non-hidden reason why recovery is impossible.

---

## 12. P4 — Close G0–G3: correctness, validity, design readiness, reproducibility

**Priority:** critical path
**Dependencies:** P3
**Parallel with:** independent-review preparation and evidence/result packaging

The concise reviewer handoff is
[`independent_review_brief_2026-10-16.md`](independent_review_brief_2026-10-16.md);
the detailed artifact inventory and questions remain in
[`independent_research_review_checklist_2026-10-14.md`](independent_research_review_checklist_2026-10-14.md).
The brief is preparation only; independent sign-off remains open and must be
recorded against the repaired source and final evidence, not the current
contaminated campaign.

### G0 — execution/accounting correctness

Reconfirm on the actual reported implementation:

- action/observation schema and checkpoint compatibility;
- long swing-low / short swing-high causal SL behavior;
- IFVG stop candidates include both the far edge and the causally observed
  through-zone extreme where available, with mirrored long/short tests;
- configured HTFs expose the frozen recent confirmed swing candidate set rather
  than only an undocumented single latest swing, with timeframe/confirmation
  provenance and deterministic ordering/deduplication;
- target ordering, structural-reference provenance, partial exits, residual
  rounding, commissions, and realized-reward accounting;
- decision/submission/fill/valuation clocks;
- bid/ask, spread, slippage, latency, terminal liquidation, and ambiguous OHLC
  convention;
- hard-risk behavior and training/evaluation penalty consistency;
- position-level trade metrics;
- deterministic checkpoint evaluation.

Do not rerun every bounded unit test merely for ceremony. Revalidate the parts
whose source or integration boundary changed and run the repository-required
full quality gate before the final experiment source is frozen.

### G1 — data/feature/evaluation validity

Complete a thesis-ready data audit:

- complete retained calendar and row counts;
- symbols/feed identity and data terms;
- duplicates, gaps, invalid OHLCV, cleaning/resampling;
- timezone/session boundaries and DST implications;
- train/test masks and purged walk-forward construction;
- fold-local fitting and frozen validation transforms;
- assembled-feature causality/prefix checks;
- MTF close/confirmation alignment;
- cache identities and strict mismatch behavior;
- full-split episode coverage;
- explicit classification of the exposed 2026 period.

The 2026-10-16 bounded M1 scan is recorded in
`docs/thesis/data_quality_audit_2026-10-16.json`. It confirms campaign-manifest
SHA-256 matches for both M1 inputs, 528,760 US100 rows and 527,139 US500 rows
from 2024-12-30 through 2026-07-03, zero duplicate or out-of-order timestamps,
zero rows with invalid numeric OHLCV/spread fields, and zero invalid OHLC
ordering. It reconciles 351,550 US100 and 350,098 US500 train rows through
2025-12-31; 177,210 US100 and 177,041 US500 rows fall in the already-exposed
2026 interval. The scan records timestamp discontinuities but does not call
them missing bars because market closures/holidays have not been checked
against an authoritative schedule. This is a bounded bar-file integrity and
coverage result, not G1 closure: feed/export provenance and data-use terms,
broker timezone/DST behavior, schedule-aware gap adjudication, quote-age/trade
impact from missing or duplicate ticks, and the separate fold/episode gates
remain open. Supplemental M1, tick, feature-cache, historical
prefix, and budget evidence from the prior seed-50 source snapshot is archived
in `docs/thesis/evidence/g1_2026-10-16/README.md`. The historical cache/prefix
reports predate changes to `features/build.py`, `structure.py`, `swings.py`, and
`config/default.yaml`.
On source-identical `main` code at `4efbc88`, the focused split/causality/
alignment/fitting/walk-forward suite passed 42 tests and the tick/fill/quote
suite passed 14 tests on the existing GH200 allocation. The exact commands and
scope limits are in `docs/thesis/g1_contract_tests_2026-10-16.json`; these are
unit/contract tests, not substitutes for real-data checks. A fresh raw-feature
prefix audit, `docs/thesis/g1_feature_prefix_current_source_2026-10-16.json`,
then passed all 12 profile/cutoff comparisons across four feature groups
representing all 18 resolved variants, with zero feature deltas. The audit ran
on job `20990395`; its 32 feature/config source hashes, two M1 input hashes,
and script hash matched the files at verification. This covers only the listed
cutoffs using uncached raw builds. The current-source raw-to-cache report,
`docs/thesis/g1_feature_cache_raw_parity_current_source_2026-10-16.json`, also
passed all four representative feature profiles: indexes/columns and content
hash sidecars matched, with zero value, NaN-mask, or numeric deltas. Cache files
were read-only. Its 32 source/config hashes, two M1 hashes, four cache-file
hashes, and the audit-script hash were verified; this does not replace checking
the resolved cache mapping for every variant at final freeze. Fold-local fit
validation, other causal paths, and all other listed validity items remain open;
rerun both audits after relevant source or input changes. The user reported
that no broker/feed identity or data-use/publication terms record is available,
so that provenance gate also remains open.

### G2 — experiment-design readiness

Before any thesis-final seed matrix begins, freeze:

- exact RQ1 control and treatment;
- the single declared structural difference;
- common action/hold/management interface;
- common reward;
- common encoder/architecture;
- data/splits/costs/guardrails;
- seed list and training budget;
- checkpoint-selection rule;
- deterministic evaluation rule;
- cost-stress grid;
- metrics/statistics and stop rules;
- whether any RQ3 mechanism pair is promoted to final paired testing.

### G3 — reproducibility readiness

Verify before final runs:

- exact source and dirty-state identity;
- resolved config and schema versions;
- data/cache hashes;
- immutable run IDs and output paths;
- checkpoints and load/recovery path;
- raw-to-summary metric reconstruction;
- search/trial ledger including invalid/failed/inactive attempts;
- explicit data-role/holdout-exposure record;
- reproduction command/environment.

### Exit criterion

The final experiment can be launched without any unresolved semantic ambiguity
that would change interpretation after seeing its results.

---

## 13. P5 — Freeze the thesis experiment and baseline hierarchy

**Priority:** critical path
**Dependencies:** P4
**Parallel with:** P6 profiling setup, review, evidence/result packaging

### 13.1 RQ1 matched control/treatment

The matched comparison must hold constant:

- PPO implementation and frozen encoder;
- continuous action/hold/management semantics;
- reward and reward weights;
- observation set except the predeclared structural intervention if the
  intervention necessarily contributes state;
- costs, risk limits, execution and session rules;
- training/validation data and fold procedure;
- seed list, rollout/update budget and checkpoint rule;
- deterministic evaluation and metrics.

The treatment introduces only the declared structural eligibility/side
restriction/context. If more than one dimension must differ, state that the
comparison is a bundled system intervention rather than claiming single-rule
causality.

The active seed-50 screen's resolved configurations have already been compared
for design match. `ladder_a0_flat` is the pure legacy A0 reference (default
features, DSR reward, legacy action semantics); `ladder_a0_flat_po3` and
`ladder_a0_flat_dist` use the full PO3/MTF profile, different strategies and
PNL reward, strategy actions, and PD context. These rows are therefore bundled
system comparisons, not matched RQ1 arms. The hash-backed comparison is
`docs/thesis/seed50_screen_config_match_audit_2026-10-16.json`; its source
campaign copy is retained under `outputs/`.
Keep the distinction when reporting the screen; define a separate matched pair
only after P4 and do not select it from the previously inspected holdout.

The `ladder_a1_sl` row is not a matched control for the Idea 1 rungs either.
Its [saved config](../../outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/runs/attempts/ladder_a1_sl__seed50/attempt1/ladder_a1_sl__seed50/20261016_173854_725508_rl_train_seed50_tcn/config.yaml)
uses DSR, `strategy_actions=true`, a signed direction-control coordinate, and
`strategy.name=baseline`; the [saved `ladder_a0_flat_po3` config](../../outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/runs/attempts/ladder_a0_flat_po3__seed50/attempt1/ladder_a0_flat_po3__seed50/20261016_173544_291860_rl_train_seed50_tcn/config.yaml)
uses PnL, no signed direction coordinate, the `po3_ifvg` strategy, and PD
context. Their action schemas and reward/strategy semantics differ. The current
18-row screen therefore contains no ready-made RQ1 pair. P5 must specify a new
matched run identity after P4, aligning the reward, action schema, feature
exposure, risk/management candidates, costs, and runtime before changing only
the predeclared structural entry intervention.

### 13.2 RQ3 mechanism evidence

Use the 18-way screen to choose the smallest set of management mechanisms worth
confirming. A clean mechanism claim requires a matched pair. Do not automatically
run all six rungs × all strategies × five seeds.

Priority order if time permits:

1. stop-management effect;
2. target-management effect;
3. multi-target/leg-weight effect;
4. entry lifecycle/FSM effect, distinguishing eligibility change from visibility
   change.

If an existing ladder comparison is structurally confounded, report it as a
system comparison or omit the causal wording.

### 13.3 Freeze stop-candidate semantics before confirmatory training

Before P7, freeze and version the exact stop candidate menu used by both RQ1 arms:

- preserve the already implemented IFVG far-edge and through-zone-extreme
  candidates;
- add the recent multi-HTF confirmed swing-low/swing-high candidate set required
  by Section 4.2;
- freeze candidate count/lookback, age/staleness rules, buffers, minimum-distance
  rule, manipulation filter, deduplication and deterministic menu ordering;
- verify that both control and treatment receive the same stop-menu capability
  unless stop-menu availability is itself the declared treatment;
- version the stop-candidate schema in run manifests/checkpoints so a model
  trained under the old latest-swing-only menu is never silently evaluated as if
  it had the expanded menu.

The current 18-way seed-42 screen remains valid as **screening evidence for its
own frozen semantics**. If the stop menu changes afterward, do not rewrite its
identity or claim that it tested the expanded menu. Any confirmatory RQ3 stop-loss
claim must use runs trained under the final frozen candidate contract.

### 13.4 Baseline hierarchy

Use baselines for different purposes rather than forcing them into one ranking:

- **sanity controls:** no-trade/null and random policy where useful for pipeline
  validation;
- **classical context:** existing EMA/MACD/RSI and other already-stable classical
  baselines on the same declared cost/split conventions;
- **domain/rule context:** a non-learned liquidity/PO3-style baseline only if its
  implementation and data contract are already stable enough not to displace
  RQ1;
- **system reference:** the pure legacy A0 system, explicitly labelled as a
  complete-system comparator rather than a matched structural control;
- **causal thesis control:** the continuous matched RQ1 control defined above.

An existing supervised LSTM/buy-and-hold/session-breakout baseline may be
reported as context if already reproducible; do not build or retune it merely to
make the table larger.

### 13.5 RQ2 decision

By the end of P5, choose exactly one:

- **DEFER:** encoder dependence is reported as not tested; no thesis-critical
  compute is spent on it; or
- **PROMOTE:** a tightly controlled encoder comparison is added with explicit
  compute/time budget and may not displace RQ1/RQ4 work.

Default under the current deadline is `DEFER`.

### 13.6 Freeze seed, budget, and runtime geometry

The thesis-critical protocol is explicitly resource-aware:

- **One-seed discovery/development stage:** use **seed `50`** as the canonical first
  seed for newly launched/relaunched comparable variants, `20,000,000` PPO
  environment steps per variant, and `n_envs=64`. This stage is for implementation
  validation, system-behaviour screening, candidate selection, and thesis finding
  generation. It is **not** robustness proof.
- Existing seed-42 artifacts remain valid for their own recorded lineage. They may be
  used as historical/exploratory evidence, debugging evidence, or an additional seed
  only when scientifically compatible, but they are not silently substituted into a
  seed-50 one-seed comparison. Do not retrain every completed variant merely to make
  an old seed label uniform unless the comparison actually requires it.
- **Final matched RQ1 confirmation is the last major training goal:** only after the
  implementation, stop-candidate contract, evaluation path, single-seed findings,
  selected RQ1 arms, runtime, costs, metrics, and checkpoint rule are frozen, run at
  least **five paired seeds** for the **small final control/treatment set only**. Do not
  train every variant/rung with five seeds.
- The final paired runs use `20,000,000` PPO environment steps **per seed and per
  arm** and `n_envs=64`, with identical seed values in each control/treatment pair.
  Freeze the seed list before launch. A natural default is `[50, 51, 52, 53, 54]` if
  no other list has already been registered.
- If the selected control and treatment already have a seed-50 run whose source,
  data, stop-candidate schema, action/observation semantics, reward, costs, runtime
  geometry, training budget, and checkpoint-selection rule are **exactly** identical
  to the final frozen protocol, that seed-50 pair may count as the first paired seed.
  This reduces the final incremental compute to four additional seed pairs. Otherwise
  seed 50 is retrained under a new final-run identity.
- `n_envs=64` is the accepted thesis-runtime target. Do **not** change to 32 merely
  to match stale manuscript prose. Any later geometry change requires a pre-result
  runtime/scientific justification and creates a newly frozen protocol.
- Requested steps, completed steps and any early termination are reported per run;
  never label a shorter/incomplete run as a completed 20M run.

### Exit criterion

A one-page final experiment specification exists, and P5 ends only when RQ1 arms,
stop-candidate schema, seed policy, 20M budget, `n_envs=64`, costs, checkpoint-
selection rule, data roles, metrics and evaluation protocol are frozen before
confirmatory outcomes are inspected. The seed-50 one-seed finding packs are complete
for all variants needed to make the final selection. No final paired run begins with an
unresolved control, seed, budget, data, cost, metric, selection field, or missing
checkpoint/log/provenance bundle from the development stage.

---

## 14. P6 — Freeze the production runtime before expensive paired runs

**Priority:** thesis-critical efficiency gate
**Dependencies:** P4 plus the safe end of the screen; may overlap P5 design work
**Parallel with:** review and result-pipeline work

### 14.1 Exact-config end-to-end baseline

Profile one trainer using the exact final effective configuration and separate:

1. startup/cache load;
2. environment reset/observation construction;
3. rollout/IPC waiting;
4. PPO update;
5. callbacks/dashboard/diagnostics;
6. checkpoint writing;
7. deterministic numerical evaluation;
8. graphics/export.

Capture matching CPU pressure/throttling, GPU utilization/VRAM, cgroup
memory.current/peak/events, PSS/USS, process count, I/O, and wall time.

### 14.2 Reuse completed throughput knowledge

An approximately 30-minute live resource sample, including frozen input logs,
a collection script, and their SHA-256 values, is archived at
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/p6_resource_window_20261016_2000/`.
This is preliminary resource-planning evidence only: its campaign has a known
model-visible feature leak in 17/18 configurations, the sample is not a
full-run profile, and it does not establish final-runtime makespan or phase
costs. It suggests the current 18-worker setup is GPU-saturated while assigned
CPU and RAM retain headroom; confirm against a valid matched run before
freezing P6.

Treat full-rollout device staging as the leading fast-path candidate because it
already has exact minibatch parity and repeat measurements. Do not reopen
closed/no-gain ideas such as pinned prefetch, deferred scalar synchronization,
broader precision, or GPU-environment rewriting unless the exact final profile
shows a new bottleneck.

Use the measured high-concurrency result only as a starting candidate. Reconfirm
campaign-level behavior because PPO-core throughput excludes callbacks,
evaluation, checkpointing, report export, and multi-run coordination.

### 14.3 Concurrency and `n_envs`

For the thesis-critical P7 protocol, use **`n_envs=64`** unless P6 discovers a
blocking correctness/resource problem before the protocol freeze. There is no
requirement to return to 32 environments to match draft text.

Choose worker/GPU concurrency by reliable **aggregate** throughput and complete
run makespan with adequate memory/CPU/GPU headroom. Compare alternative `n_envs`
values only with rollout/update semantics explicitly controlled. If changing batch
size, `n_steps`, or epochs changes the number of optimizer updates/learning
dynamics, treat it as a new scientific configuration rather than a transparent
speed fix. The broader `16/32/64` geometry comparison belongs to required Q2
unless a pre-P7 runtime defect makes it necessary for the thesis configuration.

### 14.4 Reporting overhead

Keep numerical evaluation mandatory. Expensive plots may be deferred to
post-processing after numerical completion. If interim stochastic dashboard
prediction remains in training, preserve/restore all relevant RNG state or prove
that the callback cannot perturb later training actions. Logging cadence that
also controls algorithmically meaningful updates must not be reduced merely to
print less.

### Runtime acceptance

Promote the final runtime only if:

- source/data/effective config/hardware are matched;
- the reduced phase is measured;
- aggregate end-to-end throughput improves reproducibly;
- no semantic, RNG, numerical, evaluation, provenance, or resource regression
  appears;
- training still covers the intended split and all mandatory evaluation remains.

### Exit criterion

One runtime implementation/configuration is frozen for P7. The final experiment
will not be launched on an unnecessarily slow or scientifically altered runtime.

---

## 15. P7 — Last empirical compute gate: final paired confirmation and cost stress (G4)

**Priority:** final empirical thesis gate and **last major training goal**
**Dependencies:** all required single-seed variant work and finding packs complete + P5 experiment freeze + P6 runtime freeze + no blocking review defect
**Parallelism:** independent final seed pairs may run concurrently on authorized resources

### 15.1 Main matched run matrix

Default target for RQ1:

- **only the final selected matched control/treatment pair** receives the >=5-seed
  robustness treatment by default;
- use at least five paired seeds total, with identical seed values per pair and the
  list frozen before launch; seed 50 is the preferred first pair;
- **20,000,000 PPO environment steps per seed per arm**;
- **`n_envs=64`** for every final paired run unless the whole protocol was
  pre-result re-frozen under P6;
- identical data, costs, risk, stop-candidate schema, budget, architecture,
  runtime, selection rule, and evaluation;
- reuse the seed-50 discovery run for a selected arm only when its full identity is
  exactly compatible with the frozen final protocol; otherwise train a new explicit
  final-run identity;
- add a five-seed RQ3 mechanism pair only if it is specifically promoted before
  P7 and resources permit; this is not the default.

**Do not expand the 18 configurations to five seeds.** Finish implementation and
one-seed findings first, select the smallest final comparison set, and spend robustness
compute only there.

### 15.2 Selection and checkpoint rule

Model/config selection occurs only from authorized training/walk-forward
information. The exposed 2026 report interval may be replayed for descriptive
reporting after the frozen policy exists, but cannot determine which seed,
variant, checkpoint, or hyperparameter is kept.

### 15.3 Cost sensitivity

Before looking at stress outcomes, freeze a realistic grid that varies at least:

- spread;
- slippage;

and, where meaningful, execution latency. Units and broker assumptions must be
explicit. Cost stress is primarily evaluation sensitivity on frozen policies;
do not retrain per cost cell unless that is separately predeclared as a different
research question.

### 15.4 Required result outputs

Per seed and aggregate:

- completed training budget/coverage;
- survival and breach outcomes;
- net PnL/return/final equity;
- Sharpe/Sortino/Calmar and maximum drawdown;
- profit factor, expectancy, win rate;
- completed positions, fills, turnover, exposure, holding duration;
- transaction costs;
- action/activity distributions and entry funnel;
- long/short/flat or hold/defer behavior;
- reward components versus economic PnL/equity;
- relevant structural/entry timing and level/session breakdown;
- paired control-treatment differences;
- dependence-aware uncertainty;
- cost-stress cells;
- all zero-trade, failed, incomplete, or invalid seeds visibly labelled.

### 15.5 Stop rule

If the treatment loses, is inactive, fails survival, or is inconclusive, keep the
result. Do not:

- cherry-pick successful seeds;
- silently extend only one arm's budget;
- change the gate after outcomes are known;
- alter the treatment and call it the same experiment;
- repeatedly inspect the exposed period until a favorable version appears.

### Exit criterion

G4 has a complete paired evidence pack supporting one of four conclusions:
`treatment improves`, `no material advantage established`, `treatment is worse`,
or `insufficient valid evidence`, with the reason and uncertainty explicit.

---

## 16. P8 — Independent review and thesis evidence freeze

**Priority:** critical path
**Dependencies:** review preparation begins at P4; final disposition uses P7 evidence
**Parallel with:** statistics/figure generation and result aggregation

### 16.1 Reviewer questions

Obtain a durable reviewer disposition on the actual evidence for:

- experiment design and attribution boundaries;
- fill/accounting/execution assumptions;
- reward and risk interpretation;
- feature causality/fitting;
- model selection and exposed-period handling;
- training duration/full-split exposure;
- performance/generalization claims;
- resource/provenance/reproducibility limitations.

If a reviewer finds a blocking defect, invalidate only the downstream evidence
contaminated by that defect and return to the earliest affected work package.

### 16.2 Freeze result identities

Create a single thesis result manifest mapping:

```text
claim / RQ
  -> comparison/run IDs
  -> source/config/data/checkpoint hashes
  -> raw evidence
  -> metric/statistical output
  -> table/figure
  -> allowed interpretation / limitation
```

No later experiment silently replaces a run in this manifest.

### 16.3 Evidence categories

Keep separate:

- software/correctness evidence;
- one-seed screening evidence;
- matched multi-seed thesis evidence;
- classical/rule baselines;
- exposed-2026 descriptive evidence;
- fresh holdout evidence, if any;
- runtime/resource benchmarks;
- Q-stage evidence, kept separate from thesis evidence unless formally promoted.

### Exit criterion

Every thesis number and claim is traceable, its evidence role is clear, and the
thesis evidence set is frozen.

---

## 17. E0/E1 — Results packaging runs continuously with the technical path

The agent must not spend the critical path drafting thesis chapters. Instead it builds the
**results-writing substrate** continuously so every terminal experiment becomes immediately
usable later.

### E0 — build before final results

Prepare and validate now:

1. the canonical per-run evidence-bundle writer and artifact index;
2. resolved-config, source/data/cache/checkpoint fingerprint capture;
3. durable stdout/stderr/Slurm/controller/resource-log collection;
4. checkpoint hashing, metadata, and load-validation tooling;
5. deterministic evaluation export for metrics, equity, positions, fills, breaches,
   actions/behavior, reward components, entry funnels, and costs where applicable;
6. machine-readable seed/run summary tables;
7. reproducible statistics code for paired differences, uncertainty, cost sensitivity,
   survival/breach outcomes, and dependence-aware reporting;
8. figure generation for learning curves, equity, drawdown, activity/actions, survival,
   mechanism comparisons, and cost sensitivity;
9. result-card templates containing evidence role, allowed claim, forbidden overclaim,
   limitations, exact artifact paths, and likely thesis destination;
10. global experiment/trial ledger including failed, inactive, interrupted, and zero-trade
    attempts;
11. final result-manifest schema mapping question/RQ -> run IDs -> seeds -> checkpoint hashes
    -> raw eval -> statistics -> figure/table artifacts -> interpretation boundary;
12. validation that older seed-42 artifacts remain historical/exploratory and newly comparable
    one-seed work uses seed 50 / 20M / `n_envs=64` unless a frozen run identity says otherwise.

### E0 paired-seed analysis utility — prepared, not yet applied to thesis results

`scripts/thesis/analyze_paired_seed_results.py` provides a separate paired-seed
analysis stage. It fails closed unless each arm is a succeeded
`matched_confirmation` or `robustness` run, every declared match field is
present and equal except the explicit intervention, and at least five unique
seed pairs are supplied, matching P7's minimum. It preserves per-seed arm values and run provenance,
reports a deterministic paired-seed bootstrap interval, an exact sign-flip
test where the number of pairs permits it (otherwise a seeded Monte Carlo
version), and Holm-adjusted p-values across requested metrics. It labels the
interval as between-seed uncertainty and explicitly defers temporal-path
uncertainty to the existing moving-block procedure. Its synthetic tests verify
the mechanics only; no campaign or thesis claim has been analyzed with it.

### E1 — populate continuously as runs terminate

For every terminal run or comparison, the agent immediately:

- seals the evidence bundle before moving on;
- validates the checkpoint and all expected raw outputs;
- records completed timesteps, wall time, effective throughput, resource use, and failure state;
- reconstructs metrics from raw records and cross-checks reported summaries;
- generates the applicable numerical tables and figures;
- creates/updates `finding.md` with exact evidence links and claim boundaries;
- updates the global trial ledger and result manifest;
- marks missing artifacts as blockers rather than silently leaving them for the end.

The final >=5 paired-seed evidence supersedes only the robustness/causal conclusion. It does
not erase valid single-seed implementation, behavior, failure, or mechanism findings.

---

## 18. P9 — Freeze and hand off the complete results package

**Priority:** results-critical terminal package
**Dependencies:** P8 final evidence/review disposition + E0/E1 artifact pipeline

### 18.1 Results handoff requirements

1. Populate the global result manifest only from frozen evidence.
2. Include failed, incomplete, inactive, and zero-trade runs with explicit dispositions.
3. Keep partial-exit legs separate from completed-position counts.
4. Distinguish system-level comparisons from isolated mechanism effects.
5. Report effect size/uncertainty and cost sensitivity, not only a winner label.
6. If fresh OOS is unavailable, record confirmatory temporal generalization as untested rather
   than weakening the holdout definition.
7. Verify every table/figure can be regenerated from retained machine-readable artifacts.
8. Verify every final checkpoint has a hash, metadata, load check, exact config, seed, steps,
   `n_envs`, source/data/cache identity, and deterministic evaluation reference.
9. Produce one compact `results_summary.md` that enumerates every empirical question, its
   disposition, the strongest supported interpretation, limitations, and artifact links. This is
   an evidence index for later thesis writing, not chapter prose.

### 18.2 RQ/result dispositions

For each retained question record exactly one supported disposition:

- supported;
- unsupported / evidence of worse performance;
- no advantage established / inconclusive;
- not tested.

Do not turn `no advantage established` into `no effect exists`, and do not convert a single-seed
screen into a robust treatment claim.

### Exit criterion — `RESULTS_PACKAGE_COMPLETE`

A new agent or the thesis author can answer **what ran, what happened, how certain it is, where
the raw evidence is, which checkpoint produced it, and what may safely be claimed** without
reading terminal history or rerunning an experiment.

---

## 19. P10 — Supervisor PDF freeze — 2026-11-09

**In scope for the active user goal, downstream of P9.** Use the frozen results
package to write or revise and assemble the thesis. By **2026-11-09**, prepare the complete PDF rather than waiting for optional scope.
The package should include or link the evidence manifest and a short change/
open-issue summary.

**Authoring preparation check — 2026-10-16:** the repository file inventory
contains no identifiable thesis manuscript source or thesis PDF (`.tex`,
`.docx`, `.odt`, or `.pdf`) under the thesis-related paths. P10 must therefore
locate the author's external draft, if one exists, or start the manuscript from
the frozen P9 evidence package. Do not fill result-dependent chapters before P9.

If a non-core experiment is still running, do not hold the PDF for it. Record it
as future/ongoing work unless the supervisor explicitly requests inclusion.

---

## 20. P11 — Supervisor feedback and faculty submission

### 2026-11-09 to 2026-11-15 — feedback cycle

- triage comments into scientific blocker, clarity/citation, formatting, and
  optional extension;
- scientific blockers return to the earliest contaminated package;
- do not change frozen results merely to improve the narrative;
- regenerate affected tables/figures from evidence, not manual edits;
- rerun PDF/reproducibility checks after substantive corrections.

### 2026-11-16 to 2026-11-22 — final freeze

- complete language/formatting/reference polish;
- recheck metadata, licence, disclosure, data/model availability;
- verify that abstract, RQ table, conclusions, and defense summary agree;
- create final source/PDF/evidence backup;
- stop adding optional experiments unless a true blocker requires them.

### 2026-11-23 — submission

Submit the faculty version and record the exact submitted PDF/source archive
identity.

### Exit criterion

`SUBMISSION_COMPLETE`.

---

## 21. Deadline-aware descope order

If time or compute becomes constrained, remove scope in this order:

1. RQ2 encoder comparison;
2. optional supervised/classical baseline expansion beyond already stable
   comparators;
3. broad five-seed replication of every mechanism rung;
4. nonessential per-trade visualizations and cosmetic dashboards;
5. optional live/paper-trading work.

Q0–Q7 are not descoped from the overall user goal. If needed to protect the
Nov 9/Nov 23 thesis milestones, execute them after `SUBMISSION_COMPLETE` rather
than delaying the thesis evidence freeze or submission.

Do **not** cut:

- correctness/provenance needed to trust reported evidence;
- the matched RQ1 design once it is the thesis core;
- honest failure/holdout disclosure;
- the minimum robustness needed to state uncertainty responsibly;
- the complete evidence/review window simply to add another model.

If valid empirical evidence cannot be completed, the fallback is an honest
limited/negative/inconclusive thesis or a supervisor-approved design/audit scope,
not fabricated performance.

---

# PART II — REQUIRED Q0–Q7 RESEARCH PROGRAM

## 22. Entry rule for the Q program

The Q0–Q7 program is mandatory for the overall mission. Its default execution
window is after `SUBMISSION_COMPLETE`; earlier execution is allowed only when
all of these are true:

- no thesis-critical work waits for the same developer/reviewer/compute resource;
- its parent/control is frozen;
- required implementation and schema/parity tests pass;
- the run has an explicit research role and immutable identity;
- the author explicitly authorizes the compute launch (authorized on
  2026-10-16 for the Q0–Q7 program; each new Slurm allocation still follows the
  repository's allocation-approval rule);
- the result will not silently redefine the frozen thesis experiment.

If any condition is false, continue the thesis-critical path and start the Q
program after submission. Do not skip a Q stage or mark the overall mission
complete because thesis submission has been made.

### Standard stage pattern

```text
implementation/schema verification
        -> freeze parent control
        -> one declared discovery seed per arm
        -> deterministic full-split evaluation
        -> provenance + learning-quality review
        -> winner / no-winner decision
        -> paired multi-seed follow-up only if justified
        -> freeze next parent
```

Child stages never run against an unresolved symbolic winner.

---

## 23. Q0 — Establish the A3 quality-program control

Freeze the A3/account-state-v1 control with complete source/data/config/action/
observation provenance, full chronological exposure, deterministic evaluation,
and diagnostics. This is a new quality lineage, not a substitute for the thesis
screen reference unless explicitly promoted.

---

## 24. Q1 — Account-state v2

Change only the registered account-state observation schema relative to Q0.
Verify checkpoint metadata, frame/shared-bundle parity, no-position and
short-side coordinate semantics, and deterministic evaluation.

Q2 cannot start until the Q1 disposition is frozen.

---

## 25. Q2 — Rollout geometry

This is a required post-core quality experiment, not a request to change the
thesis-critical runtime. The thesis-critical P7 configuration remains
`n_envs=64`. Compare `n_envs = 16, 32, 64` from the frozen Q1 parent.

These arms may run in parallel on independent/non-confounding resources.
Evaluate both:

- learning quality and behavior;
- aggregate throughput/makespan/resource efficiency.

Do not choose a geometry on FPS alone.

---

## 26. Q3 — Reward/risk alternatives

From the frozen Q2 parent, compare the registered RR/fixed-risk and
fixed-risk equity-delta alternatives. Predeclare whether the stage is one
multi-arm family or separate sub-comparisons.

Reward changes are scientific treatments. Keep architecture/action/data/costs
fixed and report reward diagnostics separately from economic PnL/equity.

---

## 27. Q4 — Action/policy distribution

From the frozen Q3 parent:

- Q4A: semantic Gaussian heads;
- Q4B: hybrid categorical-continuous distribution, after implementation.

Q4A and Q4B may run in parallel once both are valid. Test probability/log-prob
math, action bounds, schema persistence, gradients, save/load, and deterministic
evaluation before training claims.

---

## 28. Q5 — Learned constrained RL / Lagrangian line

The opt-in implementation uses primal-dual scalarized-return PPO. On each
training transition, its scalar cost is the largest nonnegative increment
among the environment's normalized daily-loss, max-loss, and trailing-risk
costs. Episode cost is the sum across the complete chronological training
split. PPO receives `environment_reward - lambda * cost`; after completed
episodes, projected dual ascent updates `lambda` against the configured
episode-cost budget. Initial defaults are budget `1.0`, dual step `0.1`,
initial multiplier `0.0`, and cap `100.0`. The multiplier and method settings
are stored in the model checkpoint and run protocol. This is a learned
objective, not a formal risk guarantee, and it does not replace the hard
account guardrails. Q5 is gated to full-split PPO; MVP and risk-terminated
episodes are rejected.

Before any Q5 run, verify:

- the explicit cost/constraint signal and episode aggregation;
- the selected scalarized-return PPO objective and dual update;
- multiplier initialization, bounds, and checkpoint/log persistence;
- diagnostic separation between learned constraint objective and hard account
  guardrails;
- deterministic evaluation under the same reporting gates.

Compare to the frozen Q4 parent. Do not call the method a formal constraint
guarantee unless the actual algorithm and assumptions justify that statement.

---

## 29. Q6 — Entry-state machine visibility

From the frozen Q5 parent compare three arms:

- no-FSM control;
- FSM enabled with state hidden from the policy;
- FSM enabled with state visible to the policy.

Preserve the accepted post-IFVG semantics. Compare no-FSM to hidden-FSM to
estimate the effect of changing eligibility/state transitions; compare
hidden-FSM to visible-FSM to estimate the effect of exposing that state to the
policy. The existing two-arm visible/no-FSM comparison confounds these effects
and is not sufficient. Freeze the same parent/configuration for all three arms;
run them only after the Q5 parent is frozen.

---

## 30. Q7 — Encoder comparison

From the frozen Q6 parent compare LiteTCN and TCN with identical data, budget,
seeds, action/reward/risk semantics, and evaluation.

This is the clean place for encoder dependence. A broader GRU/Transformer/VAE
program is a separate post-thesis research decision rather than an implicit
continuation of Q7.

After Q7, declare a winner/no-winner or stop the lineage. Do not retroactively
replace the frozen thesis results without deliberately reopening the thesis
experiment and repeating the required robustness checks.

---

# PART III — OPTIONAL POST-THESIS PRODUCTION / LIVE PARITY

## 31. Productionization is not thesis evidence

Only after the scientific/reproducibility archive is stable should the system be
promoted toward paper/live execution.

Required before any live-money interpretation:

- checkpoint/action/observation schema parity in live inference;
- feature/session/timezone parity between research and live streams;
- bid/ask, tick-size, lot-step, latency, slippage, partial-fill, and stop-order
  behavior verified against the broker interface;
- aggregate account-risk and concurrent-position limits;
- disconnection/stale-feed/fail-safe behavior;
- deterministic replay of representative live decisions against research logic;
- paper-trading/shadow period with complete logs;
- explicit statement of all simulator/live gaps.

Live profitability is not required for the thesis and should not delay
submission.

---

# PART IV — OPERATING PROTOCOLS

## 32. Experiment lifecycle for every new run

### Before launch

Record:

- research question/hypothesis;
- control/treatment and exact changed factor;
- run role;
- data/split/fitting scope;
- source/config/cache hashes;
- action/observation/feature/metric versions;
- seed, budget, rollout/update geometry;
- cost/risk/execution rules;
- checkpoint-selection and evaluation rules;
- expected artifacts;
- acceptance/stop criteria;
- compute/resource allocation.

### During execution

Track:

- progress and actual completed budget;
- episode coverage/end reasons;
- checkpoint creation;
- controller/worker failure events;
- resource telemetry;
- diagnostics that do not alter the policy trajectory.

Do not inspect a fresh final holdout during this phase.

### After execution

1. verify checkpoint load;
2. run deterministic evaluation;
3. reconstruct metrics from raw records;
4. run provenance/artifact audit;
5. assign valid/failed/incomplete/inactive status;
6. compare only against compatible runs;
7. record the conclusion and next decision.

---

## 33. Root-cause protocol for anomalies

For every failed or surprising result, use:

```text
symptom
-> earliest stage that diverged
-> hypothesis
-> diagnostic test
-> evidence
-> confirmed/rejected cause
-> correction/workaround
-> rerun result
-> effect on thesis evidence
```

Do not jump directly from an unexpected final Sharpe/equity curve to changing
reward, features, or model architecture.

---

## 34. Fair-comparison checklist

Before calling two runs comparable, verify:

- same authorized data/calendar;
- same fitting/split procedure;
- same execution/cost/risk rules;
- same action/observation semantics except the declared treatment;
- same model/training budget and selection rule where attribution requires it;
- same seeds or paired seed design;
- same deterministic evaluation/metrics;
- complete provenance and compatible run roles.

If any required dimension differs, describe the comparison as system-level or
exploratory rather than an isolated treatment effect.

---

## 35. Status/reporting cadence

Maintain one concise current-status record containing:

1. executive state;
2. current active package(s);
3. completed evidence since the last update;
4. failed/blocked/inactive runs;
5. data/provenance changes;
6. experiment matrix and comparability;
7. key metrics/uncertainty only from valid artifacts;
8. correctness/validity issues;
9. review decisions needed;
10. exact next action and acceptance criterion.

The thesis supervisor should be able to understand what is correct, what failed,
what is uncertain, and what will happen next without reading raw logs.

---

## 36. Final deliverables and results artifact set

The Codex agent research mission must finish with enough structured evidence that the thesis can be
written afterward, checked, and reproduced without reconstructing a campaign from tmux history or
human memory. Deliverables accumulate throughout the plan; do not wait until P9 to create them.

### 36.1 Deliverables by evidence stage

| Evidence stage | Mandatory deliverables | Later evidence/review use |
|---|---|---|
| Correctness / implementation | focused test logs, golden traces, source/config snapshot, issue disposition, finding cards | Methods, implementation validity, limitations |
| Data / feature validity | dataset audit, split/calendar report, causality/fold-fitting checks, feature/cache hashes | Data and Experimental Design |
| Stop/TP/action semantics | contract tests, candidate/provenance examples, schema versions, resolved config | Methodology, mechanism definitions, appendices |
| Seed-50 one-seed variant findings | full per-run bundle, stdout/stderr/Slurm logs, final/selected checkpoint(s), hashes, training traces, deterministic eval, raw trades/fills/equity, metrics, finding card | Experimental coverage, learning behaviour, mechanism/system screening, failure analysis |
| Classical/rule baselines | resolved settings, logs, raw outputs, metrics, comparable-cost declaration, finding cards | Baseline context |
| Runtime/resource freeze | profiler/resource logs, accepted runtime config, throughput/memory report, parity checks | Reproducibility appendix; not strategy-performance evidence |
| Final >=5 paired RQ1 confirmation | paired run bundles for the selected control/treatment only, paired seed table, aggregate metrics, effect intervals, cost stress, final checkpoints | Headline Results, RQ1, RQ4, Abstract/Conclusion |
| Optional promoted RQ3 pair | same paired bundle standard if actually promoted; otherwise retain one-seed/system-level evidence only | RQ3 with correct attribution label |
| Failed/inactive/zero-trade runs | config + command + logs + partial artifacts + explicit failure/inactivity reason | Coverage, threats to validity, negative findings |
| Evidence freeze | claim-to-run manifest, final RQ answer table, independent review disposition | Results-to-claim consistency |
| Results-package freeze | `results_summary.md`, global artifact/result manifest, all figure/table sources, archive manifest, reproduction notes | Direct handoff for thesis writing and review |

### 36.2 Final retained artifact set

Before supervisor freeze and again before submission, retain:

- `results_summary.md` and global result/artifact manifests;
- figure/table source artifacts;
- result manifest mapping claims to evidence;
- final RQ answer table;
- dataset audit and calendar-role record;
- experiment/trial ledger including failures;
- final control/treatment configs;
- per-run commands and resolved configs;
- durable stdout/stderr/Slurm/controller logs for every thesis-relevant attempt;
- training traces (`progress.csv`, TensorBoard/monitor data where used);
- final and selected checkpoint files where applicable, plus successful load checks and SHA-256 hashes;
- per-run `finding.md` cards and a global artifact index;
- seed-level and aggregate metric tables;
- raw equity/trades/fills/breach records or durable references;
- source/config/data/cache identities;
- independent-review disposition;
- cost-stress output;
- reproduction commands/environment notes;
- data/model availability and reproducibility notes needed to describe the evidence later;
- thesis/defense-ready result figures generated from the same frozen evidence.

Do not include an unverifiable artifact merely to make the archive appear more
complete.

---

## 37. Results-first execution window

These are target windows for a continuously running Codex agent. Scheduler
availability and scientific gates override dates. P0–P9 evidence remains the
critical path; P10/P11 manuscript and submission work is also in the active
goal and starts from the frozen evidence package.

| Window | Critical result target | Horizontal work while compute runs |
|---|---|---|
| **16–18 Oct** | P0/P2/E0: safe boundary, evaluator/schema repair, evidence-bundle and result-index tooling | correctness tests, HTF stop-candidate completion/verification, recovery tooling |
| **18–20 Oct** | P1/P3: complete/reconcile the one-seed screen and canonical deterministic reports | package each terminal run immediately; prepare stats/figures; audit failures |
| **20–22 Oct** | P4/P5: validity audit, select/freeze final matched pair and exact semantics | independent review prep; final configs; result-manifest population |
| **22–23 Oct** | P6: benchmark/freeze final training concurrency and runtime using aggregate makespan | prepare paired-seed launch matrix and automated evaluation |
| **23–25 Oct** | P7: final >=5 paired seeds for the selected control/treatment only + cost stress | evaluate/package finished seeds as they terminate; no idle waiting |
| **25–27 Oct** | S0/E1/P8: final statistics, uncertainty, figures, review disposition, evidence freeze | regenerate/cross-check tables and figures from raw artifacts |
| **by ~27 Oct target** | **P9 `RESULTS_PACKAGE_COMPLETE`** | complete logs + checkpoints + raw outputs + findings + global manifest handed off for thesis writing |
| **by 2026-11-09** | **P10 supervisor PDF** | write/revise from frozen findings; render, inspect, and package source, evidence links, and open-issue summary |
| **2026-11-09 to 2026-11-23** | **P11 feedback and faculty submission** | resolve supervisor comments, freeze final PDF/source/evidence archive, record submitted artifact identity |
| **after 2026-11-23 submission** | **Required Q0–Q7 program** | execute Q0 through Q7 in dependency order; freeze each parent, evaluate and package each stage before opening the next; obtain allocation approval for each new Slurm request |

The date is a target, not permission to bypass gates. If compute queues or a correctness defect delay
P7, the agent should keep every other READY lane moving and preserve the result package rather than
launching extra variants.

### Reforecast after the G1 causality failure — 2026-10-16

The October 27 P9 target is now at high risk and should not be treated as the
current expected completion date. The active seed-50 screen has a model-visible
NY-session lookahead in 17/18 configurations; the current source regression
fails as expected. A valid screen therefore requires a post-boundary source
repair, a passing same-session future-poison test, versioned rebuilds of all
three prebuilt caches, and passing real-market prefix/fold audits before fresh
training can inform selection.

For compute planning only, the current 18-worker logs' observed 300–500
environment steps/second per worker imply roughly 11–19 hours of PPO collection
for a 20M-step run. If 18 workers sustain that rate, a fresh one-seed screen's
collection phase is also about 11–19 hours; this excludes evaluation, checkpoint
export, reporting, allocation gaps, and any rate change after the feature repair.
The final five paired seeds require 200M aggregate environment steps if no
discovery pair qualifies for reuse. At two concurrent workers sustaining
300–500 steps/second each, that is about 56–93 hours of training alone. These
are arithmetic scenarios, not a runtime acceptance result; P6 must measure the
frozen configuration and choose concurrency by reliable makespan.

Current planning range from this gate failure is **about 2–4 weeks to P9** and
**about 3–6 weeks through P11 submission**, assuming allocations can be
reacquired without long gaps, the repair closes G1 on the first attempt, and
the independent review does not find a new blocking defect. The November 9 PDF
date is at risk; November 23 remains possible but is not assured. Reforecast
after the cache rebuild, fresh screen, and P6 throughput measurement. Do not
compress or skip evidence gates to meet either date.

### Runtime planning assumptions

- Treat historical FPS figures as measurements tied to their own run, phase, and
  hardware. Do not use a single-worker or brief startup sample as the current
  campaign forecast.
- Estimate remaining time from stabilized aggregate completed timesteps/hour
  across the active workers, then include startup, checkpointing, evaluation,
  report export, queue gaps, and allocation boundaries.
- Do **not** assume 18-way concurrency is optimal for the final paired matrix;
  P6 must choose the concurrency that minimizes reliable total makespan.
- Training wall time overlaps with evaluation tooling, packaging, correctness
  work, statistics code, reviewer preparation, and processing already-terminal runs.

## 38. Status board

Update this table as the persistent agent control surface. It is status, not a second roadmap.

Latest transient Slurm/process state is recorded in the dated
[`live status note`](live_status_2026-10-16_2148.md); the immutable allocation
snapshot remains separate. Keep job IDs, process IDs, countdowns, and live
worker counts in that note rather than in this roadmap.

| ID | Package | Current state | Depends on | Horizontal-safe work |
|---|---|---|---|---|
| P0 | Safe boundary/current snapshot | ACTIVE — immutable campaign snapshot, manifests, run fingerprints, user data, and failed exploratory results remain preserved. H200 job `21004765` completed with exit 0 after its final code gate passed; stopping the approved recovery supervisor also ended the child H200 `salloc` earlier than requested, unintentionally releasing that reservation. No `scancel` was issued. MI210 job `21012389` remains allocated on `gpuamd1` (1 GPU/8 CPUs/128 GiB); ROCm smoke and the full suite passed, with no active trainer step. No GH200 is allocated to this user now. No new 18-variant campaign has launched. See [`live_status_2026-10-17_0336.md`](live_status_2026-10-17_0336.md) | — | P1, P2, E0 |
| P1 | 18-way/full-split one-seed screen reconciliation | ACTIVE BUT SCIENTIFICALLY CONTAMINATED — preserve the old screen as exploratory: the real-market fold-fit audit found model-visible `prev_ny_high`/`prev_ny_low` leakage in 17/18 saved run configurations. The terminal recovery report records 11 succeeded, 4 train-survival failures, 3 interrupted, and current-attempt full-split telemetry for 17/18 variants (1,088 environments); `ladder_a0_flat` lacks a current-attempt completed-split record. Eight variants are inactive with zero train/test trades. No result is both causally admissible and gate-passing. The causal repair now uses the configured clock-completed `08:30–15:00 CT` NY session and cache identity `v22-causal-configured-ny-session`. Post-repair validation passed the four-profile fold poison (one early purged fold) and 12/12 sampled feature-prefix comparisons; the M1 inventory is recorded in `data_quality_audit_2026-10-16_v4.json`. The three named caches were transactionally rebuilt and hash/parquet-verified; report: `prebuilt_cache_rebuild_2026-10-17_mi210.json`, with prior files preserved in the dated backup directory. These checks do not establish exhaustive causality or market-calendar/DST correctness. Vendor/use provenance, remaining reviews, post-boundary CI, and a fresh admissible screen are still open. Finding: `docs/thesis/evidence/p1_18_variant_screen_disposition_2026-10-16.md` | P0 | P2, E0 |
| P2 | Deterministic evaluator/schema repair | ACTIVE — stable lexical tie-break, management-value forwarding, evaluation-worker guardrail parity, and action/observation schema checks are implemented. The complete source gate now passes on the H200: full-tree Ruff format/lint, full Mypy, and 1,594 tests. The same full test suite passes on the separate ROCm/MI210 runtime. This verifies code/runtime checks, not all required report determinism or acceptance for a completed selected checkpoint. Remaining exact checkpoint recovery/parity, repeated deterministic report, and schema/compatibility requirements remain open | P0 | P1, E0 |
| E0 | Evidence/result pipeline + indexes | ACTIVE — provenance index, dispatcher, paired-seed analysis utility, six-campaign trial ledger, and all-attempt screen reconciler exist. The refreshed recovery index is `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/thesis_evidence_index_post_causal_repair_20261017.json` (fingerprint `91d2f05c4c13e67753a47b88a1c900f4887ced66dad51980a018eb09465dd10e`; 11 succeeded, 4 failed, 3 interrupted; 230 source hashes: 220 verified, 10 mismatched, none missing; 172 checkpoint artifacts). The separate v3 index was also refreshed and records 18 interrupted runs, 228 source hashes (207 verified, 21 mismatched, none missing), and 188 checkpoint artifacts. Both indices explicitly do not establish chronological split exposure, successful checkpoint deserialization, or strategy validity; those require their separate audits. Post-repair feature evidence remains bounded to four profiles/one early fold and 12 sampled prefixes. The M1 inventory is PARTIAL because vendor/use provenance, broker calendar/DST, and tick quality are not established. Final canonical reports still await P1/P2 closure | P0 | P1, P2 |
| P3 | Canonical numerical screen reports | BLOCKED | P1 + P2 | per-checkpoint eval/package |
| P4 | G0–G3 validity/reproducibility | BLOCKED | P3 | Reviewer preparation includes the bounded notes in `docs/thesis/related_work.md` and `docs/thesis/threats_to_validity.md`, plus open decision items in `docs/thesis/independent_research_review_checklist_2026-10-14.md`; the latest real-market causal audit fails and no independent reviewer has signed off |
| P5 | Freeze matched design/baselines | BLOCKED | P4 | P6 prep, review |
| P6 | Production runtime/resource freeze | PREP-ONLY | P4/screen-safe | paired-launch prep |
| P7 | Final selected-pair >=5-seed robustness + cost stress | BLOCKED | all single-seed findings + P5 + P6 | eval/package terminal seeds |
| S0 | Statistics + final figures | PREP-ONLY | schema can be built early; final values need P7 | finished-seed analysis |
| E1 | Result aggregation/finding cards | ACTIVE-AS-ARTIFACTS-ARRIVE | E0 | every terminal run |
| P8 | Review + evidence freeze | PREP-ONLY | final P7 evidence | figure/result cross-checks |
| P9 | `RESULTS_PACKAGE_COMPLETE` handoff | BLOCKED | P8 + E1 | archive/index QA |
| Q0–Q7 | Required research program | COMPULSORY, NO RESEARCH RUNS — no stage result or winner exists. Q0/Q1 controls and schema, Q2 geometry/logging, Q3 reward/risk code, Q4A/B policy code, Q6 FSM arms, and Q7 encoders have differing levels of source/test support; source presence is not a completed experiment, and several arms remain deliberately fail-closed. Q5 now has an opt-in primal-dual scalarized-return PPO implementation and tests, but full verification and Q4-parent freeze remain open. See [`q0_q7_implementation_audit_2026-10-17.md`](q0_q7_implementation_audit_2026-10-17.md). Run in dependency order after P11 by default; earlier preparation or execution requires the stated entry gates and must not displace thesis-critical work | ordered stage parents | implementation, validation, and evidence preparation |

**GH200 allocation guard (user direction, 2026-10-16):** the thesis campaign
may have at most **2 GH200 GPUs total across all active allocations**, and no
single request may ask for more than 2. Count requests across jobs and nodes:
1 GH200 on gpuarm1 plus 2 on gpuarm2 (or 2 on gpuarm1 plus 1 on gpuarm2) is
forbidden because the total is 3. The intended 18-trainer shape is one GPU per
node across two nodes. The cluster's four configured GH200s describe hardware
capacity, not this campaign's allowance. For continuity, one 2-GH200 successor
may be queued behind an explicit Slurm dependency on the current job's
termination; verify the dependency prevents overlap and include pending jobs in
the aggregate count. As of the 2026-10-16 resource choice, the active pair is
still two GH200s and its single dependent successor has been retargeted to one
H200 to reach compute sooner; see the live status note for the current
scheduler forecast.

### G1 causality gate failure — 2026-10-16

The reproducible real-market audit
`docs/thesis/audit_real_market_fold_fit_scope_current_source_2026-10-16.py`
and report `docs/thesis/g1_real_market_fold_fit_scope_2026-10-16.json` apply a
20% OHLC perturbation only to a purged validation window. The baseline profile
passes; `ladder_a1_sl`, `ladder_a0_flat_po3`, and `ladder_a0_flat_dist` fail
because training features `prev_ny_high` and `prev_ny_low` change from
`2025-02-12 01:31+03`. The changed fields remain in the model observation
schema. Those profiles map to 17 of the active 18 worker configs, and the
campaign manifest's `quant_rl/features/session_ohlc.py` hash matches the audit
source hash. The recovery workers' logs confirm hash-validated use of the
baseline, Idea 1, and Idea 2 named prebuilt caches. Their refresh record
(`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/prebuilt_feature_cache_refresh_20261016.json`)
shows all three caches were regenerated from the current source before this
allocation; the cache sidecar hashes matched the generated caches. The earlier
12 prefix cutoffs were at 12:37+03 and did not probe the NY-session opening
interval where this defect appears.

Source inspection identifies the cause in `prev_ny_high_low`: it broadcasts
each NY session's final high/low, then applies a one-row shift within that same
session group. The resulting fields expose the current session's eventual
extremes from its second bar onward, before the session has closed. The current
training source and active workers are left unchanged to preserve the live
fingerprint. A same-session future-poison regression was run on `gpuarm1` with
CUDA disabled while the allocation remained active; it failed on current
source as expected (`1 failed, 17 deselected`), observing the first session's
eventual 110/90 range on later bars of that same session. This establishes that
the regression detects the defect, not that a repair passes. Treat the current
screen as exploratory/inadmissible for variant
selection. At the source boundary, repair the session shift, add a regression
that perturbs later bars in the same NY session, bump the feature-cache version,
rebuild all three prebuilt feature caches, rerun the real-data prefix/fold
audits, and launch a fresh matched 18-run lineage from initialization. A
transactional builder is now available at
`scripts/matrix/rebuild_named_prebuilt_feature_caches.py`; its plan-only mode
was checked on `gpuarm1` and resolves the baseline, Idea 1, and Idea 2 named
caches with matching raw-data/split settings. The cache-writing path has not
been run and must remain unused until the causal fix passes its regression.
After that gate, run `--plan-only`, then invoke the builder with a new backup
directory and report path under the fresh campaign's `artifacts/` and
`reports/` folders. The builder stages all three caches, verifies their
content-hash sidecars and exact parquet round-trip values, retains the previous
cache files, and publishes only after staging succeeds. Do not resume the 17 affected
policy checkpoints into that lineage.

The builder's transaction helper passed three focused
[tests](../../tests/test_rebuild_named_prebuilt_feature_caches.py) on `gpuarm1`
with CUDA disabled: successful publication, rollback after post-publish
validation failure, and rollback after a partial file replacement. Ruff format
and lint passed for the helper and its test. The real cache-writing path
remains unexecuted; these tests do not validate the causal repair or rebuilt
cache contents.

Source-contract note for the repair: `previous_labeled_session_ohlc` promotes
only sessions with every expected M1 bar present, while the current
`prev_ny_high_low` regression specifies same-session causality and extrema at
the close but does not specify how missing NY bars affect the reference. Do not
silently inherit that stricter missing-bar policy from the general helper; make
the choice explicit and cover it with a focused regression if the repair
changes that behavior. Before deciding, use the authorized H200 allocation to
measure NY-session completeness in the training data; the existing M1 quality
audit deliberately does not classify timestamp gaps as missing bars because
no authoritative broker/exchange calendar is recorded.

The regression `test_prev_ny_high_low_uses_only_completed_ny_sessions` is now
present in `tests/test_features/test_session_ohlc.py`. It checks that poisoning
future bars inside the second NY session cannot alter its visible prior-session
range, and that the completed range becomes visible at session close. The test
was run on `gpuarm1` on 2026-10-16 with CUDA disabled; it failed on the current
source at the first-session all-NaN assertion (`1 failed, 17 deselected`), as
expected from the audit. It has not passed on a repaired implementation. The
campaign's immutable `source_snapshot.zip` excludes `tests/`, so adding this
test does not change the active training fingerprint.

### P1 checkpoint and telemetry refresh — 2026-10-16 21:14 Helsinki

The formatted checkpoint-budget generator was rerun on `gpuarm1`; the JSON
snapshot records its matching generator SHA-256. It verifies 18/18 prior
checkpoint lineages with one 99,504-step `data_end` episode in each of 64
environments (1,152 source-attempt episode records total). In the current
attempt, three variants (`ladder_a0_flat_dist`, `ladder_a2_tp`, and
`ladder_a3_multi_tp`) have 64 completed train-split episode records each; the
baseline has an episode telemetry file but no completed-split records yet, and
the other 14 variants have no current-attempt episode file at this snapshot.
The baseline log had reached 19,996,544 of 20M steps and was in in-sample
evaluation; no final report artifact was verified. The controller log at
21:14:54 reported 18 active, 0 failed, 0 succeeded. Job `20990395` remained
RUNNING.

These records establish chronological train-split coverage for the three
current-attempt variants and prior checkpoint lineages, not active trading
throughout the year, strategy quality, completed 20M runs, or candidate
validity. The same snapshot records the failed fold-fit audit affecting 17/18
configs; all campaign outcomes remain inadmissible for selection. Machine-
readable snapshot:
[`p1_checkpoint_full_split_budget_2026-10-16.json`](p1_checkpoint_full_split_budget_2026-10-16.json).

The full-tree Ruff format check, Ruff lint, and Mypy completed on `gpuarm1`.
Mypy passed 446 source files after excluding generated `outputs/` artifacts
from discovery; this preserves the immutable resource-report script and its
recorded hash. The focused cache-builder tests passed (3/3). The
`session_ohlc` test file had 17 passes and one expected failure in the staged
same-session causal regression; the feature implementation has not yet been
changed because the campaign remains active.

At the 2026-10-16 20:07 Helsinki check, no `episode_equity.csv` existed under
the active recovery campaign's `runs/attempts/` tree. A later lineage audit
found the source campaign's attempt-2 episode records and verified complete
chronological split coverage for all 18 current checkpoint lineages; see the
20:36 evidence update and
[`p1_checkpoint_full_split_budget_2026-10-16.json`](p1_checkpoint_full_split_budget_2026-10-16.json).
This supersedes the earlier “unverified” conclusion about whether any
environment completed the split. Aggregate timesteps remain insufficient by
themselves; retain the per-environment telemetry and provenance chain.

## 39. Immediate actions for one long-running Codex agent

### P1/E0 evidence update — 2026-10-16 13:29 Helsinki

Generated `docs/thesis/trial_ledger_2026-10-16.json` from the six structured
campaign manifests currently present under `outputs/gpu_campaigns/`. The
inventory contains 108 run records (76 interrupted, 32 failed) and 162 attempt
records (112 interrupted, 50 failed). It preserves missing identity values as
null and records 198 complete and 72 partial identities across run and attempt
records. Slurm job IDs and states are taken from each attempt's provenance when
available; the campaign manifest job ID is retained as a fallback and in the
run-level job-ID list. The v3 campaign therefore maps its four recorded attempt
jobs to `20982572`, `20982793`, `20984134`, and `20985281`, rather than assigning
the old manifest job ID to every retry. The JSON validates against
`trial_ledger.schema.json`; the focused builder tests pass on gpuarm1.

This is an inventory and scheduler-state reconciliation, not completion of the
18-run screen. Its stated limitations exclude standalone/deleted legacy runs,
do not establish chronological train/test exposure or strategy validity, and
retain unresolved parent/child allocation relationships where provenance is
missing. The current GH200 allocation `20985591` is a recovery allocation, not
a campaign attempt; it remains held behind the absent release marker, and no
trainer processes were observed. Preserve that allocation and fingerprint
while the P1/P2 gate work continues. Readiness remains NOT READY.

### P1 checkpoint-lineage handoff — 2026-10-16 13:41 Helsinki

The v3 manifest records Git revision
`8a98f7792f83ce4bbe5cb84f4a2137657d791c02`, while canonical `main` is
`e110f18a2b9f684f796333aa4313afc85c6f6e04`. The campaign controller includes
`git_sha` in its resume fingerprint, so resuming v3 directly from current `main`
would fail before training. A read-only audit found all 228 recorded source-file
hashes matched the current tree before the checkpoint-map fallback change; this
confirms code-file parity at that audit point but does not erase the strict
commit-identity mismatch.

To preserve v3 and its fingerprint, its 18 attempt-3 checkpoints were validated
and staged as parent lineage for
`ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016`. They cover
11–14M saved steps and were previously CPU-load audited. The new spec keeps the
same 18 variants, seed 50, 20M total-step target, shared bundle, and exposed
selection split. The checkpoint-map builder now searches backward for the
newest valid checkpoint if a later infrastructure-failed attempt has none; the
map records attempt 3 for all 18 keys and the old v3 fingerprint. Focused tests
verify that lineage and a handoff resolver checks the source fingerprint,
target spec, and exact run-key coverage. The allocation supervisor will follow
the explicit continuation handoff after the currently held allocation returns.
The GH200 node-capacity preflight finds one additional card per node, while the
current job still reserves its existing pair. A new 2-node request `20990395`
was submitted after the commit gate; it is pending with
`AssocGrpGRESRunMinutes`. Its child start marker is staged, but no new
allocation has begun and no continuation trainer has launched. The old v3
manifest/checkpoints were not rewritten. This continuation remains exploratory
selection evidence, not confirmatory thesis evidence. Readiness remains NOT
READY.

### P1 attempt-by-attempt screen reconciliation — 2026-10-16 14:19 Helsinki

Generated
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/p1_attempt_reconciliation_20261016.json`
from the campaign manifest, every attempt's provenance and worker log, saved
checkpoint files, and `episode_equity.csv` records. It includes all 18 run
identities and all 72 recorded attempts. Every variant has 64/64 environments
with `completed_data_split=true` during attempt 2; each recorded episode is
99,504 steps and ended at `data_end`. This directly confirms one full train
split episode for every environment in that attempt, rather than inferring
coverage from aggregate PPO steps.

Latest valid saved checkpoints are 11M–14M steps, below the requested 20M, so
none of the 18 identities is a completed-budget screen result. All 18 attempt-4
logs show exit code 1 and the same pre-training `FileExistsError` at
`cache_dir.mkdir`: `cache` was a self-referential symlink. The current `cache`
path is now a real directory, and `prebuild_campaign_bar_cache.py` succeeded
on gpuarm1, validating all 14 configured bar-cache files for both symbols.
Original attempt logs/checkpoints and the v3 campaign fingerprint remain
preserved. The later `p1_reconciliation_20261016.md` snapshot reports only the
latest attempt and therefore shows its missing episode file as unavailable;
use the all-attempt JSON above for exposure reconciliation. The controller's
status snapshot is timestamped 08:17 UTC and says failed=18, while the current
campaign manifest says interrupted=18; retain both raw sources and do not
collapse them into one status without the per-attempt evidence.

The screen still has no 20M terminal identity, and this episode-coverage result
does not establish survival, evaluation validity, or strategy quality. The
recovery continuation remains exploratory. Readiness remains NOT READY.

### P2 deterministic replay and management-configuration repair — 2026-10-16 14:12 Helsinki

`TradingEnv._matched_entry_level` now selects the lexicographically smallest
candidate among levels with the most recent crossing timestamp. This keeps the
previous recency rule while making equal-time labels independent of set/hash
iteration order. The root `risk` config now declares `sl_mode`, `tp_mode`, and
`tp_reference`; training and deterministic evaluation explicitly forward these
values along with the SL/TP action thresholds and breakeven alpha. Tests verify
that the same configured management values reach both environment paths.

Validation on gpuarm1: 34 focused tests passed; the campaign GPU gate passed
full-tree formatting, Ruff, Mypy (444 source files), and 30 campaign artifact /
checkpoint tests; the full suite passed 1,543 tests in 162.78 seconds. A replay
of the preserved `ladder_a0_flat` attempt-3 14M checkpoint under
`PYTHONHASHSEED=1` and `12` produced byte-identical train and test equity CSVs,
metrics JSONs, and trade CSVs. The checkpoint SHA-256 remains
`a5f4b47f8b1e523bde501c71a56f1f3e9ccae62674951fc83fd38e89fb422530`.
Detailed hashes and replay paths are in
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/p2_replay_determinism_audit_20261016.json`.

This is bounded repeatability evidence for one interrupted 14M checkpoint and
one runtime/data/cache path. The historic test split has already been
inspected; it is not pristine holdout or confirmatory evidence. The replay
metrics are negative (train return -7.32%, test return -0.69%), so it does not
support a successful-strategy claim. No campaign trainer has launched, and
overall thesis readiness remains NOT READY.

Execute horizontally. The agent should always select the highest-priority READY task and must not wait
idly for training when another independent lane can advance. Conversely, when **all** useful lanes
are blocked only on long-running compute, the agent should enter the `WAITING_COMPUTE` quiet-wait state
from Section 7A.6 rather than spending tokens on frequent status checks.

### Lane A — correctness / implementation

1. Establish P0 source/config/data/cache boundaries and keep active run fingerprints immutable.
2. Implement and verify the HTF multi-swing stop-candidate contract before final P7 training;
   preserve the already-implemented IFVG through-zone extreme and its regression coverage. The
   opt-in three-candidate profile and causal stop-menu audit are implemented and pass focused and
   full-suite checks. Final P7 acceptance still requires independent review and selected-pair
   end-to-end evaluation under the frozen profile; the active screen remains latest-only.
3. Run focused tests first, then the required full regression/lint/type gates after integration.
4. Any defect capable of invalidating active/future scientific evidence outranks throughput or
   optional feature work.

### Lane B — active training / campaign operations

5. Preserve and reconcile the current one-seed screen until all required identities are terminal.
6. Use seed 50 as the canonical first seed for newly launched comparable variants, 20M steps,
   `n_envs=64`; retain historical seed-42 attempts under their true identities.
7. Never mutate an active run identity. Resume only from validated matching checkpoints.
8. Do not launch >=5 paired seeds until implementation, one-seed findings, selected pair, evaluator,
   and runtime are frozen.
8a. After launch/stability validation, use adaptive coarse polling or an external watcher; do not
    generate minute-by-minute training updates. Stable multi-hour jobs should normally wake the agent no
    more often than every 45–120 minutes unless a terminal/failure/checkpoint event occurs.

### Lane C — evaluator / recovery

9. Fix the observation-schema evaluator contract in isolation and add saved-checkpoint regressions.
10. Recover valid completed training from checkpoints rather than retraining because export/plots failed.
11. Make numerical deterministic evaluation mandatory; optional visualization remains post-processing.

### Lane D — results/evidence pipeline

12. Build E0 immediately: bundle writer, hashes, checkpoint metadata/load checks, raw eval exports,
    artifact index, result-card template, global ledger, and result-manifest schema.
13. Every terminal run is packaged before the agent considers it fully closed: config + command + logs +
    checkpoint/hash + raw eval + metrics + finding card.
14. Generate tables/figures from machine-readable outputs as soon as their input is terminal.
15. Keep failures, inactive runs, and zero-trade outcomes as first-class results with dispositions.

### Lane E — final design / review / statistics

16. Apply G0–G3 against actual artifacts, then freeze the smallest valid RQ1 control/treatment pair.
17. Prepare paired statistics and cost-stress tooling before the final seeds finish.
18. Prepare the independent-review packet early so review latency does not sit after P7.

### Lane F — runtime optimization

19. Treat throughput as specific to its campaign, phase, and sampling window. Base forecasts on a
    stabilized aggregate rate from the current campaign, not a brief per-worker sample.
20. Before P7, benchmark a small concurrency set for the final paired matrix and choose by reliable
    aggregate completed 20M runs/hour / total makespan, while preserving `n_envs=64` and scientific
    parity.
21. Reuse proven device-resident rollout staging; do not reopen measured no-gain optimizations unless
    the bottleneck changes.

### Codex agent scheduler rule

At every decision point, choose work in this order:

1. correctness blocker that could invalidate compute;
2. recovery/evaluation of already-completed expensive training;
3. packaging/verification of newly terminal runs;
4. launch/continue an authorized READY critical training job;
5. dependency work that unlocks the next scientific gate;
6. statistics/figures/indexes against already-frozen data;
7. required Q-stage preparation/execution only when its parent is frozen and it consumes no
   critical compute/integration capacity; otherwise continue the thesis-critical dependency chain.

### Do not do now

- do not draft result-dependent thesis chapters before P9; after P9, manuscript completion is required by this active goal;
- do not launch a Q stage before its parent/control is frozen or while it would compete with an
  unresolved thesis-critical gate; complete all Q0–Q7 stages after submission if they cannot run
  safely earlier;
- do not expand the current variants to a five-seed matrix;
- do not start the RQ2 encoder tournament unless explicitly promoted;
- do not launch final paired seeds before the final implementation/design/runtime freeze;
- do not choose models from the exposed 2026 report interval;
- do not retrain because optional chart generation failed;
- do not add SAC/VAE/new encoder families to the critical path;
- do not treat throughput, unit tests, or dashboard reward as strategy evidence.

## 40. Failure and decision rules

1. **Training completed; evaluation failed:** repair evaluation and recover from
   the checkpoint.
2. **Checkpoint loads but provenance is incomplete:** do not use it for a final
   claim; recover authoritative provenance or rerun under a new explicit identity.
3. **Training did not receive required full-split exposure:** the run cannot
   satisfy the final training contract; rerun only if that evidence is needed.
4. **Candidate fails survival/economic behavior:** preserve the negative result;
   do not weaken the gate after seeing it.
5. **Candidate makes zero trades:** report inactivity; do not call survival alone
   success.
6. **Matched control cannot be built without multiple confounds:** downgrade the
   claim to a system-level comparison or redesign before running.
7. **Throughput optimization changes PPO/evaluation behavior:** reject it for the
   same lineage or register it as a scientific treatment.
8. **No fresh holdout exists:** report the generalization limitation; do not
   relabel exposed data.
9. **Independent review finds a blocker:** invalidate only affected downstream
   evidence and return to the earliest contaminated package.
10. **RQ2 cannot finish before the deadline:** record `not tested`.
11. **RQ3 ladder remains partly confounded:** report only supported mechanism
    claims and label the rest as system comparisons.
12. **No positive strategy result exists:** finish the thesis with a valid
    negative/inconclusive result and implications.
13. **Time is short:** follow the descope order in Section 21; do not sacrifice
    correctness, provenance, or the evidence/review window to add breadth.

---

## 41. Definition of Codex agent mission end

The thesis results mission reaches `RESULTS_PACKAGE_COMPLETE` at P9, and the
thesis submission reaches `SUBMISSION_COMPLETE` at P11. The overall Codex
mission is complete only after both thesis submission and
`RESEARCH_EXTENSION_COMPLETE` are achieved:

- `RESULTS_PACKAGE_COMPLETE` is satisfied;
- every empirical finding is tied to a frozen evidence manifest and exact run identity;
- every final run has durable resolved config, command, source/data/cache fingerprints, logs,
  checkpoint(s), hashes, load validation, raw deterministic evaluation outputs, and metrics;
- the final matched comparison has the required paired-seed evidence, uncertainty/statistics, and
  cost-stress results, or an explicit valid negative/inconclusive disposition;
- failures, inactive runs, interrupted attempts, and zero-trade runs remain visible in the ledger;
- all final tables and figures regenerate from machine-readable retained artifacts;
- every `finding.md` states evidence role, allowed claim, forbidden overclaim, limitations, and exact
  artifact paths;
- `results_summary.md` and the global result/artifact manifest let the thesis author determine what
  happened and what may be claimed without reading terminal scrollback or rerunning experiments;
- the evidence archive is backed up and a new agent can resume/review from repository artifacts
  rather than conversational memory;
- all Q0–Q7 stages have a recorded winner/no-winner disposition, reproducible evidence, and a
  frozen parent decision; their results remain outside the thesis evidence lineage unless formally
  promoted under the thesis evidence gates.

The supervisor PDF, prose integration, references, formatting, and faculty submission follow P9.
They remain part of the active user goal; any discovered scientific blocker returns work to the
earliest contaminated package. The overall goal ends only when `SUBMISSION_COMPLETE` and
`RESEARCH_EXTENSION_COMPLETE` are both satisfied. Live-production work remains outside this goal.
