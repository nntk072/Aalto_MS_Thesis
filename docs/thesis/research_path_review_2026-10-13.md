# Training, execution, reward, and model-selection path review

**Review date:** 2026-10-13 (Europe/Helsinki)<br>
**Repository:** `/scratch/work/nguyenl37/Aalto_MS_Thesis`<br>
**Branch / HEAD:** `main` / `b00b5e85622ec1410c8a11cc063cd52c87d3976d`

## Scope and evidence

This is a source-path review of the current checkout, not a new backtest, model
selection decision, or independent academic sign-off. The execution path was
traced in `quant_rl/envs/trading_env.py` (`_step_untraced`, `_check_guardrails`,
`_apply_peak_trailing_dd`, `_calculate_reward`, and episode finalization). The
training and evaluation path was checked in `quant_rl/train/callbacks.py`,
`quant_rl/train/train_rl.py`, and `quant_rl/eval/eval_run.py`. Existing tests
and resource/campaign artifacts were treated as regression evidence only.

## Findings

- **Bar timing and fills:** the environment separates decision, order submission,
  quote fill, and valuation timestamps. Existing positions are marked/exited on
  the next market bar; delayed orders wait for the scheduled eligible bar. A
  bar-close entry is not exposed to that bar's earlier high/low for stop/target
  handling. Episode-end pending orders are cancelled. These are implemented
  invariants, not proof that every broker/slippage model matches live execution.
- **Hard breaches and episode status:** `_check_guardrails` force-closes an open
  position, records the breach, and terminates episodic training. Non-episodic
  evaluation latches a breach to the session; trailing/max-drawdown failure also
  latches the evaluation year. Data/year end returns truncation. The final
  episode record includes equity, trade count, breach reason, equity slope and
  peak trailing drawdown for the existing train-equity gate.
- **Reward handoff:** `_calculate_reward` receives the equity change and current
  account/guardrail state, plus the breach flag and configured sweep/strategy
  context where enabled. The environment also applies the configured peak-DD
  behavior penalty when that ceiling fires. This review did not independently
  validate the economics, scale, or desirability of every reward component.
- **Checkpoint evaluation:** `BestCheckpointEvalCallback` evaluates the first
  20,000 training bars (or the shorter available train set), uses one rollout,
  and runs at the configured final timestep. Its policy actions are now
  deterministic. Since it runs only once at the final timestep in the campaign
  configuration, `ppo_best` is a copy of the final policy with an in-sample
  equity diagnostic, not a selection among training checkpoints. The ordinary
  report still evaluates the final in-memory model.
- **Ordinary report and explicit evaluation:** `train_rl.py` evaluates the
  final in-memory model deterministically on the configured test split after
  training. `eval_run.py` prefers an explicit checkpoint, then `ppo_final.zip`,
  then the highest-step training checkpoint. No automatic locked-test metric
  selection path was found in these source paths. Human selection after viewing
  exposed-period reports remains possible; an explicit checkpoint hash and
  selection record are still required for any such comparison.

## Decision and limits

The traced source paths are documented well enough to close the narrow source
inspection task. They do **not** close scientific acceptance: no independent
review of every fill/accounting edge case was performed here; human
model/variant selection can leak exposed-period results; and previous A0/A1
pilots fail the mandatory equity/survival gate.
The 2026 test interval through 2026-07-03 is authorized for exploratory,
`selection_acknowledged` work, but is permanently exposed and cannot be called a
fresh confirmatory holdout. Do not promote this review or passing software
tests to evidence of strategy viability. The external sign-off questions and
review record are prepared in
[`independent_research_review_checklist_2026-10-14.md`](independent_research_review_checklist_2026-10-14.md);
the checklist is unsigned and does not close this gate.

## Bounded execution/reward regression refresh — 2026-10-14

The current worktree's scoped regression set passed **107 tests with no
warnings** across tick/bar fill reconciliation, trailing-stop causality,
episode termination, golden environment traces, entry-state transitions,
fixed-risk sizing, execution contracts, broker quote parity, cross-validation
parity, reward attribution, and PO3/distribution reward behavior. The golden
trace case that previously omitted its independent London/Asian entry-gate
inputs now provides gate values that allow the attempted long, while leaving
swing-stop levels NaN. It therefore tests the no-naked-position behavior
without triggering the unrelated legacy allow-on-missing-gate fallback.

Command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_trailing_stop_causality.py tests/test_envs/test_year_episode_survival.py tests/test_envs/test_trading_env_golden_traces.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_fixed_risk_sizing.py tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_backtest/test_broker_quote_parity_unittest.py tests/test_backtest/test_engine_cross_validation.py tests/test_eval/test_reward_attribution_unittest.py tests/test_envs/test_sweep_reward.py tests/test_envs/test_po3_reward.py tests/test_envs/test_distribution_reward.py -q --maxfail=8`

This is bounded deterministic regression evidence, not exhaustive production
TradingEnv/backtest/live parity, independent academic review, campaign
validation, or evidence of a profitable strategy. Those acceptance gaps stay
open.

## Walk-forward model-selection action mode — 2026-10-14

The current canonical `quant_rl.train.train_rl --walk-forward` path previously
called `evaluate_model(..., deterministic=False)` for each purged validation
fold, even though `docs/thesis/validation_protocol.md` assigns these folds to
architecture/hyperparameter selection. This made the fold scores depend on
sampled policy actions. The fold replay now goes through
`_evaluate_final_split`, the same helper that forces deterministic actions for
the final train/test reports. The periodic dashboard remains an explicitly
stochastic progress diagnostic. The separate `eval_seed_variance` utility is
also now labeled as an intentional stochastic-action sensitivity analysis,
not as a reproduction of the standard final reports.

This is a source-level consistency correction for future walk-forward runs.
No new fold training or selection metrics were generated in this update, and it
does not revise any prior candidate ranking or the failed 100k probe. The
selection protocol, exposed-data limit and independent research sign-off
remain open.

The existing `tests/test_train/test_checkpoint_eval_determinism.py` passes
2/2 after this correction. It verifies the deterministic helper and checkpoint
callback; the walk-forward call site's use of that helper was confirmed by
source inspection rather than an end-to-end fold run.

## Live A0 reward/equity observation — 2026-10-14 00:38 EEST

During the exploratory 18-run campaign, A0 had completed 3,407,872 of
20,000,000 requested steps. Its append-only worker log contained 59 callback
warnings for completed training episodes whose shaped reward sum was positive
while end equity was lower than episode start. Every recorded warning had
`breach=trailing_dd`; the logged reward sums ranged from 0.0457 to 20.9638,
and equity losses ranged from $24.70 to $7,370.56 (median $4,723.84). These
are partial-run observations, not independent trades or a final model result.

The source path explains why reward and equity can diverge: `train_rl.make_env`
defaults to `episodic=True`; `TradingEnv` sets the reward function's hard
breach penalty to 0 for episodic training and -10 for non-episodic evaluation.
The live A0 `debug_wiring.json` independently records `episodic: true`,
`breach_penalty: 0.0`, and `eval_breach_penalty: -10.0`. The nearby source
comment calls the training value a small terminal spike, but its actual value
is zero.
DSR shaping can accumulate positive reward before the breach, and the
peak-drawdown behavior penalty is only -1 per bar while its condition remains
fired. Thus the warning is consistent with the configured shaped objective;
it does not by itself prove an accounting defect. It does show that a positive
training reward total cannot be read as profitable or risk-surviving behavior.

The independent train-equity gate remains stricter: end equity must exceed
start, the equity slope must be positive, peak trailing drawdown must stay at
or below 10%, and no max-loss breach may stop trading. The campaign artifact
verifier rejects a run whose `train_equity_ok` is false. Keep that hard gate;
do not treat the reward warning as a strategy result or soften the gate to
match shaped reward. Review whether the 0-versus--10 training/evaluation breach
penalty is an intentional objective choice before any revised confirmatory
study. The live attempt's fingerprinted source was left unchanged while it was
running.

## Training/evaluation hard-breach penalty decision — 2026-10-14

The terminal breach penalty is now `-10.0` in both episodic training and
non-episodic evaluation. Previously training used `0.0`, so a hard risk breach
could terminate an episode without a negative terminal reward even though the
same breach was penalized during evaluation. The training equity/survival gate
remains an independent hard acceptance check; this reward change does not
weaken it or turn shaped reward into a profitability measure. The diagnostic
flight recorder now reports a train/evaluation penalty mismatch instead of
flagging any nonzero training penalty.

Verification: `tests/test_envs/test_year_episode_survival.py` and
`tests/test_envs/test_trading_env_golden_traces.py` pass (14 tests); scoped
Ruff check passes. The checkpoint callback now uses deterministic actions,
and `tests/test_train/test_checkpoint_eval_determinism.py` asserts this
contract. The A0 failure remains a failed result. This reward choice
has not been evaluated through a new full training campaign and has no
independent academic sign-off.

## Output purge and evidence status — 2026-10-14

At the researcher's request, the entire `outputs/` directory contents were
deleted without an archive. This included the live A0 logs/checkpoints,
resource samples, historical audit reports, and the run manifests cited in
this document. The campaign job `20938752` was confirmed cancelled before the
purge; historical progress values above describe observations made while that
job was live and are not currently re-openable artifacts. A later exploratory
campaign is configured separately in
`config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_20261014.yaml`.
Its exposed-period results, if produced, remain selection-acknowledged and
cannot constitute independent review or confirmatory evidence. The reward
penalty alignment described above is a source-level change with targeted
regression evidence; it has not yet been validated through a completed long
training run.

## Configurable higher-timeframe alignment review — 2026-10-14

The data resampler's standard timeframes are M1, M5, M15, M30, H1, H4 and D1.
Aligned feature blocks accept explicit timeframe lists; bounded defaults remain
M5/M15/H1 for the lightweight SMT, structure/BOS and liquidity blocks, and
Idea 1/2 keep M5/M15/H1 as their explicit IFVG and PO3 sweep-reference
defaults. The PO3 reference list is now configurable across the standard
M5/M15/M30/H1/H4/D1 set. M1 remains the decision clock, not a sweep reference.
This is the supported standard set, not arbitrary-duration resampling.

The PO3 pivot-reference completion timestamp now comes from the next actual
resampled-bar label instead of adding a fixed duration to every bar. The final
D1 boundary uses a calendar-day offset, preserving local-midnight completion
across DST. Tests cover all six HTF reference codes, structure features using
all six frames, prefix invariance, invalid/duplicate lists and a D1 pivot over
the Helsinki DST boundary. Focused feature suites pass 44 tests; Ruff format
and lint checks pass on the touched Python files.

This closes the implementation/configuration gap for the standard timeframe
set. It does not prove full-history real-data causality for every profile or
timezone, nor that the features improve strategy performance. Existing default
feature schemas and Idea 1/2 timeframe defaults are preserved; config content
hashing distinguishes a changed timeframe list.

## Deterministic final train-gate replay — 2026-10-14

The train-equity acceptance gate consumes the full in-sample replay. A source
audit found that this final replay used `deterministic=False`, even though the
campaign evaluation contract requires deterministic actions; the report-only
test replay was deterministic in the current checkout. Updated both final
splits to use `_evaluate_final_split`, which forces deterministic policy
actions. The periodic train-prefix dashboard monitor remains stochastic and
is labeled as a progress diagnostic. The focused determinism tests pass 2/2,
and scoped Ruff format/lint pass.

This correction applies to future launches. The active 100k probe records a
different `train_rl.py` source hash and has no retained source snapshot, so its
exact train-evaluation action mode cannot be reconstructed from the current
source. Do not reinterpret its gate outcomes as corrected deterministic
replays. The code audit does not provide independent academic sign-off,
selection-independent holdout evidence, or proof that any strategy is viable.

## 100k probe source-fidelity check — 2026-10-14

All 18 attempts record the same 194-file source-hash inventory. Comparison
against the current repository finds 186 exact matches and eight changed
files; no paths are missing. The attempt provenance and run manifests retain
the launch-time hashes and dirty-file hashes but not the corresponding source
contents. Exact retraining from the current tree is therefore not established,
although the saved checkpoints, metrics, equity curves and PnL reconciliations
remain inspectable. Re-run the audit with
`scripts/matrix/audit_campaign_source_fidelity.py`; the generated report is
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/source_fidelity_audit.md`.

This is a reproducibility limitation for the resource/strategy probe, not
evidence that its recorded metrics were fabricated or incorrect. It does mean
those metrics must not be promoted to a replayable confirmatory result.

## Live order-stop and aggregate-risk correction — 2026-10-14

The live RL robot sized a trade against `stop_loss_pips` but submitted
`sl=None`, so the broker order had no corresponding stop. In addition,
`RiskManager.check_risk_limits()` estimated existing aggregate risk from
absolute floating PnL, which is not risk-to-stop and could admit exposure above
the configured cap. The live and multi-symbol risk robots now calculate the
broker-side stop from the same tick-distance conversion used for sizing and
submit it with the order. Position size is floored to the broker volume step;
it is no longer rounded/clamped upward beyond the cash risk budget. Aggregate
open risk is estimated from each position's entry, stop, volume and MT5 tick
value. Missing stops, failed position queries, or incomplete symbol data now
block further entries. `RiskManager()` defaults now match the checked-in 1%
per-position / 2% aggregate config, and the RL robot class defaults to paper
trading.

Verification: the entire headless live test package passes **47 tests**;
scoped Ruff format/lint and `git diff --check` pass. These tests use mocked
MT5 data; no broker integration was run. This closes the missing-stop and
floating-PnL aggregation defects, but it does not make live execution
equivalent to the training policy: the RL bridge currently exposes direction
signals with a fixed 50-pip live stop rather than the model's structural stop,
SL/TP-mode, and target behavior, and the live path does not enforce the
training/evaluation cumulative daily and max-loss guardrails. Keep
`PAPER_TRADING=true`; live deployment remains not ready. This correction does
not alter campaign results or the 18-run launch decision.

## Explicit checkpoint reanalysis action mode — 2026-10-14

The checkpoint reanalysis CLI claimed its actions were sampled and warned that
an unseeded run would use a different trade set, while `evaluate_model` defaults
to deterministic policy actions. The reanalysis CLI now passes the action mode
explicitly: deterministic is the default, and sampled actions require
`--stochastic`. `--seed` is documented as making stochastic reanalysis
repeatable, and the selected mode and seed are saved in both the reanalysis
manifest and `training_log.json`.

Verification: the focused reanalysis provenance tests pass 15/15, including
parser default/opt-in and action-mode wiring; scoped Ruff format/lint and
`git diff --check` pass. No checkpoint was re-evaluated, and no prior campaign
metric or scientific conclusion changes. This does not close independent
research review, real-data causal certification, final-holdout evidence, or
the failed strategy acceptance gates.
