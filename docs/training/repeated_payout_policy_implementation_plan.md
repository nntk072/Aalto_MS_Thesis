# Repeated Payout Policy Implementation Plan

## 1. Scope and non-goals

Introduce a new **selectable repeated-payout policy** while preserving the current policy as the legacy/default-compatible alternative.

Required behavior:

- operating balance: **$100,000**
- payout target: **$110,000**
- maximum daily loss: **$5,000**
- maximum account loss: static **$90,000 equity floor**
- no account-level trailing drawdown in repeated-payout mode
- agent-controlled continuous risk: **0.00% to 0.90%**
- single-cycle training initially
- repeated chronological cycles in evaluation
- cumulative payouts, cycle success/failure, time-to-payout, and sustainability metrics

Planning only: do not implement source changes or launch training.

The repository was audited directly on Triton at `/scratch/work/nguyenl37/Aalto_MS_Thesis`. The working tree is already heavily modified by pre-existing work; implementation must not clean/reset unrelated state.

## 2. Current implementation audit

### Account accounting - `trading/account/state.py`

`AccountState` currently has `balance`, `equity`, `peak_equity`, `session_start_balance`, `daily_loss`, `max_drawdown`, and `open_pnl`.

Verified semantics:

- `equity = balance + open_pnl`.
- `loss_from_initial() = max(0, initial_balance - equity)`.
- `max_drawdown` is maximum historical loss from initial balance, not trailing HWM DD.
- `reset_daily()` stores current realized balance, not equity, then zeros `daily_loss`.
- `close_trade()` adds only losing realized closes to `daily_loss`.
- profitable closes do not offset `daily_loss`.
- unrealized loss does not enter `daily_loss`.

Thus the current env-side daily-loss variable is **sum of losing realized closes**, not loss from session-opening equity.

### Confirmed daily-loss inconsistency - `evaluation/plotting/plot_series.py`

Verified functions:

- `session_start_equity()`
- `daily_loss_limit_series()`
- `daily_drawdown_pct()`
- `daily_drawdown_usd()`

Plotting uses:

```text
daily_floor = session_start_equity - daily_loss_limit
```

and keeps that floor fixed through the session.

Environment/guardrail/reward/constraint logic instead uses `AccountState.daily_loss`. This is a confirmed semantic inconsistency, not merely a chart issue.

### Guardrails - `trading/risk/guardrails.py`

`FTMOGuardrails` currently:

- checks hard daily with `acc.daily_loss >= daily_loss_limit`;
- checks soft daily with the same variable;
- checks max loss via current loss from initial balance;
- checks training trailing DD as `peak_equity - equity >= trailing_dd_limit * initial_balance`;
- rejects per-trade risk only when risk is greater than the cap, so equality is allowed;
- exposes `breach_reason()` used by the environment.

### Training/evaluation trailing-DD difference

Files:

- `training/runner/training.py`
- `simulation/environment/worker.py`

Verified helpers in `train_rl.py`:

- `_strategy_risk_ranges`
- `_max_loss_per_trade`
- `_guardrail_kwargs`
- `_eval_guardrail_kwargs`
- `_training_episodic`
- `make_env`

Current training may use `ftmo.trailing_dd_limit=0.07`. Evaluation explicitly sets hard and soft trailing DD to zero. `worker.py::make_bundle_env()` mirrors that. This difference is intentional for the current policy and must not be treated as a defect.

Repeated-payout mode should set account-level trailing DD to zero in both training and evaluation.

### Dynamic risk and sizing

Files/functions:

- `trading/risk/sizing.py::compute_lots`
- `TradingEnv._risk_budget_usd`
- `TradingEnv._compute_lots_for`
- `TradingEnv._actual_risk_usd`
- `TradingEnv._try_enter_position`
- `TradingEnv._decode_action`

Current dynamic budget is based on live sizing equity times agent risk fraction. `compute_lots()` clips lots to `[min_lot,max_lot]` and applies the fractional `max_loss_cap`.

Fixed-risk mode supplies explicit `risk_usd` and deliberately bypasses the fractional max-loss cap.

After entry, `TradingEnv._try_enter_position()` stores planned stop exposure as stop distance x position size x contract size.

No current entry-sizing path was found that also caps by remaining daily capacity or remaining account-floor capacity.

### Zero/low-risk behavior

`TradingEnv.__init__()` currently defaults `risk_floor=0.0005`. In dynamic mode, `_decode_action()` converts entry requests at or below that floor to hold.

For repeated-payout mode use:

```yaml
risk_frac_range: [0.0, 0.009]
risk_floor: 0.0
```

Zero then remains a valid no-trade action. If action decoding needs numerical tolerance, use a tiny numerical epsilon rather than a 5-bp policy floor.

### Per-trade cap precedence

`train_rl.py::_max_loss_per_trade(cfg)` currently uses:

- `cfg.ftmo.risk_per_trade_limit` for strategy-action RL (default about $1,000);
- `cfg.backtest.validation.max_loss_per_trade_usd` for legacy/non-strategy paths (default about $100).

`worker.py` mirrors this.

`simulation/backtest/engine.py::run_backtest()` additionally calls the guardrail trade-risk check after sizing.

Repeated-payout mode must use one resolved risk contract across direct env construction, worker env construction, evaluation, and any standalone backtest used for comparison.

### Multi-TP and position management

Relevant files:

- `simulation/environment/env.py`
- `tests/test_envs/test_tp_mode.py`
- `tests/test_envs/test_risk_alignment.py`

Existing tests already cover initial-size-based partial closes, TP hit masks, final residual allocation, multiple target crossings, cooldown after full lifecycle, session-end partial closure, stop-before-target on ambiguous bars, tick partial TP then stop, multi-target tick crossing, bar/tick parity, and monotonic trailing behavior.

Keep distinct:

- `_apply_tp_breakeven()`: TP pullback behavior;
- `_apply_breakeven()`: SL breakeven behavior.

Repeated-payout work should not redesign multi-TP.

### Reward and constrained metrics

Files:

- `simulation/environment/rewards.py`
- `TradingEnv._calculate_reward`
- `evaluation/rollout/model.py`

Reward and constraint diagnostics consume daily loss, loss from initial, trailing DD, limits, and breach flags. Therefore the daily-loss correction must be centralized in canonical account/policy state. Fixing plotting alone would leave learning and evaluation semantics inconsistent.

### Episode/session behavior

`TradingEnv.__init__()` documents:

- `episodic=True`: guardrail breach terminates episode;
- `episodic=False`: force-close/block remainder of session, then continue into later sessions.

Relevant methods include `reset()`, `_check_guardrails()`, `step()`, `_step_untraced()`, `_progress_market()`, `_note_session_open()`, `_is_ny_session_boundary()`, `_advance_ny_step()`, and `_force_close_guard()`.

Initial repeated-payout training should use one-cycle episodes. Chronological evaluation must continue after successful payout resets.

### Evaluation/export

Files:

- `evaluation/rollout/model.py::evaluate_model`
- `evaluation/reporting/export.py::_write_split`
- `evaluation/reporting/export.py::save_run`
- `evaluation/reanalysis/runner.py`

Current evaluation success/breach diagnostics originate from environment breach events. Export writes equity, trades, fills, positions, breach events, constraint costs, session activity, metrics, and evaluation diagnostics.

Repeated-payout state must be emitted by the environment and exported directly rather than independently inferred from the equity curve.

### Baselines

`training/runner/baselines.py` calls `simulation.backtest.engine.run_backtest`, not `TradingEnv`.

If thesis comparisons require repeated-payout baseline parity, the same policy/accounting/risk rules must be propagated into `run_backtest()`. Otherwise explicitly document the methodological exception.

### Observation/checkpoint compatibility

`simulation/contracts/observations.py` has `account-state-v1` and `account-state-v2`. V2 includes `daily_loss_over_limit` and `remaining_daily_loss_budget_over_limit`, and explicitly states that `daily_loss` preserves the existing realized-loss-only meaning.

Changing that meaning is an observation-contract change even if vector width stays unchanged. Add a new observation schema version for repeated-payout semantics.

Action metadata already exists via `_action_schema_version`, `_action_layout_key`, and `_action_schema` in `models/factory/agent.py`, `models/policies/ppo.py`, dashboard metadata, and `eval_run.py`.

No new action dimension is needed if the existing risk coordinate is reused.

## 3. Proposed selectable policy architecture

Add a config selector; keep legacy default:

```yaml
trading_policy:
  mode: legacy
  repeated_payout:
    operating_balance: 100000.0
    payout_target_balance: 110000.0
    account_floor_equity: 90000.0
    daily_loss_limit: 5000.0
    soft_daily_loss_limit: 2000.0
    trailing_dd_limit: 0.0
    payout_trigger: realized_balance_flat
    continue_after_payout: true
    agent_risk_min_frac: 0.0
    agent_risk_max_frac: 0.009
    risk_floor: 0.0
    reserve_cost_buffer_usd: <explicit calibrated value or model-derived estimate>
    enforce_remaining_daily_capacity: true
    enforce_remaining_account_capacity: true
```

`configuration/loader.py::load_config()` already merges defaults, variant fragments, then CLI overrides; no new loader architecture is required.

Recommended ownership:

- `AccountState`: realized/unrealized ledger.
- Small policy/accounting state object: session anchor, cycle state, payout counters, policy floors/capacities.
- `TradingEnv`: chronological transitions.
- Guardrails: consume canonical state, preserving legacy behavior.
- Export/plots: consume canonical emitted state/events.

Avoid broad broker, TP, reward, feature, or execution refactors.

## 4. Exact mathematical definitions

Let:

- (B_t): realized balance
- (U_t): unrealized PnL
- (E_t=B_t+U_t): equity
- (E_{d,0}): equity captured once at session start
- (F=90{,}000): static account floor
- (T=110{,}000): payout target
- (L_d=5{,}000): daily allowance

### Actual daily-loss breach

```text
daily_floor_d = E_d,0 - 5,000
daily_loss_t = max(0, E_d,0 - E_t)
daily_breach_t = E_t <= daily_floor_d
```

Consequences:

- floor is fixed for the session;
- unrealized PnL counts;
- intraday profit does not ratchet the floor;
- same-day realized gains can offset later losses because breach uses current equity against a fixed anchor.

Example:

```text
session opens 108,000 -> floor 103,000
equity rises 111,000 -> floor stays 103,000
equity falls 106,000 -> daily loss 2,000; no breach
equity reaches 103,000 -> breach
```

Do not implement `floor_t=current_equity_t-5000`; that moving value can never independently detect the intended breach.

### Prospective daily risk capacity

This is a sizing quantity, not a breach rule:

```text
remaining_daily_capacity_t = max(0, E_t - daily_floor_d)
```

At equity $106k with a $103k fixed floor, capacity is $3k.

### Static account loss

```text
remaining_account_capacity_t = max(0, E_t - 90,000)
account_breach_t = E_t <= 90,000
```

The $90k floor never trails profits and never rises after payout.

### Agent risk and execution-aware budget

```text
r_t in [0.0, 0.009]
requested_risk_usd_t = E_t * r_t

policy_trade_cap_t =
    min(0.009 * E_t,
        remaining_daily_capacity_t,
        remaining_account_capacity_t)

usable_trade_cap_t =
    max(0, policy_trade_cap_t - execution_reserve_t)

approved_risk_t =
    min(requested_risk_usd_t, usable_trade_cap_t)
```

`execution_reserve_t` must be derived from the existing spread/commission/slippage model or exposed as an explicit reproducible config value.

After lot rounding/min-lot clipping, recompute modeled stop risk and reject entry if it exceeds usable capacity.

Track separately requested risk fraction/USD, approved risk USD, executed lot size, modeled stop risk, execution reserve, and realized loss.

A gap/slippage loss may exceed modeled stop risk; report this as realized execution loss, not as retroactive agent-requested risk.

### Payout trigger/state transition

Recommended auditable trigger:

```text
position is flat AND realized balance >= 110,000
```

Do not trigger from unrealized equity.

On success:

1. record cycle success/time;
2. record payout = `balance - 100,000`;
3. add cumulative payout;
4. increment cycle/success counters;
5. reset operating balance/equity to $100,000;
6. clear cycle-local state;
7. continue from the next chronological observation.

A payout reset is not a new market day. To avoid an immediate false daily breach after resetting from about $110k to $100k, emit an explicit `payout_reset` and re-anchor post-reset policy session state to the new operating equity.

## 5. Complete configuration specification

Default must remain `trading_policy.mode=legacy`.

Repeated-payout config validation should reject target <= operating balance, account floor >= operating balance, negative daily limit, agent max risk >= 1%, nonzero account trailing DD, or incompatible observation schema.

Legacy mode must preserve current FTMO limits, current training/eval trailing-DD behavior, current risk range/floor, current episode semantics, and checkpoint behavior.

## 6. File-by-file implementation tasks

### `trading/account/state.py`

- add session-start-equity support;
- add canonical daily-loss/remaining-capacity accessors;
- preserve legacy realized-loss semantics behind legacy mode or migrate callers atomically;
- add safe operating-balance reset for payout;
- ensure mark-to-market can drive daily/account breach state.

### `trading/risk/guardrails.py`

- consume canonical daily state;
- add/route static repeated-payout account floor;
- disable account trailing DD for repeated-payout mode;
- preserve `FTMOGuardrails` legacy behavior;
- retain explicit trade-risk validation for standalone backtests.

### `trading/risk/sizing.py`

- do not treat min-lot clipping as automatic approval;
- expose enough sizing detail to validate rounded actual stop risk;
- preserve fixed-risk legacy behavior.

### `simulation/environment/env.py`

Update only relevant paths:

- `__init__()`: resolved policy config/state;
- `_create_account()`: policy-compatible state;
- `reset()`: episode/cycle/session reset distinction;
- `_check_guardrails()`: canonical breach state;
- `_decode_action()`: [0,0.009] repeated-payout risk and zero-risk hold;
- `_risk_budget_usd()`: requested/capacity-limited risk;
- `_compute_lots_for()`: post-rounding validation;
- `_actual_risk_usd()`: canonical modeled stop risk;
- `_try_enter_position()`: requested/approved/executed risk logging;
- `_progress_market()` / `_step_untraced()`: payout success transition;
- `_note_session_open()`, `_is_ny_session_boundary()`, `_advance_ny_step()`: fixed session-opening-equity anchor;
- `_calculate_reward()`: canonical policy values;
- `_get_observation()`: repeated-payout observation schema;
- `_force_close_guard()`: explicit hard-failure path.

Do not change multi-TP, entry geometry, tick execution, or unrelated features.

### `simulation/contracts/observations.py`

- add a new schema version for changed daily-loss semantics;
- optionally expose cycle progress and remaining capacities;
- reject silent legacy checkpoint reuse under changed semantics.

### `training/runner/training.py`

Update `_strategy_risk_ranges()`, `_max_loss_per_trade()`, `_guardrail_kwargs()`, `_eval_guardrail_kwargs()`, `_training_episodic()`, and `make_env()`.

Add one policy-resolution helper to avoid divergent reconstruction.

Repeated-payout training initially: one cycle per episode; payout success or hard failure terminates; no account trailing DD.

### `simulation/environment/worker.py`

`make_bundle_env()` independently reconstructs config today. It must consume the same resolved policy values as direct env construction.

Require parity for daily floor, account floor, risk range/capacity, trailing-DD disablement, payout trigger, observation schema, and evaluation/episodic mode.

### `evaluation/rollout/model.py`

`evaluate_model()` must propagate repeated-payout policy and return cumulative payouts, cycle counts/events, payout times/amounts, per-cycle duration, failure reason/time, and risk summaries.

Repeated chronological evaluation must continue after payout success.

### `evaluation/reporting/export.py`

Extend `_write_split()` / `save_run()` to write canonical repeated-payout artifacts, e.g. `cycle_events.csv`, `payouts.csv`, risk execution diagnostics, and repeated-payout fields in `evaluation_diagnostics.json`.

Do not re-infer policy state independently.

### `evaluation/plotting/plot_series.py`

Keep `session_start_equity()` and `daily_loss_limit_series()` aligned with environment semantics. Add a regression test that plotting and environment produce the same daily floor.

### `evaluation/plotting/plots.py` and `evaluation/plotting/plots_interactive.py`

- label static $90k floor;
- show payout-reset events;
- omit account trailing-DD rule in repeated-payout mode;
- use canonical daily-floor state.

### `evaluation/reanalysis/runner.py`

- resolve policy before env creation;
- validate policy/observation/action metadata;
- include resolved policy parameters in provenance;
- reject silent legacy-to-repeated-payout semantic reinterpretation.

### `models/factory/agent.py`, `models/policies/ppo.py`

No action-width change if existing risk coordinate is reused. Only update action schema metadata if coordinate layout/meaning itself changes. Observation schema remains independently versioned.

### `training/callbacks/dashboard.py`

Record policy mode, observation schema, risk range, and repeated-payout metrics in run snapshots.

### `simulation/backtest/engine.py`

If baseline parity is required, wire the same policy/account/risk transitions and emit the same cycle/payout events. Otherwise leave unchanged and document the methodological exception.

### `training/runner/baselines.py`

Modify only if repeated-payout baseline parity is required. Do not silently compare repeated-payout RL against legacy $100 sizing/account rules.

### Config/docs

Relevant surfaces include `config/base/default.yaml`, experiment/campaign YAMLs, `docs/thesis/eval_pack.md`, and `docs/reproducibility.md`. Archive the fully resolved policy with each run.

## 7. Interface and state-management changes

Track explicitly:

- policy mode;
- session-start equity/fixed daily floor;
- current daily loss/remaining daily capacity;
- static account floor/remaining account capacity;
- cycle number/start time/start balance/target;
- cumulative payouts;
- successful/failed cycles;
- last payout amount/time;
- payout-reset event count;
- failure reason/latch;
- requested risk fraction/USD;
- approved risk;
- modeled executed stop risk;
- realized loss diagnostics.

Do not overload `AccountState.max_drawdown` for cycle/payout semantics.

## 8. Training and evaluation integration

### Single-cycle training

Start each episode at $100k.

Success terminal: flat realized balance >= $110k.

Failure terminal: current equity reaches the daily floor or $90k static floor.

No account trailing-DD termination.

### Repeated chronological OOS evaluation

Start once at the first OOS observation and advance monotonically.

On payout, emit event, reset operating trading capital to $100k, preserve cumulative payout/counters, and continue at the next observation. No rewind, random restart, or reused observations.

## 9. Risk, execution, and accounting edge cases

Tests must cover:

1. $108k session open -> $103k fixed floor;
2. equity $106k -> $3k remaining daily capacity;
3. intraday profit does not ratchet daily floor;
4. unrealized loss breaches before close;
5. realized gain then loss still uses fixed anchor/current equity;
6. boundary reset captures anchor once;
7. default block-overnight behavior;
8. future overnight position marked-to-market anchor;
9. equity exactly $90k;
10. gap below $90k;
11. unrealized touch of $110k while open -> no payout;
12. flat close above target -> actual overshoot payout, reset $100k;
13. mid-session payout reset/re-anchor;
14. multiple payouts;
15. failure after prior payouts preserves cumulative payout;
16. requested risk zero;
17. tiny positive risk/min-lot case;
18. max request 0.009;
19. request exceeds remaining daily capacity;
20. request exceeds remaining account capacity;
21. execution reserve consumes remaining capacity;
22. rounded lot exceeds approved risk;
23. realized loss exceeds modeled stop due to gap/slippage;
24. partial TP preserves original risk diagnostics;
25. SL breakeven and TP pullback remain distinct.

## 10. Test requirements and acceptance criteria

Extend focused tests near:

- `tests/test_guardrails.py`
- `tests/test_backtest/`
- `tests/test_envs/test_account_normalization.py`
- `tests/test_risk.py`
- `tests/test_envs/test_risk_alignment.py`
- `tests/test_envs/test_fixed_risk_sizing.py`
- `tests/test_envs/test_year_episode_survival.py`
- `tests/test_envs/test_ny_steps_eod.py`
- `tests/test_envs/test_trading_env_golden_traces.py`
- `tests/test_envs/test_bundle_worker.py`
- `tests/test_envs/test_observation_schema.py`
- `tests/test_rollout.py`
- `tests/test_eval/test_eval_run_provenance.py`
- `tests/test_plots_export.py`
- relevant plotting tests.

Acceptance criteria:

- fixed session-opening-equity daily floor;
- unrealized loss counts;
- no intraday-profit ratchet;
- correct session reset;
- static $90k floor;
- no repeated-payout account trailing DD;
- $110k flat-realized payout trigger;
- exact payout/reset/cumulative accounting;
- repeated cycles;
- failure after prior payouts;
- risk mapping [0,0.009];
- zero-risk hold;
- post-rounding/min-lot hard-cap validation;
- daily/account capacity enforcement;
- direct-env/worker parity;
- train/eval accounting parity;
- plot/export/environment parity;
- checkpoint/schema mismatch rejection;
- legacy/default policy golden behavior unchanged.

## 11. Backward compatibility requirements

- default remains legacy;
- no existing policy is removed;
- existing config keys remain valid;
- legacy training trailing-DD behavior remains;
- legacy evaluation trailing-DD disablement remains;
- legacy fixed-risk behavior remains;
- multi-TP/execution behavior remains;
- checkpoints only load under compatible semantics or fail clearly;
- never silently reinterpret old `daily_loss_over_limit`.

## 12. Partial-implementation traps

Avoid:

- fixing plots while env/reward/constraints remain on realized losing closes;
- changing daily-loss meaning without observation-schema versioning;
- updating direct env but not worker;
- clipping requested risk without checking rounded actual risk;
- fixed-risk bypass of repeated-payout capacity;
- payout on unrealized equity;
- balance reset with stale pre-payout daily anchor;
- cumulative-payout reset during cycle reset;
- payout cash treated as trading equity;
- legacy trailing DD accidentally active in repeated-payout mode;
- standalone backtest rule drift from claimed baseline methodology;
- independent policy recomputation in exporters/plots;
- labeling gap/slippage overshoot as agent-requested risk.

## 13. Ordered implementation phases

1. **Freeze contracts:** config validation, equations, payout/session/risk definitions, unit tests.
2. **Minimal accounting:** session-equity daily rule, static floor, cycle state/reset, legacy default.
3. **Environment/risk wiring:** [0,0.009], capacity enforcement, post-rounding validation, risk logs, payout transition.
4. **Worker/reward/observation/checkpoints:** parity, canonical constraints/reward, new observation version, provenance.
5. **Evaluation/export/plots:** chronological cycles, payout/risk artifacts, sustainability metrics.
6. **Regression/parity:** repeated-payout tests plus legacy golden suites; decide baseline parity scope.
7. **Smoke training.**
8. **Main training** with existing Triton/Slurm/GH200 workflow.
9. **Chronological OOS evaluation** and thesis artifact regeneration.

Use the existing external watcher mechanism for long-running jobs instead of repeatedly polling job state.

## 14. Smoke-training procedure

1. deterministic handcrafted payout/failure traces;
2. very short PPO repeated-payout smoke run;
3. verify nonzero trades and plausible requested/executed risk;
4. verify success/failure episode reasons;
5. load checkpoint through normal evaluation;
6. small held-out repeated-cycle evaluation;
7. verify payout/cycle/risk exports;
8. verify direct/worker parity.

Do not proceed if action semantics differ, observation mismatch is bypassed, payout reuses observations, rounded risk exceeds capacity, or exported/plot daily floors differ from the environment.

## 15. Full-training and OOS evaluation procedure

For each final seed/variant:

- archive resolved config;
- record source revision and current dirty-state provenance;
- record data/feature fingerprints;
- train single-cycle episodes;
- preserve checkpoints through current infrastructure;
- evaluate frozen checkpoints on the same chronological OOS split;
- use identical execution/account policy across compared repeated-payout runs;
- separate policy learning, execution assumptions, accounting constraints, and sustainability conclusions in thesis reporting.

## 16. Final metrics and expected deliverables

Required metrics:

- cumulative payouts;
- payout count and amounts;
- successful/failed cycles and rates;
- time/steps/sessions per successful cycle;
- payout frequency;
- longest no-payout interval;
- final operating equity;
- total economic value = final operating equity + cumulative payouts;
- daily/static-floor breach counts;
- requested risk fraction/USD distributions;
- approved risk distribution;
- modeled stop-risk distribution;
- requested-to-executed risk ratio;
- rejection counts by daily capacity/account capacity/min-lot rounding;
- realized loss beyond modeled risk due to gap/slippage;
- existing return/Sharpe/drawdown/trade metrics.

Expected artifacts:

- resolved config;
- checkpoints;
- run manifest/provenance;
- equity/trades/fills/positions;
- breach events;
- cycle events;
- payout events;
- risk diagnostics;
- evaluation diagnostics;
- repeated-payout sustainability summary;
- regenerated thesis figures/tables from canonical exports.

## 17. Recommended policy configuration and unresolved decision

Freeze these implementation defaults:

1. Daily loss uses a fixed session-opening-equity floor and current equity including unrealized PnL.
2. Intraday profits do not ratchet that floor.
3. Prospective risk capacity is current equity minus the fixed floor and is separate from breach detection.
4. Account loss is a static $90k equity floor.
5. Account trailing DD is disabled in repeated-payout mode.
6. Payout requires flat position and realized balance >= $110k.
7. Payout removes balance above $100k and resets operating capital to $100k while preserving cumulative payouts.
8. Evaluation continues at the next chronological observation.
9. Agent risk is continuous [0,0.009], zero allowed, with execution-aware capacity validation.
10. Daily-loss semantic changes get a new observation schema version.
11. Legacy remains default and retains current behavior.

The main unresolved numeric design choice is the conservative execution reserve used to convert remaining policy capacity into usable pre-trade risk. Derive it from the repository's existing spread/commission/slippage model or expose it as an explicit reproducible configuration value; do not hide it in an unexplained constant.
