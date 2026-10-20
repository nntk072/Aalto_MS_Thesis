# Environment Domain

## Primary Files

| File | Role |
|------|------|
| `simulation/environment/env.py` | Simulator facade with native transition and Gymnasium adapter |
| `simulation/contracts/transition.py` | Framework-neutral result of one simulation transition |
| `simulation/environment/initialization_runtime.py` | Initialization-stage composition root |
| `simulation/environment/market_source_initialization.py` | Market source, frame, tick, and bundle setup |
| `simulation/environment/account_policy_initialization.py` | Account, broker, cost, and policy setup |
| `simulation/environment/risk_entry_initialization.py` | Risk controls and entry state-machine setup |
| `simulation/environment/execution_controls_initialization.py` | Fill, latency, episode, and execution controls |
| `simulation/environment/reward_initialization.py` | Reward function construction |
| `simulation/environment/strategy_initialization.py` | Strategy and close-accounting state initialization |
| `simulation/environment/observation_arrays_initialization.py` | Observation feature and market arrays |
| `simulation/environment/action_episode_initialization.py` | Action schema and episode state initialization |
| `simulation/environment/episode_spaces_initialization.py` | Observation-space construction |
| `simulation/environment/step_runtime.py` | Step transition stages and result construction |
| `simulation/environment/action_intent_runtime.py` | Strategy action and entry validation |
| `simulation/environment/step_constraints_runtime.py` | Risk, session, and delayed-fill constraints |
| `simulation/environment/step_action_execution.py` | Entry and discretionary action execution |
| `simulation/environment/step_accounting_runtime.py` | Reward and equity accounting |
| `simulation/environment/entry_funnel_runtime.py` | Candidate evidence and rejection funnel updates |
| `simulation/environment/step_info_runtime.py` | Step result and constraint telemetry |
| `simulation/environment/step_end_runtime.py` | Episode progression and termination semantics |
| `simulation/environment/execution_runtime.py` | Execution service composition root |
| `simulation/environment/protective_level_runtime.py` | Bar and chronological multi-TP protective-level transitions |
| `simulation/environment/tick_quote_runtime.py` | Tick exit lookup and randomized quote resolution |
| `simulation/environment/fill_quote_runtime.py` | Stop, target, and bar quote construction |
| `simulation/environment/trade_accounting_runtime.py` | Exit provenance and realized PnL/R bookkeeping |
| `simulation/environment/ema_exit_runtime.py` | EMA-based protective exit decisions |
| `simulation/environment/action_diagnostics.py` | Action coordinate telemetry |
| `simulation/environment/entry_diagnostics.py` | Entry rejection and funnel contracts |
| `simulation/environment/timebase.py` | Bar spacing and broker-clock semantics |
| `simulation/environment/mixin_base.py` | Typing support for composed environment services |
| `simulation/environment/position_constants.py` | Position mode ordering and price tolerance |
| `simulation/environment/factory.py` | Builds environments and resolves shared risk, action, cost, and execution kwargs |
| `decisioning/observation_state.py` | Stable account width and entry-state observation coordinates |
| `simulation/environment/rewards.py` | Differential Sharpe Ratio (DSR) reward |
| `simulation/environment/sweep_reward.py` | Sweep confirmation + CompositeReward |
| `simulation/environment/po3_reward.py` | PO3 strategy alignment reward |
| `simulation/environment/distribution_reward.py` | Distribution strategy alignment reward |
| `trading/strategies/base.py` | TradingStrategy base class |
| `trading/strategies/baseline.py` | Baseline (no-op) strategy |
| `trading/strategies/po3_ifvg.py` | PO3 + IFVG strategy |
| `trading/strategies/distribution.py` | Distribution strategy |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `TradingEnv` | `simulation/environment/env.py` | Main Gymnasium environment class |
| `TradingEnv.step()` | `simulation/environment/env.py` | Execute one step (action → reward) |
| `DSRReward` | `simulation/environment/rewards.py` | Differential Sharpe Ratio reward |
| `SweepConfirmationReward` | `simulation/environment/sweep_reward.py` | Sweep-based reward |
| `CompositeReward` | `simulation/environment/sweep_reward.py` | Weighted composite of rewards |
| `PO3Reward` | `simulation/environment/po3_reward.py` | PO3 alignment reward (entry transitions only) |
| `DistributionReward` | `simulation/environment/distribution_reward.py` | Distribution alignment reward |
| `TradingStrategy` | `trading/strategies/base.py` | Strategy semantics base class |
| `BaselineStrategy` | `trading/strategies/baseline.py` | No-op baseline (Idea 3) |
| `PO3IFVGStrategy` | `trading/strategies/po3_ifvg.py` | PO3 + IFVG entry logic |
| `DistributionStrategy` | `trading/strategies/distribution.py` | Distribution entry logic |

## Direct Dependencies

- `simulation/backtest/engine.py` for event-driven replay
- `trading/` for broker execution, account state, risk, orders, and strategies
- `features/pipeline/builder.py` (feature matrix, `MTF_RAW_SUFFIXES`)
- `market_data/transforms/sessions.py` (`ny_session_mask`)
- `features/context/reachability.py` (causal available-target distance used by environment entry checks)

## Consumers (Callers)

- `training/runner/training.py` (creates env, trains model)
- `evaluation/rollout/model.py` (evaluates model through env)
- `evaluation/rollout/runner.py` (runs episodes)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_envs/test_trading_env_decomposition.py` | Env structure |
| `tests/test_envs/test_trading_env_golden_traces.py` | Golden trace regression |
| `tests/test_envs/test_account_normalization.py` | Account state normalization |
| `tests/test_envs/test_baseline_regression.py` | Overlay invariant (baseline unchanged) |
| `tests/test_envs/test_continuous_actions.py` | Continuous action space |
| `tests/test_envs/test_ny_steps_eod.py` | NY session + EOD logic |
| `tests/test_envs/test_strategy_env.py` | Strategy-mode env wiring |
| `tests/test_envs/test_sweep_reward.py` | Sweep reward |
| `tests/test_envs/test_po3_reward.py` | PO3 reward |
| `tests/test_envs/test_distribution_reward.py` | Distribution reward |
| `tests/test_reward.py` | DSR reward |

## Do NOT Read For Env Tasks

- `features/technical/indicators.py` (indicator implementation detail)
- `models/encoders/core.py` (encoder detail — only need `build_agent`)
- `training/runner/training.py` (training orchestration)
- `integrations/mt5/` (live trading — separate domain)

## Critical Invariants

## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Change a state transition | simulation service | `simulation/environment/step_runtime.py` | action/observation contracts; `tests/test_envs/test_sl_tp_fill.py` | one step or execution service plus golden checks | model family and learner workflow |
| Change observation layout | schema evolution | `decisioning/observations.py` | versioned observation schema; `tests/test_envs/test_observation_schema.py` | schema and compatibility boundary | market-data source and report metrics |

## Axis of Change

- **Axis:** simulation clock, state transitions, fills, account/risk application, observations, and episode rules.
- **Owner:** `simulation/`; trading owns reusable account/order/risk semantics, features owns market-derived context.
- **Stable boundary:** `SimulationTransition` and the environment factory; Gymnasium `step()` adapts the native transition for RL callers.
- **Expected edit radius:** one transition/service module and its focused environment tests.
- **Must not depend on:** model classes, PPO/SAC implementations, or training runners.

1. **Observation space**: `obs["seq"]` is `[T, F]` float32. `obs["account"]` has 6 float32 values (equity, direction, open PnL, unrealised R, SL distance, trailing drawdown), plus 4 one-hot entry-state values when `entry_state_observation` is enabled with an entry-state machine.
2. **Discrete actions**: `{hold=0, enter_long=1-9, enter_short=10-18, exit=19}`
3. **Continuous actions**: `Box(-1, 1)` for proportional position sizing
4. **Strategy mode**: flag-derived `Box[-1,1]` action coordinates are defined by `decisioning/actions.py:build_action_schema` (schema version `overlay-action-v2`). Base: `[intensity, stop, risk, target, exit]` (5-D); optional direction is first; multi-TP replaces `target` with `tp1_sel, tp2_sel, tp3_sel`; optional SL/TP mode coordinates precede optional `z1,z2` simplex logits; `exit` is last. With every option enabled, the layout is 12-D. A width alone cannot identify semantics; retain the schema flags and version in model/checkpoint metadata.
5. **Overlay invariant**: with `strategy_actions=false`, baseline behaviour is bit-for-bit unchanged
6. **Block overnight**: `block_overnight=True` closes positions at last NY bar
## Verified Source Paths

- `simulation/environment/env.py`
- `simulation/environment/initialization_runtime.py`
- `simulation/environment/step_runtime.py`
- `simulation/environment/action_intent_runtime.py`
- `simulation/environment/step_constraints_runtime.py`
- `simulation/environment/step_action_execution.py`
- `simulation/environment/step_accounting_runtime.py`
- `simulation/environment/entry_funnel_runtime.py`
- `simulation/environment/step_info_runtime.py`
- `simulation/environment/step_end_runtime.py`
- `simulation/environment/execution_runtime.py`
- `simulation/environment/action_diagnostics.py`
- `simulation/environment/entry_diagnostics.py`
- `simulation/environment/timebase.py`
- `simulation/environment/mixin_base.py`
- `simulation/environment/position_constants.py`
- `simulation/environment/factory.py`
- `decisioning/observation_state.py`
- `tests/test_envs/test_observation_schema.py`
