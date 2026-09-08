# Quantitative RL Trading System: Architecture

- [Introduction](#introduction)
- [Terminology](#terminology)
- [Disclaimers](#disclaimers)
- [Scope](#scope)
- [End-to-end procedure](#end-to-end-procedure)
- [Architecture](#architecture)
- [Lifecycle of one cycle](#lifecycle-of-one-cycle)
  - [Data and cache](#data-and-cache)
  - [Feature chains and strategy state](#feature-chains-and-strategy-state)
  - [Environment step](#environment-step)
  - [Training, evaluation and walk-forward](#training-evaluation-and-walk-forward)
  - [Live: paper first, then promote](#live-paper-first-then-promote)
- [Budgets and constraints](#budgets-and-constraints)
- [Causal-only rule](#causal-only-rule)
- [Adding a strategy](#adding-a-strategy)
- [References](#references)

## Introduction

This describes how the repository turns raw M1 price files into a trained
reinforcement-learning agent and, optionally, a live trading robot: the data
pipeline, the feature builders, the `TradingEnv` contract, the backtest
engine, the training and evaluation entry points, and the live bridge.

This page documents the **framework**: the pipeline stages, the contracts
between them, and the invariants that hold across every configuration.
*What* a particular strategy trades - the PO3/IFVG chain of Idea 1 or the
distribution chain of Idea 2 - is documented in the modules that own it
(`quant_rl/envs/strategies/` plus `config/idea1_po3_ifvg.yaml` /
`config/idea2_distribution.yaml`) and is linked rather than duplicated here
as a tutorial. [Adding a strategy](#adding-a-strategy) describes how a new
one is plugged in.

How-to material lives elsewhere and is not repeated:

- [`README.md`](../README.md) — installation, quick start, project layout.
- [`RUNNING_COMMANDS.md`](../doc/RUNNING_COMMANDS.md) — every command a typical
  research or evaluation session needs.
- [`DEPLOYMENT.md`](../doc/DEPLOYMENT.md) — the paper-to-live promotion protocol.

## Terminology

**M1 spine**
   One-minute bars are the only true source data. Every higher timeframe is
   resampled from the M1 files named under `data.m1_files` in
   `quant_rl/config/default.yaml`; separately downloaded higher-TF CSVs are
   ignored (see [Disclaimers](#disclaimers)).

**HTF / LTF**
   Higher timeframe (M15 FVG zones) and lower timeframe (M5 confirmations)
   in the multi-timeframe signal stack.

**OOS split**
   The locked train/out-of-sample boundary in `data.split`:
   in-sample is `train_end` inclusive, out-of-sample starts at
   `test_start`. Ideas may not move these dates (see
   [Budgets and constraints](#budgets-and-constraints)).

**Causal feature**
   A feature column whose value at bar `t` is computed only from
   information available at decision time `t`. See
   [Causal-only rule](#causal-only-rule).

**Overlay vs baseline**
   The strategy machinery added on top of the baseline (Idea 3)
   configuration. Baseline behaviour is the reference; Idea 1/2 are strict
   opt-in overlays (see [Overlay invariant](#overlay-invariant)).

**`TradingStrategy` vs `TradingEnv`**
   A strategy object owns *semantics only*: whether an entry is valid at
   bar `t`, where the structural stop reference sits, and which
   take-profit targets are candidates. The environment owns all *mechanics*:
   fills, lot sizing, account state, costs, guardrails and logging
   (`quant_rl/envs/strategies/__init__.py`).

**Observation `seq` / `account`**
   The environment's observation is a dictionary: `seq` is the normalised
   60-bar feature window (plus optional VAE latent `vae_z`), `account` is
   the scalar account state. Raw price-level columns requested by a
   strategy (`raw_columns`) are kept for execution but excluded from the
   normalised sequence.

**Discrete vs strategy action space**
   Legacy spaces: `Discrete(20)` (`0` hold, `1-9` enter-long variants,
   `10-18` enter-short variants, `19` exit) or a 1-D `Box(-1, 1)` for
   proportional sizing. Strategy mode replaces these with one 4-D
   `Box` `[direction, risk, rr, tp]` (see [Environment step](#environment-step)).

**DSR vs sweep vs entry-event rewards**
   Three reward layers combined by `CompositeReward`: the economic
   Differential Sharpe Ratio (`DSRReward`), an optional
   `SweepConfirmationReward`, and an optional strategy *alignment* reward
   (`PO3Reward` / `DistributionReward`) that fires on entry transitions
   only.

**Liquidity sweep vs BOS**
   A sweep is a push through a prior swing level followed by a reclaim; a
   break of structure is a confirmed close beyond a swing level. Both are
   detected causally in `quant_rl/features/liquidity.py`.

**Manipulation vs distribution phase**
   The two PO3 stages tracked in `quant_rl/features/po3_state.py`:
   the manipulation sweep that runs against the eventual direction, and the
   distribution leg that follows it.

**FTMO limits vs `live_risk_overrides`**
   Training/evaluation enforce the *dollar* kill-switches of the `ftmo:`
   block via guardrails; live MT5 sizing uses the *percentage-of-balance*
   block `live_risk_overrides:`. The two must stay aligned
   (`quant_rl/config/default.yaml`).

**Paper vs live**
   `PAPER_TRADING=true` (the default) logs every signal and intended order
   without placing anything; `PAPER_TRADING=false` routes orders to the
   broker.

## Disclaimers

- **Higher-TF CSVs are ignored.** All timeframes are resampled from the
  true M1 files named in `data.m1_files`; a separately downloaded M5/H1
  file is never read. Tick files (`data.tick_files`) are used only for
  fill pricing when `costs.use_tick_execution` is on.
- **The feature cache is versioned.** Feature pipelines are cached as
  parquet keyed by `FEATURE_CACHE_VERSION` (currently
  `v5-idea1-strategy-state` in `quant_rl/features/build.py`). A change to
  the feature schema must bump the version; stale caches are otherwise
  silently reused.
- **Idea 1/2 runs use a distinct cache key.** `quant_rl/train/train_rl.py`
  selects `v5_idea1` when `features.include_strategy_state` is on and
  `v4_po3causal` otherwise, so overlay runs never poison the baseline
  cache.
- **OOS dates are locked in `default.yaml`.** `data.split.train_end` /
  `test_start` are the single source of truth for the split; variant
  configs must not override them.
- **`PAPER_TRADING` defaults to true.** The live entry point
  (`live_trading_rl.py`) logs signals only unless paper mode is
  explicitly disabled.
- **Training dollar caps and live percentage sizing must stay aligned.**
  `ftmo:` (dollar kill-switches) and `live_risk_overrides:` (% of
  balance) encode the same risk appetite for different units; changing
  one without the other silently diverges training from live sizing.
- **Strategy registration is config-time only.** A strategy is selected
  through `_STRATEGY_CONFIGS` in `quant_rl/train/train_rl.py` and the
  merged YAML; there is no runtime plugin loader.
- **The overlay is strict (see [Overlay invariant](#overlay-invariant)).** With
  `env.strategy_actions=false` (default), the environment keeps the
  legacy discrete or 1-D continuous action space, the DSR (or sweep)
  reward, swing-level SL/TP and the full observation column set —
  bit-for-bit baseline behaviour, locked by
  `tests/test_envs/test_baseline_regression.py`. Idea 1/2 are *not* a
  replacement of Idea 3; they activate only when a variant YAML sets
  `strategy_actions: true` and `features.include_strategy_state: true`.
- **Alignment rewards fire on entry transitions only, and stay small.**
  `PO3Reward` / `DistributionReward` return `0.0` unless the position
  changed this step, and their configured weights (`strategy_weight:
  0.1`) are an order of magnitude below the economic reward — they shape
  *where* entries happen, never dominate PnL.

### Overlay invariant

PR #3 added the strategy machinery as a **strict overlay**. When
`env.strategy_actions=false` (default / Idea 3 baseline),
`quant_rl/envs/trading_env.py` must keep:

- the legacy discrete (`Discrete(20)`) or 1-D continuous action space,
- the DSR (or sweep) reward with no alignment component,
- swing-level SL/TP, not structural strategy stops,
- the full observation column set.

`_strategy_from_cfg` in `quant_rl/train/train_rl.py` returns
`(BaselineStrategy(), None, 0.0)` unless a variant config enables
`strategy_actions`; only then does it return
`(PO3IFVGStrategy, PO3Reward, weight)` or
`(DistributionStrategy, DistributionReward, weight)`. The invariant is
locked by `tests/test_envs/test_baseline_regression.py`.

## Scope

- `quant_rl/` owns research: data, features, environment, backtest
  primitives, training and evaluation.
- `mt5_trading/` owns broker I/O only. The environment never places MT5
  orders; the live path never retrains.
- Feature detectors never look ahead (see [Causal-only rule](#causal-only-rule)).
- Strategy objects own semantics only; fills, lots and logging stay in
  the environment.

## End-to-end procedure

Research pipeline, one configuration run (`quant_rl/train/train_rl.py`):

```mermaid
sequenceDiagram
    participant P as run_pipeline (data/pipeline.py)
    participant B as build_features (features/build.py)
    participant L as liquidity / po3_state
    participant S as split_train_test (data/split.py)
    participant E as TradingEnv
    participant ST as _strategy_from_cfg (train_rl.py)
    participant A as build_agent (models/agent.py)
    participant V as evaluate_model (eval/rollout.py)
    P->>B: true M1 CSVs (data.m1_files)
    B->>L: only when include_strategy_state
    L-->>B: sweep / BOS / PO3 / IFVG columns
    B-->>P: feature frames (versioned parquet cache)
    P->>S: feature frames
    S-->>E: in-sample / OOS frames (locked dates)
    ST-->>E: strategy + alignment reward (or baseline)
    E-->>A: observation space
    A-->>V: trained checkpoint
    V-->>V: metrics, trades, plots -> outputs/
```

The optional feature stage is the overlay: `detect_liquidity_sweeps`,
`detect_bos`, `build_po3_state` and `build_ifvg_zone_features` run only
when `features.include_strategy_state` is enabled by a variant config
(`config/idea1_po3_ifvg.yaml`, `config/idea2_distribution.yaml`).

Live loop, one bar (`live_trading_rl.py` driving
`mt5_trading/robot/rl_robot.py`):

```mermaid
sequenceDiagram
    participant MT5 as MT5 terminal
    participant R as RLRobot (mt5_trading/robot/rl_robot.py)
    participant SA as RLStrategyAdapter (quant_rl/live/rl_strategy.py)
    participant RM as RiskManager
    participant TR as Trader
    MT5->>R: new M1 bar
    R->>SA: bar -> model observation
    SA-->>R: signal (direction, sizing intent)
    R->>RM: position sizing (live_risk_overrides)
    R->>TR: open/close via Trader, or paper log when PAPER_TRADING
```

## Architecture

```mermaid
flowchart LR
  data[data_pipeline]
  feat[features_build]
  liq[liquidity_po3_state]
  env[TradingEnv]
  strat[TradingStrategy]
  reward[alignment_reward]
  engine[backtest_engine]
  risk[structural_risk]
  agent[build_agent]
  eval[evaluation]
  live[mt5_RLRobot]
  data --> feat
  feat --> liq
  feat --> env
  liq --> env
  strat --> env
  reward --> env
  engine --> env
  risk --> env
  env --> agent
  agent --> eval
  agent --> live
```

**Data** — `quant_rl/data/pipeline.py` (`run_pipeline`) ingests the true
M1 files, resamples every higher timeframe from them, and hands feature
frames to `quant_rl/data/split.py` (`split_train_test`), which applies the
locked in-sample/OOS dates. Tick files are consulted by the backtest for
fill pricing only.

**Features** — `quant_rl/features/build.py` (`build_features`) is the
single entry point. It owns `FEATURE_CACHE_VERSION` and the opt-in
`include_strategy_state` block that appends the overlay columns from
`quant_rl/features/liquidity.py` (`detect_liquidity_sweeps`,
`detect_bos`) and `quant_rl/features/po3_state.py` (`build_po3_state`,
`build_ifvg_zone_features`).

**Environment and strategies** — `quant_rl/envs/trading_env.py`
(`TradingEnv`) hosts the gym interface, mechanics and composite reward.
`quant_rl/envs/strategies/` holds the semantics objects
(`TradingStrategy`, `BaselineStrategy`, `PO3IFVGStrategy`,
`DistributionStrategy`).

**Alignment rewards** — `quant_rl/envs/po3_reward.py` (`PO3Reward`),
`quant_rl/envs/distribution_reward.py` (`DistributionReward`) and
`quant_rl/envs/sweep_reward.py` (`SweepConfirmationReward`,
`CompositeReward`). `CompositeReward` filters strategy-reward keyword
inputs through each component's `required_inputs`.

**Backtest primitives** — `quant_rl/backtest/engine.py` plus the account,
broker, costs and guardrails modules; `quant_rl/backtest/risk.py`
provides the structural risk helpers `compute_sl_tp_from_structure`
(`sl_mode: exact` when `buffer_pts=0.0`, `buffered` with a positive
buffer) and `resolve_tp_target` (`rr`, liquidity and
previous-day-high/low targets with `rr_fallback` for invalid ones).

**Models** — `quant_rl/models/agent.py` (`build_agent`) assembles the
PPO/SAC agent with its sequence encoder; optional VAE latents are frozen
when present.

**Training** — `quant_rl/train/train_rl.py` wires everything: `_STRATEGY_CONFIGS`
maps strategy names to variant YAMLs, `_strategy_from_cfg` builds
`(strategy, strategy_reward, weight)`, and the merged config drives
`make_env`, `build_agent` and learning.

**Evaluation** — `quant_rl/eval/` produces run artifacts (rollouts,
exports, metrics, bootstrap CIs, calibration, walk-forward); the
directory `quant_rl/evaluation/` holds reporting helpers
(`calculate_metrics`, plots, chart indicators). Checkpoints, trades and
plots land under `outputs/`.

**Live** — `live_trading_rl.py` loads a checkpoint and runs
`quant_rl/live/rl_strategy.py` (`RLStrategyAdapter`) inside
`mt5_trading/robot/rl_robot.py` (`RLRobot`), which sizes through
`RiskManager` and executes through a `Trader` — or only logs, when
`PAPER_TRADING` is on.

**Config** — `quant_rl/config/default.yaml` is the base; variant YAMLs
under `config/` are merged over it. A variant is opt-in end to end: it
must flip `features.include_strategy_state`, `env.strategy_actions` and
declare the `strategy:` block (see the two example files).

Full-system class view. Packages are namespaces; `<<module>>` boxes group
free functions of one file. Every name below exists at `main` (`3e56c5b`).

```mermaid
classDiagram
    direction TB

    namespace data {
        class loader {
            <<module>>
            +load_bars(path)
            +load_ticks(path)
            +iter_ticks_chunks(path, chunksize)
        }
        class clean_mod {
            <<module>>
            +clean(df, tz)
        }
        class resample_mod {
            <<module>>
            +resample(m1, tf)
            +build_all_timeframes(m1)
        }
        class align_mod {
            <<module>>
            +align_timeframes()
            +join_symbols()
        }
        class session_mod {
            <<module>>
            +filter_session()
            +add_session_id(df)
        }
        class TickBook {
            <<tick bid/ask store>>
        }
        class pipeline_mod {
            <<module>>
            +run_pipeline(cfg, force)
            +build_tick_books(cfg)
        }
        class split_mod {
            <<module>>
            +split_train_test()
            +get_split_config(cfg)
        }
    }

    namespace features {
        class build_mod {
            <<module>>
            +build_features()
            +FEATURE_CACHE_VERSION
        }
        class liquidity_mod {
            <<module>>
            +detect_liquidity_sweeps()
            +detect_bos(bars, structure)
        }
        class po3_state_mod {
            <<module>>
            +build_po3_state()
            +build_ifvg_zone_features()
        }
    }

    namespace envs {
        class TradingEnv {
            +strategy: TradingStrategy
            +reward_fn: DSRReward | CompositeReward
            +strategy_actions: bool
            +action_space
            +observation_space
            +step(action)
            +reset()
        }
        class TradingStrategy {
            <<abstract>>
            +str name
            +tuple~str~ required_features
            +tuple~str~ raw_columns
            +validate_entry(direction, row)* bool
            +sl_reference(direction, row)* float
            +target_candidates(direction, row)* dict
        }
        class BaselineStrategy
        class PO3IFVGStrategy {
            +enforce_gate: bool
            +require_asian_context: bool
        }
        class DistributionStrategy {
            +enforce_gate: bool
        }
        class DSRReward {
            +eta: float
            +reset()
            +__call__(pnl_step)
        }
        class SweepConfirmationReward {
            +alpha: float
            +beta: float
            +hold_bars: int
        }
        class CompositeReward {
            +sweep_reward: SweepConfirmationReward
            +_dsr_fn: DSRReward
            +strategy_reward: PO3Reward | DistributionReward | None
            +strategy_weight: float
        }
        class PO3Reward {
            +required_inputs
        }
        class DistributionReward {
            +required_inputs
        }
    }

    namespace backtest {
        class run_backtest_mod {
            <<module>>
            +run_backtest(bars, features, policy)
        }
        class AccountState {
            +initial_balance: float
            +balance: float
            +equity: float
            +peak_equity: float
            +daily_loss: float
            +max_drawdown: float
            +update_equity(open_pnl)
        }
        class Broker {
            +leverage: int
            +margin_pct: float
            +contract_size: float
            +cost_model: CostModel
            +required_margin(price, lots)
            +open_position(acc, quote, lots, direction) Position
        }
        class Position {
            +direction: int
            +size: float
            +entry_price: float
            +sl_price: float
            +tp_price: float
        }
        class CostModel {
            +fill_price(bid, ask, direction)
        }
        class costs_mod {
            <<module>>
            +COST_US100
        }
        class TradingCostModel
        class FTMOGuardrails {
            +daily_loss_limit: float
            +max_loss_limit: float
            +risk_per_trade_limit: float
            +check_daily(acc) bool
            +check_max_drawdown(acc) bool
            +check_trade_risk(risk) bool
            +any_breach(acc) bool
            +breach_reason(acc) str
        }
        class risk_mod {
            <<module>>
            +compute_sl_tp_from_structure()
            +resolve_tp_target()
        }
    }

    pipeline_mod ..> loader : M1 CSVs
    pipeline_mod ..> clean_mod
    pipeline_mod ..> resample_mod
    pipeline_mod ..> session_mod
    pipeline_mod ..> TickBook : build_tick_books
    pipeline_mod ..> split_mod : locked dates
    split_mod ..> build_mod : feature frames
    build_mod ..> liquidity_mod : include_strategy_state
    build_mod ..> po3_state_mod : include_strategy_state

    TradingEnv *-- TradingStrategy : semantics only
    TradingEnv *-- CompositeReward : reward_fn (overlay)
    TradingEnv ..> DSRReward : reward_fn (baseline)
    TradingEnv ..> AccountState
    TradingEnv ..> Broker
    TradingEnv ..> CostModel
    TradingEnv ..> FTMOGuardrails
    TradingEnv ..> risk_mod : structural SL/TP
    TradingStrategy <|-- BaselineStrategy
    TradingStrategy <|-- PO3IFVGStrategy
    TradingStrategy <|-- DistributionStrategy
    CompositeReward *-- SweepConfirmationReward
    CompositeReward *-- DSRReward
    CompositeReward o-- PO3Reward : optional, entry transitions
    CompositeReward o-- DistributionReward : optional, entry transitions

    run_backtest_mod ..> Broker
    run_backtest_mod ..> AccountState
    run_backtest_mod ..> CostModel
    run_backtest_mod ..> FTMOGuardrails
    run_backtest_mod ..> TickBook : tick fills
    Broker ..> Position : creates
    Broker *-- CostModel

    namespace models {
        class build_agent_mod {
            <<module>>
            +build_agent()
        }
        class Encoder {
            <<abstract>>
            +obs_dim: int
            +seq_len: int
            +latent_dim: int
            +forward(seq)
        }
        class Policy {
            <<abstract>>
            +forward(obs)
        }
        class TCNEncoder {
            <<BaseFeaturesExtractor>>
        }
        class TransformerEncoder {
            <<BaseFeaturesExtractor>>
        }
        class GRUEncoder {
            <<BaseFeaturesExtractor>>
        }
        class VAE {
            +VAEEncoder encoder
            +VAEDecoder decoder
        }
        class VAEEncoder
        class VAEDecoder
        class VAEFeatureExtractor {
            <<BaseFeaturesExtractor>>
        }
    }

    namespace train {
        class train_rl_mod {
            <<module>>
            +train_rl()
            +make_env()
            +_STRATEGY_CONFIGS
        }
        class ProgressLoggerCallback {
            <<BaseCallback>>
            +_on_rollout_end()
            +_on_training_end()
        }
        class BestCheckpointEvalCallback {
            <<BaseCallback>>
            +eval_env_factory
            +best_mean_reward: float
            +_run_eval()
        }
        class AuxiliaryTrainerCallback {
            <<BaseCallback>>
        }
        class auxiliary_mod {
            <<module>>
            +build_supervised_windows()
        }
    }

    namespace eval {
        class rollout_mod {
            <<module>>
            +evaluate_model()
            +make_action_fn()
        }
        class export_mod {
            <<module>>
            +build_run_dir(base, name)
            +save_run()
        }
    }

    namespace evaluation {
        class metrics_mod {
            <<module>>
            +calculate_metrics()
        }
        class runner_mod {
            <<module>>
            +run_episode()
        }
        class WFSplit {
            <<walk-forward split>>
        }
        class walkforward_mod {
            <<module>>
            +purged_walk_forward()
        }
        class CI {
            +as_tuple()
        }
        class bootstrap_mod {
            <<module>>
            +bootstrap_ci()
            +metrics_with_ci()
            +sharpe_stat()
            +sortino_stat()
            +max_drawdown_stat()
            +win_rate_stat()
        }
        class CalibrationReport {
            +summary() str
        }
        class calibration_mod {
            <<module>>
            +calibration_report()
            +plot_reliability_diagram()
        }
    }

    namespace live {
        class live_entry_mod {
            <<module>>
            +PAPER_TRADING default true
        }
        class RLStrategyAdapter {
            +update_bars(bars, secondary_bars)
            +build_observation()
            +predict_signal(account_state) int
            +as_strategy()
        }
        class RLRobot {
            +trade()
            +calculate_position_size(symbol)
            +check_risk_before_trade(symbol, position_size)
        }
    }

    namespace mt5_trading {
        class Trader {
            <<abstract>>
            +open_position()*
            +close_positions()*
            +get_opened_positions()*
            +get_all_positions()*
            +send_to_break_even()*
            +calculate_position_size()*
        }
        class TradingData {
            <<abstract>>
            +get_data()*
            +get_symbol()*
        }
        class MT5Strategy {
            <<abstract mt5_trading.adapters.strategy.TradingStrategy>>
            +signal()*
        }
        class RiskManager {
            +calculate_position_size()
            +check_risk_limits()
            +get_total_exposure(symbols)
            +get_volatility_multiplier(atr_pct, base_atr)
        }
    }

    build_mod ..> VAE : optional latent
    build_agent_mod ..> Encoder
    build_agent_mod ..> Policy
    build_agent_mod ..> TCNEncoder
    build_agent_mod ..> TransformerEncoder
    build_agent_mod ..> GRUEncoder
    build_agent_mod ..> VAEFeatureExtractor
    VAE *-- VAEEncoder
    VAE *-- VAEDecoder
    VAEFeatureExtractor ..> VAE

    train_rl_mod ..> pipeline_mod : run_pipeline
    train_rl_mod ..> build_mod : build_features
    train_rl_mod ..> TradingEnv : make_env
    train_rl_mod ..> envs.TradingStrategy : _strategy_from_cfg
    train_rl_mod ..> envs.PO3Reward
    train_rl_mod ..> envs.DistributionReward
    train_rl_mod ..> build_agent_mod
    train_rl_mod ..> ProgressLoggerCallback
    train_rl_mod ..> BestCheckpointEvalCallback
    train_rl_mod ..> AuxiliaryTrainerCallback
    train_rl_mod ..> rollout_mod : evaluate_model
    train_rl_mod ..> export_mod : save_run
    BestCheckpointEvalCallback ..> TradingEnv : fresh eval env
    AuxiliaryTrainerCallback ..> auxiliary_mod

    rollout_mod ..> runner_mod
    rollout_mod ..> metrics_mod
    bootstrap_mod ..> CI : returns
    calibration_mod ..> CalibrationReport : returns
    walkforward_mod ..> WFSplit : returns

    live_entry_mod ..> RLStrategyAdapter
    live_entry_mod ..> RLRobot
    RLStrategyAdapter ..> build_mod : _rebuild_features
    RLStrategyAdapter ..> MT5Strategy : as_strategy
    RLRobot ..> MT5Strategy : signal
    RLRobot *-- Trader
    RLRobot *-- RiskManager
    RLRobot ..> TradingData : bars
```

Packages map 1:1 to directories; the only cross-tree boundary is the
`live` namespace: `RLStrategyAdapter` (in `quant_rl/live/`) bridges the
model into `mt5_trading/`, which knows nothing about RL training. The
`_strategy_from_cfg` helper lives in `quant_rl/train/train_rl.py` and
builds the `envs` strategy/reward pair shown above.

## Lifecycle of one cycle

### Data and cache

`run_pipeline` reads only the M1 files named in `data.m1_files` and the
tick files in `data.tick_files`. Higher timeframes are resampled from
M1; sessions are evaluated in `session.tz` (`Etc/GMT-3`, NY-open proxy
`16:30`–`23:00`, with a DST flag caveat). Feature frames are cached as
parquet under a version key so identical configurations skip
recomputation; the key changes whenever the feature schema changes.

### Feature chains and strategy state

`build_features` always builds the base technical and PO3/FVG columns.
When `features.include_strategy_state` is on, it additionally runs the
causal detectors and appends their columns:

- `detect_liquidity_sweeps` — pushes through prior swing levels with
  reclaim confirmation (`swing_period`, `require_reclaim`,
  `max_age_bars`).
- `detect_bos` — break-of-structure flags against confirmed swings.
- `build_po3_state` — manipulation/distribution phase state for the
  PO3 chain (`require_manipulation_sweep`,
  `require_distribution_confirmation`).
- `build_ifvg_zone_features` — inversion-FVG zones
  (`max_age_bars`, `require_price_retest`).

Overlay columns are consumed two ways: strategy classes read them per
bar (`required_features` plus execution-side `raw_columns`), and the
alignment rewards read their boolean flags at entry transitions.

### Environment step

One `TradingEnv` step:

1. **Observation** — `seq`: the normalised 60-bar feature window
   (`obs_window: 60`), plus `account` scalars, plus an optional frozen
   `vae_z` latent.
2. **Action** — legacy mode: `Discrete(20)` or 1-D `Box(-1, 1)`
   proportional sizing. Strategy mode (`env.strategy_actions=true`): one
   4-D `Box` — `action[0]` direction/entry intensity (`|x| < 0.25`
   holds), `action[1]` risk selector mapped into `risk_frac_range`,
   `action[2]` RR selector mapped into `rr_range`, `action[3]`
   TP-target selector over `tp.allowed_targets`.
3. **Strategy gate** — the strategy's `validate_entry` decides whether
   the chosen entry is valid at this bar. `entry.enforce_gate` defaults
   to `false`: the gate is then diagnostic (invalid entries are logged
   and scored, not blocked); `true` makes it a hard veto. Which mode a
   checkpoint was trained under must be recorded and reused live.
4. **Structural SL/TP** — `sl_reference` feeds
   `compute_sl_tp_from_structure` (exact or buffered stop), and
   `target_candidates` feeds `resolve_tp_target`; invalid targets fall
   back per `invalid_target_policy` (`rr_fallback`).
5. **Reward** — `CompositeReward` combines `DSRReward` (economic),
   `SweepConfirmationReward` (optional) and the strategy alignment
   reward (optional, entry transitions only, weight
   `strategy.reward.strategy_weight`). Each alignment component
   receives only the keywords in its `required_inputs`.
6. **Guardrails** — `FTMOGuardrails` enforces the `ftmo:` dollar limits
   (daily loss, max loss, per-trade risk) inside training/evaluation
   episodes.

### Training, evaluation and walk-forward

`train_rl.py` merges `quant_rl/config/default.yaml` with the variant
YAML selected from `_STRATEGY_CONFIGS`, builds the environment via
`make_env` and `_strategy_from_cfg`, the agent via `build_agent`, and
trains. `BestCheckpointEvalCallback` checkpoints during training;
`quant_rl/eval/rollout.py` (`evaluate_model`) replays checkpoints on the
locked OOS window, and `quant_rl/eval/` writes run artifacts
(metrics, trades, bootstrap CIs, calibration, walk-forward reports) to
`outputs/`. The cache key distinction (`v5_idea1` vs `v4_po3causal`)
keeps overlay and baseline feature caches separate.

### Live: paper first, then promote

`live_trading_rl.py` loads a trained checkpoint into
`RLStrategyAdapter`, which converts each MT5 M1 bar into the same
observation the environment produced during training. `RLRobot.trade()`
derives the signal, sizes it via `RiskManager` (percent-of-balance under
`live_risk_overrides:`), and either places orders through `Trader` or
only logs them. Promotion from paper to live follows
`doc/DEPLOYMENT.md`; a live adapter must be paper-first, and an SMT
secondary symbol is wired only when the checkpoint was trained with it.

## Budgets and constraints

1. **The OOS split is locked.** `data.split.train_end: "2025-12-31"` /
   `test_start: "2026-01-01"` in `quant_rl/config/default.yaml`. Idea 1/2
   must not redefine the split, the cost model, or the baseline action
   space.
2. **FTMO dollar limits are hard kill-switches in training/eval.**
   `daily_loss_limit` (5000), `max_loss_limit` (10000) and
   `risk_per_trade_limit` (1000) are enforced by `FTMOGuardrails`;
   `profit_target` is reported only.
3. **Live sizing is percent-of-balance.** `live_risk_overrides:` —
   `risk_per_symbol` 1%, aggregate open risk `max_total_risk` 2%,
   `default_lot_size` fallback, `stop_loss_pips` for sizing. Keep the
   ratio aligned with `ftmo:` for the configured account size.
4. **Observation window is 60 bars** (`obs_window: 60`); episodes cap at
   `max_episode_steps: 1000`. Checkpoints and live adapters must agree
   with these.
5. **The overlay stays an overlay.** With `strategy_actions=false`
   everything is bit-for-bit baseline ([Overlay
   invariant](#overlay-invariant)); variant configs activate Idea 1/2
   only through their own YAMLs.
6. **Alignment rewards stay secondary.** Entry-transition only, and
   `strategy_weight` (0.1 in both shipped variants) small relative to
   the economic reward.

## Causal-only rule

Every feature detector uses only information available at decision time
`t`:

- Sweep and BOS detections reference swing levels *confirmed* before the
  decision bar; the level used at `t` is shifted by one bar so the
  confirming close of the swing is never inside the same bar's decision.
- The PO3 HTF/LTF mapping is shifted one period in the v4 feature
  generation: an M15 value mapped to an M1 decision bar comes from a
  completed M15 bar only.

Adding future bars must never change historical feature values. This is
the same discipline as the NRT-only rule it mirrors: a feature that
repaints is a bug, and the causal tests under `tests/test_features/`
exist to catch it.

Live mirrors training: the SMT divergence secondary symbol
(`data.secondary`, `US500.cash`) participates only if the checkpoint was
trained with it, and the `enforce_gate` mode recorded at training time
is the mode used live.

## Adding a strategy

A strategy is a semantics object, a config file, and a registry entry.
Implement `TradingStrategy` (`quant_rl/envs/strategies/base.py`):

```python
from quant_rl.envs.strategies.base import TradingStrategy


class MyStrategy(TradingStrategy):
    name = "my_strategy"

    #: Feature columns that must exist in the feature matrix.
    required_features = ("my_sweep_flag", "my_zone_level")
    #: Raw price columns kept for execution, excluded from the
    #: normalised sequence observation.
    raw_columns = ("my_zone_level_raw",)

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Whether an entry in ``direction`` is allowed at this bar."""

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Structural SL reference price, or None if unavailable."""

    def target_candidates(self, *, direction: int, row: pd.Series) -> dict[str, float]:
        """Named TP candidate levels for the TP-target resolver."""
```

Optionally add an entry-event alignment reward next to
`PO3Reward` / `DistributionReward`, declaring the keywords it consumes:

```python
class MyReward:
    required_inputs = ("position_changed", "direction", "my_sweep_flag")

    def reset(self) -> None: ...

    def __call__(self, *, position_changed: bool, direction: int,
                 my_sweep_flag: bool) -> float:
        if not position_changed or direction == 0:
            return 0.0
        ...
```

`CompositeReward` passes only the declared `required_inputs` to the
component, so a reward can never silently read the wrong column.

Then:

1. Add a YAML under `config/` merged over `default.yaml`, flipping
   `features.include_strategy_state: true` and
   `env.strategy_actions: true` and declaring the `strategy:` block
   (name, gates, risk ranges, TP targets, reward weights). Copy the
   shape of `config/idea1_po3_ifvg.yaml`.
2. Register the name in `_STRATEGY_CONFIGS` in
   `quant_rl/train/train_rl.py` and handle it in `_strategy_from_cfg`.
3. Add contract tests mirroring
   `tests/test_envs/test_baseline_regression.py` (overlay safety) and
   `tests/test_envs/test_strategy_env.py` (env wiring).
4. Run `scripts/verify_strategy_features.py` to confirm every
   `required_features` / `raw_columns` name exists in the produced
   feature matrix.

**Must guarantee**

- **Causal features.** Every column in `required_features` obeys the
  [Causal-only rule](#causal-only-rule).
- **Required columns present.** A missing column fails loudly at env
  construction, not silently mid-episode; keep
  `scripts/verify_strategy_features.py` green.
- **Overlay-safe.** With `strategy_actions=false` the strategy is never
  constructed; baseline behaviour is untouched.
- **Repeatable across bars.** `validate_entry` / `sl_reference` /
  `target_candidates` answer from the feature row of the decision bar
  only — no hidden per-episode state that changes past answers.
- **Gate mode documented.** State explicitly whether `entry.enforce_gate`
  is diagnostic (default) or a hard veto for this strategy, and keep
  training and live modes identical.
- **Paper first.** Any live adapter ships paper-mode only until it
  passes the promotion criteria in `doc/DEPLOYMENT.md`.

## References

- [`README.md`](../README.md) — installation, quick start, project
  structure.
- [`RUNNING_COMMANDS.md`](../doc/RUNNING_COMMANDS.md) — command reference.
- [`DEPLOYMENT.md`](../doc/DEPLOYMENT.md) — paper-to-live promotion
  protocol.
- [`quant_rl/config/default.yaml`](../quant_rl/config/default.yaml) —
  base configuration (split dates, FTMO limits, live risk overrides).
- Strategy semantics: [`quant_rl/envs/strategies/`](../quant_rl/envs/strategies/),
  [`config/idea1_po3_ifvg.yaml`](../config/idea1_po3_ifvg.yaml),
  [`config/idea2_distribution.yaml`](../config/idea2_distribution.yaml).
- Feature diagnostics:
  [`scripts/verify_strategy_features.py`](../scripts/verify_strategy_features.py).

Contract tests:

- [`tests/test_envs/test_baseline_regression.py`](../tests/test_envs/test_baseline_regression.py)
  — overlay invariant.
- [`tests/test_envs/test_strategy_env.py`](../tests/test_envs/test_strategy_env.py)
  — strategy-mode env wiring.
- [`tests/test_envs/test_po3_reward.py`](../tests/test_envs/test_po3_reward.py)
  and
  [`tests/test_envs/test_distribution_reward.py`](../tests/test_envs/test_distribution_reward.py)
  — alignment rewards.
- [`tests/test_features/test_liquidity.py`](../tests/test_features/test_liquidity.py)
  and
  [`tests/test_features/test_po3_state.py`](../tests/test_features/test_po3_state.py)
  — causal detectors.
- [`tests/test_integration/test_strategy_integration.py`](../tests/test_integration/test_strategy_integration.py)
  — features-to-env integration.
- [`tests/test_risk.py`](../tests/test_risk.py) — structural risk
  helpers.
- [`tests/test_backtest/test_engine_cross_validation.py`](../tests/test_backtest/test_engine_cross_validation.py)
  — engine validation.
- [`tests/test_live/`](../tests/test_live/) — live bridge and entry
  point.






