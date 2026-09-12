# Environment Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/envs/trading_env.py` | Gymnasium trading environment | 1425 |
| `quant_rl/envs/reward.py` | Differential Sharpe Ratio (DSR) reward | 75 |
| `quant_rl/envs/sweep_reward.py` | Sweep confirmation reward | ~80 |
| `quant_rl/envs/po3_reward.py` | PO3 strategy reward | ~60 |
| `quant_rl/envs/distribution_reward.py` | Distribution strategy reward | ~60 |
| `quant_rl/envs/strategies/base.py` | Trading strategy base class | ~50 |
| `quant_rl/envs/strategies/baseline.py` | Baseline (no-op) strategy | ~40 |
| `quant_rl/envs/strategies/po3_ifvg.py` | PO3 + IFVG strategy | ~100 |
| `quant_rl/envs/strategies/distribution.py` | Distribution strategy | ~80 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `TradingEnv` | `trading_env.py` | Main Gymnasium environment class |
| `TradingEnv.__init__()` | `trading_env.py` | Initialize with bars, features, cost model, etc. |
| `TradingEnv.step()` | `trading_env.py` | Execute one step (action → reward) |
| `TradingEnv.reset()` | `trading_env.py` | Reset environment state |
| `DSRReward` | `reward.py` | Differential Sharpe Ratio reward |
| `DSRReward.__call__()` | `reward.py` | Compute DSR for one step |
| `SweepConfirmationReward` | `sweep_reward.py` | Sweep-based reward |
| `CompositeReward` | `sweep_reward.py` | Weighted composite of rewards |
| `PO3Reward` | `po3_reward.py` | PO3 strategy reward |
| `DistributionReward` | `distribution_reward.py` | Distribution strategy reward |
| `TradingStrategy` | `strategies/base.py` | Strategy base class |
| `BaselineStrategy` | `strategies/baseline.py` | No-op baseline strategy |
| `PO3IFVGStrategy` | `strategies/po3_ifvg.py` | PO3 + IFVG entry/exit logic |
| `DistributionStrategy` | `strategies/distribution.py` | Distribution entry/exit logic |

## Direct Dependencies

- `quant_rl/backtest/` (engine, broker, costs, guardrails, risk)
- `quant_rl/features/build.py` (feature matrix, `MTF_RAW_SUFFIXES`)
- `quant_rl/data/session.py` (`ny_session_mask`)
- `quant_rl/models/vae.py` (optional VAE feature extractor)

## Consumers (Callers)

- `quant_rl/train/train_rl.py` (creates env, trains model)
- `quant_rl/eval/rollout.py` (evaluates model through env)
- `quant_rl/evaluation/runner.py` (runs episodes)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_envs/test_trading_env_decomposition.py` | Env structure |
| `tests/test_envs/test_trading_env_golden_traces.py` | Golden trace regression |
| `tests/test_envs/test_account_normalization.py` | Account state normalization |
| `tests/test_envs/test_baseline_regression.py` | Baseline strategy regression |
| `tests/test_envs/test_continuous_actions.py` | Continuous action space |
| `tests/test_envs/test_ny_steps_eod.py` | NY session + EOD logic |
| `tests/test_envs/test_strategy_env.py` | Strategy integration |
| `tests/test_envs/test_sweep_reward.py` | Sweep reward |
| `tests/test_envs/test_po3_reward.py` | PO3 reward |
| `tests/test_envs/test_distribution_reward.py` | Distribution reward |
| `tests/test_reward.py` | DSR reward |

## Do NOT Read For Env Tasks

- `quant_rl/features/indicators.py` (indicator implementation detail)
- `quant_rl/models/encoder.py` (encoder detail — only need `build_agent`)
- `quant_rl/train/train_rl.py` (training orchestration)
- `mt5_trading/` (live trading — separate domain)

## Critical Invariants

1. **Observation space**: `obs["seq"]` is `[T, F]` float32, `obs["account"]` is `[5]` float32
2. **Discrete actions**: `{hold=0, enter_long=1-9, enter_short=10-18, exit=19}`
3. **Continuous actions**: `Box(-1, 1)` for proportional position sizing
4. **Risk params**: `risk_frac_range` and `rr_ratio_range` define the 3x3 grid
5. **Block overnight**: `block_overnight=True` closes positions at session end