# Environment Domain

## Primary Files

| File | Role |
|------|------|
| `quant_rl/envs/trading_env.py` | Gymnasium trading environment |
| `quant_rl/envs/reward.py` | Differential Sharpe Ratio (DSR) reward |
| `quant_rl/envs/sweep_reward.py` | Sweep confirmation + CompositeReward |
| `quant_rl/envs/po3_reward.py` | PO3 strategy alignment reward |
| `quant_rl/envs/distribution_reward.py` | Distribution strategy alignment reward |
| `quant_rl/envs/strategies/base.py` | TradingStrategy base class |
| `quant_rl/envs/strategies/baseline.py` | Baseline (no-op) strategy |
| `quant_rl/envs/strategies/po3_ifvg.py` | PO3 + IFVG strategy |
| `quant_rl/envs/strategies/distribution.py` | Distribution strategy |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `TradingEnv` | `trading_env.py` | Main Gymnasium environment class |
| `TradingEnv.step()` | `trading_env.py` | Execute one step (action → reward) |
| `DSRReward` | `reward.py` | Differential Sharpe Ratio reward |
| `SweepConfirmationReward` | `sweep_reward.py` | Sweep-based reward |
| `CompositeReward` | `sweep_reward.py` | Weighted composite of rewards |
| `PO3Reward` | `po3_reward.py` | PO3 alignment reward (entry transitions only) |
| `DistributionReward` | `distribution_reward.py` | Distribution alignment reward |
| `TradingStrategy` | `strategies/base.py` | Strategy semantics base class |
| `BaselineStrategy` | `strategies/baseline.py` | No-op baseline (Idea 3) |
| `PO3IFVGStrategy` | `strategies/po3_ifvg.py` | PO3 + IFVG entry logic |
| `DistributionStrategy` | `strategies/distribution.py` | Distribution entry logic |

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
| `tests/test_envs/test_baseline_regression.py` | Overlay invariant (baseline unchanged) |
| `tests/test_envs/test_continuous_actions.py` | Continuous action space |
| `tests/test_envs/test_ny_steps_eod.py` | NY session + EOD logic |
| `tests/test_envs/test_strategy_env.py` | Strategy-mode env wiring |
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
4. **Strategy mode**: 4-D `Box [direction, risk, rr, tp]` when `env.strategy_actions=true`
5. **Overlay invariant**: with `strategy_actions=false`, baseline behaviour is bit-for-bit unchanged
6. **Block overnight**: `block_overnight=True` closes positions at last NY bar
