# Live Trading Domain

## Primary Files

| File | Role |
|------|------|
| `live/runtime/strategy.py` | `RLStrategyAdapter` — trained RL policy → MT5 bridge |
| `integrations/mt5/robot/rl_robot.py` | RL robot for MT5 |
| `integrations/mt5/robot/cross_over_robot.py` | Cross-over strategy robot |
| `integrations/mt5/robot/multi_symbol_robot.py` | Multi-symbol robot |
| `integrations/mt5/adapters/data.py` | Data adapter for MT5 |
| `integrations/mt5/adapters/strategy.py` | Strategy adapter for MT5 |
| `integrations/mt5/adapters/trader.py` | Trader adapter for MT5 |
| `integrations/mt5/domain/trader.py` | Domain trader logic |
| `integrations/mt5/domain/risk_manager.py` | Risk manager for live trading |
| `integrations/mt5/domain/signal.py` | Signal data container |
| `integrations/mt5/domain/mt5_connection.py` | MT5 connection management |
| `integrations/mt5/domain/multi_symbol_manager.py` | Multi-symbol management |
| `integrations/mt5/domain/volatility_analyzer.py` | Volatility analysis |
| `integrations/mt5/domain/strategies/*.py` | Rule-based MT5 strategies |
| `integrations/mt5/domain/data_sources/mt5_data.py` | MT5 data source |
| `integrations/mt5/logging_config.py` | Logging configuration |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `RLStrategyAdapter` | `rl_strategy.py` | Bridge between the trained RL model and MT5 |
| `RLRobot` | `rl_robot.py` | RL-based MT5 robot |
| `RiskManager` | `risk_manager.py` | Live risk management (`live_risk_overrides`) |
| `Signal` | `signal.py` | Trading signal container |

## Direct Dependencies

- `models/factory/agent.py` (model loading)
- `features/pipeline/builder.py` (feature rebuild for live observations)
- `simulation/environment/` (action semantics — discrete vs continuous vs strategy mode)

## Consumers (Callers)

- `live_trading_rl.py` (live RL trading entry point)
- `live_trading.py` (live rule-based trading entry point)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_live/test_rl_robot.py` | RL robot |
| `tests/test_live/test_rl_bridge.py` | RLStrategyAdapter bridge |
| `tests/test_live/test_live_entrypoint.py` | Live entry point |
| `tests/test_live/test_rule_based_strategies.py` | Rule-based strategies |

## Do NOT Read For Live Tasks

- `features/technical/indicators.py` (feature implementation detail)
- `simulation/backtest/` (backtest — separate domain)
- `training/` (training — separate domain)
- `evaluation/` (evaluation — separate domain)

## Critical Invariants

1. **Live risk ≈ training risk** — `live_risk_overrides` must stay aligned with `ftmo` dollar limits
2. **Silent divergence is the failure mode** — training under $1000 cap, live under $5000 is a bug
3. **Action semantics must match** — discrete (PPO), continuous (SAC), or
   flag-derived strategy-overlay `Box` coordinates from
   `simulation/contracts/actions.py`; checkpoint schema metadata must match the live
   environment configuration
4. **PAPER_TRADING defaults to true** — no orders placed unless explicitly disabled
5. **Model loading** — must load with same encoder architecture used in training
