# Live Trading Domain

## Primary Files

| File | Role |
|------|------|
| `quant_rl/live/rl_strategy.py` | `RLStrategyAdapter` — quant_rl → MT5 bridge |
| `mt5_trading/robot/rl_robot.py` | RL robot for MT5 |
| `mt5_trading/robot/cross_over_robot.py` | Cross-over strategy robot |
| `mt5_trading/robot/multi_symbol_robot.py` | Multi-symbol robot |
| `mt5_trading/adapters/data.py` | Data adapter for MT5 |
| `mt5_trading/adapters/strategy.py` | Strategy adapter for MT5 |
| `mt5_trading/adapters/trader.py` | Trader adapter for MT5 |
| `mt5_trading/domain/trader.py` | Domain trader logic |
| `mt5_trading/domain/risk_manager.py` | Risk manager for live trading |
| `mt5_trading/domain/signal.py` | Signal data container |
| `mt5_trading/domain/mt5_connection.py` | MT5 connection management |
| `mt5_trading/domain/multi_symbol_manager.py` | Multi-symbol management |
| `mt5_trading/domain/volatility_analyzer.py` | Volatility analysis |
| `mt5_trading/domain/strategies/*.py` | Rule-based MT5 strategies |
| `mt5_trading/domain/data_sources/mt5_data.py` | MT5 data source |
| `mt5_trading/logging_config.py` | Logging configuration |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `RLStrategyAdapter` | `rl_strategy.py` | Bridge between quant_rl model and MT5 |
| `RLRobot` | `rl_robot.py` | RL-based MT5 robot |
| `RiskManager` | `risk_manager.py` | Live risk management (`live_risk_overrides`) |
| `Signal` | `signal.py` | Trading signal container |

## Direct Dependencies

- `quant_rl/models/agent.py` (model loading)
- `quant_rl/features/build.py` (feature rebuild for live observations)
- `quant_rl/envs/` (action semantics — discrete vs continuous vs strategy mode)

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

- `quant_rl/features/indicators.py` (feature implementation detail)
- `quant_rl/backtest/` (backtest — separate domain)
- `quant_rl/train/` (training — separate domain)
- `quant_rl/evaluation/` (evaluation — separate domain)

## Critical Invariants

1. **Live risk ≈ training risk** — `live_risk_overrides` must stay aligned with `ftmo` dollar limits
2. **Silent divergence is the failure mode** — training under $1000 cap, live under $5000 is a bug
3. **Action semantics must match** — discrete (PPO) vs continuous (SAC) vs strategy 4-D Box
4. **PAPER_TRADING defaults to true** — no orders placed unless explicitly disabled
5. **Model loading** — must load with same encoder architecture used in training
