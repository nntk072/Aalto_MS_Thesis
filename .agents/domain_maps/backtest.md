# Backtest Domain

## Primary Files

| File | Role |
|------|------|
| `quant_rl/backtest/engine.py` | Event-driven backtest engine |
| `quant_rl/backtest/broker.py` | Broker: position open/close, fill logic |
| `quant_rl/backtest/account.py` | Account state tracking |
| `quant_rl/backtest/costs.py` | Cost model (spread, slippage, commission) |
| `quant_rl/backtest/guardrails.py` | FTMO guardrails (daily loss, max drawdown) |
| `quant_rl/backtest/risk.py` | Risk calculations (SL/TP, position sizing) |
| `quant_rl/backtest/cross_validation/` | Engine cross-validation against backtrader |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_backtest()` | `engine.py` | Main backtest entry point |
| `Broker` | `broker.py` | Position management, fill execution |
| `Position` | `broker.py` | Position data container |
| `AccountState` | `account.py` | Account equity, balance, margin |
| `CostModel` | `costs.py` | Spread, slippage, commission model |
| `FTMOGuardrails` | `guardrails.py` | FTMO rule enforcement |
| `compute_sl_tp_from_structure()` | `risk.py` | Structure-aware SL/TP |
| `resolve_tp_target()` | `risk.py` | TP target resolution |

## Direct Dependencies

- `quant_rl/data/ticks.py` (optional TickBook for fill pricing)

## Consumers (Callers)

- `quant_rl/envs/trading_env.py` (uses engine internally)
- `quant_rl/train/run_backtest.py` (CLI entry point)
- `quant_rl/evaluation/runner.py` (evaluation loop)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_backtest/test_engine_cross_validation.py` | Engine + cross-validation |
| `tests/test_cost_model.py` | Cost model calculations |
| `tests/test_guardrails.py` | FTMO guardrails |
| `tests/test_risk.py` | Risk calculations |

## Do NOT Read For Backtest Tasks

- `quant_rl/features/` (feature engineering — separate domain)
- `quant_rl/models/` (models — separate domain)
- `quant_rl/train/` (training — separate domain)
- `mt5_trading/` (live trading — separate domain)

## Fill-Price Semantics

1. **Tick data** (priority): `TickBook.quote_at(fill_instant)` at next bar open
2. **Bar-spread fallback**: `CostModel.bar_quote(close)`
3. **Tick rejection**: If tick quote outside bar `[low, high]` range, reject for bar quote

## Critical Invariants

1. **MT5 spread is in broker points** — must convert via `point_size` (e.g., 0.01 for US100)
2. **Mark-to-market uses bar-close quote** — long @ bid, short @ ask
3. **Guardrail breach blocks new trading** — for the rest of the session (episodic=False)
