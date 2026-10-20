# Backtest Domain

## Primary Files

| File | Role |
|------|------|
| `simulation/backtest/engine.py` | Event-driven backtest engine |
| `trading/execution/broker.py` | Broker: position open/close, fill logic |
| `trading/account/state.py` | Account state tracking |
| `trading/execution/costs.py` | Cost model (spread, slippage, commission) |
| `trading/risk/guardrails.py` | FTMO guardrails (daily loss, max drawdown) |
| `trading/risk/sizing.py` | Risk calculations (SL/TP, position sizing) |
| `simulation/backtest/cross_validation/` | Engine cross-validation against backtrader |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_backtest()` | `simulation/backtest/engine.py` | Main backtest entry point |
| `Broker` | `trading/execution/broker.py` | Position management, fill execution |
| `Position` | `trading/execution/broker.py` | Position data container |
| `AccountState` | `trading/account/state.py` | Account equity, balance, margin |
| `CostModel` | `trading/execution/costs.py` | Spread, slippage, commission model |
| `FTMOGuardrails` | `trading/risk/guardrails.py` | FTMO rule enforcement |
| `compute_sl_tp_from_structure()` | `trading/risk/sizing.py` | Structure-aware SL/TP |
| `resolve_tp_target()` | `trading/risk/sizing.py` | TP target resolution |

## Direct Dependencies

- `market_data/loading/ticks.py` (optional TickBook for fill pricing)

## Consumers (Callers)

- `simulation/environment/env.py` (uses engine internally)
- `training/runner/backtest.py` (CLI entry point)
- `evaluation/rollout/runner.py` (evaluation loop)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_backtest/test_engine_cross_validation.py` | Engine + cross-validation |
| `tests/test_cost_model.py` | Cost model calculations |
| `tests/test_guardrails.py` | FTMO guardrails |
| `tests/test_risk.py` | Risk calculations |

## Do NOT Read For Backtest Tasks

- `features/` (feature engineering — separate domain)
- `models/` (models — separate domain)
- `training/` (training — separate domain)
- `integrations/mt5/` (live trading — separate domain)

## Fill-Price Semantics

1. **Tick data** (priority): `TickBook.quote_at(fill_instant)` at next bar open
2. **Bar-spread fallback**: `CostModel.bar_quote(close)`
3. **Tick rejection**: If tick quote outside bar `[low, high]` range, reject for bar quote

## Critical Invariants

1. **MT5 spread is in broker points** — must convert via `point_size` (e.g., 0.01 for US100)
2. **Mark-to-market uses bar-close quote** — long @ bid, short @ ask
3. **Guardrail breach blocks new trading** — for the rest of the session (episodic=False)
