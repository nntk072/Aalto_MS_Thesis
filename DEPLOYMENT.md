# DEPLOYMENT.md — Paper → Live Promotion Protocol

This document defines, **before any real capital is at risk**, how a trained
checkpoint goes from `outputs/` to paper trading, and from paper to live.
The goal is that "when do we go live" is a written criterion, not a judgment
call made under the pressure of "the backtest looked good."

---

## 1. Entrypoint and safeguards

- **Entrypoint:** `live_trading_rl.py` (repo root).
- **Default is paper:** `PAPER_TRADING` defaults to `true` — signals and
  intended orders are logged, no orders are placed. Real orders require an
  explicit `PAPER_TRADING=false`.
- **Checkpoint selection:** `RL_MODEL_PATH=outputs/<run>/model/ppo_final`
  (or `--model`). The run's saved `config.yaml` is loaded automatically for
  train/live feature parity; SMT/secondary-symbol data is wired iff the run
  was trained with it.
- **Live risk sizing** comes from `live_risk_overrides:` in
  `quant_rl/config/default.yaml`. These are intentionally a
  percentage-of-balance model (MT5 sizing), aligned with the `ftmo:` dollar
  kill-switches enforced by `FTMOGuardrails` during training/eval
  (`risk_per_symbol` 1% ≈ `risk_per_trade_limit` $1000 on a $100k account).
  If you change one, keep the other aligned — silent divergence between the
  training risk budget and the live one is a behavior change, not a detail.

## 2. Rule-based baseline (`live_trading.py`)

- **Entrypoint:** `live_trading.py` (repo root).
- **Default is paper:** `PAPER_TRADING` defaults to `true` — same guardrail
  semantics as `live_trading_rl.py`. Real orders require an explicit
  `PAPER_TRADING=false`.
- **MT5 terminal:** Uses `MT5_TERMINAL_PATH` environment variable (defaults to
  standard MT5 install path).
- **Strategy selection:** `STRATEGY_TYPE` env var: `crossover`, `smc`, 
  `trend_breakout`, or `combined` (default).
- **Symbol selection:** Reads `config/symbols_config.yaml` and uses
  `VolatilityAnalyzer` to pick the top volatile symbols (up to `MAX_SYMBOLS`).

**Promotion criteria (to go live):** The rule-based baseline does NOT require the
full Sharpe-parity protocol because there is no learned model to promote. Instead,
the gating criteria are:
- Minimum paper trial length: ≥ 10 trading days of continuous paper execution
- Trade count: ≥ 20 closed paper trades across all selected symbols
- Guardrails: Zero breach of `quant_rl/backtest/guardrails.py` logic (daily loss
  limit, max drawdown limit, risk per trade limit) during paper trading
- Ops: No data-feed gaps > 5 minutes during the trial

Fail any one → investigate and fix, then restart the trial clock.

## 4. Paper-trading trial period (pass/fail decided now)

Before flipping `PAPER_TRADING=false`, a run must satisfy **all** of:

| Criterion | Threshold |
|---|---|
| Trial length | ≥ 20 trading days of continuous paper execution |
| Trade count | ≥ 30 closed paper trades on the trial symbol |
| Sharpe parity | Paper Sharpe within 0.5 of the backtested OOS Sharpe for the same checkpoint |
| Guardrails | Zero daily-loss / max-drawdown kill-switch breaches in paper that did not occur in the OOS backtest |
| Ops | No data-feed gaps > 5 minutes during the trial (check logs); no unresolved exceptions |

Fail any one → fix the cause, redeploy, restart the trial clock. Do not
average across failed trials.

## 5. Model versioning / promotion

The checkpoint that `live_trading_rl.py` loads is **explicit, not tribal
knowledge**:

1. A checkpoint is promoted by copying it into `models/production/`:
   ```bash
   mkdir -p models/production
   cp -r outputs/<run>/model models/production/<run>_ppo_final
   cp outputs/<run>/config.yaml models/production/<run>_config.yaml
   echo "promoted <run> on <date>; trial: <dates>; report: <path>" \
       >> models/production/PROMOTIONS.log
   ```
2. `RL_MODEL_PATH` in the deployment environment (`.env`) points at
   `models/production/<run>_ppo_final` — never at a mutable `outputs/`
   directory that future training runs write into.
3. Rollback = repoint `RL_MODEL_PATH` at the previous production copy and
   restart. The `PROMOTIONS.log` line is the audit trail.
4. Never overwrite an existing `models/production/<run>_*` directory;
   promote a retrained model under a new run id.

## 6. Going live — final checklist

- [ ] Section 2 criteria all pass for the exact promoted checkpoint
- [ ] `PAPER_TRADING=false` set deliberately, in the deployment env only
- [ ] Live account balance matches the account size assumed by
      `live_risk_overrides` (defaults assume $100k)
- [ ] Kill-switch limits re-checked against `ftmo:` block
- [ ] First live session supervised end-to-end, then reviewed before unattended runs
