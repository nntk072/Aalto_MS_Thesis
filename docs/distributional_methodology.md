# Distributional Performance Methodology

This note documents the distribution-focused evaluation layer added in
PLAN 9 so it can be cited directly in the thesis.

## Observation semantics

| Quantity | One observation means |
|---|---|
| Position PnL | Net realised PnL of one completed position, summing its partial and final fills once |
| Holding time | Elapsed seconds from entry to final exit of one completed position |
| Sweep delay | Seconds between the first close beyond a liquidity level and the agent's entry |

Partial fills are execution events, not independent completed trades. A
position that has realized a partial exit but remains open is excluded from
completed-position trade statistics. Fill-level counts and realized-PnL
reconciliation remain separate diagnostics.

## Target reference and mode lifecycle

Reference-selected targets retain their TP1/TP2/TP3 slot identities, allocated
quantities, and source metadata. A valid set may contain one or two targets;
disabled slots stay disabled, and duplicate prices are not repaired with
artificial offsets. EMA exits remain a separate exit mechanism.

The current versioned legacy mode contract is:

- `fixed` leaves all targets at their entry-time references.
- `breakeven` preserves the legacy TP pullback: once its favorable-excursion
  threshold is reached, the nearest remaining target may move once toward
  `entry ± alpha × initial risk`. It does not move the stop. A price already
  crossed by the current bar is not installed as a retroactive target.
- `trailing` may extend only the nearest remaining, unhit slot to a confirmed
  favorable swing (high for long, low for short), rounded outward to the
  instrument tick. The update is accepted only if it remains beyond the
  current close and does not overtake a later active slot.

Exit checks run before position guards. A target update is logged with its
source, old and new prices, decision timestamp, and earliest effective next
bar. Hit slots and their fills are immutable. Mode changes never alter the
position size allocation, initial risk, or previously realized quantity.

## Statistics reported

For each observation family we report: count, mean, median, sample std,
min/max, empirical quantiles P01/P05/P25/P75/P95/P99, sample skewness (g1)
and excess kurtosis (g2).

## Tail risk

- **VaR5** = empirical 5th percentile of trade PnL (sign convention kept:
  negative values denote losses).
- **CVaR5** = mean of all observations at or below VaR5, i.e. the average
  outcome *inside* the worst 5% tail.

Both are purely empirical; no normality assumption is made.

## Conditional regimes

PnL distributions are computed per group: overall, long, short, London
levels and Asian levels, using FIFO pairing of open and close entries in
the environment trade log.

## Robustness conventions

- Non-finite observations are excluded from statistics and their count is
  reported separately - never silently mixed in.
- Empty or degenerate inputs yield count=0 with NaN fields rather than
  raising, so batch evaluation cannot be derailed by one bad episode. A run
  with zero completed positions is labeled `INACTIVE`; it is not evidence of
  successful trading performance.
- Existing point-estimate metrics are unchanged by this layer; the
  distribution output is additive under the `pnl_distribution` key of the
  evaluation report.

## Interpretation guardrails

Superiority claims must reference the full distribution, e.g. higher
*median* profitability with a narrower downside tail, or an edge that
persists across the central 90% of outcomes instead of being driven by a
few extreme winners.

Also apply **TI-7**: do not claim superiority on default transaction costs
alone — cite the cost-stress grid (`scripts/eval/test_oos.py --cost-multipliers`).
When many variants share the locked OOS, state the trial count; full DSR/PBO
is out of scope for Thesis_plan (see Integrated_startup_plan plan 09).
