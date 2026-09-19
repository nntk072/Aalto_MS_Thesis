# Distributional Performance Methodology

This note documents the distribution-focused evaluation layer added in
PLAN 9 so it can be cited directly in the thesis.

## Observation semantics

| Quantity | One observation means |
|---|---|
| Trade PnL | The realised PnL of one completed trade |
| Holding time | Elapsed seconds from entry to exit of one completed trade |
| Sweep delay | Seconds between the first close beyond a liquidity level and the agent's entry |

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
  raising, so batch evaluation cannot be derailed by one bad episode.
- Existing point-estimate metrics are unchanged by this layer; the
  distribution output is additive under the `pnl_distribution` key of the
  evaluation report.

## Interpretation guardrails

Superiority claims must reference the full distribution, e.g. higher
*median* profitability with a narrower downside tail, or an edge that
persists across the central 90% of outcomes instead of being driven by a
few extreme winners.
