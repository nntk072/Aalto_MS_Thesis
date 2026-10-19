# Thesis validity limits

The active data split uses all available observations through 2026-10-16:
training from 2024-12-30 through 2025-12-31 and testing from 2026-01-01.
The 2026 test period
has previously been inspected, so it is exposed development evidence and must
not be described as a pristine holdout.

The source is user-attested FTMO Demo MT5 data. The exact demo server and
liquidity provider are unknown, and no export or license record was supplied.
Internal model training is user-authorized; permission to redistribute the raw
CSV data has not been established. Report these provenance limits and do not
claim the raw data can be redistributed.

Short-session dates and all their bars remain in both partitions. The scheduled
close is supplied as a date-level input before the first observation for that
date. The schedule is an experimental input and should be named in the methods;
the model is not expected to infer it from the day's price path.

Tick-based fills use a seeded random arrival latency in the configured
0–30 millisecond interval and the first eligible bid/ask tick after arrival.
These fills model the declared execution assumption; they do not establish
real-world broker latency, queue position, market impact, or profitability.

The M1 sequence is the policy timestep. M3 and higher bars are contextual
inputs and may enter only after their source bars have closed. Report all
selected feature and execution assumptions with each final result.
