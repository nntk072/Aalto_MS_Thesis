# Threats to Validity

Examiner-facing note for the MSc. Maps claim risks to mitigations on
`feature/thesis-plan-validity` (Thesis_plan T01–T04) or accepted limitations.

| Threat | Finding | Mitigation | Residual risk |
|--------|---------|------------|---------------|
| Short sample / few regimes | F1 | Accepted limitation; do not claim decade-scale robustness | High for regime generalisation |
| Broker TZ / DST session mislabel | F3 | T-02.3 diagnostic + documented assumption `Etc/GMT-3`; CT session-OHLC remains DST-aware | Medium if broker DST differs from config |
| Locked OOS reused for selection | F4 | T-03.2 validation protocol: WF/val for selection; locked OOS = final report only | Medium if protocol ignored |
| Overlapping WF test folds | F5 | T-03.1 non-overlapping expanding WF + `label_horizon` purge | Low after fix; legacy API deprecated |
| Stale feature cache | F6 | T-02.1 content-hash cache key | Low if callers use helper path |
| Fit-scope leakage under WF | F7 | T-02.4 `train_mask` on locked split; poison test | Medium: WF folds still reuse IS-scoped features (not rebuilt per fold) |
| Incomplete causality coverage | F8 | T-02.2 assembled-matrix truncation harness (+ existing block tests) | Low–medium for exotic flags |
| Optimistic costs | F10 | T-03.3 cost grid; TI-7 — no superiority on default costs alone | Medium for live expectancy |
| Multi-simulator divergence | F9 | Eval pack names engine per table | Medium for live parity claims |

## Declared assumptions

1. Broker server time for session/PD features is **`Etc/GMT-3`** (UTC+3 fixed).
2. Primary thesis tables use **`TradingEnv`** episode evaluation unless a table caption says `run_backtest`.
3. Locked OOS dates in `config/default.yaml` / `experiments.yaml` are a **final holdout**, not a free hyperparameter tuner.
4. Cost-stressed results (see `scripts/test_oos.py` and eval pack) bound default-cost claims.

## Related docs

- [thesis_audit_2026.md](../architecture/thesis_audit_2026.md)
- [validation_protocol.md](validation_protocol.md)
- [eval_pack.md](eval_pack.md)
- [distributional_methodology.md](../distributional_methodology.md)
