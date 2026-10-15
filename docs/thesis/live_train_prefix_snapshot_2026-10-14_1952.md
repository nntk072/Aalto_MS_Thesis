# Live training-prefix monitor snapshot

Captured: 2026-10-14T19:57:18+00:00  
Campaign: `ladder18_seed42_20m_policy_migration_nenv64_20261014` · Slurm job `20963043` · state `RUNNING|3:53:11|6:00:00|gpuarm[1-2]`

> **Interim diagnostic only.** Each row is a deterministic 20,000-bar training-prefix monitor. It is not a full-year training result, final train evaluation, test evaluation, strategy ranking, or confirmatory evidence. The campaign data role is selection-acknowledged.

All 18 variants produced monitors. **9** had no trades; observed trade counts were 0–32 (median 0). PnL ranged from **$-628.71** to **$521.82** (median $0.00, including zero-trade rows). **7** monitors carried action, stop, or exit-collapse warnings. These are in-sample diagnostics, not full-period performance.

| Variant | Monitor step | Trades | PnL (USD) | Max DD | Decision warnings |
| --- | ---: | ---: | ---: | ---: | --- |
| `ladder_a0_flat__seed42` | 13,005,312 | 8 | $-388.66 | 0.48% | — |
| `ladder_a0_flat_dist__seed42` | 9,907,584 | 32 | $-628.71 | 2.57% | — |
| `ladder_a0_flat_po3__seed42` | 10,104,192 | 1 | $-142.79 | 0.31% | SL COLLAPSE  top htf 100%; EXIT COLLAPSE  ema_21 100.0% |
| `ladder_a1_sl__seed42` | 10,202,496 | 0 | $0.00 | 0.00% | — |
| `ladder_a1_sl_dist__seed42` | 10,104,192 | 0 | $0.00 | 0.00% | — |
| `ladder_a1_sl_po3__seed42` | 10,104,192 | 13 | $-527.24 | 0.63% | ACTION COLLAPSE  stop_u std 0.030 |
| `ladder_a2_tp__seed42` | 10,300,800 | 1 | $-98.28 | 0.10% | SL COLLAPSE  top deviation 100%; EXIT COLLAPSE  ema_21 100.0% |
| `ladder_a2_tp_dist__seed42` | 10,202,496 | 9 | $201.42 | 0.30% | EXIT COLLAPSE  ema_21 100.0% |
| `ladder_a2_tp_po3__seed42` | 10,005,888 | 8 | $521.82 | 0.95% | EXIT COLLAPSE  structural 100.0% |
| `ladder_a3_multi_tp__seed42` | 10,300,800 | 2 | $-184.40 | 0.19% | SL COLLAPSE  top deviation 100%; EXIT COLLAPSE  ema_21 100.0% |
| `ladder_a3_multi_tp_dist__seed42` | 10,104,192 | 0 | $0.00 | 0.00% | — |
| `ladder_a3_multi_tp_po3__seed42` | 10,104,192 | 6 | $-303.08 | 0.30% | EXIT COLLAPSE  structural 100.0% |
| `ladder_a4a_entry_fsm_hidden__seed42` | 8,301,376 | 0 | $0.00 | 0.00% | — |
| `ladder_a4a_entry_fsm_hidden_dist__seed42` | 8,203,072 | 0 | $0.00 | 0.00% | — |
| `ladder_a4a_entry_fsm_hidden_po3__seed42` | 8,301,376 | 0 | $0.00 | 0.00% | — |
| `ladder_a4b_distribution__seed42` | 8,203,072 | 0 | $0.00 | 0.00% | — |
| `ladder_a4b_entry_fsm__seed42` | 8,506,176 | 0 | $0.00 | 0.00% | — |
| `ladder_a4b_po3_ifvg__seed42` | 8,301,376 | 0 | $0.00 | 0.00% | — |

Monitor timesteps differ by variant and may lag each worker’s latest PPO log. Raw JSONL copies, hashes, monitor records, and source-size stability flags are retained under campaign `reports/live_train_prefix_snapshot_20261014T1952Z/`. The machine-readable summary is `docs/thesis/live_train_prefix_snapshot_2026-10-14_1952.json`. Complete-split evaluations remain required.
