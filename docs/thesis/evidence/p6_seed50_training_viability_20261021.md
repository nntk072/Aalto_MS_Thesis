# P6 seed-50 training viability

**Gate:** PASS — minimum training viability only; exploratory, one seed per arm.
**Campaign:** `p6_entry_fsm_viability_seed50_train2025_20261021_scratch1`

**Fingerprint:** `3ca5c8e0cd3df51b29a5cdcb8f3d1c0f602a6c6721bf5aed3f6986a851b769b7`

**Slurm:** `21103722` completed with exit code `0:0`; training finished before evaluation and allocation release.

## Training-only results

| Arm | Completed steps | Training trades | Net PnL | Return | Max drawdown | Full-year survival | Cumulative DSR reward | Reward reconciliation | Lifecycle |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `ladder_a4a_entry_fsm_hidden` | 20,004,864 | 7 | $4,579.86 | 4.58% | 1.70% | yes | -5.6819 | error 0.0; unattributed 0.0 | 4577 candidates; 7 entries / 7 closes |
| `ladder_a4b_entry_fsm` | 20,004,864 | 7 | $6,349.80 | 6.35% | 2.76% | yes | -4.0182 | error 0.0; unattributed 0.0 | 4530 candidates; 7 entries / 7 closes |

Each arm started from scratch with seed 50 and a requested 20,000,000-step budget. The final run manifest and in-sample training evaluation record 20,004,864 completed training timesteps. Both full-year training replays completed with no breach and no risk-gate failure; each produced seven trades with fourteen tick-sourced fill-quote events and nonzero, fully reconciled DSR reward. Both cumulative DSR reward sums are negative, so this pass is limited to the predeclared minimum viability gate and does not imply reward improvement or comparative efficacy.

The report-only 2026 evaluation was generated automatically by the training runner. Its metrics are excluded from this gate and from P6 selection. That interval was already exposed and is not confirmatory evidence.

## Provenance

- Campaign manifest SHA-256: `df4ae5c024e0e44e5673cbd53b644b2ef3c6d5a523cee1c831a653540f62ca71`
- Source snapshot SHA-256: `4f96de9f173a59e04c5ae9049749c3401b8717e61faa687f251a06eced4ab159`; manifest SHA-256: `14dbfedb88fffc0a6550f46ca041ef605b914409ec0c178ff0074cb63d220c9b`
- Git SHA: `77fa156a260a383f20d2ca8488a65c3fba21e182`; runtime: Python 3.12.15, NVIDIA GH200 144G HBM3e, GPU-404813aa-6b38-6245-ed8d-3ed4af1952a7, 580.173.02, Torch 2.14.0+cu130, SB3 2.9.0.
- Training input and schedule hashes, per-attempt config/provenance hashes, training diagnostics hashes, and checkpoint hashes are recorded in the companion JSON.
- Fingerprint overrides: none. Initial checkpoint lineage: empty.

This viability pass does not satisfy the five-seed RQ1 estimate, selected-pair cost stress, independent review, or result freeze. P7 remains necessary. The configured 5-point extra slippage was not applied to P6 strict tick fills.

Full machine-readable evidence: [`p6_seed50_training_viability_20261021.json`](p6_seed50_training_viability_20261021.json).
