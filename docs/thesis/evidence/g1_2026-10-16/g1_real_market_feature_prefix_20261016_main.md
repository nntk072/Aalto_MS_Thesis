# Real-market feature-prefix audit — 2026-10-16

## Result

- **12 / 12 comparisons passed**, across four feature profiles and three cutoffs; maximum absolute feature delta was **0**.
- Profiles: legacy baseline; baseline management ladder; PO3/IFVG; distribution. The report lists the exact variants grouped under each profile.
- Cutoffs: 2025-12-15T12:37:00+03:00, 2026-03-09T12:37:00+03:00, 2026-06-15T12:37:00+03:00.
- The comparison rebuilt each profile on the full market history and on prefixes ending at each cutoff, then compared all feature values through the cutoff.

## Provenance

- Repository source: `/scratch/work/nguyenl37/Aalto_MS_Thesis-main`, `main`, HEAD `8a98f7792f83ce4bbe5cb84f4a2137657d791c02`.
- Campaign: `ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016`, fingerprint `3474621efd6a1481bd2a20bce0cfd2c0b69969df1849c6cd738d19dc5bb4ba34`.
- Source files used by the audit: 32; all hashes match the current checkout. Both raw M1 input hashes match the campaign manifest.
- The repository audit script's default serializer failed because the M1 inputs resolve through a symlink outside the checkout. The rerun used an ignored copy with only explicit-root and external-path serialization changes; original and executed script hashes are recorded in the JSON. The feature-building implementation and resolved variant configs were not changed.
- Executed on existing Slurm job `20982793`; CPU-only, no GPU kernels.

## Interpretation limits

This supports **prefix invariance for the assembled features at these real-market cutoffs**. It does not prove every label, execution, stateful environment, checkpoint-selection, or statistical path is leakage-free. It does not establish feed identity or data-use rights, timezone/DST truth, or strategy quality. The 2026 interval is previously inspected and this audit is not confirmatory holdout evidence.

Machine-readable evidence: [`g1_real_market_feature_prefix_20261016_main.json`](g1_real_market_feature_prefix_20261016_main.json).
