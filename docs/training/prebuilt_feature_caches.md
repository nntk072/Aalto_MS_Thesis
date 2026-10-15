# Prebuilt strategy feature caches

> **Plan routing:** Cache use and acceptance are governed by the [master project execution plan](../thesis/master_project_execution_plan.md). This file is the cache contract reference, not a separate work queue.

The three named Parquet matrices are stored locally under
`data/prebuilt_feature_cache/`:

| Variant | Cache |
| --- | --- |
| `ladder_a0_flat` (baseline) | `baseline.parquet` |
| `ladder_a0_flat_po3` (Idea 1) | `idea1_po3_ifvg.parquet` |
| `ladder_a0_flat_dist` (Idea 2) | `idea2_distribution.parquet` |

`variant_resolver` selects the matching path for these three variants. Before
loading, `train_rl` checks the `.hash` sidecar against effective feature config,
market data, and the training split. Idea caches also require their
`.htf_sweeps.parquet` audit data. A mismatch or missing file fails closed rather
than rebuilding or overwriting a named prebuilt cache. Other variants keep the
existing content-addressed cache behavior.

The local cache files are generated artifacts and are excluded from Git. They
must be copied with the project data if training on another checkout or node
that does not share `/scratch`.
