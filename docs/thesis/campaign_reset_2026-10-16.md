# Campaign reset — 2026-10-16

## Previous run stopped and removed

- Campaign: `ladder18_seed42_20m_full_split_n64_20261015`
- Source checkout: `/scratch/work/nguyenl37/Aalto_MS_Thesis`, source commit
  `061987e15b124a616e924d014a0be778826ca305`
- Slurm job `20974801` was interrupted at the user's request. No `scancel` command
  was used. The supervisor was stopped to prevent automatic resubmission, and the
  job left `squeue` before its artifacts were removed.
- All 18 variants had partial checkpoints only; none completed 20M steps. Highest
  checkpoints ranged from 5M to 9M steps. The dashboard evaluation failed because
  `evaluate_model()` did not accept `observation_schema_version` and
  `observation_age_scale_bars` from the training evaluator configuration.
- The user requested deletion of this run. Its 3.2G campaign output directory was
  removed after recording this provenance summary. No shared feature cache or other
  campaign directory was included in the deletion.

## Replacement run

- New campaign spec: `config/campaigns/18_variant_seed50_20m_full_split_n64_20261016.yaml`
- Identity follows the master plan's new-run policy: seed 50, 20M steps per
  variant, full-split episodes, `n_envs=64`, shared-bundle data, and the same 18
  registered variants. It has a separate output identity and does not resume seed 42.
- The evaluator now forwards both observation schema arguments into `TradingEnv`.
- Verification before launch: all 18 identities resolved in campaign dry-run;
  `ruff format --check .`, `ruff check .`, `mypy .`, and
  `pytest tests/test_rollout.py -q` passed.
- The new Slurm run is launched only after this source/config state is committed to
  `main`; its campaign manifest is the operational source for the live job ID and
  runtime status.
