# Changes: variant resolver, launcher, cache and bench fixes

Based on the debug/standardization plan (v4). Plan item IDs are in brackets.

## What was wrong (verified by replaying all 29 variants)

- **Baseline-named ladder rungs ran PO3-IFVG.** `ladder_a0_flat` … `ladder_a4a_entry_fsm_hidden` set `strategy_actions: true` without a strategy, so `default.yaml`'s `strategy.name: po3_ifvg` was used. [A1]
- **Variant fields were dead config.** The launcher hard-coded `--arch tcn`, passed no `--algo`, no `--use-vae`, and used 20M steps with one seed. `ablation_sac_agent` ran PPO, `ablation_transformer_encoder` ran TCN, `ablation_conditional_narrative` ran without its VAE. [A4]
- **Done/failed markers were per variant.** A multi-seed run would skip seeds 2..N. [B1]
- **Resume was broken.** The matrix passed `RESUME`, which `train_rl` ignores. [B2]
- **Feature-cache digest used 64-bar endpoints.** A corrected mid-series bar with the same length and endpoints reused stale features. [B3]
- **Bundle input signature used endpoints only.** Same weakness as above. [E3]
- **Hard-coded `/scratch/work/nguyenl37` paths.** [B4]
- **Benchmark probe measured only the default config**, with a placeholder feature hash and a hard-coded bundle path. [E6]
- **PnL + sweep reward** would silently drop the PnL term. Not active in any config, but one flag away. [E5]

## What changed

| File | Change |
|---|---|
| `quant_rl/train/variant_resolver.py` (new) | Single source of truth. Resolves a variant into base config, `--strategy`, algo, arch, steps, seeds, and explicit overrides. Refuses undeclared `reward_mode`, unknown strategy/algo/arch, and `use_vae=1` without a VAE path. Writes `resolved_config.json` per run. Pure Python + PyYAML. [A1, A4] |
| `scripts/train/train_one_variant.sh` | Rewritten as a thin wrapper around the resolver. Runs one seed at a time; markers and output dirs are keyed `<variant>__seed<N>`; no resume branch; `DRY_RUN=1` prints the exact commands; `VAE_PATH` required for VAE variants. [A4, B1, B2, B4] |
| `scripts/matrix/run_18variant_matrix.sh` | Repo path is script-relative; `STEPS` defaults to the YAML value; skip check uses the seed-keyed marker; resume lookup removed. [B1, B2, B4] |
| `config/experiments.yaml` | Every variant declares `reward_mode` explicitly, using the effective value from before this change (no behavior change). Comments kept. |
| `quant_rl/train/ablation_utils.py` | `merge_variant_cfg` pins `strategy.name=baseline` for baseline rungs, so the legacy `--variant` path agrees with the resolver. [A1] |
| `quant_rl/features/build.py` | Cache digest covers every OHLCV/spread value and the full train mask. [B3] |
| `quant_rl/envs/env_bundle_builder.py` | Bundle input signature includes a digest of stored bar values. [E3] |
| `quant_rl/envs/trading_env.py` | `reward_mode='pnl'` with `use_sweep_reward=True` raises at construction. [E5] |
| `scripts/bench/measure_ppo_throughput.py`, `run_fps_probe.sh` | `--variant` option uses that variant's base config, overrides, algo, arch. Wrapper requires `BUNDLE_DIR`; no hard-coded hash. [E6] |
| `scripts/matrix/ablation_runner.py` | Prints a warning: it bypasses the resolver and shares one features CSV. [A5] |
| `scripts/eval/report_ablations.py` | Docstring path corrected. [B6] |
| `baseline/A3_*`, `baseline/runbook_*` | "SUPERSEDED" banner. [B5] |
| `tests/test_train/test_variant_resolver.py` | 12 tests. Runs without OmegaConf or torch. **All 12 pass** in the sandbox. |
| `tests/test_train/test_variant_overrides_match_legacy.py` | Parametrized over all 29 variants: resolver overrides must reproduce the legacy `merge_variant_cfg` config, except `strategy.name` for baseline rungs. **Not run here** (needs OmegaConf). |
| `tests/test_features/test_cache_hash_full_column.py` | Mid-series change changes the digest. **Not run here** (needs pandas/numpy). |

## Behavior changes to expect

1. Baseline-named ladder rungs now run `BaselineStrategy` (plus full-PO3 features and whatever `reward_mode` they declare). Earlier results for them were PO3-IFVG runs and are not comparable.
2. Controls run their declared arch and algo: `ablation_sac_agent` is SAC/GRU, `ablation_transformer_encoder` is transformer.
3. Steps and seeds come from `experiments.yaml` (100k, seeds 42–46) unless `STEPS` / `SEEDS` / `SEED` are set. The old default was 20M and one seed.
4. The `ftmo.trailing_dd_limit=0.07` override was dropped. It equaled the default.
5. `ablation_conditional_narrative` fails unless `VAE_PATH` is set.

## Not done (and why)

- **Resume is removed, not implemented.** Implementing it needs restoring `num_timesteps` and optimizer state in `train_rl`, which I can't test without torch. Rollover restarts a variant from scratch; finished seeds are skipped.
- **Bundle mode stays off** (`env.data_source: frames`). The bundle parity tests still cover only `BaselineStrategy`. Add PO3/distribution parity tests before enabling it.
- **Reward magnitudes are not normalized.** PnL reward is a fraction of initial balance (~1e-3 to 1e-2 per trade), DSR is clipped to ±10, and PO3 shaping adds 0.01–0.02. Measure per-variant reward distributions with a short rollout before the sweep.
- **`ablation_runner.py` is not retired**, only flagged.
- **Earlier plan item E1 is withdrawn.** `build_env_bundle` injects `strategy.raw_columns` into the config before `bundle_key` hashes it, so PO3 and distribution bundles already get different keys.

## Decisions still needed from the owner

1. **Baseline ladder rungs: `reward_mode`.** They currently declare `dsr`, while PO3 and distribution rungs declare `pnl`. The table (`python -m quant_rl.train.variant_resolver table`) shows this confound. Either align them or accept it as a documented reference.
2. **Baseline ladder rungs: PD context.** Off for baseline rungs, on for PO3 and distribution rungs.
3. **Ladder reference.** Should `ladder_a0_flat` be a pure baseline (default features, no strategy actions) rather than full-PO3 with `BaselineStrategy`?
4. **Final steps and seeds.** 100k with 5 seeds, or 20M with 1 seed?
5. **Variant set.** 18 (the matrix list) or 29 (including knobs and controls)?

## How to run

```bash
# inspect every variant's effective settings
PYTHONPATH=. python -m quant_rl.train.variant_resolver table
PYTHONPATH=. python -m quant_rl.train.variant_resolver meta ladder_a2_tp

# dry run (no training, no markers)
DRY_RUN=1 VARIANT=ladder_a2_tp scripts/train/train_one_variant.sh

# tests that run without the training stack
PYTHONPATH=. python tests/test_train/test_variant_resolver.py

# tests that need the repo environment (OmegaConf, pandas, pytest)
pytest tests/test_train/test_variant_overrides_match_legacy.py tests/test_features/test_cache_hash_full_column.py -q
```

## Verification status

- Resolver tests: 12/12 pass here.
- Launcher dry run and VAE refusal: pass here.
- Python syntax: all changed files parse.
- Shell syntax: all changed scripts pass `bash -n`.
- **Not run:** the full pytest suite, the OmegaConf equivalence test, the cache-hash test, any training, and the throughput probe. These need the repo's Python environment, which was not available in the sandbox.
