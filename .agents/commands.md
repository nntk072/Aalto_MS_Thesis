# Commands Reference — Aalto_MS_Thesis

> Agent-oriented examples. The canonical command catalog is
> [RUNNING_COMMANDS](../docs/operations/RUNNING_COMMANDS.md); CI policy
> lives in [ci-verification](rules/ci-verification.md).

## Setup

```bash
uv sync                                    # install all deps
uv sync --extra dev                        # include dev tooling
```

## Agent commit/push gate — blocking

**Required** before commit/push. Full-tree format/lint/types (`mypy .` — never
path-scoped mypy); **scoped** pytest only (see `.agents/rules/ci-verification.md`).
Do not run `pytest tests/ -v` unless asked.

```bash
uv run ruff format --check .    # if fails: uv run ruff format . && re-check
uv run ruff check .
uv run mypy .
uv run pytest <relevant test paths> -q
```

Merge CI bar (GitHub / Orchestra phase 9) still runs `pytest tests/ -v`.

## Data Pipeline

```bash
python market_data/commands/prepare.py             # raw CSV → parquet → features
python market_data/commands/prepare.py --force     # ignore cache, reprocess
```

## Training

```bash
uv run thesis-train train --mvp              # MVP smoke test (30 days)
uv run thesis-train train                    # full training run
uv run thesis-train train --algo sac --arch gru
uv run thesis-train train --strategy po3_ifvg  # Idea 1 overlay
uv run thesis-train train --config config/features/full_po3_mtf.yaml
uv run thesis-train train --walk-forward --wf-splits 5 --purge-bars 60
uv run thesis-train backtest                # random policy backtest
uv run thesis-train baselines                 # baseline strategies
uv run python scripts/research/benchmarks/compare_encoders.py                     # encoder comparison
```

## Evaluation

```bash
uv run thesis-evaluate reanalyze --run outputs/<run_dir>
uv run python scripts/research/thesis/evaluation/report_g3.py --runs-dir outputs
```

## Testing

```bash
uv run pytest tests/ -v                      # full suite (commit / push)
pytest -m "not slow"                         # local shortcut
pytest -m unit                               # unit tests only
pytest -m integration                        # integration tests only
pytest tests/test_features/                  # feature tests only
pytest tests/test_envs/                      # env tests only
pytest tests/test_orchestra/                 # orchestra tests
./tools/check unit
```

## Linting & Type Checking

```bash
uv run ruff check .                          # lint (whole tree)
uv run ruff format .                         # format
uv run ruff check --fix .                    # auto-fix
uv run mypy .                                # type check (whole tree)
```

## Orchestra

```bash
orchestra doctor                             # Tier-0 health check
orchestra run "task description"             # full pipeline (verify runs CI gate)
```

## Docker

```bash
make docker-build
make docker-build-test
make docker-test
make docker-lint
make docker-pipeline
```

## Dependency Management

```bash
make lock
make lock-export
make lock-check
make deps-check
```

## Config Overrides

```bash
python market_data/commands/prepare.py data.cache_dir=my_cache env.obs_window=30
uv run thesis-train train training.max_days=60
```
