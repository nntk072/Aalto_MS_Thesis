# Commands Reference — Aalto_MS_Thesis

## Setup

```bash
uv sync                                    # install all deps
uv sync --extra dev                        # include dev tooling
```

## CI gate (commit / push)

Mirrors `orchestra/ci_gate.py` and `.github/workflows/ci.yml`. Run on commit/push
only; while coding, use scoped ruff + pytest on touched paths.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

Local shortcut (not merge bar): `pytest -m "not slow"`

## Data Pipeline

```bash
python scripts/prepare_data.py             # raw CSV → parquet → features
python scripts/prepare_data.py --force     # ignore cache, reprocess
```

## Training

```bash
uv run python -m quant_rl.train.train_rl --mvp              # MVP smoke test (30 days)
uv run python -m quant_rl.train.train_rl                    # full training run
uv run python -m quant_rl.train.train_rl --algo sac --arch gru
uv run python -m quant_rl.train.train_rl --strategy po3_ifvg  # Idea 1 overlay
uv run python -m quant_rl.train.train_rl --config config/features_full_po3_mtf.yaml
uv run python -m quant_rl.train.train_rl --walk-forward --wf-splits 5 --purge-bars 60
uv run python -m quant_rl.train.run_backtest                # random policy backtest
uv run python -m quant_rl.train.run_baselines                 # baseline strategies
uv run python scripts/compare_encoders.py                     # encoder comparison
```

## Evaluation

```bash
uv run python -m quant_rl.eval.eval_run --run outputs/<run_dir>
uv run python scripts/report_g3.py --runs-dir outputs
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
pytest --cov=quant_rl --cov-report=term-missing -q
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
python scripts/prepare_data.py data.cache_dir=my_cache env.obs_window=30
uv run python -m quant_rl.train.train_rl training.max_days=60
```
