# Commands Reference — Aalto_MS_Thesis

## Setup

```bash
uv sync                                    # install all deps
uv sync --extra dev                        # include dev tooling
```

## Data Pipeline

```bash
python scripts/prepare_data.py             # raw CSV → parquet → features
python scripts/prepare_data.py --force     # ignore cache, reprocess
```

## Training

```bash
python -m quant_rl.train.train_rl --mvp    # MVP smoke test (30 days)
python -m quant_rl.train.train_rl          # full training run
python -m quant_rl.train.train_rl --algo sac --arch gru   # SAC with GRU encoder
python -m quant_rl.train.run_backtest      # backtest with random policy
python -m quant_rl.train.run_baselines     # run baseline strategies
python scripts/compare_encoders.py         # compare encoder architectures
```

## Testing

```bash
pytest                                     # full test suite
pytest -m "not slow"                       # skip slow torch/SB3 tests
pytest -m unit                             # unit tests only
pytest -m integration                      # integration tests only
pytest -k "swing"                          # tests matching "swing"
pytest tests/test_features/                # feature tests only
pytest --cov=quant_rl --cov-report=term-missing -q   # coverage
```

## Linting & Type Checking

```bash
ruff check quant_rl/                       # lint
ruff format quant_rl/                      # format
ruff check --fix quant_rl/                 # auto-fix lint issues
mypy quant_rl/                             # type check
```

## Docker

```bash
make docker-build                          # build runtime image
make docker-build-test                     # build test image
make docker-test                           # run tests in Docker
make docker-lint                           # run lint in Docker
make docker-pipeline                       # full pipeline (lint → typecheck → test)
```

## Dependency Management

```bash
make lock                                  # re-resolve uv.lock
make lock-export                           # regenerate pip pin files
make lock-check                            # verify uv.lock is current
make deps-check                            # verify pip pins match uv.lock
```

## Config Overrides

```bash
python scripts/prepare_data.py data.cache_dir=my_cache env.obs_window=30
python -m quant_rl.train.train_rl training.max_days=60
```