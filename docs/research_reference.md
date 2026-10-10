# Implementation-verified research reference

> Historical proposals for delayed trade-close reward credit assignment, an
> EMA-21 trade-chart overlay, and GPU-vectorized simulation did not establish
> implemented functionality. Verify these features in source and tests before
> documenting them as supported.
Evidence baseline: Triton checkout `main` at `76684a3`, inspected 2026-10-10. This page describes code at that commit; later changes require verification. It supplements [architecture](../docs/architecture.md) and [config](../docs/config.md). Refer to the source when branch state differs.

## Environment and execution

The package requires Python >=3.12 (`pyproject.toml`). Installation uses `uv`; dependency groups and platform-specific Torch indexes are defined in `pyproject.toml`. Use the architecture-specific virtual environment on Triton (`.venv` on aarch64 GH200, `.venv-x86` on x86_64 H200); consult [Triton scripts](../scripts/triton/README.md) and [running commands](../docs/operations/RUNNING_COMMANDS.md). Avoid GPU allocations for documentation verification; use existing allocations if authorized. `scripts/data/prepare_data.py`, `quant_rl/train/train_rl.py`, and `scripts/eval/` are the pipeline entry points. Inspect their `--help` output before invoking a long-running command.

## Pipeline and source ownership

| Pipeline stage | Authoritative modules | Verification |
| --- | --- | --- |
| M1 ingestion / timeframe resampling | `quant_rl/data/{loader,pipeline,resample,align,split}.py` | `tests/test_data_split.py` |
| Feature computation / normalization | `quant_rl/features/{build,normalize,po3_config}.py` | Feature and leakage tests under `tests/` |
| Strategy context / action interpretation | `quant_rl/envs/{trading_env,tp_decoders}.py` and `quant_rl/envs/strategies/` | `tests/test_envs/test_tp_mode.py` |
| RL agent / policies | `quant_rl/models/{agent,ppo_policy,encoder}.py` | Model tests under `tests/` |
| Reward and account limits | `quant_rl/envs/{reward,sweep_reward,po3_reward}.py`, `quant_rl/backtest/{account,guardrails}.py` | `tests/test_reward.py`; `tests/test_envs/test_sweep_reward.py` |
| Train and checkpointing | `quant_rl/train/{train_rl,variant_resolver,callbacks}.py` | Document actual CLI options from source |
| Evaluation / plotting | `quant_rl/evaluation/`, `quant_rl/eval/`, `scripts/eval/` | Thesis [validation protocol](../docs/thesis/validation_protocol.md) |

The environment is backed by the event-driven broker and account components under `quant_rl/backtest/`. Keep training execution, evaluation output and any MT5 live adapter distinct when making risk or fidelity claims.

## Action-space layout: current implementation

`TradingEnv` constructs a strategy-overlay `Box(-1, 1)` in `quant_rl/envs/trading_env.py`. Its dimension formula is:

```text
5 + direction_control + sl_mode + tp_mode + (2 × multi_tp) + (2 × multi_tp × simplex)
```

Each indicator is 1 if its feature flag is enabled, otherwise 0. The complete ordered action vector is:

```text
[direction?] intensity, stop, risk, [tp1_sel, tp2_sel, tp3_sel | target],
[sl_mode?], [tp_mode?], [z1,z2 if multi_tp and simplex], exit
```

`direction` is present when `agent_direction_control=true`. With `allow_multi_tp=false`, there is one `target` dimension; enabling it replaces that selector with **three** TP selectors (a net gain of two dimensions). `allow_simplex` adds `z1,z2` only when multi-TP is enabled. The final action dimension remains `exit`. Source references: `quant_rl/envs/trading_env.py` constructor/action setup and `_decode_action`; `quant_rl/envs/tp_decoders.py` for TP selection, fractions and validation.

| Example overlay flags | Dimensions |
| --- | ---: |
| No optional dimensions | 5 |
| Default direction off, SL on, TP mode off, multi-TP off | 6 |
| Direction on, SL on, TP mode on, multi-TP on, simplex on | 12 |

At `76684a3`, `quant_rl/config/default.yaml` sets `agent_direction_control: false`, `allow_agent_sl_mode: true`, `allow_agent_tp_mode: false`, `allow_multi_tp: false` and `tp_breakeven_alpha: 0.5`. Verify `allow_simplex` in active merged config before declaring a 12-D experiment. Action dimensionality also depends on `strategy_actions` and `continuous_actions`; the non-overlay alternatives are one-dimensional continuous and `Discrete(20)`.

**Important decoder detail:** TP mode handling and fraction splitting depend on the final vector shape and associated flags. Do not assume that enabling `allow_agent_tp_mode` alone proves all TP-management branches executed in a training result. Verify run config and trade logs.

## Train / OOS boundaries and normalization

`quant_rl/data/split.py:split_train_test` intersects bars/features timestamps, includes the complete `train_end` day in training and keeps rows from `test_start` for testing. Its default dates are `2025-12-31` and `2026-01-01`. `get_split_config` reads `cfg.data.split`. The module explicitly instructs callers to pass `train_mask` when building features so `rolling_zscore` uses training-only statistics. Date splitting alone does **not** establish absence of look-ahead; inspect each preprocessing caller, timeframe alignment, labels and evaluation code. See [validation protocol](../docs/thesis/validation_protocol.md).

## Reward / FTMO-style constraints

Reward classes include `DSRReward`, `PnLReward`, `RMultipleReward` in `quant_rl/envs/reward.py`, plus separate strategy-specific rewards. The active reward is configuration-dependent; do not describe every run as differential Sharpe. Inspect reward initialization inside `quant_rl/envs/trading_env.py` and the resolved experiment YAML before reporting an equation or a result.

`FTMOGuardrails` (`quant_rl/backtest/guardrails.py`) provides separate hard checks for `daily_loss_limit`, fixed loss from initial balance (`max_loss_limit`), and optional peak-to-current-equity trailing drawdown (`trailing_dd_limit × initial_balance`). Soft daily and trailing thresholds inhibit entries but are not hard kill-switches. The `risk_per_trade_limit` check is independent. These are *FTMO-style simulation constraints*, not evidence of live broker compliance. Report limit values from the actual resolved run configuration rather than the dataclass defaults.

## Experiments, results, and claims

Training runs, variant comparisons, ablation tables and thesis figures must always identify experiment ID, git SHA, merged configuration, data provenance, seeds, training steps, checkpoint choice, IS/OOS dates and the path of raw metrics. Results without those fields are **unverified**. Check `scripts/matrix/`, `scripts/eval/`, `outputs/` and [evaluation pack](../docs/thesis/eval_pack.md), then distinguish failed or incomplete attempts. Do not copy numerical results from old slides or design plans.

## Documentation maintenance checklist

For any change to `quant_rl/envs/trading_env.py`, `quant_rl/envs/tp_decoders.py`, `quant_rl/config/default.yaml`, `quant_rl/data/split.py` or `quant_rl/backtest/guardrails.py`, re-check this page and `docs/config.md`. For changes to model/reward algorithms or metrics, check the thesis methodology and evaluation docs. Record evidence and validation outcomes in the maintained reference page; do not conflate an unexecuted test with a pass.
