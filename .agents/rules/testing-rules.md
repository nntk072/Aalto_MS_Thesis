# Testing rules

Applies to this repo's pytest setup (`tests/`).

## Coverage target: 80% minimum

```bash
pytest --cov=quant_rl --cov-report=term-missing -q
```

Test types required for features:
1. **Unit tests** — individual functions, reward shaping, indicators.
2. **Integration tests** — env loop, agent + model interaction (see `tests/test_integration/`).

## Test-driven development (mandatory for new features)

1. Write the test first (RED) — it should FAIL.
2. Run it to confirm failure.
3. Write minimal implementation (GREEN).
4. Refactor (IMPROVE) while keeping coverage ≥ 80%.

Fix the implementation, not the tests — unless the tests are wrong.

## Structure

- **AAA pattern**: Arrange → Act → Assert.
- **Descriptive names** explaining behavior:
  - `test_sweep_reward_penalizes_early_entry`
  - `test_env_raises_when_action_out_of_bounds`
- Categorize with pytest markers:

```python
@pytest.mark.unit
def test_cosine_similarity(): ...

@pytest.mark.integration
def test_trading_env_episode_loop(): ...
```

## Relations

- Activates: any new feature, bug fix, or change to `quant_rl/`, `backtest/`, or env logic.
- Enforced at commit by [git-commit-rules.md](git-commit-rules.md); orchestrated by [development-workflow.md](development-workflow.md); test runs follow [token-efficient-shell.md](token-efficient-shell.md).