# Python coding standards

- PEP 8 + Google-style docstrings + PEP 484 type hints.
- Max line length: 120 characters.
- Tools: Ruff (format/lint), mypy (strict), pytest.

## Code quality
- No mutable default args (`def f(x=None): x = x or []`).
- No bare `except:` — catch specific exceptions.
- Use `pathlib.Path` over `os.path`.
- Public API re-exported via `__init__.py`.
- Prefer immutable data structures for configs and data records:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    lot_size: float
```


## Module structure
- Module: 400 lines max; Function: 60 lines max; Commit: 500 changed lines max.
- Split by concern, each with its own tests.

## Relations

- Activates: whenever writing or editing `.py` files, tests, or configs for tooling.
- Enforced at commit time by [git-commit-rules.md](git-commit-rules.md) pre-commit checks.
- Read slices follow [token-efficient-context-replies.md](token-efficient-context-replies.md); test runs follow [token-efficient-shell.md](token-efficient-shell.md).
