# Security rules


## Before ANY commit

- [ ] No hardcoded secrets (API keys, passwords, tokens, MT5 credentials).
- [ ] Secrets come from environment variables / `.env` (never committed; see `.env.example`).
- [ ] External inputs (market data, config files, model artifacts) validated at load boundaries — fail fast with clear messages.
- [ ] Error messages do not leak paths, credentials, or account info.
- [ ] Model/data artifacts from untrusted sources are never executed unpickled without checks.

## If a security issue is found

1. STOP other work immediately.
2. Fix CRITICAL issues before continuing anything else.
3. Rotate any exposed secrets.
4. Search the codebase for similar patterns.

## Relations

- Activates: always, checked at every commit via [git-commit-rules.md](git-commit-rules.md); deep check whenever touching credentials, network calls, or artifact loading.
