# Orchestra Probe Matrix (Plan v3 Phase 0)

Evidence tags: **A** = verified locally, **B** = official docs, **C** = unproven assumption.

CLI versions (**A**): Cline 3.0.61, Kilo 7.5.16, OpenCode 1.18.30, Vibe 2.25.2, Gemini CLI 0.59.0.

## Tier-0 (local, no inference)

| Backend | Command / check | Proves | Tag | Notes |
|---------|---------------|--------|-----|-------|
| All | `which <cli>` | binary exists | A | |
| Cline | read `~/.cline/data/settings/providers.json` (ids + model only) | provider configured | A | Never log tokens |
| Kilo/OpenCode | binary present | CLI installed | A | Auth cache optional from prior doctor |
| Vibe | `~/.vibe/config.toml` + `MISTRAL_API_KEY` or local Ollama | config readable | A | `--agent` is permission profile, not model (**A**) |
| Gemini | `GEMINI_API_KEY` or `~/.gemini/settings.json` auth type | auth configured | A | |

## Tier-1 (doctor only, no generation)

| Backend | Command | Proves | Does NOT prove | Tag |
|---------|---------|--------|----------------|-----|
| Gemini | `GET /v1beta/models` | catalog + auth | RPD remaining, generate access | B |
| Mistral/Vibe | `GET {api_base}/v1/models` | model exists | quota | B |
| Kilo | `kilo models` (no `--refresh`), `kilo auth list` | catalog + creds | live RPM/RPD | A |
| OpenCode | `opencode models`, `opencode providers list` | catalog + creds | live RPM/RPD | A |
| Cline | parse `providers.json` only | selected model id | model served live | A |

## Forbidden probes

- `kilo roll-call`, any `run`/`-p` prompt, `cline auth`, `cline doctor fix`, `generateContent`, Vibe ping (**A**)

## Session / resume (**C** unless noted)

| Backend | Mechanism | Tag |
|---------|-----------|-----|
| Cline | `--id <session-id>` resume; sessions under `~/.cline/data/sessions/` | A for flag; C for Orchestra capture |
| Kilo/OpenCode | `-s/--session`, `--fork` with `-m` model switch | A flags; C fork+model |
| Vibe | `-c` continue latest; `--resume [ID]` separate | A |
| Gemini | `--resume latest\|index\|UUID`; `--session-id` is NEW uuid only | B |

## Effort flags (**A**)

| CLI | Effort mechanism |
|-----|------------------|
| Cline | `--thinking none\|low\|medium\|high\|xhigh` |
| Kilo/OpenCode | `--variant` (not `--thinking`) |
| Vibe | per-model `thinking` in config; `VIBE_ACTIVE_MODEL` for model (**C** per-run override) |
| Gemini | no native effort flag in CLI |

## Phase 5 gate

- `native_resume: false` default in `models.yaml` until operator enables per model
- `native_model_switch: false` — fork+model unproven (**C**)
- OpenCode/Kilo JSON logs may include `"sessionID"` — Orchestra parses when `native_resume: true` (**C**)
- Cline session dirs under `~/.cline/data/sessions/<id>/` — regex capture attempted (**C**)
- Safe default: handoff bundle on model change; same-model retry uses native id only when enabled

## VIBE_ACTIVE_MODEL

- Orchestra sets env `VIBE_ACTIVE_MODEL={model_id}` when spawning Vibe (**A** design)
- Whether Vibe honors it at runtime: **C** — fallback is `active_model` in config.toml
