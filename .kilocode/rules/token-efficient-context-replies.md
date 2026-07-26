# Token-efficient context and replies

Rules add to every request — keep behavior tight; do not compensate with long chat.

## Before reading files

- **Locate first**: `Grep` / `Glob` with narrow patterns; then `Read` only likely files.
- **Slice reads**: default ~80–150 lines around the target; widen only when needed.
- **Skip unless the task needs them**: `node_modules`, `dist`, `build`, `.venv`, `*-lock.*`, `uv.lock`, minified assets, large fixtures, full migration histories.
- **Prefer diffs**: `git diff -- <path>` over reading old + new full files.
- **No duplicate work**: if content is already in the thread or a prior tool result, do not re-read or re-run the same command.

## Scope

- Touch only files required for the task; no drive-by refactors or doc sweeps.
- For repo-wide questions, sample key paths — do not read every similar file.
- Use the `task` tool with the **explore** subagent when a keyword search suffices.

## Assistant output (saves output tokens)

- Answer first; avoid restating the question and long preambles.
- Summarize tool results in prose; do not paste large logs or directory listings into chat.
- Show code via **line citations** (`path:line`) or small hunks — not whole files after an edit.
- Match depth to the ask: short task → short reply; deep review only when requested.
- Do not repeat the same explanation across turns; refer to earlier messages briefly.