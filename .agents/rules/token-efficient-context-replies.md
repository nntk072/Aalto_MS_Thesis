# Token-efficient context and replies

Rules add to every request — keep behavior tight; do not compensate with long chat.

## Before reading files

- **Locate first**: use search/grep with narrow patterns; then open only likely files.
- **Slice reads**: default ~80–150 lines around the target; widen only when needed.
- **Skip unless needed**: `node_modules`, `dist`, `build`, `.venv`, `*-lock.*`, `uv.lock`, minified assets, large fixtures, full migration histories.
- **Prefer diffs**: `git diff -- <path>` over reading old + new full files.
- **No duplicate work**: do not re-read or re-run what is already in the thread.

## Scope

- Touch only files required for the task; no drive-by refactors or doc sweeps.
- For repo-wide questions, sample key paths — do not read every similar file.

## Assistant output

- Answer first; avoid restating the question and long preambles.
- Summarize tool results in prose; do not paste large logs or directory listings.
- Show code via line citations or small hunks — not whole files after an edit.
- Match depth to the ask; do not repeat explanations across turns.