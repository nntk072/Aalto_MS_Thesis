# P3 External Seed-50 Artifact Search — 2026-10-18

**Disposition:** no external exact-name campaign directory or named report-pack
archive was found in the searched user roots. This does not establish that an
unnamed archive or backup on an unsearched system does not exist.

## Scope and method

Read-only searches covered `/scratch/work/nguyenl37` and `/home/nguyenl37`,
excluding the canonical checkout at
`/scratch/work/nguyenl37/Aalto_MS_Thesis` because its `outputs/` tree was
already inventoried by the linked recoverability audit. The searches looked
for:

- directories named `ladder18_seed50*` or `ladder_a*_seed50`;
- archive or JSON files whose names contain `ladder18`, `seed50`, `runpack`,
  or `reportpack` and whose suffix is `.zip`, `.tar`, `.tar.gz`, `.tgz`, or
  `.7z`;
- checkpoint files named `ppo_ckpt_*.zip` or `best_model.zip`.

The external-only search commands were:

```bash
repo=/scratch/work/nguyenl37/Aalto_MS_Thesis
find /scratch/work/nguyenl37 /home/nguyenl37 -path "$repo" -prune -o \
  -type d \( -name 'ladder18_seed50*' -o -name 'ladder_a*_seed50' \) -print
find /scratch/work/nguyenl37 /home/nguyenl37 -path "$repo" -prune -o \
  -type f \( -iname '*ladder18*' -o -iname '*seed50*' -o \
  -iname '*runpack*' -o -iname '*reportpack*' \) \
  \( -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.gz' -o \
  -iname '*.tgz' -o -iname '*.7z' \) -print
find /scratch/work/nguyenl37 /home/nguyenl37 -path "$repo" -prune -o \
  -type f \( -name 'ppo_ckpt_*.zip' -o -name 'best_model.zip' \) -print
```

## Findings

No matching campaign directory or named archive was found outside the
canonical checkout. The checkpoint-name search found eight `best_model.zip`
files under `/scratch/work/nguyenl37/.tmp/pytest-of-nguyenl37/`, in test-case
directories for `test_best_checkpoint_*`; each was 107,255 bytes. Inspection
of a representative archive showed the small Stable-Baselines3 test fixture
contents (`policy.pth`, optimizer state, and `system_info.txt`), not a saved
thesis run. No external `ppo_ckpt_*.zip` file was returned.

## Limits

This name-based search did not scan the contents of arbitrarily named archives,
follow directory symlinks, scan other users' directories, or inspect remote/cloud
backups and other filesystems. Combined with the recursive 612-archive
`outputs/` inventory, it still does not prove that no external backup exists.
Exact recovery of the 18 raw report packs remains
unavailable from the currently located artifacts, and the summary-derived
finding cards remain noncanonical.
