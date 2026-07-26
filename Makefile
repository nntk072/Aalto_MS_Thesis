# Dependency management helpers.
#
# uv.lock is the single source of truth. The pip pin files
# (requirements.txt, requirements-dev.txt, constraints.txt) are generated from
# it so CI (plain pip) and local `uv sync` stay consistent.

# Recipes use bash features (process substitution).
SHELL := bash

UV ?= uv
EXPORT = $(UV) export --frozen --no-hashes --no-emit-project --format requirements-txt

# torch is installed from the PyTorch CPU index in an index-scoped step, so it is
# stripped from the exported pip requirements to avoid --extra-index-url shadowing.
STRIP_TORCH = grep -v '^torch=='

.PHONY: lock lock-check lock-export deps-check

## Re-resolve dependencies and update uv.lock.
lock:
	$(UV) lock

## Fail if uv.lock is out of date with pyproject.toml.
lock-check:
	$(UV) lock --check

## Regenerate the pip pin files from uv.lock.
lock-export:
	@printf '%s\n' \
	  '# Pinned runtime dependencies, exported from uv.lock (CPU-only torch).' \
	  '# Single source of truth is uv.lock; regenerate with `make lock-export`.' \
	  '#' \
	  '# NOTE: torch is intentionally NOT listed here. Its CPU wheels live on the' \
	  '# PyTorch index, and mixing that index with PyPI via --extra-index-url lets it' \
	  '# shadow common PyPI packages (certifi, numpy, ...). Install torch separately' \
	  '# from an index-scoped command instead (see the CI workflow / Makefile):' \
	  '#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu' \
	  > requirements.txt
	$(EXPORT) | grep -vE '^\s*#|^\s*$$|^-e ' | $(STRIP_TORCH) >> requirements.txt
	@printf '%s\n' \
	  '# Pinned dev/CI tooling, exported from uv.lock (dev extra).' \
	  '# Installs the runtime pins plus lint/type-check tooling:' \
	  '#   pip install -r requirements.txt -r requirements-dev.txt' \
	  '# Single source of truth is uv.lock; regenerate with `make lock-export`.' \
	  '# (torch is installed separately from the PyTorch index; see requirements.txt.)' \
	  '-r requirements.txt' \
	  '' \
	  > requirements-dev.txt
	@comm -13 \
	  <($(EXPORT) | grep -vE '^\s*#|^\s*$$|^-e ' | sort) \
	  <($(EXPORT) --extra dev | grep -vE '^\s*#|^\s*$$|^-e ' | sort) \
	  >> requirements-dev.txt
	@printf '%s\n' \
	  '# CPU-only version constraints, derived from uv.lock.' \
	  '#' \
	  '# Purpose: hold every (in)direct dependency to the locked versions and force the' \
	  '# CPU build of torch, so ad-hoc `pip install <pkg>` cannot pull a newer version' \
	  '# or the multi-GB CUDA torch wheels. Constraints only cap versions; they never' \
	  '# install packages or add indexes, so listing torch here is safe.' \
	  '#' \
	  '# Usage:' \
	  '#   pip install -c constraints.txt <anything>' \
	  '#   pip install -c constraints.txt -r requirements.txt -r requirements-dev.txt' \
	  '#   # torch (index-scoped, installed WITHOUT -c: constraints pin PyPI-only' \
	  '#   # transitive versions absent from the torch index):' \
	  '#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu' \
	  '#' \
	  '# Single source of truth is uv.lock; regenerate with `make lock-export`.' \
	  '' \
	  > constraints.txt
	$(EXPORT) --extra dev | grep -vE '^\s*#|^\s*$$|^-e ' | sort >> constraints.txt

## Fail if the committed pip pin files drift from uv.lock.
deps-check: lock-check
	@tmp=$$(mktemp -d); cp requirements.txt requirements-dev.txt constraints.txt $$tmp/; \
	$(MAKE) --no-print-directory lock-export; \
	if ! diff -q $$tmp/requirements.txt requirements.txt \
	  || ! diff -q $$tmp/requirements-dev.txt requirements-dev.txt \
	  || ! diff -q $$tmp/constraints.txt constraints.txt; then \
	    echo 'ERROR: pip pin files are out of date with uv.lock. Run `make lock-export`.'; \
	    cp $$tmp/requirements.txt $$tmp/requirements-dev.txt $$tmp/constraints.txt .; \
	    rm -rf $$tmp; exit 1; \
	fi; \
	rm -rf $$tmp; echo 'pip pin files are in sync with uv.lock.'
