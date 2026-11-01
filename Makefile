# AgentFarm — optional convenience targets (use a venv with deps installed).
PYTHON ?= python3
UV ?= uv
LOCK_FILE := requirements-lock.txt
# Universal (all platforms, Python >= 3.10) pin set for CI. Re-run `make lock`
# after editing [project.dependencies] or the dev extra in pyproject.toml.
# Without --upgrade uv keeps existing pins, so the output only changes when
# pyproject.toml changes; use `make lock-upgrade` to move every pin forward.
UV_COMPILE := $(UV) pip compile pyproject.toml --extra dev --universal \
	--python-version 3.10 --annotation-style line \
	--custom-compile-command "make lock"

.PHONY: lock lock-upgrade lock-check crossover-search-smoke crossover-search-help

lock:
	$(UV_COMPILE) -o $(LOCK_FILE)

lock-upgrade:
	$(UV_COMPILE) --upgrade -o $(LOCK_FILE)

# Fails when requirements-lock.txt is out of sync with pyproject.toml.
lock-check:
	@tmp=$$(mktemp) && cp $(LOCK_FILE) $$tmp && $(UV_COMPILE) -q -o $$tmp >/dev/null && \
	if ! diff -u $(LOCK_FILE) $$tmp; then \
		echo "::error::$(LOCK_FILE) is out of date; run 'make lock' and commit the result."; \
		rm -f $$tmp; exit 1; \
	fi; rm -f $$tmp

# Fast wiring check: 2 children, synthetic states (no checkpoint paths required).
crossover-search-smoke:
	$(PYTHON) scripts/run_crossover_search.py \
		--search-space minimal \
		--max-runs 2 \
		--n-states 120 \
		--seed 0 \
		--run-dir /tmp/agentfarm_crossover_smoke

crossover-search-help:
	$(PYTHON) scripts/run_crossover_search.py --help
