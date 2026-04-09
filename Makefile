# AgentFarm — optional convenience targets (use a venv with deps installed).
PYTHON ?= python3

.PHONY: crossover-search-smoke crossover-search-help

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
