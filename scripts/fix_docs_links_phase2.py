#!/usr/bin/env python3
"""Safer docs-only link updates for Phase 2 (prefix-aware, docs/ tree only)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# Order matters: longer / more specific replacements first.
REPLACEMENTS: list[tuple[str, str]] = [
    # directories
    ("docs/config/", "docs/reference/config/"),
    ("docs/data/", "docs/reference/data/"),
    ("docs/api/", "docs/reference/api/"),
    ("docs/devlog/", "docs/research/devlog/"),
    ("docs/experiments/", "docs/research/experiments/"),
    ("docs/features/", "docs/archive/features/"),
    ("docs/howto/neural_recombination_runbook.md", "docs/guides/neural-recombination-runbook.md"),
    ("docs/analysis/modules/", "docs/reference/analysis/modules/"),
    ("docs/spatial/", "docs/concepts/spatial/"),
    ("docs/electron/", "docs/reference/electron/"),
    # root docs files (explicit)
    ("docs/ExperimentQuickStart.md", "docs/getting-started/experiments-quickstart.md"),
    ("docs/LOGGING_QUICK_REFERENCE.md", "docs/reference/logging-quick-reference.md"),
    ("docs/PROFILING_AND_BENCHMARKING_PLAN.md", "docs/archive/profiling-and-benchmarking-plan.md"),
    ("docs/action_data.md", "docs/concepts/action-data.md"),
    ("docs/action_system.md", "docs/concepts/actions.md"),
    ("docs/agents.md", "docs/concepts/agents-and-decisions.md"),
    ("docs/api_reference.md", "docs/reference/api-reference.md"),
    ("docs/core_architecture.md", "docs/concepts/core-architecture.md"),
    ("docs/deployment.md", "docs/guides/deployment.md"),
    ("docs/developer-guide.md", "docs/guides/development-setup.md"),
    ("docs/user-guide.md", "docs/guides/development-setup.md"),
    ("docs/experiments.md", "docs/research/experiments-catalog.md"),
    ("docs/experiment_runner.md", "docs/guides/experiment-runner.md"),
    ("docs/experiment_analysis.md", "docs/guides/experiment-analysis.md"),
    ("docs/logging_guide.md", "docs/guides/logging.md"),
    ("docs/module_overview.md", "docs/concepts/module-overview.md"),
    ("docs/monitoring.md", "docs/guides/monitoring.md"),
    ("docs/usage_examples.md", "docs/guides/usage-examples.md"),
    ("docs/ipc-api.md", "docs/reference/ipc-api.md"),
    ("docs/deterministic_simulations.md", "docs/guides/deterministic-simulations.md"),
    ("docs/genetics_analysis.md", "docs/guides/genetics-analysis.md"),
    ("docs/generic_combat_simulation_howto.md", "docs/guides/generic-combat-simulation.md"),
    ("docs/generic_simulation_scenario_howto.md", "docs/guides/generic-simulation-scenario.md"),
    ("docs/health_resource_analysis.md", "docs/guides/health-resource-analysis.md"),
    ("docs/glossary.md", "docs/reference/glossary.md"),
    ("docs/metrics.md", "docs/reference/metrics.md"),
    ("docs/repositories.md", "docs/reference/repositories-overview.md"),
    ("docs/deep_q_learning.md", "docs/concepts/deep-q-learning.md"),
    ("docs/initial_diversity.md", "docs/concepts/initial-diversity.md"),
    ("docs/observation_channels.md", "docs/concepts/observation-channels.md"),
    ("docs/perception_system.md", "docs/concepts/perception-system.md"),
    ("docs/perception_system_design.md", "docs/concepts/perception-system-design.md"),
    ("docs/dynamic_channel_system.md", "docs/concepts/dynamic-channel-system.md"),
    ("docs/state_system.md", "docs/concepts/state-system.md"),
    ("docs/dependency-injection.md", "docs/concepts/dependency-injection.md"),
    ("docs/redis_agent_memory.md", "docs/concepts/redis-agent-memory.md"),
    # relative within docs (no docs/ prefix) — apply only in markdown links
    ("](../config/", "](../reference/config/"),
    ("](config/", "](reference/config/"),
    ("](../data/", "](../reference/data/"),
    ("](data/", "](reference/data/"),
    ("](../devlog/", "](../research/devlog/"),
    ("](devlog/", "](research/devlog/"),
    ("](../experiments/", "](../research/experiments/"),
    ("](experiments/", "](research/experiments/"),
    ("](../howto/neural_recombination_runbook.md", "](../guides/neural-recombination-runbook.md"),
    ("](../analysis/modules/", "](../reference/analysis/modules/"),
    ("](analysis/modules/", "](reference/analysis/modules/"),
    ("](../spatial/", "](../concepts/spatial/"),
    ("](../electron/", "](../reference/electron/"),
    ("](../ExperimentQuickStart.md", "](../getting-started/experiments-quickstart.md"),
    ("](../api_reference.md", "](../reference/api-reference.md"),
    ("](../deployment.md", "](../guides/deployment.md"),
    ("](../developer-guide.md", "](../guides/development-setup.md"),
    ("](../user-guide.md", "](../guides/development-setup.md"),
    ("](../experiments.md", "](../research/experiments-catalog.md"),
    ("](../experiment_runner.md", "](../guides/experiment-runner.md"),
    ("](../experiment_analysis.md", "](../guides/experiment-analysis.md"),
    ("](../logging_guide.md", "](../guides/logging.md"),
    ("](../module_overview.md", "](../concepts/module-overview.md"),
    ("](../usage_examples.md", "](../guides/usage-examples.md"),
    ("](../ipc-api.md", "](../reference/ipc-api.md"),
    ("](../core_architecture.md", "](../concepts/core-architecture.md"),
    ("](../agents.md", "](../concepts/agents-and-decisions.md"),
    ("](../action_system.md", "](../concepts/actions.md"),
    ("](../deep_q_learning.md", "](../concepts/deep-q-learning.md"),
    ("](../genetics_analysis.md", "](../guides/genetics-analysis.md"),
    ("](../features/", "](../archive/features/"),
]


def fix_docs() -> None:
    for path in DOCS.rglob("*"):
        if path.suffix not in {".md", ".html", ".yml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"fixed: {path.relative_to(REPO)}")


def fix_repo_md_outside_docs() -> None:
    """Update README and farm READMEs that reference docs paths."""
    targets = list(REPO.glob("README.md")) + list(REPO.glob("farm/**/README.md"))
    targets += list(REPO.glob("farm/**/ARCHITECTURE.md"))
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        original = text
        for old, new in REPLACEMENTS:
            if old.startswith("docs/"):
                text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"fixed: {path.relative_to(REPO)}")


if __name__ == "__main__":
    fix_docs()
    fix_repo_md_outside_docs()
