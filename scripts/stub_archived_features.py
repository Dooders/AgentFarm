#!/usr/bin/env python3
"""Replace archived feature pages with short redirect stubs."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "docs" / "archive" / "features"

STUBS: dict[str, tuple[str, str]] = {
    "FEATURES.md": (
        "AgentFarm Features",
        "[Architecture](../../concepts/architecture.md) · "
        "[Usage examples](../../guides/usage-examples.md) · "
        "[Documentation hub](../../README.md)",
    ),
    "agent_based_modeling_analysis.md": (
        "Agent-Based Modeling & Analysis",
        "[Architecture](../../concepts/architecture.md) · "
        "[Agents and decisions](../../concepts/agents-and-decisions.md) · "
        "[Experiment analysis](../../guides/experiment-analysis.md)",
    ),
    "customization_flexibility.md": (
        "Customization & Flexibility",
        "[Configuration guide](../../reference/config/configuration_guide.md) · "
        "[Usage examples](../../guides/usage-examples.md) · "
        "[Initial diversity](../../concepts/initial-diversity.md)",
    ),
    "ai_machine_learning.md": (
        "AI & Machine Learning",
        "[Deep Q-learning](../../concepts/deep-q-learning.md) · "
        "[Neural recombination](../../guides/neural-recombination.md) · "
        "[Decision module README](../../../farm/core/decision/README.md)",
    ),
    "data_visualization.md": (
        "Data & Visualization",
        "[Analysis modules](../../reference/analysis/modules/README.md) · "
        "[Data API](../../reference/data/data_api.md) · "
        "[Genetics analysis](../../guides/genetics-analysis.md)",
    ),
    "research_tools.md": (
        "Research Tools",
        "[Experiment runner](../../guides/experiment-runner.md) · "
        "[Experiments catalog](../../research/experiments-catalog.md) · "
        "[Deterministic simulations](../../guides/deterministic-simulations.md)",
    ),
    "data_system.md": (
        "Data System",
        "[Data API](../../reference/data/data_api.md) · "
        "[Database schema](../../reference/data/database_schema.md) · "
        "[Repositories](../../reference/data/repositories.md)",
    ),
    "spatial_indexing_performance.md": (
        "Spatial Indexing & Performance",
        "[Spatial indexing](../../concepts/spatial/spatial_indexing.md) · "
        "[Architecture](../../concepts/architecture.md) · "
        "[Benchmarks](../../../benchmarks/README.md)",
    ),
}


def main() -> None:
    for filename, (title, links) in STUBS.items():
        path = FEATURES / filename
        path.write_text(
            f"# {title}\n\n"
            f"> **Archived:** Legacy feature documentation replaced during the docs overhaul. "
            f"Use the links below for current material.\n\n"
            f"{links}\n",
            encoding="utf-8",
        )
        print(f"stubbed {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
