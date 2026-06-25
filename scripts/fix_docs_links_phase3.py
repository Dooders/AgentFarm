#!/usr/bin/env python3
"""Fix common broken relative links after docs Phase 2/3 moves."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

REPLACEMENTS: list[tuple[str, str]] = [
    # devlog depth fixes (file under docs/research/devlog/)
    ("](../design/", "](../../design/"),
    ("](../glossary.md", "](../../reference/glossary.md"),
    ("](../research/experiments/", "](../experiments/"),
    ("](../concepts/", "](../../concepts/"),
    ("](../reference/", "](../../reference/"),
    ("](../guides/", "](../../guides/"),
    ("](../getting-started/", "](../../getting-started/"),
    # concepts cross-links
    ("](perception_system_design.md", "](perception-system-design.md"),
    ("](dynamic_channel_system.md", "](dynamic-channel-system.md"),
    ("](observation_channels.md", "](observation-channels.md"),
    ("](memmap_optimization.md", "](memmap-optimization.md"),
    ("](api_reference.md", "](../reference/api-reference.md"),
    ("](reference/config/", "](../reference/config/"),
    ("](genetics_analysis.md", "](../guides/genetics-analysis.md"),
    ("](experiments.md", "](../research/experiments-catalog.md"),
    ("](logging_guide.md", "](../guides/logging.md"),
    ("](electron/config_explorer_architecture.md", "](../reference/electron/config_explorer_architecture.md"),
    ("](../distillation_soft_label_comparison.md", "](../research/distillation-soft-label-comparison.md"),
    ("](../perception_system_design.md", "](../perception-system-design.md"),
    ("](../redis_agent_memory.md", "](../redis-agent-memory.md"),
    ("](spatial/spatial_indexing.md", "](spatial_indexing.md"),
    ("](../API_REFERENCE.md", "](../api/API_REFERENCE.md"),
    ("](../QUICK_REFERENCE.md", "](../api-reference.md"),
    ("](../INDEX.md", "](../api-reference.md"),
    ("](reference/analysis/modules/", "](../reference/analysis/modules/"),
    ("](reference/config/", "](../reference/config/"),
    ("](reference/data/", "](../reference/data/"),
    ("](research/experiments/intrinsic_evolution/", "](../research/experiments/intrinsic_evolution/"),
    ("](../farm/core/initial_diversity.py", "](../../../farm/core/initial_diversity.py"),
]


def main() -> None:
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"fixed {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
