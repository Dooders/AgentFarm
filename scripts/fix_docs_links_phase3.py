#!/usr/bin/env python3
"""Fix common broken relative links after docs Phase 2/3 moves."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

DEVLOG_REPLACEMENTS: list[tuple[str, str]] = [
    # depth fixes when the source file is under docs/research/devlog/
    ("](../design/", "](../../design/"),
    ("](../glossary.md", "](../../reference/glossary.md"),
    ("](../research/experiments/", "](../experiments/"),
    ("](../concepts/", "](../../concepts/"),
    ("](../reference/", "](../../reference/"),
    ("](../guides/", "](../../guides/"),
    ("](../getting-started/", "](../../getting-started/"),
]

SPATIAL_REPLACEMENTS: list[tuple[str, str]] = [
    # same-directory link when the source file is under docs/concepts/spatial/
    ("](spatial/spatial_indexing.md", "](spatial_indexing.md"),
]

REPLACEMENTS: list[tuple[str, str]] = [
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
    ("](../API_REFERENCE.md", "](../api/API_REFERENCE.md"),
    ("](../QUICK_REFERENCE.md", "](../api-reference.md"),
    ("](../INDEX.md", "](../api-reference.md"),
    ("](reference/analysis/modules/", "](../reference/analysis/modules/"),
    ("](reference/config/", "](../reference/config/"),
    ("](reference/data/", "](../reference/data/"),
    ("](research/experiments/intrinsic_evolution/", "](../research/experiments/intrinsic_evolution/"),
    ("](../farm/core/initial_diversity.py", "](../../../farm/core/initial_diversity.py"),
]


def replacements_for(path: Path) -> list[tuple[str, str]]:
    rel = path.relative_to(DOCS)
    parts = rel.parts
    replacements = list(REPLACEMENTS)
    if len(parts) >= 2 and parts[:2] == ("research", "devlog"):
        replacements = DEVLOG_REPLACEMENTS + replacements
    if len(parts) >= 2 and parts[:2] == ("concepts", "spatial"):
        replacements = replacements + SPATIAL_REPLACEMENTS
    return replacements


def main() -> None:
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in replacements_for(path):
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"fixed {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
