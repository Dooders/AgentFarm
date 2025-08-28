"""Helper script to create a new research project with template files."""

import argparse
from typing import List, Optional

from research.research import ResearchProject

EXAMPLE_HYPOTHESIS = """# Research Hypotheses

## Primary Hypothesis
Describe your main research hypothesis here.

## Secondary Hypotheses
1. First secondary hypothesis
2. Second secondary hypothesis

## Background
Provide relevant background information.

## Methodology Overview
Describe your general approach.
"""

EXAMPLE_PROTOCOL = """# {name} Protocol

## Purpose
Describe the purpose of this protocol.

## Steps
1. First step
2. Second step
3. Third step

## Validation Criteria
- Criterion 1
- Criterion 2

## Notes
Additional notes and considerations.
"""


def create_research_template(
    name: str, description: str = "", tags: Optional[List[str]] = None
):
    """Create a new research project with template files."""
    # Create project
    project = ResearchProject(name, description, tags=tags)

    # Add example hypothesis
    with open(project.project_path / "hypothesis.md", "w") as f:
        f.write(EXAMPLE_HYPOTHESIS)

    # Add example protocols
    for protocol_type in ["validation", "analysis"]:
        content = EXAMPLE_PROTOCOL.format(name=protocol_type.title())
        project.add_protocol(protocol_type, content, protocol_type)

    # Add example literature reference
    project.add_literature_reference(
        "Example Paper Title",
        ["Author One", "Author Two"],
        2023,
        "example2023",
        notes="Example paper demonstrating reference format.",
    )

    print(
        f"""
Research project '{name}' created with template files:
- hypothesis.md
- protocols/validation.md
- protocols/analysis.md
- literature/bibliography.bib
- literature/papers/example2023_notes.md

Next steps:
1. Edit hypothesis.md to define your research questions
2. Customize protocols for your needs
3. Add relevant literature references
4. Create your first experiment
    """
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create research project from template"
    )
    parser.add_argument("name", help="Name of the research project")
    parser.add_argument("--description", default="")
    parser.add_argument("--tags", nargs="+", default=[])

    args = parser.parse_args()
    create_research_template(args.name, args.description, args.tags)


if __name__ == "__main__":
    main()
