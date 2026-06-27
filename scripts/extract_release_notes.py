#!/usr/bin/env python3
"""Extract a version section from CHANGELOG.md for GitHub Releases."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_release_notes(changelog_text: str, version: str) -> str | None:
    """Return the markdown block for ``## [version]`` if present."""
    header_pattern = re.compile(rf"^## \[{re.escape(version)}\][^\n]*$")
    boundary_pattern = re.compile(r"^## \[|^## Entries$")
    lines = changelog_text.splitlines()
    entries_idx = next((i for i, line in enumerate(lines) if line == "## Entries"), len(lines))

    header_idx: int | None = None
    in_fence = False
    for i, line in enumerate(lines):
        if i >= entries_idx:
            break
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and header_pattern.match(line):
            header_idx = i

    if header_idx is None:
        return None

    body_lines: list[str] = []
    in_fence = False
    for line in lines[header_idx + 1 :]:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        if not in_fence and boundary_pattern.match(line):
            break
        body_lines.append(line)

    header_line = lines[header_idx]
    body = "\n".join(body_lines).strip()
    return f"{header_line}\n\n{body}".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release version without a leading v (e.g. 0.2.0)")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to CHANGELOG.md (default: CHANGELOG.md)",
    )
    args = parser.parse_args(argv)

    changelog_path = args.changelog
    if not changelog_path.is_file():
        print(f"Changelog not found: {changelog_path}", file=sys.stderr)
        return 1

    notes = extract_release_notes(changelog_path.read_text(encoding="utf-8"), args.version)
    if notes is None:
        print(
            f"No '## [{args.version}]' section found in {changelog_path}. "
            "Add a version header before running the release workflow.",
            file=sys.stderr,
        )
        return 1

    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
