#!/usr/bin/env python3
"""Extract a version section from CHANGELOG.md for GitHub Releases."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_release_notes(changelog_text: str, version: str) -> str | None:
    """Return the markdown block for ``## [version]`` if present."""
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)"
    match = re.search(pattern, changelog_text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return None
    header = re.search(rf"^## \[{re.escape(version)}\][^\n]*", changelog_text, flags=re.MULTILINE)
    body = match.group(1).strip()
    return f"{header.group(0)}\n\n{body}".strip() if header else body


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
