#!/usr/bin/env python3
"""Verify internal markdown links in documentation resolve to markdown files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    REPO / "docs",
    REPO / "README.md",
    REPO / "AGENTS.md",
    REPO / "CONTRIBUTING.md",
]

LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Skip links to code, assets, CI, or non-doc paths
SKIP_SUBSTRINGS = (
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".png",
    ".gif",
    ".jpg",
    ".csv",
    ".db",
    ".html",
    ".js",
    ".ts",
    ".tsx",
    ".toml",
    ".github/",
    "/farm/",
    "../farm/",
    "../../farm/",
    "../../../farm/",
    "../../scripts/",
    "../../../scripts/",
    "RepositoryProtocol",
    "LICENSE",
    ".ipynb",
    "../../../experiments/",
    "benchmarks/results/",
)


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
    return files


def is_checkable(target: str) -> bool:
    if not target or target.startswith("#"):
        return False
    if target.startswith(("http://", "https://", "mailto:", "ftp://")):
        return False
    if target.startswith("/"):
        return False
    if any(part in target for part in SKIP_SUBSTRINGS):
        return False
    return True


def resolve_link(source: Path, target: str) -> Path | None:
    clean = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean:
        return source
    candidate = (source.parent / clean).resolve()
    if candidate.is_file() and candidate.suffix == ".md":
        return candidate
    if candidate.with_suffix(".md").is_file():
        return candidate.with_suffix(".md")
    if candidate.is_dir() and (candidate / "README.md").is_file():
        return candidate / "README.md"
    if candidate.is_dir() and (candidate / "index.md").is_file():
        return candidate / "index.md"
    return None


def main() -> int:
    errors: list[str] = []
    md_files = iter_markdown_files()
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{md_file.relative_to(REPO)}: read failed ({exc})")
            continue
        for match in LINK_PATTERN.finditer(text):
            raw = match.group(1).strip()
            if not is_checkable(raw):
                continue
            if resolve_link(md_file, raw) is None:
                errors.append(f"{md_file.relative_to(REPO)}: {raw}")
    if errors:
        print("Broken internal markdown links:", file=sys.stderr)
        for err in sorted(set(errors)):
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"OK: checked markdown links in {len(md_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
