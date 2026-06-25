#!/usr/bin/env python3
"""Generate Jekyll redirect stub pages for legacy documentation URLs."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
MANIFEST = DOCS / "redirects.yml"

DEVLOG_SKIP = {"index.md"}


def load_manifest() -> tuple[dict[str, str], dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return data.get("files", {}), data.get("directories", {})


def stub_content(redirect_to: str) -> str:
    target = redirect_to.strip("/").split("#")[0]
    return (
        "---\n"
        "layout: redirect\n"
        f"redirect_to: /{target}/\n"
        "sitemap: false\n"
        "---\n"
    )


def write_stub(path: Path, redirect_to: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stub_content(redirect_to), encoding="utf-8")


def devlog_redirects() -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    devlog_dir = DOCS / "research" / "devlog"
    if not devlog_dir.is_dir():
        return mapping
    for post in sorted(devlog_dir.glob("*.md")):
        if post.name in DEVLOG_SKIP:
            continue
        slug = post.stem
        mapping[DOCS / "devlog" / post.name] = f"research/devlog/{slug}"
    mapping[DOCS / "devlog" / "index.md"] = "research/devlog"
    return mapping


def experiments_redirects() -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    root = DOCS / "research" / "experiments"
    if not root.is_dir():
        return mapping
    for md in sorted(root.rglob("*.md")):
        rel = md.relative_to(root)
        mapping[DOCS / "experiments" / rel] = f"research/experiments/{rel.with_suffix('').as_posix()}"
    return mapping


def mirror_redirects(old_prefix: str, new_prefix: str) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    src = DOCS / new_prefix
    if not src.is_dir():
        return mapping
    for md in sorted(src.rglob("*.md")):
        rel = md.relative_to(src)
        old_path = DOCS / old_prefix / rel
        new_target = f"{new_prefix}/{rel.with_suffix('').as_posix()}"
        mapping[old_path] = new_target
    return mapping


MIRROR_PAIRS = (
    ("config", "reference/config"),
    ("data", "reference/data"),
    ("api", "reference/api"),
    ("spatial", "concepts/spatial"),
    ("electron", "reference/electron"),
    ("features", "archive/features"),
    ("analysis", "reference/analysis"),
)


def main() -> None:
    file_map, dir_map = load_manifest()
    written = 0

    for old_rel, new_target in file_map.items():
        write_stub(DOCS / old_rel, new_target)
        written += 1

    for old_dir, new_target in dir_map.items():
        write_stub(DOCS / old_dir / "index.md", new_target)
        written += 1

    for stub_path, new_target in devlog_redirects().items():
        write_stub(stub_path, new_target)
        written += 1

    for stub_path, new_target in experiments_redirects().items():
        write_stub(stub_path, new_target)
        written += 1

    for old_prefix, new_prefix in MIRROR_PAIRS:
        for stub_path, new_target in mirror_redirects(old_prefix, new_prefix).items():
            write_stub(stub_path, new_target)
            written += 1

    print(f"Wrote {written} redirect stubs under docs/")


if __name__ == "__main__":
    main()
