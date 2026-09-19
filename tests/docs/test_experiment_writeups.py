"""Guardrails for GitHub Pages experiment writeups.

Each writeup is one whole experiment. Variant runs live as ``##`` sections
on that page, not as a stack of separate posts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_WRITEUPS = _REPO / "docs" / "research" / "writeups"
_INDEX = _WRITEUPS / "index.md"
_CATALOG = _REPO / "docs" / "research" / "experiments-catalog.md"
_CONFIG = _REPO / "docs" / "_config.yml"
_LAYOUT = _REPO / "docs" / "_layouts" / "experiment.html"

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_REQUIRED = ("layout", "title", "status", "updated", "category", "excerpt", "variants")
_STATUSES = frozenset({"complete", "implemented", "in-progress", "design"})


def _writeup_paths() -> list[Path]:
    return sorted(path for path in _WRITEUPS.glob("*.md") if path.name != "index.md")


def _parse(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    assert match, f"{path.name} is missing YAML front matter"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict), f"{path.name} front matter is not a mapping"
    return data, text[match.end() :]


@pytest.mark.unit
def test_writeup_layout_and_index_exist() -> None:
    assert _LAYOUT.is_file()
    assert _INDEX.is_file()
    index_meta, index_body = _parse(_INDEX)
    assert index_meta.get("layout") == "page"
    assert 'where: "layout", "experiment"' in index_body


@pytest.mark.unit
def test_nav_includes_writeups() -> None:
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    links = config.get("nav_links") or []
    match = next((link for link in links if link.get("title") == "Writeups"), None)
    assert match is not None, "docs/_config.yml nav_links is missing Writeups"
    assert match.get("url") == "/research/writeups/"


@pytest.mark.unit
def test_each_writeup_is_a_whole_experiment_with_variant_runs() -> None:
    paths = _writeup_paths()
    assert paths, "expected experiment writeups under docs/research/writeups/"
    catalog = _CATALOG.read_text(encoding="utf-8")

    for path in paths:
        meta, body = _parse(path)
        missing = [key for key in _REQUIRED if key not in meta]
        assert not missing, f"{path.name} missing front matter: {missing}"
        assert meta["layout"] == "experiment"
        assert meta["status"] in _STATUSES, f"{path.name} has unknown status {meta['status']!r}"
        variants = meta["variants"]
        assert isinstance(variants, list) and variants, f"{path.name} needs a non-empty variants list"

        ids = []
        for variant in variants:
            assert "id" in variant and "title" in variant, f"{path.name} variant is missing id/title"
            variant_id = str(variant["id"])
            ids.append(variant_id)
            heading_pattern = r"^## .+\{#" + re.escape(variant_id) + r"\}\s*$"
            heading = re.search(heading_pattern, body, re.MULTILINE)
            assert heading, (
                f"{path.name} variant {variant_id!r} must be an H2 with {{#{variant_id}}} "
                "so the run stays inside the writeup"
            )

        assert len(ids) == len(set(ids)), f"{path.name} has duplicate variant ids"
        assert f"writeups/{path.stem}.md" in catalog, f"catalog does not link to writeups/{path.stem}.md"
