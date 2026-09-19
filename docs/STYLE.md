# Documentation style guide

Conventions for AgentFarm docs under `docs/`.

## Layout

| Tier | Directory | Purpose |
|------|-----------|---------|
| Tutorials | `getting-started/` | First-run paths (install, first simulation) |
| How-to | `guides/` | Task-oriented workflows |
| Concepts | `concepts/` | Architecture and subsystem explanations |
| Reference | `reference/` | API, config, data schema, lookup |
| Research | `research/` | Devlog, experiment catalog, and whole-experiment writeups |
| Design | `design/` | RFCs and design proposals |
| Archive | `archive/` | Deprecated or superseded material |

## Experiment writeups

Whole-experiment GitHub Pages narratives live in `research/writeups/`
(`layout: experiment`). One page per experiment; variant runs are `##`
sections with `{#id}` matching the front-matter `variants` list. The
[writeups index](research/writeups/index.md) auto-lists those pages. Keep
runner/CLI details on the [experiments catalog](research/experiments-catalog.md);
keep dated field notes in `research/devlog/`.

## Filenames

- Use **kebab-case** for new pages: `experiment-runner.md`, not `experiment_runner.md`.
- One `#` title per page; match the title to the H1.

## Links

- Prefer relative links between docs: `[Architecture](concepts/architecture.md)`.
- After moving pages, run `python scripts/check_docs_links.py`.
- Legacy GitHub Pages URLs are preserved via redirect stubs — regenerate with `python scripts/generate_doc_redirects.py` after changing `docs/redirects.yml`.

## Redirects

- Manifest: `docs/redirects.yml`
- Generator: `scripts/generate_doc_redirects.py`
- Do not hand-edit generated stubs under old paths (e.g. `docs/devlog/`, `docs/api_reference.md`).

## Status labels (design RFCs)

Use in `docs/design/README.md`: **Accepted**, **Proposed**, **Reference**.

## CI

- `docs-links` job: internal markdown link check
- `docs-external-links` job: lychee on `docs/` and root entry points (see `.lychee.toml`)
