# Releasing AgentFarm

This guide is for **maintainers** cutting a new release. Contributors should target `dev`; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Overview

AgentFarm uses [Semantic Versioning](https://semver.org/) while the project is in active development:

| Bump | When |
|------|------|
| **PATCH** (`0.1.0` → `0.1.1`) | Bug fixes only, no API or config changes |
| **MINOR** (`0.1.0` → `0.2.0`) | New features, backward-compatible additions |
| **MAJOR** (`0.x` → `1.0.0`) | First stable API contract; breaking changes after `1.0.0` require a major bump |

While on `0.x`, breaking changes are allowed but must be documented under a `### Breaking` heading in `CHANGELOG.md`.

**Branches:**

- `dev` — integration branch for all contributor PRs
- `main` — stable release branch; updated only via maintainer release PRs

## Release cadence

Choose one approach and stick to it so contributors know what to expect:

- **Milestone-based** (recommended today): cut a release when a feature set or research milestone is ready.
- **Time-based**: e.g. every 2–4 weeks once contributor velocity is steady.

Track planned work with [GitHub Milestones](https://github.com/Dooders/AgentFarm/milestones) named after the target version (`0.2.0`, etc.).

Each release should have a scope document under [`docs/milestones/`](milestones/) (for example [`0.2.0.md`](milestones/0.2.0.md)). Create the GitHub milestone with **Actions → Create release milestone** (`.github/workflows/create-release-milestone.yml`) or locally:

```bash
gh workflow run create-release-milestone.yml \
  -f version=0.2.0 \
  -f assign_issues=954,944,930,952,953
```

## Pre-release checklist

Before opening a release PR:

1. **`dev` CI is green** — the [`tests` workflow](../.github/workflows/tests.yml) must pass on the latest `dev` commit.
2. **Heavy regressions** — for releases touching simulation core, RL, or evolution code, confirm the [nightly](../.github/workflows/nightly-heavy-tests.yml), [deterministic-simulation](../.github/workflows/deterministic-simulation.yml), and [evolution-regression](../.github/workflows/evolution-regression.yml) workflows are green (run manually or wait for the latest scheduled run).
3. **Changelog is ready** — roll up recent dated entries into a version section (see below).
4. **Version is bumped** — update `farm/_version.py` (the single source of truth).

## Step-by-step

### 1. Bump the version

Edit `farm/_version.py`:

```python
__version__ = "0.2.0"
```

`pyproject.toml` reads this value automatically. Do not hard-code versions elsewhere.

Verify locally:

```bash
source venv/bin/activate
pip install -e .
python -c "from farm import __version__; print(__version__)"
python -c "from importlib.metadata import version; print(version('agentfarm'))"
```

Both commands must print the same version.

### 2. Update `CHANGELOG.md`

Add a version header **above** the dated entries:

```markdown
## [0.2.0] - 2026-06-25

### Added
- ...

### Fixed
- ...

### Breaking
- ...
```

Keep the existing dated entries below for history, or consolidate them into the version section and remove duplicates.

The release workflow extracts notes from the `## [X.Y.Z]` section using `scripts/extract_release_notes.py`.

### 3. Open a release PR (`dev` → `main`)

```bash
git checkout dev
git pull origin dev
git checkout -b release/0.2.0
# commit version + changelog changes
git push -u origin release/0.2.0
```

Open a PR **from your release branch into `main`** (not into `dev`). Title: `Release v0.2.0`.

The PR should contain only:

- `farm/_version.py` version bump
- `CHANGELOG.md` version section
- Any last-minute release-only doc tweaks

Get review, ensure CI passes, then merge to `main`.

### 4. Publish the GitHub Release

After the release PR merges to `main`:

1. Go to **Actions → Release → Run workflow**.
2. Enter the version **without** a leading `v` (e.g. `0.2.0`).
3. Leave **Publish to PyPI** unchecked unless PyPI trusted publishing is configured (see below).

The workflow will:

- Verify `farm/_version.py` matches the input version
- Confirm the tag `vX.Y.Z` does not already exist
- Run the fast pytest suite (`-m "not slow and not integration"`, excluding `tests/test_deterministic_pytest.py`)
- Build sdist and wheel artifacts
- Extract release notes from `CHANGELOG.md`
- Create tag `vX.Y.Z` and a GitHub Release with attached artifacts

### 5. Sync `main` back into `dev`

After releasing, merge `main` into `dev` so the version bump and changelog land on the integration branch:

```bash
git checkout dev
git pull origin dev
git merge origin/main
git push origin dev
```

Resolve conflicts if any, then delete the release branch.

### 6. Announce (optional)

For releases with user-visible changes (config schema, DB format, CLI):

- Post a short note in [GitHub Discussions](https://github.com/Dooders/AgentFarm/discussions)
- Link the GitHub Release and call out any `### Breaking` items

## Stability policy

| Area | Stability |
|------|-----------|
| Documented CLI entrypoints (`run_simulation.py`, `farm.core.cli`) | Stable within a minor release |
| `SimulationConfig` keys documented in `docs/config/` | Stable within a minor release |
| Simulation database schema | Breaking changes require changelog + migration note |
| `farm/core/decision/`, evolvable genomes, inheritance payloads | **Experimental** — may change between minors on `0.x` |
| Internal module layout under `farm/` | Not guaranteed stable until `1.0.0` |

## Labels for contributors

Maintainers may use these GitHub labels when triaging:

| Label | Meaning |
|-------|---------|
| `changelog-worthy` | Should appear in the next release notes |
| `breaking-change` | Requires `### Breaking` in changelog |
| `release-blocker` | Must be fixed before the next release |

## PyPI publishing (optional)

Releases are distributed via **GitHub Releases** by default. To also publish to PyPI:

1. Create a PyPI project for `agentfarm`.
2. Configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) for this repository's `release.yml` workflow.
3. Re-run the release workflow with **Publish to PyPI** enabled.

Until then, users install from source or from release artifacts:

```bash
pip install git+https://github.com/Dooders/AgentFarm@v0.2.0
# or
pip install agentfarm-0.2.0-py3-none-any.whl
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Release workflow: version mismatch | Ensure `farm/_version.py` on `main` matches the workflow input |
| Release workflow: no changelog section | Add `## [X.Y.Z] - YYYY-MM-DD` to `CHANGELOG.md` and merge to `main` |
| Tag already exists | Delete the tag locally/remotely only if the release was mistaken; never reuse a published version number |
| Copilot changelog PR after merge | Expected on `main` merges; use `[skip-changelog]` in commit messages to skip |

## Related docs

- [CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow
- [CHANGELOG.md](../CHANGELOG.md) — release notes source
- [deployment.md](deployment.md) — running AgentFarm in production
