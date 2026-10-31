# Continuous integration

How AgentFarm is tested and validated in GitHub Actions: what each workflow covers, what actually blocks a merge, and the known gaps with recommended fixes.

This page is the single reference for the CI suite. Keep it in sync when adding, removing, or re-scoping a workflow under [`.github/workflows/`](../../.github/workflows/).

## What CI is responsible for

AgentFarm is a simulation and analysis platform whose headline claims are **deterministic, reproducible simulations**, a **tested RL/evolution stack**, and an **installable Python package** with a REST/WebSocket API. The CI suite should therefore prove, on every change:

1. The fast test suite passes and the simulation core stays above the coverage gate.
2. Determinism holds: same seed, same output, within and across processes.
3. Reproducibility holds: recorded experiments re-run byte-identically in a pinned environment.
4. Evolution-optimizer behaviour stays inside its regression thresholds.
5. Code quality (lint, docs links) is maintained.
6. The package builds, installs, and imports cleanly on the supported Python versions and platforms.
7. Releases are cut from a verified `main` with matching version and changelog.

Items 1 through 5 are covered today to varying degrees; item 6 is not covered; item 7 is covered at release time only. See [Gaps and recommendations](#gaps-and-recommendations).

## Workflow inventory

All workflows live under [`.github/workflows/`](../../.github/workflows/). Runner is `ubuntu-latest` unless noted.

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| `tests` | `tests.yml` | push / PR to `main`, `dev` | Fast pytest suite with coverage gate, editor Jest tests, Ruff, docs link checks |
| `Deterministic Simulation Check` | `deterministic-simulation.yml` | push / PR to `main`, `dev`; manual | Component, cross-process, and CLI determinism checks |
| `Evolution Regression` | `evolution-regression.yml` | push / PR to `main` (path-scoped); weekly Mon 08:00 UTC | Threshold-based regression checks for the evolution optimizer |
| `Reproduce consensus experiment` | `reproduce-consensus.yml` | push / PR touching `farm/experiments/consensus/**`, `farm/provenance/**`, `run_experiment.py` | Run, notarize, re-run, and byte-compare the consensus experiment |
| `nightly-heavy-tests` | `nightly-heavy-tests.yml` | daily 09:00 UTC; manual | Full pytest suite including `slow` and `integration` markers |
| `Deploy Docs to GitHub Pages` | `pages.yml` | push to `main` touching `docs/**`; manual | Jekyll build and deploy of `docs/` |
| `Release` | `release.yml` | manual (`main` only) | Verify version/tag/changelog, run fast suite, build, create GitHub Release, optional PyPI publish |
| `Create release milestone` | `create-release-milestone.yml` | manual | Create or reuse a GitHub milestone and assign issues |
| `Copilot docs task after merge` | `copilot-docs-after-merge.yml` | push to `main` | Open an issue assigned to Copilot to sync documentation |
| `Copilot changelog task after merge` | `copilot-changelog-after-merge.yml` | push to `main` (ignores changelog/devlog-only pushes) | Open an issue assigned to Copilot to update `CHANGELOG.md` |

### `tests` jobs

| Job | What it runs | Blocking? |
|-----|--------------|-----------|
| `python` | `pytest -m "not slow and not integration" --ignore=tests/test_deterministic_pytest.py --cov=farm --cov-fail-under=90`, then two smoke tests (`test_simulation_repeatability_fixed_seed`, `test_simulation_id_generation`). Python 3.10. | Yes (required status check) |
| `js-ui` | `npm ci && npm test -- --runInBand` in `farm/editor/`. Node 20. | Yes (required status check) |
| `lint` | `ruff check . --output-format=github` with Ruff pinned to `0.15.18`. | **No** — `continue-on-error: true` |
| `docs-links` | `python scripts/check_docs_links.py` (internal markdown links). | No (not in required checks) |
| `docs-external-links` | `lychee` over `docs/`, `README.md`, `AGENTS.md`, `CONTRIBUTING.md` using `.lychee.toml`. | **No** — `continue-on-error: true` |

### `Deterministic Simulation Check` steps

Runs with `PYTHONHASHSEED=0` and a 40-minute job timeout. Each step has its own `timeout`:

1. `tests/test_component_determinism.py` (`-m "determinism and not slow and not integration"`)
2. `tests/test_deterministic_pytest.py::test_determinism_regression_critical` and `::test_different_seeds_produce_different_results`
3. `tests/test_cross_process_determinism.py`
4. `python tests/test_deterministic.py --environment testing --steps 50 --seed 42`
5. `python tests/test_deterministic.py --environment testing --steps 100 --seed 123`

On failure, `simulations/` is uploaded as an artifact.

### `nightly-heavy-tests`

Runs `pytest -m ""` (every marker, including `slow` and `integration`) with the same 90% coverage gate, then runs `tests/test_deterministic_pytest.py` again with `-m integration`. Recent runs take 10 to 30 minutes and report roughly 7,200 passed and 18 skipped (CUDA- and optional-dependency-gated tests).

### `Reproduce consensus experiment`

Installs from `requirements-ci-lock.txt` (pinned `numpy`, `pandas`, `matplotlib`, `scipy`) plus `pip install -e . --no-deps` and `farm-notary@v0.1.0` on Python 3.12, then:

1. `run_experiment.py` with a fixed seed into `/tmp/consensus-run`
2. `run_experiment.py verify-report` (derived artifacts follow from `trials.csv`)
3. `farm.provenance.notary.notarize(...)`
4. `farm-notary reproduce` (byte-compare every artifact)
5. `farm-notary verify` (manifest integrity)

This is the machine check behind the "bitwise reproducible from the committed seed in a pinned environment" claim. See [Farm Notary](farm-notary.md).

## Merge gating

Branch protection is enforced by the repository ruleset **Protect main and dev** on `refs/heads/main` and `refs/heads/dev`:

- Deletions and non-fast-forward pushes are blocked.
- Required status checks: **`python`** and **`js-ui`** only. `strict_required_status_checks_policy` is `false` (branches do not need to be up to date with the base before merging).

Everything else, including `Deterministic Simulation Check`, `lint`, `docs-links`, `Evolution Regression`, and `Reproduce consensus experiment`, can fail without blocking a merge.

## Test selection and markers

`pytest.ini` sets `addopts = -m "not slow and not integration"` with `--strict-markers` and `xfail_strict = true`. Markers in use:

| Marker | Default suite | Where it runs in CI |
|--------|---------------|---------------------|
| `unit`, `ml`, `db`, `analysis` | Yes | `tests` → `python`; nightly |
| `determinism`, `determinism_regression` | Yes (unless also `slow`/`integration`) | `Deterministic Simulation Check`; `tests` smoke step; nightly |
| `slow` | No | nightly only |
| `integration` | No | nightly; `Deterministic Simulation Check` (explicit module) |
| `evolution_regression` | Yes | `Evolution Regression`; nightly |

Local equivalents are listed in [AGENTS.md](../../AGENTS.md) and [Development setup](development-setup.md).

## Coverage gate

`--cov-fail-under=90` is enforced in `tests` and `nightly-heavy-tests`. The denominator is controlled by `.coveragerc`, which omits roughly 60 modules: matplotlib/plot pipelines, Tk GUI code, CUDA-only paths, optional-dependency wrappers (`tianshou`, `manim`, FAISS/ViT), and standalone analysis scripts. The gate therefore measures the simulation core, database, runners, API, and config layers rather than the whole package. The report is `term-missing` only; no coverage artifact or per-PR diff is produced.

## Environment pinning

| Workflow | Python | Dependency source |
|----------|--------|-------------------|
| `tests`, `nightly-heavy-tests`, `Deterministic Simulation Check`, `Evolution Regression`, `Release` | 3.10 | `requirements.txt` (unpinned lower bounds) |
| `Reproduce consensus experiment` | 3.12 | `requirements-ci-lock.txt` (4 pins) + `--no-deps` |

GitHub Actions are SHA-pinned in every workflow except `reproduce-consensus.yml`, which uses floating `@v4` / `@v5` tags. Pinned major versions are not consistent across files (`tests.yml` is on `checkout@v7` / `setup-python@v6`; others are on `v4` / `v5`).

## Gaps and recommendations

Ordered by impact. Each item is scoped to workflow or packaging-config changes; none require runtime code changes except clearing the Ruff baseline.

### 1. Lint is non-blocking and currently failing

`lint` runs with `continue-on-error: true`, so Ruff failures show as a red job inside a green workflow. The PR template asks contributors to run `ruff check .`, but CI does not enforce it.

**Recommendation:** clear the Ruff baseline once, remove `continue-on-error`, add `ruff format --check`, and add `lint` to the required status checks. If the baseline is too large to clear in one change, make a diff-scoped check blocking (`ruff check $(git diff --name-only origin/dev...HEAD -- '*.py')`) and keep the full-repo run informational.

### 2. Required checks are too narrow

Only `python` and `js-ui` gate merges. Determinism is the library's headline property and `Deterministic Simulation Check` can fail without blocking.

**Recommendation:** add `deterministic-test`, `lint`, and `docs-links` to the ruleset's required status checks.

### 3. Single Python version, single OS, unpinned dependencies

Every job runs on Python 3.10 on Ubuntu (3.12 for the consensus job). The README advertises Python 3.8+, Ruff targets `py38`, and `pyproject.toml` has no `requires-python`. `requirements.txt` includes `pywin32` and the docs give Windows instructions, but no job runs on Windows or macOS. The main suite installs unpinned dependencies, so a new `torch`, `numpy`, `pydantic`, or `tianshou` release can break `dev` with no code change.

**Recommendation:**

- Matrix `{3.10, 3.12}` × `ubuntu-latest` as required legs; add `3.13`, `windows-latest`, and `macos-latest` as `continue-on-error` informational legs.
- Generate a full lock file (`pip-compile` or `uv pip compile`) for the required legs.
- Add a weekly "latest dependencies" job that installs from the unpinned `requirements.txt` so dependency breakage is detected and attributed separately from code changes.
- Set `requires-python` in `pyproject.toml` and update the README to match what CI actually tests.

### 4. Packaging is never validated

`pyproject.toml` declares no `[project] dependencies`; the built wheel's metadata lists only the `notary` extra. A `pip install agentfarm` from a wheel installs zero runtime dependencies. CI never builds a wheel except in `Release`, never runs `twine check`, and never installs the wheel into a clean environment.

**Recommendation:** add a `package` job on PRs that runs `python -m build`, `twine check dist/*`, installs the wheel into a fresh venv, imports `farm`, and runs a short simulation (`run_simulation.py --environment testing --steps 10`). Mirror runtime dependencies from `requirements.txt` into `[project] dependencies`.

### 5. Evolution regression never runs on contributor PRs

`evolution-regression.yml` triggers on `main` only, but [CONTRIBUTING.md](../../CONTRIBUTING.md) directs all PRs to `dev`. The threshold checks therefore run weekly on `dev` and on `dev → main` merges, never on the PR that introduces a regression. Its path filter also omits `farm/core/decision/**`.

**Recommendation:** add `dev` to both `push.branches` and `pull_request.branches`; add `farm/core/decision/**` to the path filters.

### 6. No type checking

The codebase relies on Pydantic models, `typing.Protocol` interfaces (`farm/core/interfaces.py`), and `TYPE_CHECKING` blocks, but no `mypy` or `pyright` job exists.

**Recommendation:** add an informational type-check job scoped to `farm/core`, `farm/api`, and `farm/config`; promote to blocking once the baseline is clean.

### 7. No dependency or security scanning

There is no `.github/dependabot.yml`, no CodeQL, no `pip-audit`, and no `npm audit`. The `overrides: js-yaml` entry in `farm/editor/package.json` indicates a vulnerability was patched by hand.

**Recommendation:** enable Dependabot for the `pip`, `npm`, and `github-actions` ecosystems (the latter also normalizes action pinning), add a `pip-audit` step to `tests`, and enable CodeQL default setup.

### 8. Workflow mechanics

- No `concurrency` group on `tests.yml` or `deterministic-simulation.yml`; back-to-back pushes to a PR run in parallel. Add `concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true }`.
- No `timeout-minutes` on `tests.yml` or `nightly-heavy-tests.yml` (GitHub default is 6 hours).
- `nightly-heavy-tests` runs `tests/test_deterministic_pytest.py` twice (once via `-m ""`, once explicitly).
- Scheduled failures have no notification path. Add a `failure()` step that opens or updates a tracking issue.
- Upload `coverage.xml` as an artifact and consider a patch-coverage comment (`diff-cover` or Codecov) so reviewers can see what a PR did to coverage.
- `pages.yml` builds Jekyll only on `main`; a broken docs build on `dev` is discovered at release time. Run the build step (without deploy) on PRs touching `docs/**`.
- `deterministic-simulation.yml` lists `setup.py` in `cache-dependency-path`; the file does not exist.

### 9. Surfaces with no CI coverage

- `benchmarks/` is never executed. `tests/benchmarks/test_spatial_benchmark_verified_artifact.py` checks committed artifacts, but nothing regenerates or trends them. A nightly benchmark smoke with timing artifacts would catch performance regressions in the spatial index.
- The FastAPI server is exercised through `TestClient` only. Add a smoke step that starts `uvicorn farm.api.server:app`, polls a health endpoint, and shuts down, to catch import-time and lifespan failures that `TestClient` masks.
- `farm/editor/` has one Jest file (`__tests__/explorer.test.js`) and no JS lint or build step.
- `scripts/` has roughly 50 CLI entry points with 9 test files under `tests/scripts/`; the `Makefile` smoke target `crossover-search-smoke` is not run anywhere.

## Suggested implementation order

1. Make `lint` blocking; add `deterministic-test`, `lint`, and `docs-links` to required checks.
2. Add `dev` to `evolution-regression.yml` triggers and broaden its path filter.
3. Add a `package` job and fix `pyproject.toml` dependencies / `requires-python`.
4. Lock the main test environment; add a weekly unpinned-dependencies job.
5. Python version and OS matrix.
6. Dependabot, `pip-audit`, CodeQL.
7. `concurrency` and `timeout-minutes` on all PR workflows; nightly failure issue.
8. Type checking, coverage artifact, Jekyll build on docs PRs, API startup smoke, benchmark smoke.

## Related

- [Development setup](development-setup.md) — running the same checks locally
- [Release process](../RELEASE.md) — pre-release CI checklist and the `Release` workflow
- [Deterministic simulations](deterministic-simulations.md) — what the determinism checks guarantee
- [Farm Notary](farm-notary.md) — the reproduction and notarization tooling behind the consensus workflow
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — branching model and PR checklist
