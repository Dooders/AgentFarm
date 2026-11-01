# Continuous integration

How AgentFarm is tested and validated in GitHub Actions: what each workflow covers, what actually blocks a merge, and the remaining gaps.

This page is the single reference for the CI suite. Keep it in sync when adding, removing, or re-scoping a workflow under [`.github/workflows/`](../../.github/workflows/).

## What CI is responsible for

AgentFarm is a simulation and analysis platform whose headline claims are **deterministic, reproducible simulations**, a **tested RL/evolution stack**, and an **installable Python package** with a REST/WebSocket API. The CI suite proves, on every change:

1. The fast test suite passes on the supported Python versions and the simulation core stays above the coverage gate.
2. Determinism holds: same seed, same output, within and across processes.
3. Reproducibility holds: recorded experiments re-run byte-identically in a pinned environment.
4. Evolution-optimizer behaviour stays inside its regression thresholds.
5. Code quality is maintained: Ruff passes, the dependency lock is in sync, internal docs links resolve.
6. The package builds, passes `twine check`, installs into a clean environment from the wheel alone, imports, and runs a simulation.
7. The API server starts as a real process and answers requests.
8. Known vulnerabilities in the locked dependency set are surfaced (and new ones block).
9. Releases are cut from a verified `main` with matching version and changelog.

## Workflow inventory

All workflows live under [`.github/workflows/`](../../.github/workflows/). Runner is `ubuntu-latest` unless noted. Every PR-triggered workflow has a `concurrency` group that cancels superseded runs on the same ref, and every job has a `timeout-minutes`.

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| `tests` | `tests.yml` | push / PR to `main`, `dev` | Python/OS matrix, packaging smoke, API smoke, editor Jest, Ruff + lock check, mypy, docs links |
| `Deterministic Simulation Check` | `deterministic-simulation.yml` | push / PR to `main`, `dev`; manual | Component, cross-process, and CLI determinism checks |
| `Evolution Regression` | `evolution-regression.yml` | push / PR to `main`, `dev` (path-scoped); weekly Mon 08:00 UTC; manual | Threshold-based regression checks for the evolution optimizer |
| `Reproduce consensus experiment` | `reproduce-consensus.yml` | push / PR touching `farm/experiments/consensus/**`, `farm/provenance/**`, `run_experiment.py` | Run, notarize, re-run, and byte-compare the consensus experiment |
| `security` | `security.yml` | push / PR touching dependency files; weekly Mon 06:00 UTC; manual | `pip-audit` on the lock (blocking), `npm audit` on the editor (informational) |
| `CodeQL` | `codeql.yml` | push / PR to `main`, `dev`; weekly Mon 05:00 UTC; manual | Static security analysis for Python and JavaScript |
| `nightly-heavy-tests` | `nightly-heavy-tests.yml` | daily 09:00 UTC; manual | Full pytest suite including `slow` and `integration`; benchmark smoke |
| `latest-deps` | `latest-deps.yml` | weekly Mon 07:00 UTC; manual | Fast suite against the *unpinned* `[project.dependencies]` (upstream-breakage canary) |
| `Deploy Docs to GitHub Pages` | `pages.yml` | push to `main` touching `docs/**`; PRs touching `docs/**` (build only); manual | Jekyll build and deploy of `docs/` |
| `Release` | `release.yml` | manual (`main` only) | Verify version/tag/changelog, run fast suite, build + `twine check`, create GitHub Release, optional PyPI publish |
| `Create release milestone` | `create-release-milestone.yml` | manual | Create or reuse a GitHub milestone and assign issues |
| `Copilot docs task after merge` | `copilot-docs-after-merge.yml` | push to `main` | Open an issue assigned to Copilot to sync documentation |
| `Copilot changelog task after merge` | `copilot-changelog-after-merge.yml` | push to `main` (ignores changelog/devlog-only pushes) | Open an issue assigned to Copilot to update `CHANGELOG.md` |

Dependabot (`.github/dependabot.yml`) opens weekly grouped minor/patch PRs for pip, the `farm/editor` npm toolchain, and GitHub Actions.

### `tests` jobs

| Job | What it runs | Blocking? |
|-----|--------------|-----------|
| `pytest (3.10, ubuntu-latest)`, `pytest (3.12, ubuntu-latest)` | `pytest -m "not slow and not integration" --ignore=tests/test_deterministic_pytest.py --cov=farm --cov-fail-under=90 --cov-report=xml`, two determinism/ID smoke tests, then `scripts/ci/api_smoke.sh` (starts `uvicorn farm.api.server:app`, probes `/openapi.json`, `/docs`, `/api/simulations`, `/api/analysis/modules`). Uploads `coverage.xml` as an artifact. | Yes, via `python` |
| `pytest (3.13, ubuntu-latest)`, `pytest (3.12, windows-latest)`, `pytest (3.12, macos-latest)` | Same suite without the coverage gate. `continue-on-error: true`. | No (informational) |
| `package` | `python -m build`, `twine check dist/*`, install the wheel into a fresh venv constrained by the lock, run `scripts/ci/package_smoke.py` from outside the checkout (imports the main subsystems, checks the bundled YAML config ships, runs a 5-step simulation). Uploads `dist/`. | Yes, via `python` |
| `lint` | `ruff check . --output-format=github` (Ruff `0.15.18`) and `make lock-check`. | Yes, via `python` |
| `docs-links` | `python scripts/check_docs_links.py` (internal markdown links). | Yes, via `python` |
| `python` | Gate job: `needs: [pytest, lint, docs-links, package]`, fails if any of them did not succeed. This is the status-check name the branch ruleset requires. | **Required status check** |
| `js-ui` | `npm ci && npm test -- --runInBand` in `farm/editor/`. Node 20. | **Required status check** |
| `typecheck` | `mypy` scoped by `[tool.mypy]` in `pyproject.toml` to `farm/core`, `farm/api`, `farm/config`. | No — `continue-on-error: true` until the baseline (~400 errors) is cleared |
| `docs-external-links` | `lychee` over `docs/`, `README.md`, `AGENTS.md`, `CONTRIBUTING.md` using `.lychee.toml`. | No — `continue-on-error: true` (external links flake) |

### `Deterministic Simulation Check` steps

Runs with `PYTHONHASHSEED=0` and a 40-minute job timeout. Each step has its own `timeout`:

1. `tests/test_component_determinism.py` (`-m "determinism and not slow and not integration"`)
2. `tests/test_deterministic_pytest.py::test_determinism_regression_critical` and `::test_different_seeds_produce_different_results`
3. `tests/test_cross_process_determinism.py`
4. `python tests/test_deterministic.py --environment testing --steps 50 --seed 42`
5. `python tests/test_deterministic.py --environment testing --steps 100 --seed 123`

On failure, `simulations/` is uploaded as an artifact.

### `nightly-heavy-tests`

- `python-heavy`: `pytest -m ""` (every marker, including `slow` and `integration`) with the same 90% coverage gate. Recent runs take 10 to 30 minutes and report roughly 7,200 passed and 18 skipped (CUDA- and optional-dependency-gated tests).
- `benchmark-smoke`: `python -m benchmarks.run_benchmarks --list`, then `--spec benchmarks/specs/ci_smoke.yaml` (a few-second `observation_flow` run). The results JSON is uploaded as an artifact so timings can be trended.

Both jobs open or comment on a `ci-failure` issue when they fail (see [Scheduled-failure reporting](#scheduled-failure-reporting)).

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
- Required status checks: **`python`** and **`js-ui`**. `strict_required_status_checks_policy` is `false` (branches do not need to be up to date with the base before merging).

Because `python` is a gate job, the required pytest legs, `lint`, `docs-links`, and `package` all block merges without a ruleset change. The following still do **not** block and should be added to the ruleset's required checks by a repository admin:

- `deterministic-test` (from `Deterministic Simulation Check`)
- `pip-audit (3.10)` and `pip-audit (3.12)` (from `security`, path-filtered so they only appear on dependency changes)

```bash
# Admin only. Adds deterministic-test to the required checks; keeps the existing ones.
gh api -X PUT repos/Dooders/AgentFarm/rulesets/18047834 --input - <<'EOF'
{
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": false,
      "do_not_enforce_on_create": false,
      "required_status_checks": [
        {"context": "python"},
        {"context": "js-ui"},
        {"context": "deterministic-test"}
      ]}}
  ]
}
EOF
```

## Test selection and markers

`pytest.ini` sets `addopts = -m "not slow and not integration"` with `--strict-markers` and `xfail_strict = true`. Markers in use:

| Marker | Default suite | Where it runs in CI |
|--------|---------------|---------------------|
| `unit`, `ml`, `db`, `analysis` | Yes | `tests` → `pytest` legs; nightly; `latest-deps` |
| `determinism`, `determinism_regression` | Yes (unless also `slow`/`integration`) | `Deterministic Simulation Check`; `tests` smoke step; nightly |
| `slow` | No | nightly only |
| `integration` | No | nightly; `Deterministic Simulation Check` (explicit module) |
| `evolution_regression` | Yes | `Evolution Regression`; nightly |

Local equivalents are listed in [AGENTS.md](../../AGENTS.md) and [Development setup](development-setup.md).

## Coverage gate

`--cov-fail-under=90` is enforced on the required `pytest` legs and in `nightly-heavy-tests`. The denominator is controlled by `.coveragerc`, which omits roughly 60 modules: matplotlib/plot pipelines, Tk GUI code, CUDA-only paths, optional-dependency wrappers (`tianshou`, `manim`, FAISS/ViT), and standalone analysis orchestration. The gate therefore measures the simulation core, database, runners, API, and config layers rather than the whole package. Each required leg uploads `coverage.xml` as the `coverage-<python>-<os>` artifact.

## Dependencies and environment pinning

- **Source of truth:** `pyproject.toml` `[project.dependencies]` (runtime) and the `dev` extra (test tooling). `requirements.txt` only contains `-e .[dev]`. Optional extras: `media` (moviepy, for `scripts/animate_simulation.py`) and `notary`.
- **Lock:** `requirements-lock.txt`, generated by `make lock` (`uv pip compile`, universal across platforms for Python >= 3.10). Without `--upgrade`, uv keeps existing pins, so the file only changes when `pyproject.toml` changes; `make lock-upgrade` moves every pin forward. `make lock-check` (run in the `lint` job) fails when the lock is stale.
- **Who installs what:**

| Workflow | Python | Dependency source |
|----------|--------|-------------------|
| `tests` (`pytest`, `package`, `typecheck`), `nightly-heavy-tests`, `Deterministic Simulation Check`, `Evolution Regression`, `Release` | 3.10 / 3.12 / 3.13 (matrix) or 3.10 | `requirements-lock.txt` + `pip install -e . --no-deps` |
| `latest-deps` | 3.12 | unpinned `pip install -e .[dev]` |
| `Reproduce consensus experiment` | 3.12 | `requirements-ci-lock.txt` (4 pins) + `--no-deps` |

- **Python floor:** `requires-python = ">=3.10"`. Pillow forks in the lock (12.x on 3.10, 13 on 3.11+) because Pillow 13 fixes advisories but needs 3.11.
- **Actions:** SHA-pinned in every workflow; Dependabot keeps them current.

## Security scanning

- `security.yml` → `pip-audit` runs against `requirements-lock.txt` on Python 3.10 and 3.12 with `--strict`. Advisories that cannot be fixed under current constraints are ignored explicitly, with the reason, in the workflow: the `protobuf` 3.19 advisories pinned by `tianshou 0.4.11`, and the Pillow 12.x advisories on the 3.10 leg (fixed in Pillow 13, which needs 3.11+). Remove an ignore as soon as its constraint lifts.
- `npm audit --audit-level=high` on `farm/editor` is informational; the Jest toolchain carries advisories whose fixes are major upgrades that Dependabot proposes separately.
- `codeql.yml` analyzes Python and JavaScript; findings appear under Security → Code scanning.

## Scheduled-failure reporting

`.github/actions/report-scheduled-failure` is a composite action that opens, or comments on, an open issue titled `Scheduled CI failure: <workflow>` labelled `ci-failure`. It is wired into `nightly-heavy-tests` (both jobs), `latest-deps`, and the scheduled run of `Evolution Regression`. Close the issue once the workflow is green again.

## Remaining gaps

Ordered by impact.

1. **Ruleset still lists only `python` and `js-ui`.** `deterministic-test` and `pip-audit` can fail without blocking. Requires admin access; see [Merge gating](#merge-gating).
2. **Type checking is informational.** The mypy baseline is roughly 400 errors across `farm/core`, `farm/api`, `farm/config`. Burn it down module by module, then drop `continue-on-error` from `typecheck`.
3. **`ruff format --check` is not enforced.** Roughly 70% of files would be reformatted; adopt in one formatting commit, then add the check to `lint`.
4. **Informational matrix legs have no history.** 3.13, Windows and macOS legs may fail silently behind `continue-on-error`; review them periodically and promote a leg to required once it is reliably green.
5. **Coverage is an artifact, not a PR comment.** `coverage.xml` is uploaded but no patch-coverage diff is posted; add `diff-cover` or Codecov if reviewers want it inline.
6. **Benchmark results are not trended.** The nightly smoke uploads results JSON but nothing compares runs over time. `benchmarks/run_benchmarks.py --compare` exists and could feed a scheduled report. The `spatial_comprehensive` spec in `benchmarks/specs/` references an experiment that is not registered with the runner.
7. **Editor coverage is thin.** One Jest file, no JS lint, no build step.
8. **`scripts/` has ~50 CLI entry points and 9 test files.** `scripts/ci/` is covered by the workflows; the research scripts are not.
9. **`tianshou<0.5` holds `protobuf` on 3.19.** Upgrading tianshou (and revisiting the NumPy constraint) would remove two `pip-audit` ignores.

## Related

- [Development setup](development-setup.md) — running the same checks locally
- [Release process](../RELEASE.md) — pre-release CI checklist and the `Release` workflow
- [Deterministic simulations](deterministic-simulations.md) — what the determinism checks guarantee
- [Farm Notary](farm-notary.md) — the reproduction and notarization tooling behind the consensus workflow
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — branching model and PR checklist
