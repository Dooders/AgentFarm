# Deterministic Simulation Check

This GitHub Action verifies that the AgentFarm simulation produces deterministic results when run with specific configuration parameters.

## Purpose

The workflow ensures that:

- The simulation runs to completion (all 1000 steps)
- Final agent counts match expected values (32 agents)
- Simulation timing is within expected bounds
- Any changes to the codebase don't break deterministic behavior

**Determinism Note**: Results may vary across different OS/BLAS/CUDA versions. This check validates consistency within the same environment. Cross-platform determinism requires additional validation.

## Configuration

The workflow is configured to check for these expected results:

- **No Early Termination**: The simulation should run all 1000 steps without terminating early
- **Final Agent Count**: Should be 32 (agents survive through the full simulation)
- **Simulation Time**: Should be between 60-600 seconds

## Expected Behavior

Based on your deterministic simulation results:

- Simulation runs with `--environment development --profile simulation --steps 1000 --seed 42`
- Should run to completion (all 1000 steps)
- Final agent count should be 32
- Results should be identical across runs with the same seed

## Customization

To modify the expected values, edit the environment variables in the "Verify deterministic results" step:

```yaml
env:
  EXPECTED_EARLY_TERMINATION: 0  # 1 if simulation terminates early, 0 if runs full 1000 steps
  EXPECTED_AGENT_COUNT: 32        # Final agent count from deterministic run with seed 42
  EXPECTED_MIN_TIME: 60.0         # Minimum expected simulation time in seconds
  EXPECTED_MAX_TIME: 600.0        # Maximum expected simulation time in seconds (10 minutes)
```

## Failure Handling

- **Errors**: Will fail the workflow if early termination occurs unexpectedly, final agent count is incorrect, or timing is out of bounds

## Simulation Parameters

The workflow runs the simulation with these parameters:

- Environment: `development`
- Profile: `simulation`
- Steps: `1000`
- Seed: `42` (for deterministic results)
- In-memory database (for faster execution)
- No persistence to disk

## Artifacts

The workflow uploads simulation results as artifacts, including:

- Database files (`simulations/`)
- Log files (`logs/`)

These can be downloaded for further analysis if the workflow fails.

---

## CI Test Suite (`tests.yml`)

Runs the project test suites on pushes and pull requests to `main`.

### What it runs

- Python tests
  - Sets up Python 3.10
  - Caches pip dependencies for faster builds
  - Installs dependencies from `requirements.txt`
  - Runs `pytest` with coverage reporting (`--cov=farm --cov-report=term-missing`)

- JavaScript UI tests
  - Sets up Node.js 20
  - Caches npm dependencies for faster builds
  - Working directory: `farm/editor`
  - Installs dependencies (`npm ci` with fallback to `npm install`)
  - Runs Jest tests (`npm test -- --runInBand`)

### Triggers

- `push` to `main`
- `pull_request` targeting `main`

### How to run locally

```bash
# Python
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q --cov=farm --cov-report=term-missing

# JS UI
cd farm/editor
npm install
npm test -- --runInBand
```

### Customize

- Change Python version in `actions/setup-python@v5` (`python-version`)
- Change Node version in `actions/setup-node@v4` (`node-version`)
- Adjust coverage flags in the pytest step
- Add caching (pip/npm) to speed up CI if needed
