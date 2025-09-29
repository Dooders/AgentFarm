# Deterministic Simulation Check

This GitHub Action verifies that the AgentFarm simulation produces deterministic results when run with specific configuration parameters.

## Purpose

The workflow ensures that:
- The simulation terminates consistently (e.g., at step 168 out of 1000 maximum steps)
- Final agent counts match expected values
- Simulation timing is within expected bounds
- Any changes to the codebase don't break deterministic behavior

## Configuration

The workflow is configured to check for these expected results:

- **Early Termination**: The simulation should terminate early (before reaching 1000 steps)
- **Final Agent Count**: Should be 0 (all agents died)
- **Simulation Time**: Should be between 1-300 seconds

## Expected Behavior

Based on your deterministic simulation results:
- Simulation runs with `--environment development --profile simulation --steps 1000 --seed 42`
- Should terminate early at step 168 due to all agents dying
- Final agent count should be 0
- Results should be identical across runs with the same seed

## Customization

To modify the expected values, edit the environment variables in the "Verify deterministic results" step:

```yaml
env:
  EXPECTED_EARLY_TERMINATION: 1  # 1 for early termination, 0 for full run
  EXPECTED_AGENT_COUNT: 0        # Expected final number of agents
  EXPECTED_MIN_TIME: 1.0         # Minimum simulation time in seconds
  EXPECTED_MAX_TIME: 300.0       # Maximum simulation time in seconds
```

## Failure Handling

- **Errors**: Will fail the workflow if final agent count or timing is unexpected
- **Warnings**: Will warn if early termination behavior changes (currently set as warning, can be changed to error)

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
  - Installs dependencies from `requirements.txt`
  - Runs `pytest` with coverage reporting (`--cov=farm --cov-report=term-missing`)

- JavaScript UI tests
  - Sets up Node.js 20
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
