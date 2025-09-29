# Testing Framework

This repository uses pytest for Python tests and Jest for a small set of browser-based UI tests under `farm/editor`.

### Running Python tests

- Quick default run (unit only):

```bash
pytest
```

- With verbose output and coverage:

```bash
pytest -q --cov=farm --cov-report=term-missing
```

- Include integration and slow tests:

```bash
pytest -q -m "integration or slow"
```

### Markers

- `unit`: fast tests run by default
- `integration`: slower DB/environment/end-to-end tests
- `slow`: very slow tests excluded by default
- `db`, `ml`, `analysis`: opt-in domain-specific markers

Markers can be combined, e.g. `-m "unit and not db"`.

### Shared fixtures

Common fixtures live in `tests/conftest.py`:

- `env`: lightweight `Environment` configured for fast in-memory testing
- `db`: `SimulationDatabase` with a seeded simulation record
- `tmp_db_path`: unique temporary sqlite path with cleanup
- `fast_sleep`: patches `time.sleep` to no-op
- `disable_network`: blocks outbound network calls
- `set_test_seed`: deterministic RNG seed for Python and NumPy

Utility factory helpers live in `tests/utils`.

### JavaScript UI tests

The UI test at `farm/editor/__tests__/explorer.test.js` expects a Jest + jsdom environment. To run it, install dev deps and run Jest:

```bash
npm install --prefix farm/editor --no-audit --no-fund
npm test --prefix farm/editor -- --runInBand
```

### Conventions

- Place pure unit tests near the feature under `tests/` and prefer `test_*.py`.
- Use markers to categorize runtime; integration tests should be under `tests/integration/`.
- Prefer fixtures over ad-hoc setup code; share helpers via `tests/utils`.
- Default runs exclude `integration` and `slow` to keep CI fast.
