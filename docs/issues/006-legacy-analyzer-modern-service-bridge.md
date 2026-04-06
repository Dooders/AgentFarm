# [Tech Debt] Legacy `SimulationAnalyzer` and modern `AnalysisService` coexist without a bridge

**Labels:** `technical-debt`, `analysis`

## Summary

Two parallel systems serve overlapping purposes:

- `farm/core/analysis.py::SimulationAnalyzer` — legacy, used by `farm/api/`, CLI, and experiment runner
- `farm/analysis/` (`AnalysisService`) — modern, protocol-based, 8 modules, well-tested

They are not integrated. The legacy class only has 4 real query methods and doesn't leverage any modern module capabilities.

## Problem

- New analysis features added to `AnalysisService` are invisible to API/CLI consumers
- Bug fixes or improvements in the modern modules don't reach users of `SimulationAnalyzer`
- Duplicate maintenance burden

## Proposed Resolution

One of:

**Option A (Recommended):** Make `SimulationAnalyzer` a thin facade over `AnalysisService`:

```python
class SimulationAnalyzer:
    def __init__(self, db_path, simulation_id=None):
        self._service = AnalysisService(...)

    def get_population_over_time(self):
        result = self._service.run(AnalysisRequest(module_name="population", ...))
        return result.data
```

**Option B:** Formally deprecate `SimulationAnalyzer`, update API/CLI to use `AnalysisService` directly.

## Acceptance Criteria

- [ ] Single code path for analysis logic
- [ ] API and CLI produce equivalent results to direct `AnalysisService` calls
- [ ] Deprecated class (if Option B) emits `DeprecationWarning`
- [ ] Tests updated accordingly
