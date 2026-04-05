# [Bug] `analyze_simulation()` is a stub that silently returns empty metrics

**Labels:** `bug`, `analysis`

## Summary

`farm/core/analysis.py::analyze_simulation()` is a stub function that returns an empty dictionary. It is called by the experiment runner and API layer, so callers silently receive no data without any error or warning.

## Location

`farm/core/analysis.py` lines 156–172

## Current Behavior

```python
def analyze_simulation(simulation_data):
    results = {
        "metrics": {},
        "statistics": {},
    }
    return results
```

Any caller (e.g. `farm/runners/experiment_runner.py`, `farm/core/cli.py`) receives an empty result and cannot distinguish this from a real (but empty) analysis.

## Expected Behavior

Either:
- Implement the function with real logic using the modern `AnalysisService`/modules, **or**
- Raise `NotImplementedError` so callers fail loudly until this is properly implemented

## Impact

- Misleading API results
- Silent data loss in experiment reports
- Users following documentation examples get empty data with no feedback
