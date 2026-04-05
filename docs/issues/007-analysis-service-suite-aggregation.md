# [Feature] `AnalysisService` should support multi-module batch aggregation into a unified summary

**Labels:** `enhancement`, `analysis`

## Summary

`AnalysisService.run_batch()` can run multiple modules but returns independent results with no cross-module synthesis. There is no way to request a unified summary combining population + resource + temporal + spatial results in a single call.

## Proposed Feature

Add an `AnalysisService.run_suite()` method (or similar) that:

1. Accepts a named suite (e.g. `"system_dynamics"`, `"full"`) or a list of module names
2. Runs all specified modules
3. Aggregates their outputs into a structured summary dict/JSON
4. Optionally generates a combined HTML/Markdown report

### Example API

```python
result = service.run_suite(
    suite="system_dynamics",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/")
)
# result.summary contains cross-module aggregated data
```

### Built-in Suites to Define

| Suite | Modules |
|---|---|
| `system_dynamics` | population, resources, temporal |
| `agent_behavior` | actions, agents, spatial, learning |
| `social` | social_behavior, combat, dominance |
| `full` | all modules |

## Acceptance Criteria

- [ ] `run_suite()` method on `AnalysisService`
- [ ] At least 2 named suites defined
- [ ] Aggregated result object with combined metadata
- [ ] Tests covering suite execution
