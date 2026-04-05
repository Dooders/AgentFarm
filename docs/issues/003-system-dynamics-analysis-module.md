# [Feature] Add composite System Dynamics analysis module

**Labels:** `enhancement`, `analysis`

## Summary

The individual analysis modules (population, resources, temporal, spatial) operate independently. There is no module that synthesizes them together into a cross-domain "system dynamics" view — which is the core of what the documentation promises.

## Proposed Feature

Add a `farm/analysis/system_dynamics/` module (following the existing `BaseAnalysisModule` pattern) that:

1. **Runs sub-analyses**: Invokes population, resource, and temporal modules
2. **Cross-correlates outputs**, e.g.:
   - Pearson/Granger causality between resource depletion and population decline
   - Lag-correlation between action frequencies and reward trends
   - Resource scarcity ↔ population volatility coupling
3. **Identifies feedback loops**: Detect periods where low resources → high deaths → lower consumption → partial resource recovery
4. **Produces a unified report**: A single JSON/HTML summary combining all findings

## Acceptance Criteria

- [ ] Module follows `BaseAnalysisModule` template in `farm/analysis/template/`
- [ ] Registered in `farm/analysis/registry.py`
- [ ] At least one cross-module correlation analysis (e.g. resource–population)
- [ ] Tests in `tests/analysis/test_system_dynamics.py`
- [ ] Accessible via `AnalysisService` with `module_name="system_dynamics"`
