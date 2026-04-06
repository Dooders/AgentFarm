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

- [x] Module follows `BaseAnalysisModule` pattern (same as `population` / `resources` / `temporal`; there is no `farm/analysis/template/` package in-tree)
- [x] Registered in `farm/analysis/registry.py`
- [x] Cross-module correlation: resource–population (levels + first differences), action–reward lags, scarcity vs population volatility; optional Granger on changes
- [x] Tests in `tests/analysis/test_system_dynamics.py`
- [x] Accessible via `AnalysisService` with `module_name="system_dynamics"`

**Implementation:** `farm/analysis/system_dynamics/` — merged per-step frame from population, resources, and temporal loaders; groups `all` (foundation submodule runs + synthesis + report), `synthesis` (cross-domain metrics + unified JSON/HTML), `foundation` (reference runs only).
