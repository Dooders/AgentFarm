# [Bug] Documentation describes non-existent `SimulationAnalyzer` methods

**Labels:** `bug`, `documentation`

## Summary

The "System Dynamics Analysis" section of `docs/features/agent_based_modeling_analysis.md` documents several methods on `SimulationAnalyzer` that **do not exist** in the actual class (`farm/core/analysis.py`). Users following the docs will get `AttributeError`.

## Non-Existent Methods Referenced

| Documented Method | Exists? |
|---|---|
| `analyzer.get_population_over_time()` | ❌ |
| `analyzer.calculate_survival_rates()` | ✅ |
| `analyzer.get_resource_statistics()` | ❌ |
| `analyzer.assess_resource_sustainability()` | ❌ |
| `analyzer.analyze_action_distributions()` | ❌ |
| `analyzer.get_decision_patterns()` | ❌ |
| `analyzer.measure_cooperation_levels()` | ❌ |
| `analyzer.measure_competition_intensity()` | ❌ |
| `analyzer.analyze_resource_utilization()` | ❌ |

## Location

`docs/features/agent_based_modeling_analysis.md` lines 185–255

## Resolution Options

1. Implement the missing methods (thin wrappers over the modern analysis service), **or**
2. Update the documentation to reflect the actual available API

## Resolution (2026-04-06)

**Option 2:** Updated [agent_based_modeling_analysis.md](../features/agent_based_modeling_analysis.md) so [System Dynamics Analysis](../features/agent_based_modeling_analysis.md#system-dynamics-analysis) and nearby sections only call real `SimulationAnalyzer` methods (`calculate_survival_rates`, `analyze_resource_distribution`, `analyze_resource_efficiency`, `analyze_competitive_interactions`). Removed fictional `TimeSeriesAnalyzer` and incorrect `SimulationComparison` import; fixed example SQL to use `agent_states` / `resource_states` and corrected `agent_actions` column names in the network example. Aligned `farm.database.simulation_comparison` type hints and doc examples with string `simulation_id` primary keys.
