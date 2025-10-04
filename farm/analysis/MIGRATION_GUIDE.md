# Migration Guide: Database Analyzers → Analysis Modules

## Overview

The database analyzers have been migrated to the protocol-based `farm.analysis` module system. This provides a consistent API, automatic caching, progress tracking, and better error handling.

## Quick Migration

### Before (Old Way)
```python
from farm.database.database import SimulationDatabase
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

db = SimulationDatabase("sqlite:///simulation.db")
repository = PopulationRepository(db.session_manager)
analyzer = PopulationAnalyzer(repository)
stats = analyzer.analyze_comprehensive_statistics()
```

### After (New Way)
```python
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService
from pathlib import Path

service = AnalysisService(EnvConfigService())
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/population")
)
result = service.run(request)
```

## Module Mapping

| Old Analyzer | New Module | Status | Notes |
|-------------|------------|--------|-------|
| `PopulationAnalyzer` | `population` | ✅ Complete | Full feature parity |
| `ResourceAnalyzer` | `resources` | ✅ Complete | Enhanced with efficiency metrics |
| `ActionStatsAnalyzer` | `actions` | ✅ Complete | Merged with sequence patterns |
| `AgentAnalyzer` | `agents` | ✅ Complete | Includes lifespan & behavior |
| `LearningAnalyzer` | `learning` | ✅ Complete | Same functionality |
| `SpatialAnalysis` | `spatial` | ✅ Complete | Enhanced visualization |
| `TemporalPatternAnalyzer` | `temporal` | ✅ Complete | Enhanced pattern detection |
| `CombatAnalysis` | `combat` | ✅ Complete | Combat metrics & patterns |

## Benefits of Migration

1. ✅ **Consistent API** across all analysis types
2. ✅ **Automatic result caching** for faster re-runs
3. ✅ **Progress tracking** with callbacks
4. ✅ **Batch processing** support
5. ✅ **Better error handling** with specific exceptions
6. ✅ **Type safety** with protocol interfaces
7. ✅ **Standardized validation** and data quality checks
8. ✅ **Unified configuration** and logging
9. ✅ **Comprehensive testing** and documentation
10. ✅ **Backward compatibility** maintained

## Detailed Migration Examples

### Population Analysis

**Old:**
```python
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

analyzer = PopulationAnalyzer(repository)
stats = analyzer.analyze_comprehensive_statistics()
birth_death = analyzer.analyze_birth_death_patterns()
composition = analyzer.analyze_agent_type_composition()
```

**New:**
```python
from farm.analysis.service import AnalysisService, AnalysisRequest

service = AnalysisService(EnvConfigService())
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/population"),
    group="all"  # Runs all analysis functions
)

result = service.run(request)
# Results saved to results/population/:
# - population_statistics.json (stats, rates, stability)
# - agent_composition.csv (composition data)
# - population_over_time.png (dynamics plot)
# - birth_death_rates.png (birth/death plot)
# - agent_composition.png (composition plot)
```

### Resource Analysis

**Old:**
```python
from farm.database.analyzers.resource_analyzer import ResourceAnalyzer

analyzer = ResourceAnalyzer(repository)
distribution = analyzer.analyze_resource_distribution()
consumption = analyzer.analyze_consumption_patterns()
efficiency = analyzer.analyze_efficiency_metrics()
hotspots = analyzer.find_resource_hotspots()
```

**New:**
```python
request = AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/resources"),
    group="all"
)

result = service.run(request)
# Results saved to results/resources/:
# - resource_statistics.json
# - consumption_patterns.csv
# - resource_efficiency.png
# - consumption_patterns.png
```

### Action Analysis

**Old:**
```python
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer

analyzer = ActionStatsAnalyzer(repository)
stats = analyzer.analyze_action_statistics()
patterns = analyzer.analyze_action_patterns()
success = analyzer.analyze_success_rates()
sequences = analyzer.analyze_action_sequences()
```

**New:**
```python
request = AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/actions"),
    group="all"
)

result = service.run(request)
# Results saved to results/actions/:
# - action_statistics.json
# - action_patterns.csv
# - success_rates.csv
# - action_distribution.png
# - success_rates.png
# - action_sequences.png
```

### Agent Analysis

**Old:**
```python
from farm.database.analyzers.agent_analyzer import AgentAnalyzer

analyzer = AgentAnalyzer(repository)
stats = analyzer.analyze_agent_statistics()
lifespan = analyzer.analyze_lifespan_patterns()
behavior = analyzer.analyze_behavior_patterns()
clustering = analyzer.perform_behavior_clustering()
```

**New:**
```python
request = AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents"),
    group="all"
)

result = service.run(request)
# Results saved to results/agents/:
# - agent_statistics.json
# - lifespan_analysis.csv
# - behavior_analysis.json
# - agent_statistics.png
# - lifespan_distribution.png
# - behavior_clusters.png
```

## Advanced Usage Examples

### Custom Analysis Parameters

```python
# Customize plot appearance
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/custom"),
    analysis_kwargs={
        "plot_population": {
            "figsize": (14, 8),
            "dpi": 300,
            "colors": ['blue', 'green', 'red']
        },
        "plot_births_deaths": {
            "window": 50,  # Rolling window size
            "alpha": 0.7
        }
    }
)
```

### Progress Tracking

```python
def progress_callback(message: str, progress: float):
    print(f"[{progress:.1%}] {message}")

request = AnalysisRequest(
    module_name="agents",  # Clustering can be slow
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents"),
    progress_callback=progress_callback
)
```

### Batch Processing

```python
# Analyze multiple experiments
experiments = [f"experiment_{i:03d}" for i in range(10)]

requests = [
    AnalysisRequest(
        module_name="population",
        experiment_path=Path(f"data/{exp}"),
        output_path=Path(f"results/{exp}/population"),
        group="basic"
    )
    for exp in experiments
]

results = service.run_batch(requests)

# Check results
for result in results:
    if result.success:
        print(f"✓ {result.module_name}: {result.execution_time:.2f}s")
    else:
        print(f"✗ {result.module_name}: {result.error}")
```

### Caching for Performance

```python
# First run (computes and caches)
request = AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/spatial"),
    enable_caching=True
)

result1 = service.run(request)
print(f"Cache hit: {result1.cache_hit}")  # False

# Second run (uses cache)
result2 = service.run(request)
print(f"Cache hit: {result2.cache_hit}")  # True
print(f"Speedup: {result1.execution_time / result2.execution_time:.1f}x")
```

## Migration Complete

Old analyzers have been removed. The migration to the new protocol-based analysis system is complete.

### Migration Timeline

- **Phase 1 (Weeks 1-6):** Create new analysis modules ✅
- **Phase 2 (Month 2):** Update all internal usage ✅
- **Phase 3 (Month 3):** Remove old implementations ✅

## Error Handling

The new system provides better error handling:

```python
from farm.analysis.exceptions import (
    ModuleNotFoundError,
    DataValidationError,
    DataProcessingError
)

try:
    result = service.run(request)
except ModuleNotFoundError as e:
    print(f"Module '{e.module_name}' not found")
    print(f"Available: {e.available_modules}")
except DataValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Missing columns: {e.missing_columns}")
except DataProcessingError as e:
    print(f"Processing failed at step: {e.step}")
```

## Configuration

### Environment Variables

```bash
# Custom module paths
export FARM_ANALYSIS_MODULES="my.custom.module:my_module"
```

### Programmatic Configuration

```python
from farm.core.services import EnvConfigService

# Custom config
config = EnvConfigService()
config.set('analysis.cache.enabled', True)
config.set('analysis.cache.max_size', '1GB')

service = AnalysisService(config)
```

## Troubleshooting

### Common Issues

1. **Module not found**
   ```python
   # Check available modules
   from farm.analysis.registry import get_module_names
   print("Available:", get_module_names())
   ```

2. **Validation errors**
   ```python
   # Check data structure
   print("Columns:", df.columns.tolist())
   print("Data types:", df.dtypes.to_dict())
   ```

3. **Performance issues**
   ```python
   # Enable caching
   request.enable_caching = True

   # Use batch processing for multiple analyses
   results = service.run_batch(requests)
   ```

4. **Memory issues**
   ```python
   # Process in smaller batches
   request.group = "basic"  # Run fewer functions

   # Clear cache periodically
   service.clear_cache()
   ```

## Getting Help

1. **Check the documentation:**
   - `farm/analysis/README.md` - Complete guide
   - `farm/analysis/QUICK_REFERENCE.md` - Quick reference
   - `farm/analysis/ARCHITECTURE.md` - System design

2. **Run the examples:**
   - `examples/analysis_example.py` - 7 complete examples

3. **Check the tests:**
   - `tests/analysis/test_integration.py` - End-to-end examples
   - `tests/analysis/test_{module}.py` - Module-specific tests

4. **Use the interactive help:**
   ```python
   # List all modules
   modules = service.list_modules()
   for m in modules:
       print(f"{m['name']}: {m['description']}")

   # Get detailed module info
   info = service.get_module_info("population")
   print(f"Functions: {info['functions']}")
   print(f"Groups: {info['function_groups']}")
   ```

## Migration Checklist

- [ ] Replace old analyzer imports with new module requests
- [ ] Update file paths to use Path objects
- [ ] Add error handling for new exception types
- [ ] Update configuration to use new system
- [ ] Test with sample data before production use
- [ ] Update documentation and examples
- [ ] Remove old analyzer usage (when ready)

## Next Steps

1. ✅ Review this guide
2. ✅ Test migration with your data
3. ✅ Update your analysis scripts
4. ✅ Update documentation
5. ✅ Train team on new system

The new analysis module system provides significant improvements in consistency, performance, and maintainability while maintaining backward compatibility during the transition period.
