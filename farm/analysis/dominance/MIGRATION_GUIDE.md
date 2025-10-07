# Migration Guide: Legacy API to Orchestrator

This guide helps you migrate from the legacy function-based API to the new protocol-based orchestrator API for dominance analysis.

---

## TL;DR - No Migration Required! ✅

**Good news:** All existing code continues to work without changes due to backward compatibility wrappers. Migration is **optional** and only recommended if you want to adopt the new patterns for better testability and architecture.

---

## Why Migrate?

### Benefits of the New API

1. **Better Testing** - Easy to mock components with protocols
2. **Cleaner Architecture** - No circular dependencies
3. **More Flexibility** - Swap implementations without changes
4. **Unified API** - Single orchestrator for all operations
5. **Better Type Hints** - Full type safety with protocols

### When to Migrate

✅ **Migrate if you:**
- Want to write unit tests with mocks
- Need custom implementations of analysis logic
- Prefer object-oriented APIs
- Want to leverage dependency injection

❌ **Don't migrate if you:**
- Have working code that doesn't need changes
- Prefer simple function-based APIs
- Don't need custom implementations

---

## Migration Patterns

### Pattern 1: Simple Function Calls

**BEFORE (Legacy - Still Works):**
```python
from farm.analysis.dominance.compute import compute_population_dominance

result = compute_population_dominance(session)
```

**AFTER (Orchestrator - Optional):**
```python
from farm.analysis.dominance import get_orchestrator

orchestrator = get_orchestrator()
result = orchestrator.compute_population_dominance(session)
```

**Migration Steps:**
1. Import `get_orchestrator` instead of individual functions
2. Create orchestrator instance once (reuse it)
3. Call methods on orchestrator instead of module functions

---

### Pattern 2: Multiple Function Calls

**BEFORE (Legacy):**
```python
from farm.analysis.dominance.compute import (
    compute_population_dominance,
    compute_survival_dominance,
    compute_comprehensive_dominance,
)

pop_dom = compute_population_dominance(session)
surv_dom = compute_survival_dominance(session)
comp_dom = compute_comprehensive_dominance(session)
```

**AFTER (Orchestrator):**
```python
from farm.analysis.dominance import get_orchestrator

orchestrator = get_orchestrator()

# Better: Use high-level orchestration method
results = orchestrator.run_full_analysis(session, config)
pop_dom = results['population_dominance']
surv_dom = results['survival_dominance']
comp_dom = results['comprehensive_dominance']
```

---

### Pattern 3: DataFrame Analysis

**BEFORE (Legacy):**
```python
from farm.analysis.dominance.analyze import (
    analyze_dominance_switch_factors,
    analyze_high_vs_low_switching,
    analyze_by_agent_type,
)
from farm.analysis.common.metrics import get_valid_numeric_columns

# Get numeric columns
repro_cols = [col for col in df.columns if 'reproduction' in col]
numeric_repro_cols = get_valid_numeric_columns(df, repro_cols)

# Run analyses
df = analyze_dominance_switch_factors(df)
df = analyze_high_vs_low_switching(df, numeric_repro_cols)
df = analyze_by_agent_type(df, numeric_repro_cols)
```

**AFTER (Orchestrator):**
```python
from farm.analysis.dominance import get_orchestrator

orchestrator = get_orchestrator()

# All-in-one analysis (auto-detects reproduction columns)
df = orchestrator.analyze_dataframe_comprehensively(df)
```

---

### Pattern 4: Testing Code

**BEFORE (Legacy - Hard to Test):**
```python
# Difficult to test because of direct imports
def my_analysis_function(session):
    from farm.analysis.dominance.compute import compute_population_dominance
    result = compute_population_dominance(session)
    return process_result(result)

# Test requires patching module
def test_my_analysis(mock_session):
    with patch('farm.analysis.dominance.compute.compute_population_dominance') as mock:
        mock.return_value = 'system'
        result = my_analysis_function(mock_session)
        assert result == expected
```

**AFTER (Orchestrator - Easy to Test):**
```python
# Easy to test with dependency injection
def my_analysis_function(orchestrator, session):
    result = orchestrator.compute_population_dominance(session)
    return process_result(result)

# Test with mock orchestrator
def test_my_analysis():
    mock_orchestrator = Mock()
    mock_orchestrator.compute_population_dominance.return_value = 'system'
    
    result = my_analysis_function(mock_orchestrator, mock_session)
    assert result == expected
```

---

### Pattern 5: Custom Implementations

**BEFORE (Legacy - Not Possible):**
```python
# Can't customize computation logic without modifying source
```

**AFTER (Orchestrator - Easy):**
```python
from farm.analysis.dominance import create_dominance_orchestrator
from farm.analysis.dominance.interfaces import DominanceComputerProtocol

class MyCustomComputer:
    """Custom computer with different weighting."""
    
    def compute_comprehensive_dominance(self, sim_session):
        # Your custom logic with different weights
        return custom_result
    
    # Implement other protocol methods...

# Use custom implementation
orchestrator = create_dominance_orchestrator(
    custom_computer=MyCustomComputer()
)
```

---

## Complete Migration Example

### Before (Legacy API)

```python
# analyze_simulation.py
from farm.analysis.dominance.compute import (
    compute_population_dominance,
    compute_survival_dominance,
    compute_comprehensive_dominance,
    compute_dominance_switches,
)
from farm.analysis.dominance.analyze import (
    analyze_dominance_switch_factors,
    analyze_reproduction_dominance_switching,
)
from farm.analysis.dominance.data import (
    get_agent_survival_stats,
    get_reproduction_stats,
)

def analyze_simulation(session, config):
    # Compute metrics
    pop_dom = compute_population_dominance(session)
    surv_dom = compute_survival_dominance(session)
    comp_dom = compute_comprehensive_dominance(session)
    switches = compute_dominance_switches(session)
    
    # Get data
    survival_stats = get_agent_survival_stats(session)
    repro_stats = get_reproduction_stats(session)
    
    # Combine results
    return {
        'population_dominance': pop_dom,
        'survival_dominance': surv_dom,
        'comprehensive_dominance': comp_dom,
        'switches': switches,
        'survival_stats': survival_stats,
        'reproduction_stats': repro_stats,
    }

def analyze_results_dataframe(df):
    # Analyze DataFrame
    df = analyze_dominance_switch_factors(df)
    df = analyze_reproduction_dominance_switching(df)
    return df
```

### After (Orchestrator API)

```python
# analyze_simulation.py
from farm.analysis.dominance import get_orchestrator

def analyze_simulation(session, config):
    orchestrator = get_orchestrator()
    
    # Single method for complete analysis
    return orchestrator.run_full_analysis(session, config)

def analyze_results_dataframe(df):
    orchestrator = get_orchestrator()
    
    # Single method for comprehensive DataFrame analysis
    return orchestrator.analyze_dataframe_comprehensively(df)
```

**Result:**
- ✅ 20 lines → 8 lines (60% reduction)
- ✅ Fewer imports to manage
- ✅ Single point of entry
- ✅ Easier to test with mocks

---

## Migration Checklist

### Step 1: Assess Current Usage
- [ ] Identify all files importing from `farm.analysis.dominance`
- [ ] Document which functions are being used
- [ ] Identify any custom logic that might need adaptation

### Step 2: Test Backward Compatibility
- [ ] Run existing tests to verify nothing breaks
- [ ] All tests should pass without changes

### Step 3: Gradual Migration
- [ ] Start with new code using orchestrator
- [ ] Gradually refactor old code when touched
- [ ] Keep legacy code working during transition

### Step 4: Update Tests (Optional)
- [ ] Replace patches with mock orchestrators
- [ ] Use dependency injection in test fixtures
- [ ] Verify improved test clarity

### Step 5: Documentation
- [ ] Update internal documentation
- [ ] Add orchestrator usage examples
- [ ] Document any custom implementations

---

## Compatibility Matrix

| Import | Legacy API | Orchestrator | Status |
|--------|-----------|--------------|---------|
| `from farm.analysis.dominance.compute import *` | ✅ | ✅ | Fully compatible |
| `from farm.analysis.dominance.analyze import *` | ✅ | ✅ | Fully compatible |
| `from farm.analysis.dominance import *` | ✅ | ✅ | Enhanced with orchestrator |
| Direct function calls | ✅ | ✅ | Delegated to classes |
| Custom implementations | ❌ | ✅ | New capability |
| Mock testing | Limited | ✅ | Much easier |

---

## FAQ

### Q: Do I have to migrate my code?
**A:** No! All existing code continues to work. Migration is optional.

### Q: Will the legacy API be removed?
**A:** No plans to remove it. Backward compatibility is maintained indefinitely.

### Q: What's the performance difference?
**A:** Negligible. The orchestrator adds minimal overhead (single method call indirection).

### Q: Can I mix legacy and orchestrator APIs?
**A:** Yes! They work together seamlessly.

### Q: How do I know if migration is worth it?
**A:** Migrate if you need better testing, custom implementations, or cleaner architecture. Otherwise, stick with what works.

---

## Support

For questions or issues:
- Check [ORCHESTRATOR_GUIDE.md](./ORCHESTRATOR_GUIDE.md) for API reference
- Review [orchestrator.py](./orchestrator.py) source code
- Open an issue on GitHub with migration questions

---

## Summary

✅ **Backward Compatibility:** 100% - No breaking changes  
✅ **Migration Required:** No - Optional only  
✅ **Benefits:** Better testing, cleaner architecture, more flexibility  
✅ **Effort:** Low - Simple API changes  
✅ **Risk:** None - Can migrate gradually

**Recommendation:** Use orchestrator for new code, migrate old code opportunistically when touching it for other reasons.
