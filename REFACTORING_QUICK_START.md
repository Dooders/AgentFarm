# Analysis Refactoring Quick Start Guide

This guide will help you get started with the analysis module refactoring quickly.

---

## 📋 What You Have

1. **ANALYSIS_CODE_STATE_REPORT.md** - Complete inventory of all analysis code
2. **ANALYSIS_REFACTORING_PLAN.md** - Detailed implementation plan with code examples
3. **REFACTORING_TASKS.md** - Task tracker with checklists
4. **REFACTORING_QUICK_START.md** - This quick reference guide
5. **REFACTORING_README.md** - Main project overview

---

## 🚀 Getting Started

### Step 1: Review the Current State

```bash
# Read the state report to understand what exists
cat ANALYSIS_CODE_STATE_REPORT.md | less
```

**Key Findings:**
- 6 major locations with analysis code
- 117+ analysis-related files
- Significant duplication in population, resource, and action analysis
- Existing protocol-based architecture ready to use

### Step 2: Review the Plan

```bash
# Read the refactoring plan
cat ANALYSIS_REFACTORING_PLAN.md | less
```

**Key Points:**
- 6-week timeline
- Phased approach (Foundation → Core → Specialized → Scripts → Testing)
- Backward compatibility maintained
- Template-based implementation

### Step 3: Start with Foundation

Create common utilities first:

```bash
# The plan includes the complete implementation in Phase 1
# Copy the code from ANALYSIS_REFACTORING_PLAN.md Section "Task 1.1"
```

---

## 🏗️ Creating Your First Module (Population)

Follow the detailed code examples in **ANALYSIS_REFACTORING_PLAN.md** under "Migration 2.1: Population Module".

---

## 📝 Implementation Workflow

### For Each Module:

1. **Create Structure**
   ```bash
   mkdir -p farm/analysis/<module_name>
   cd farm/analysis/<module_name>
   touch __init__.py module.py data.py compute.py analyze.py plot.py
   ```

2. **Implement Data Processing** (`data.py`)
   - Load data from database or files
   - Transform to standard DataFrame format
   - Handle missing data

3. **Implement Computations** (`compute.py`)
   - Statistical calculations
   - Metric computations
   - Use functions from `farm.analysis.common.utils`

4. **Implement Analysis** (`analyze.py`)
   - Call compute functions
   - Save results to files
   - Report progress via context

5. **Implement Visualizations** (`plot.py`)
   - Create matplotlib figures
   - Save to output directory
   - Use consistent styling

6. **Update Module Class** (`module.py`)
   - Register all functions
   - Define function groups
   - Set up validation

7. **Write Tests** (`tests/analysis/test_*.py`)
   - Unit tests for each function
   - Integration test for full workflow
   - Aim for >80% coverage

8. **Register Module**
   ```python
   # In farm/analysis/__init__.py
   from farm.analysis.<module_name> import <module_name>_module
   ```

---

## 🧪 Testing Your Module

```bash
# Run unit tests
pytest tests/analysis/test_population.py -v

# Run with coverage
pytest tests/analysis/test_population.py --cov=farm.analysis.population --cov-report=html

# Integration test
python -c "
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())
request = AnalysisRequest(
    module_name='population',
    experiment_path=Path('data/experiment_001'),
    output_path=Path('results/population'),
    group='basic'
)
result = service.run(request)
print(f'Success: {result.success}')
"
```

---

## 📚 Code Examples

### Using the Module via Service (Recommended)

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Initialize service
config_service = EnvConfigService()
service = AnalysisService(config_service)

# Create request
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/population"),
    group="all"  # or "basic", "plots", "analysis"
)

# Run analysis
result = service.run(request)

if result.success:
    print(f"Analysis complete!")
    print(f"Results: {result.output_path}")
    print(f"Time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.error}")
```

### Using Module Directly

```python
from pathlib import Path
from farm.analysis.population import population_module

# Run analysis
output_path, df = population_module.run_analysis(
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/population"),
    group="basic"
)

print(f"Results saved to: {output_path}")
print(f"Processed {len(df)} records")
```

### Adding Custom Progress Tracking

```python
def progress_callback(message: str, progress: float):
    print(f"[{progress*100:.0f}%] {message}")

request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/population"),
    progress_callback=progress_callback
)

result = service.run(request)
```

---

## 🎯 Priority Order

Follow this order for maximum impact:

### Week 1: Foundation
1. ✅ Create `farm/analysis/common/utils.py` (reusable utilities)
2. ✅ Set up testing infrastructure
3. ✅ Create module templates

### Week 2: High Priority
1. 🔴 **Population Module** (most used, clear boundaries)
2. 🔴 **Resources Module** (high usage, well-defined)

### Week 3: Medium Priority  
3. 🟡 **Actions Module** (moderate complexity)
4. 🟡 **Agents Module** (includes lifespan, behavior)

### Week 4: Specialized
5. 🟢 Learning, Spatial, Temporal, Combat modules

### Week 5: Consolidation
6. Update scripts, deprecate old code

### Week 6: Polish
7. Testing, documentation, release

---

## 🔍 Finding Source Code

Use the state report to locate source files:

```bash
# For population module
grep -A 10 "Population Analysis" ANALYSIS_CODE_STATE_REPORT.md

# Find all population-related analyzers
find . -name "*population*.py" | grep -E "(analyzers|analysis)" | grep -v __pycache__
```

**Key Source Files:**
- Population: `farm/database/analyzers/population_analyzer.py`
- Resources: `farm/database/analyzers/resource_analyzer.py`
- Actions: `farm/database/analyzers/action_stats_analyzer.py`
- Agents: `farm/database/analyzers/agent_analyzer.py`

---

## ✅ Checklist for Each Module

Use this checklist (also in REFACTORING_TASKS.md):

- [ ] Create directory structure
- [ ] Implement `data.py` - data processing
- [ ] Implement `compute.py` - statistical computations
- [ ] Implement `analyze.py` - analysis functions
- [ ] Implement `plot.py` - visualizations
- [ ] Implement `module.py` - module class
- [ ] Update `__init__.py` - exports
- [ ] Register in `farm/analysis/__init__.py`
- [ ] Write unit tests (target: >80% coverage)
- [ ] Write integration tests
- [ ] Performance benchmark vs old code
- [ ] Update documentation
- [ ] Create usage examples

---

## 🆘 Getting Help

### Look at Existing Modules

The best reference is existing working modules:

```bash
# Study the dominance module (most complete example)
ls -la farm/analysis/dominance/

# Read the module implementation
cat farm/analysis/dominance/module.py
```

### Review Documentation

```bash
# Architecture overview
cat farm/analysis/ARCHITECTURE.md

# User guide
cat farm/analysis/README.md

# Quick reference
cat farm/analysis/QUICK_REFERENCE.md
```

### Use the Template

```bash
# The template has TODOs marking what to implement
cat farm/analysis/template/standard_module.py
```

---

## 📊 Tracking Progress

Update **REFACTORING_TASKS.md** as you go:

```bash
# Mark task as complete
# Change [ ] to [x] in REFACTORING_TASKS.md

# Update progress percentages
# Update "Overall Progress" section
```

---

## 🔄 Daily Workflow

1. **Morning:** Check REFACTORING_TASKS.md for today's tasks
2. **Work:** Implement following the plan and examples
3. **Test:** Write and run tests as you go
4. **Document:** Update docstrings and comments
5. **Commit:** Small, focused commits with clear messages
6. **Evening:** Update task tracker, plan tomorrow

---

## 💡 Tips & Best Practices

### 1. Start Small
- Implement one function at a time
- Test each function before moving on
- Don't try to implement everything at once

### 2. Follow Existing Patterns
- Look at dominance module for inspiration
- Use the same structure and naming conventions
- Copy and adapt rather than rewrite from scratch

### 3. Use Type Hints
```python
def analyze_data(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """All functions should have type hints."""
    pass
```

### 4. Write Docstrings
```python
def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics.
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary with mean, std, min, max, etc.
        
    Example:
        >>> stats = compute_statistics(np.array([1, 2, 3]))
        >>> print(stats['mean'])
        2.0
    """
```

### 5. Report Progress
```python
def analyze_something(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    ctx.report_progress("Starting analysis", 0.0)
    # ... do work ...
    ctx.report_progress("Computing metrics", 0.5)
    # ... more work ...
    ctx.report_progress("Complete", 1.0)
```

### 6. Handle Errors Gracefully
```python
try:
    result = compute_something(data)
except ValueError as e:
    ctx.logger.error(f"Computation failed: {e}")
    raise AnalysisFunctionError("compute_something", e)
```

---

## 🎓 Learning Resources

### Internal Documentation
- `farm/analysis/ARCHITECTURE.md` - System design
- `farm/analysis/README.md` - User guide  
- `farm/analysis/dominance/` - Complete example module

### Python Best Practices
- Type hints: https://docs.python.org/3/library/typing.html
- Protocols: https://www.python.org/dev/peps/pep-0544/
- Docstrings: https://www.python.org/dev/peps/pep-0257/

---

## 🚦 When to Ask for Review

Request review when:
- ✅ First module complete (Population)
- ✅ Core modules complete (Population, Resources, Actions, Agents)
- ✅ Before deprecating old code
- ✅ Before final release

---

## 📈 Success Metrics

You'll know you're successful when:
- ✅ All tests passing (>80% coverage)
- ✅ Results match old implementation (regression tests)
- ✅ No performance degradation
- ✅ Documentation complete
- ✅ Examples working
- ✅ Code review approved

---

## 🎉 Next Steps

Ready to start? Here's what to do RIGHT NOW:

```bash
# 1. Create common utilities (copy from plan)
mkdir -p farm/analysis/common
# Copy content from ANALYSIS_REFACTORING_PLAN.md Task 1.1
# into farm/analysis/common/utils.py

# 2. Create your first module structure
mkdir -p farm/analysis/population
cd farm/analysis/population
touch __init__.py module.py data.py compute.py analyze.py plot.py

# 3. Implement following ANALYSIS_REFACTORING_PLAN.md
# Start with data.py - see Migration 2.1 for complete example

# 4. Create tests
mkdir -p tests/analysis
touch tests/analysis/test_population.py

# 5. Run tests as you implement
pytest tests/analysis/test_population.py -v

# 6. Iterate until working!
```

---

**Good luck! 🚀**

You have everything you need to succeed:
- ✅ Complete state report
- ✅ Detailed implementation plan  
- ✅ Working examples
- ✅ Bootstrap tooling
- ✅ Testing strategy
- ✅ This quick start guide

**Questions?** Review the plan or look at existing modules for examples.

**Stuck?** Check the dominance module - it has everything you need.

**Ready?** Follow the detailed implementation guide in ANALYSIS_REFACTORING_PLAN.md!
