# Analysis Module Refactoring Project

**Status:** 🟢 Ready to Start  
**Timeline:** 6 weeks  
**Start Date:** 2025-10-03  

---

## 📂 Project Documentation

This project consolidates all analysis code into the protocol-based `farm.analysis` module. All documentation is organized and ready for implementation.

### Core Documents

| Document | Purpose | Start Here? |
|----------|---------|-------------|
| **[ANALYSIS_CODE_STATE_REPORT.md](ANALYSIS_CODE_STATE_REPORT.md)** | Complete inventory of existing code | ✅ **START HERE** |
| **[ANALYSIS_REFACTORING_PLAN.md](ANALYSIS_REFACTORING_PLAN.md)** | Detailed implementation plan with code | 📋 Implementation |
| **[REFACTORING_TASKS.md](REFACTORING_TASKS.md)** | Task tracker and checklists | ✓ Tracking |
| **[REFACTORING_QUICK_START.md](REFACTORING_QUICK_START.md)** | Quick reference guide | 📖 Reference |

### Supporting Files

- **farm/analysis/** - Existing protocol-based framework (ready to use)
- **farm/analysis/dominance/** - Reference implementation (complete example)

---

## 🎯 Project Goals

### Primary Objectives
1. ✅ Eliminate code duplication across 6 locations
2. ✅ Provide single, consistent API for all analysis
3. ✅ Improve maintainability and testability
4. ✅ Maintain backward compatibility

### Success Criteria
- [ ] All database analyzers migrated (18 analyzers)
- [ ] All analysis scripts consolidated
- [ ] Test coverage > 80%
- [ ] Documentation complete
- [ ] No breaking changes for users

---

## 🚀 Quick Start

### For First-Time Setup

```bash
# 1. Read the current state report
cat ANALYSIS_CODE_STATE_REPORT.md

# 2. Review the detailed refactoring plan
cat ANALYSIS_REFACTORING_PLAN.md

# 3. Check the quick start guide
cat REFACTORING_QUICK_START.md

# 4. Start implementing!
# Follow examples in ANALYSIS_REFACTORING_PLAN.md
```

### For Daily Work

```bash
# 1. Check today's tasks
cat REFACTORING_TASKS.md | grep "Phase 2" -A 20

# 2. Implement following the plan
# Use code examples from ANALYSIS_REFACTORING_PLAN.md

# 3. Test as you go
pytest tests/analysis/test_population.py -v

# 4. Update task tracker
# Mark completed items in REFACTORING_TASKS.md
```

---

## 📊 Current Status

### Phase 1: Foundation (Week 1)
**Status:** 🟡 Not Started  
**Priority:** HIGH

- [ ] Create common utilities
- [ ] Test bootstrap script  
- [ ] Set up testing infrastructure
- [ ] Complete documentation setup

### Phase 2: Core Analyzers (Weeks 2-3)
**Status:** ⏳ Pending Foundation  
**Priority:** HIGH

- [ ] 🔴 Population Module
- [ ] 🔴 Resources Module
- [ ] 🟡 Actions Module
- [ ] 🟡 Agents Module

### Phase 3: Specialized Analyzers (Week 4)
**Status:** ⏳ Pending Core  
**Priority:** MEDIUM

- [ ] Learning Module
- [ ] Spatial Module
- [ ] Temporal Module
- [ ] Combat Module

### Phase 4: Scripts & Consolidation (Week 5)
**Status:** ⏳ Pending Specialized  
**Priority:** MEDIUM

- [ ] Update orchestration scripts
- [ ] Migrate utility functions
- [ ] Add deprecation warnings
- [ ] Update existing modules

### Phase 5: Testing & Release (Week 6)
**Status:** ⏳ Pending Scripts  
**Priority:** HIGH

- [ ] Comprehensive testing
- [ ] Documentation review
- [ ] Code quality checks
- [ ] Release preparation

---

## 🏗️ Architecture Overview

### Current Architecture (Ready to Use)

```
farm/analysis/
├── core.py              ✅ BaseAnalysisModule, protocols
├── protocols.py         ✅ Type-safe interfaces
├── service.py           ✅ AnalysisService orchestration
├── registry.py          ✅ Module discovery
├── validation.py        ✅ Data validators
├── common/
│   ├── context.py      ✅ Analysis context
│   ├── metrics.py      ✅ Shared metrics
│   └── utils.py        🔄 TO CREATE (Phase 1)
├── dominance/          ✅ REFERENCE EXAMPLE
├── genesis/            ✅ Existing module
├── advantage/          ✅ Existing module
└── social_behavior/    ✅ Existing module
```

### Target Architecture (To Create)

```
farm/analysis/
├── [existing modules above]
├── population/         🔄 Week 2 - HIGH PRIORITY
├── resources/          🔄 Week 2 - HIGH PRIORITY
├── actions/            🔄 Week 3 - MEDIUM PRIORITY
├── agents/             🔄 Week 3 - MEDIUM PRIORITY
├── learning/           🔄 Week 4 - LOW PRIORITY
├── spatial/            🔄 Week 4 - LOW PRIORITY
├── temporal/           🔄 Week 4 - LOW PRIORITY
└── combat/             🔄 Week 4 - LOW PRIORITY
```

---

## 📚 Key Concepts

### Protocol-Based Design

Your existing `farm.analysis` module uses protocols for type-safe, flexible design:

```python
# Protocols define interfaces
class AnalysisModule(Protocol):
    @property
    def name(self) -> str: ...
    def register_functions(self) -> None: ...
    def get_data_processor(self) -> DataProcessor: ...

# Implementations inherit from base class
class PopulationModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(name="population", description="...")
    
    def register_functions(self) -> None:
        self._functions = {...}
    
    def get_data_processor(self) -> SimpleDataProcessor:
        return SimpleDataProcessor(process_population_data)
```

### Module Structure Pattern

Every module follows the same pattern:

```
module_name/
├── __init__.py       # Public API
├── module.py         # Module class
├── data.py          # Data processing
├── compute.py       # Computations
├── analyze.py       # Analysis functions
└── plot.py          # Visualizations
```

### Service-Based Usage

Users interact via the service layer:

```python
from farm.analysis.service import AnalysisService, AnalysisRequest

service = AnalysisService(config_service)
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/exp"),
    output_path=Path("results")
)
result = service.run(request)
```

---

## 🔍 Finding Information

### "How do I...?"

| Question | Answer |
|----------|--------|
| Get started? | Read **REFACTORING_QUICK_START.md** |
| Find existing code? | See **ANALYSIS_CODE_STATE_REPORT.md** |
| Implement a module? | Follow **ANALYSIS_REFACTORING_PLAN.md** |
| Track progress? | Update **REFACTORING_TASKS.md** |
| See an example? | Look at `farm/analysis/dominance/` |
| Create module structure? | Follow examples in ANALYSIS_REFACTORING_PLAN.md |

### "Where is...?"

| Item | Location |
|------|----------|
| Existing modules | `farm/analysis/{dominance,genesis,advantage,social_behavior}` |
| Database analyzers | `farm/database/analyzers/` |
| Analysis scripts | `scripts/` |
| Tests | `tests/analysis/` |
| Documentation | `farm/analysis/{README,ARCHITECTURE,QUICK_REFERENCE}.md` |

---

## 🧪 Testing Strategy

### Test Levels

1. **Unit Tests** - Test individual functions
   ```bash
   pytest tests/analysis/test_population.py::test_compute_statistics -v
   ```

2. **Integration Tests** - Test full workflow
   ```bash
   pytest tests/analysis/test_population.py::test_integration -v
   ```

3. **Regression Tests** - Ensure results match old code
   ```bash
   pytest tests/analysis/test_population_regression.py -v
   ```

### Coverage Goals

- **Target:** >80% code coverage
- **Minimum:** >70% for each module

```bash
# Run with coverage
pytest tests/analysis/ --cov=farm.analysis --cov-report=html

# View report
open htmlcov/index.html
```

---

## 📖 Learning Path

### For New Team Members

1. **Week 1 - Learn the System**
   - Read `farm/analysis/README.md`
   - Read `farm/analysis/ARCHITECTURE.md`
   - Study `farm/analysis/dominance/` module
   - Run existing tests

2. **Week 2 - Implement First Module**
   - Use bootstrap script
   - Follow plan for Population module
   - Write tests
   - Get code review

3. **Week 3+ - Become Expert**
   - Implement additional modules
   - Help others
   - Improve patterns
   - Update documentation

---

## 🤝 Contributing

### Workflow

1. **Pick a Task** from REFACTORING_TASKS.md
2. **Create Branch** (e.g., `feature/population-module`)
3. **Implement** following the plan
4. **Test** (unit + integration)
5. **Document** (docstrings + examples)
6. **Commit** (clear, focused commits)
7. **Update Tracker** (mark tasks complete)
8. **Request Review**

### Commit Message Format

```
feat(analysis): Add population module

- Implement data processing from PopulationRepository
- Add population statistics computation
- Create population visualizations
- Add comprehensive tests

Related: Task 2.1 in REFACTORING_TASKS.md
```

### Code Review Checklist

- [ ] Follows existing patterns
- [ ] Has type hints
- [ ] Has docstrings
- [ ] Tests passing (>80% coverage)
- [ ] Documentation updated
- [ ] No breaking changes

---

## 📈 Progress Tracking

### Weekly Check-In Template

```markdown
## Week X: [Phase Name]

### Completed
- [ ] Task A
- [ ] Task B

### In Progress
- [ ] Task C (50% done)

### Blockers
- None / [Describe blocker]

### Next Week Plan
- [ ] Task D
- [ ] Task E

### Notes
- [Any important notes or decisions]
```

### Metrics to Track

- Tasks completed vs planned
- Test coverage percentage
- Lines of code migrated
- Modules completed
- Documentation coverage

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Module not found after creation  
**Solution:** Check that you registered in `farm/analysis/__init__.py`

**Issue:** Tests failing with import errors  
**Solution:** Ensure `__init__.py` exports are correct

**Issue:** Data processing fails  
**Solution:** Check database path and connection

**Issue:** Plots not saving  
**Solution:** Verify `ctx.get_output_file()` usage

### Getting Help

1. Check existing modules for examples
2. Review documentation in `farm/analysis/`
3. Look at similar functionality in old code
4. Ask for code review early

---

## 📅 Timeline

```
Week 1: Foundation
├─ Common utilities
├─ Testing setup
└─ Documentation

Week 2: High Priority Modules
├─ Population (most important)
└─ Resources (high usage)

Week 3: Medium Priority Modules
├─ Actions (moderate complexity)
└─ Agents (includes submodules)

Week 4: Specialized Modules
├─ Learning
├─ Spatial
├─ Temporal
└─ Combat

Week 5: Consolidation
├─ Update scripts
├─ Migrate utilities
├─ Add deprecation warnings
└─ Update existing modules

Week 6: Polish & Release
├─ Comprehensive testing
├─ Documentation review
├─ Code quality
└─ Release preparation
```

---

## 🎯 Success Metrics

### Quantitative

- [ ] 18 analyzers migrated
- [ ] 8 new modules created
- [ ] >80% test coverage
- [ ] 0 breaking changes
- [ ] All tests passing

### Qualitative

- [ ] Code is maintainable
- [ ] API is intuitive
- [ ] Documentation is clear
- [ ] Examples work
- [ ] Users are happy

---

## 🔗 Related Links

### Internal
- Existing analysis modules: `farm/analysis/`
- Database infrastructure: `farm/database/`
- Test suite: `tests/analysis/`

### External Resources
- Python Protocols: https://www.python.org/dev/peps/pep-0544/
- Type Hints: https://docs.python.org/3/library/typing.html
- Matplotlib: https://matplotlib.org/
- Pandas: https://pandas.pydata.org/

---

## 📞 Contact

**Project Owner:** [Your Name]  
**Questions?** Check existing documentation first, then ask!  
**Stuck?** Look at the dominance module - it has everything you need.

---

## 🎉 Let's Get Started!

Ready to begin? Here's your immediate next steps:

```bash
# 1. Review current state
cat ANALYSIS_CODE_STATE_REPORT.md

# 2. Read the detailed refactoring plan
cat ANALYSIS_REFACTORING_PLAN.md

# 3. Check task breakdown
cat REFACTORING_TASKS.md

# 4. Start implementing!
# Follow ANALYSIS_REFACTORING_PLAN.md with detailed code examples
```

**Everything you need is ready. Time to code! 🚀**

---

**Last Updated:** 2025-10-03  
**Status:** 🟢 Ready for Implementation  
**Next Review:** End of Week 1
