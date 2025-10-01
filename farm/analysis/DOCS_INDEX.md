# Analysis Module System - Documentation Index

Complete guide to all documentation for the analysis module system.

## 📚 Getting Started

### Quick Start
1. **[README.md](README.md)** - Complete user guide
   - Quick start examples
   - Creating new modules
   - Advanced features
   - Best practices

2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Cheat sheet
   - Common patterns
   - Import reference
   - Quick lookups
   - Tips and tricks

3. **[examples/analysis_example.py](../../examples/analysis_example.py)** - Working code
   - 7 complete examples
   - Basic to advanced usage
   - Copy-paste ready

## 🏗️ Architecture & Design

### System Design
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
   - Component overview
   - Data flow diagrams
   - Design patterns used
   - Extension points

2. **[protocols.py](protocols.py)** - Protocol definitions
   - Type-safe contracts
   - Interface specifications
   - Runtime checking

3. **[core.py](core.py)** - Base implementations
   - BaseAnalysisModule
   - Data processors
   - Function wrappers

## 🔄 Migration

### Upgrading
1. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Step-by-step migration
   - What changed
   - Code examples
   - Common issues
   - Checklist

2. **[REFACTORING_SUMMARY.md](../../REFACTORING_SUMMARY.md)** - What was done
   - Executive summary
   - New capabilities
   - Metrics
   - Before/after comparisons

3. **[ANALYSIS_REFACTORING.md](../../ANALYSIS_REFACTORING.md)** - Complete overview
   - Breaking changes
   - File structure
   - Testing
   - Credits

## 📝 Templates & Examples

### Creating New Modules
1. **[template/module.py](template/module.py)** - Module template
   - Annotated template
   - Best practices
   - Example implementation

2. **Example Modules**
   - [dominance/module.py](dominance/module.py) - Full-featured module
   - [null_module.py](null_module.py) - Minimal module

## 🧪 Testing

### Test Documentation
1. **[tests/analysis/](../../tests/analysis/)** - Test suite
   - [conftest.py](../../tests/analysis/conftest.py) - Shared fixtures
   - [test_protocols.py](../../tests/analysis/test_protocols.py) - Protocol tests
   - [test_validation.py](../../tests/analysis/test_validation.py) - Validation tests
   - [test_core.py](../../tests/analysis/test_core.py) - Core functionality
   - [test_registry.py](../../tests/analysis/test_registry.py) - Registry tests
   - [test_service.py](../../tests/analysis/test_service.py) - Service tests
   - [test_exceptions.py](../../tests/analysis/test_exceptions.py) - Exception tests
   - [test_integration.py](../../tests/analysis/test_integration.py) - Integration tests

## 📖 API Reference

### Core APIs

#### Protocols
- **[protocols.py](protocols.py)**
  - `AnalysisModule` - Module protocol
  - `DataLoader` - Data loading protocol
  - `DataProcessor` - Data processing protocol
  - `DataValidator` - Validation protocol
  - `AnalysisFunction` - Function protocol
  - `Analyzer` - Analysis protocol
  - `Visualizer` - Visualization protocol

#### Core Classes
- **[core.py](core.py)**
  - `BaseAnalysisModule` - Module base class
  - `SimpleDataProcessor` - Simple processor
  - `ChainedDataProcessor` - Processor chain
  - `make_analysis_function()` - Function wrapper

#### Validation
- **[validation.py](validation.py)**
  - `ColumnValidator` - Column validation
  - `DataQualityValidator` - Quality checks
  - `CompositeValidator` - Validator composition
  - `validate_numeric_columns()` - Helper
  - `validate_simulation_data()` - Helper

#### Exceptions
- **[exceptions.py](exceptions.py)**
  - `AnalysisError` - Base exception
  - `DataValidationError` - Invalid data
  - `ModuleNotFoundError` - Missing module
  - `DataProcessingError` - Processing failure
  - `ConfigurationError` - Config issue
  - And more...

#### Service Layer
- **[service.py](service.py)**
  - `AnalysisService` - Main service
  - `AnalysisRequest` - Request object
  - `AnalysisResult` - Result object
  - `AnalysisCache` - Caching system

#### Registry
- **[registry.py](registry.py)**
  - `ModuleRegistry` - Registry class
  - `register_modules()` - Registration
  - `get_module()` - Retrieval
  - `get_module_names()` - Listing

#### Common Utilities
- **[common/context.py](common/context.py)**
  - `AnalysisContext` - Execution context

- **[common/metrics.py](common/metrics.py)**
  - `get_valid_numeric_columns()` - Column filtering
  - `analyze_correlations()` - Correlation analysis
  - `split_and_compare_groups()` - Group comparison
  - `group_and_analyze()` - Group analysis
  - `find_top_correlations()` - Top correlations

## 📊 By Topic

### Validation
- [validation.py](validation.py) - Validators
- [exceptions.py](exceptions.py) - Exception types
- [tests/analysis/test_validation.py](../../tests/analysis/test_validation.py) - Tests
- [README.md#validation](README.md#advanced-features) - Usage examples

### Caching
- [service.py](service.py) - `AnalysisCache` class
- [README.md#caching](README.md#advanced-features) - Usage guide
- [examples/analysis_example.py](../../examples/analysis_example.py) - Example code

### Progress Tracking
- [common/context.py](common/context.py) - `AnalysisContext`
- [service.py](service.py) - Service integration
- [README.md#progress-tracking](README.md#advanced-features) - Usage guide
- [examples/analysis_example.py](../../examples/analysis_example.py) - Examples

### Batch Processing
- [service.py](service.py) - `run_batch()` method
- [README.md#batch-analysis](README.md#advanced-features) - Usage guide
- [examples/analysis_example.py](../../examples/analysis_example.py) - Examples

### Error Handling
- [exceptions.py](exceptions.py) - All exception types
- [README.md#error-handling](README.md#best-practices) - Best practices
- [MIGRATION_GUIDE.md#error-handling](MIGRATION_GUIDE.md) - Migration guide

## 🎯 By Use Case

### "I want to create a new analysis module"
1. Read [README.md#creating-a-new-analysis-module](README.md)
2. Copy [template/module.py](template/module.py)
3. Study [dominance/module.py](dominance/module.py)
4. Check [examples/analysis_example.py](../../examples/analysis_example.py)

### "I want to run an analysis"
1. See [QUICK_REFERENCE.md#basic-usage](QUICK_REFERENCE.md)
2. Run [examples/analysis_example.py](../../examples/analysis_example.py)
3. Read [README.md#quick-start](README.md)

### "I want to migrate existing code"
1. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
2. Check [REFACTORING_SUMMARY.md](../../REFACTORING_SUMMARY.md)
3. Study updated modules in [dominance/module.py](dominance/module.py)

### "I want to understand the architecture"
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Study [protocols.py](protocols.py)
3. Review [core.py](core.py)

### "I want to write tests"
1. Study [tests/analysis/conftest.py](../../tests/analysis/conftest.py)
2. Check [tests/analysis/test_core.py](../../tests/analysis/test_core.py)
3. Read [tests/analysis/test_integration.py](../../tests/analysis/test_integration.py)

### "I want to validate data"
1. Read [validation.py](validation.py)
2. Check [README.md#validation](README.md)
3. Study [tests/analysis/test_validation.py](../../tests/analysis/test_validation.py)

## 📑 Quick Reference Tables

### File Purposes

| File | Purpose |
|------|---------|
| `protocols.py` | Type-safe protocol definitions |
| `core.py` | Base implementations |
| `validation.py` | Data validators |
| `exceptions.py` | Custom exception types |
| `registry.py` | Module discovery and registration |
| `service.py` | High-level service API |
| `common/context.py` | Execution context |
| `common/metrics.py` | Shared utilities |
| `base_module.py` | Legacy base (deprecated) |

### Documentation Types

| Type | Files |
|------|-------|
| User Guides | README.md, QUICK_REFERENCE.md |
| Architecture | ARCHITECTURE.md, protocols.py |
| Migration | MIGRATION_GUIDE.md, REFACTORING_SUMMARY.md |
| Templates | template/module.py |
| Examples | examples/analysis_example.py |
| Tests | tests/analysis/*.py |
| API Reference | Source code docstrings |

### Learning Path

| Level | Read This |
|-------|-----------|
| Beginner | README.md → QUICK_REFERENCE.md → examples/ |
| Intermediate | ARCHITECTURE.md → template/module.py → dominance/module.py |
| Advanced | protocols.py → core.py → tests/ |
| Migrating | MIGRATION_GUIDE.md → REFACTORING_SUMMARY.md |

## 🔍 Finding Information

### By Question

**"How do I...?"** → [README.md](README.md) or [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**"Why does it work this way?"** → [ARCHITECTURE.md](ARCHITECTURE.md)

**"What changed?"** → [REFACTORING_SUMMARY.md](../../REFACTORING_SUMMARY.md) or [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

**"Where's an example?"** → [examples/analysis_example.py](../../examples/analysis_example.py)

**"How do I test it?"** → [tests/analysis/](../../tests/analysis/)

### By Component

**Modules** → [core.py](core.py), [template/module.py](template/module.py)

**Validation** → [validation.py](validation.py), [exceptions.py](exceptions.py)

**Service** → [service.py](service.py), [registry.py](registry.py)

**Protocols** → [protocols.py](protocols.py)

**Utilities** → [common/](common/)

## 📞 Getting Help

1. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common patterns
2. Read [README.md](README.md) for detailed explanations
3. Study [examples/analysis_example.py](../../examples/analysis_example.py)
4. Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migration issues
5. Look at [tests/analysis/](../../tests/analysis/) for test examples

## 📝 Contributing

When adding documentation:
1. Update this index
2. Add examples to [examples/](../../examples/)
3. Write tests in [tests/analysis/](../../tests/analysis/)
4. Update [README.md](README.md) if adding features
5. Update [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common patterns

## Version

- **Analysis System Version:** 2.0.0
- **Last Updated:** 2025-01-15
- **Breaking Changes:** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
