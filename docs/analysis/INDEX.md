# Analysis Module Documentation Index

**Complete documentation for the Analysis Module system.**

---

## 📚 Main Documentation

### [README](../../farm/analysis/README.md)
**Start here** - User guide, quick start, and feature overview
- Getting started
- Common use cases
- Code examples
- Best practices

### [API Reference](./API_REFERENCE.md)
**Complete API documentation** - Detailed API reference for all classes and functions
- Service Layer (AnalysisService, AnalysisRequest, AnalysisResult)
- Core Classes (BaseAnalysisModule, DataProcessor, Context)
- Protocols (Type definitions and contracts)
- Validation (Validators and quality checks)
- Exceptions (Error handling)
- Registry (Module discovery and management)
- Common Utilities (Statistical functions, plotting helpers)
- All Analysis Modules
- Type Definitions

### [Quick Reference](./QUICK_REFERENCE.md)
**Quick lookup guide** - Fast reference for common tasks
- Common tasks with code snippets
- Module structure
- Function patterns
- Validation examples
- Exception handling
- Performance tips
- Testing patterns

### [Architecture](../../farm/analysis/ARCHITECTURE.md)
**System design** - Architecture and design patterns
- Architecture diagram
- Core components
- Data flow
- Extension points
- Design patterns
- Type safety
- Error handling strategy
- Performance considerations
- Testing strategy

---

## 📖 Specialized Guides

### Module-Specific Documentation

#### [📁 All Modules Index](./modules/README.md)
**Complete module catalog with quick reference**

#### Core Modules (Full Documentation)

- **[Population Analysis](./modules/Population.md)** - Population dynamics and composition
- **[Resources Analysis](./modules/Resources.md)** - Resource distribution and consumption
- **[Actions Analysis](./modules/Actions.md)** - Action patterns and success rates
- **[Agents Analysis](./modules/Agents.md)** - Individual agent behavior

#### Specialized Modules (Full Documentation)

- **[Learning Analysis](./modules/Learning.md)** - Learning performance and curves
- **[Spatial Analysis](./modules/Spatial.md)** - Spatial patterns and movement
- **[Temporal Analysis](./modules/Temporal.md)** - Temporal patterns and efficiency
- **[Combat Analysis](./modules/Combat.md)** - Combat metrics and patterns

#### Legacy Modules (Detailed Docs)

- **[Dominance Analysis](./Dominance.md)** - Dominance hierarchy analysis
- **[Genesis Analysis](./Genesis.md)** - Initial population generation analysis
- **[Advantage Analysis](./Advantage.md)** - Relative advantage analysis
- **[Social Behavior Analysis](./Social.md)** - Social interaction analysis

---

## 🚀 Getting Started

### First Time Users

1. **Read the [README](../../farm/analysis/README.md)** - Understand the system basics
2. **Try the [Quick Start](../../farm/analysis/README.md#quick-start)** - Run your first analysis
3. **Explore [Examples](../../examples/analysis_example.py)** - See working code
4. **Refer to [Quick Reference](./QUICK_REFERENCE.md)** - Learn common patterns

### Module Developers

1. **Review [Architecture](../../farm/analysis/ARCHITECTURE.md)** - Understand system design
2. **Study [API Reference](./API_REFERENCE.md)** - Learn the APIs
3. **Check [Module Template](../../farm/analysis/README.md#creating-a-new-analysis-module)** - Follow best practices
4. **Run [Tests](../../tests/analysis/)** - See test examples

### Advanced Users

1. **[API Reference](./API_REFERENCE.md)** - Master all APIs
2. **[Architecture](../../farm/analysis/ARCHITECTURE.md)** - Understand internals
3. **[Source Code](../../farm/analysis/)** - Read implementation
4. **[Tests](../../tests/analysis/)** - Learn from tests

---

## 📝 Additional Resources

### Code Examples

- **[Main Example](../../examples/analysis_example.py)** - Comprehensive working example
- **[Test Suite](../../tests/analysis/)** - 24 test files with usage examples
- **[Module Implementations](../../farm/analysis/)** - 14 built-in modules to learn from

### Development Guides

- **[Test Coverage Report](../../ANALYSIS_MODULE_TEST_COVERAGE_REPORT.md)** - Testing status
- **[Refactoring Summary](../../farm/analysis/REFACTORING_SUMMARY.md)** - What changed in v2.0
- **[Migration Guide](../../farm/analysis/MIGRATION_GUIDE.md)** - Upgrading from old system

### Configuration

- **[Config Service](../../farm/core/services.py)** - Configuration management
- **[Environment Variables](../../farm/analysis/README.md#configuration)** - Setup and configuration

---

## 🎯 Quick Access

### By Task

| Task | Documentation |
|------|---------------|
| Run an analysis | [Quick Reference - Running an Analysis](./QUICK_REFERENCE.md#running-an-analysis) |
| Create a module | [Quick Reference - Creating a Module](./QUICK_REFERENCE.md#creating-a-module) |
| Validate data | [API Reference - Validation](./API_REFERENCE.md#validation) |
| Handle errors | [Quick Reference - Exception Handling](./QUICK_REFERENCE.md#exception-handling) |
| Use utilities | [API Reference - Common Utilities](./API_REFERENCE.md#common-utilities) |
| Batch processing | [Quick Reference - Batch Processing](./QUICK_REFERENCE.md#batch-processing) |
| Cache results | [Quick Reference - Caching Control](./QUICK_REFERENCE.md#caching-control) |
| List modules | [Quick Reference - Listing Available Modules](./QUICK_REFERENCE.md#listing-available-modules) |

### By Component

| Component | Documentation |
|-----------|---------------|
| AnalysisService | [API Reference - Service Layer](./API_REFERENCE.md#service-layer) |
| BaseAnalysisModule | [API Reference - Core Classes](./API_REFERENCE.md#core-classes) |
| Protocols | [API Reference - Protocols](./API_REFERENCE.md#protocols) |
| Validators | [API Reference - Validation](./API_REFERENCE.md#validation) |
| Exceptions | [API Reference - Exceptions](./API_REFERENCE.md#exceptions) |
| Registry | [API Reference - Registry](./API_REFERENCE.md#registry) |
| Context | [API Reference - AnalysisContext](./API_REFERENCE.md#analysiscontext) |
| Utilities | [API Reference - Common Utilities](./API_REFERENCE.md#common-utilities) |

---

## 🔍 Search Guide

### Looking for information about...

#### **Running analyses**
→ [Quick Reference - Common Tasks](./QUICK_REFERENCE.md#common-tasks)

#### **Creating custom modules**
→ [README - Creating a New Analysis Module](../../farm/analysis/README.md#creating-a-new-analysis-module)  
→ [Quick Reference - Creating a Module](./QUICK_REFERENCE.md#creating-a-module)

#### **Service API**
→ [API Reference - Service Layer](./API_REFERENCE.md#service-layer)

#### **Validation**
→ [API Reference - Validation](./API_REFERENCE.md#validation)  
→ [Quick Reference - Validation](./QUICK_REFERENCE.md#validation)

#### **Error handling**
→ [API Reference - Exceptions](./API_REFERENCE.md#exceptions)  
→ [Quick Reference - Exception Handling](./QUICK_REFERENCE.md#exception-handling)

#### **Performance optimization**
→ [Quick Reference - Performance Tips](./QUICK_REFERENCE.md#performance-tips)  
→ [Architecture - Performance Considerations](../../farm/analysis/ARCHITECTURE.md#performance-considerations)

#### **Testing**
→ [Quick Reference - Testing](./QUICK_REFERENCE.md#testing)  
→ [Architecture - Testing Strategy](../../farm/analysis/ARCHITECTURE.md#testing-strategy)  
→ [Test Coverage Report](../../ANALYSIS_MODULE_TEST_COVERAGE_REPORT.md)

#### **Architecture and design**
→ [Architecture Document](../../farm/analysis/ARCHITECTURE.md)

#### **Statistical utilities**
→ [API Reference - Common Utilities](./API_REFERENCE.md#common-utilities)  
→ [Quick Reference - Common Utilities](./QUICK_REFERENCE.md#common-utilities)

#### **Specific modules**
→ See [Module-Specific Documentation](#module-specific-documentation) above

---

## 📊 Documentation Status

| Document | Status | Coverage |
|----------|--------|----------|
| README | ✅ Complete | User guide, quick start, examples |
| API Reference | ✅ Complete | All classes, methods, parameters |
| Quick Reference | ✅ Complete | Common tasks, patterns, tips |
| Architecture | ✅ Complete | Design, patterns, workflows |
| Test Coverage | ✅ Complete | 100% file coverage |
| Examples | ✅ Complete | Working code samples |

**Last Updated**: 2025-10-04  
**Documentation Version**: 2.0.0

---

## 🤝 Contributing

When contributing to the analysis module:

1. **Update documentation** when changing APIs
2. **Add examples** for new features
3. **Update quick reference** for common patterns
4. **Keep architecture** doc in sync with design changes
5. **Write tests** for all new code
6. **Follow** the existing module structure

---

## 📧 Support

- **Issues**: File an issue in the project repository
- **Questions**: Check existing documentation first
- **Examples**: See `examples/analysis_example.py`
- **Tests**: Browse `tests/analysis/` for usage patterns

---

**Analysis Module v2.0.0** | [Project Home](../../README.md)
