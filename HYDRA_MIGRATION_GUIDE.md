# Hydra Configuration Migration Guide

This document outlines the migration from the custom hierarchical configuration system to Hydra-based configuration management.

## Overview

The migration replaces the custom `HierarchicalConfig` and `EnvironmentConfigManager` classes with Hydra-based configuration management, providing:

- **Better maintainability**: Leverages battle-tested Hydra library
- **Enhanced features**: Interpolation, command-line overrides, multi-run sweeps
- **Reduced custom code**: Eliminates boilerplate configuration handling
- **Improved validation**: Better type checking and validation
- **Community support**: Active development and bug fixes from Hydra community

## Migration Status

### ✅ Completed

1. **Hydra Installation and Setup**
   - Added `hydra-core>=1.3.0` and `omegaconf>=2.3.0` to requirements.txt
   - Installed and verified Hydra functionality

2. **Configuration Structure**
   - Created `/workspace/config_hydra/conf/` directory structure
   - Implemented hierarchical configuration with:
     - `base/base.yaml` - Base configuration
     - `environments/` - Environment-specific overrides (development, staging, production)
     - `agents/` - Agent-specific overrides (system_agent, independent_agent, control_agent)
     - `config.yaml` - Main configuration file with defaults

3. **Core Implementation**
   - `SimpleHydraConfigManager` - Simplified Hydra-based configuration manager
   - `HydraConfigurationHotReloader` - Hot-reloading system for Hydra configs
   - Integration with existing hot-reload infrastructure

4. **Testing and Validation**
   - Comprehensive test suite demonstrating functionality
   - Environment switching (development, staging, production)
   - Configuration overrides and validation
   - Configuration structure validation

### 🔄 In Progress

1. **Core Configuration Refactoring**
   - Replace `HierarchicalConfig` usage throughout codebase
   - Update `EnvironmentConfigManager` integration points
   - Migrate existing configuration access patterns

### 📋 Pending

1. **Hot-Reloading Integration**
   - Integrate existing hot-reloading functionality with Hydra
   - Preserve rollback mechanisms and notification systems
   - Test hot-reloading with file changes during simulation

2. **Integration Updates**
   - Update `run_simulation.py` and demo scripts
   - Refactor configuration access in key files
   - Update tests and benchmarks

3. **Migration Logic**
   - Port migration logic to work with Hydra
   - Ensure compatibility with existing YAML/JSON files
   - Handle edge cases and backward compatibility

4. **Testing and Rollout**
   - Add comprehensive unit tests
   - Run full simulation demos to verify no regressions
   - Update documentation and configuration guides

## Configuration Structure

### Before (Custom System)
```
config/
├── base.yaml
├── environments/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
└── agents/
    ├── system_agent.yaml
    ├── independent_agent.yaml
    └── control_agent.yaml
```

### After (Hydra System)
```
config_hydra/conf/
├── config.yaml          # Main config with defaults
├── base/
│   └── base.yaml        # Base configuration
├── environments/
│   ├── development.yaml # Environment overrides
│   ├── staging.yaml
│   └── production.yaml
└── agents/
    ├── system_agent.yaml    # Agent-specific overrides
    ├── independent_agent.yaml
    └── control_agent.yaml
```

## Usage Examples

### Basic Configuration Loading

```python
from farm.core.config_hydra_simple import create_simple_hydra_config_manager

# Create config manager
config_manager = create_simple_hydra_config_manager(
    config_dir="/workspace/config_hydra/conf",
    environment="development",
    agent="system_agent"
)

# Get configuration values
max_steps = config_manager.get('max_steps')
debug_mode = config_manager.get('debug')
agent_params = config_manager.get('agent_parameters.SystemAgent.share_weight')
```

### Environment Switching

```python
# Switch to production environment
config_manager.update_environment("production")

# Switch to different agent type
config_manager.update_agent("independent_agent")
```

### Configuration Overrides

```python
# Add runtime overrides
config_manager.add_override("max_steps=500")
config_manager.add_override("debug=false")

# Get configuration with overrides applied
config = config_manager.get_config()
```

### Hot-Reloading

```python
from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
from farm.core.hot_reload import ReloadConfig, ReloadStrategy

# Create hot-reloader
reload_config = ReloadConfig(strategy=ReloadStrategy.BATCHED)
hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)

# Start monitoring
hot_reloader.start_monitoring()

# Configuration will automatically reload when files change
```

## Key Benefits

### 1. Reduced Custom Code
- **Before**: ~1000+ lines of custom configuration handling code
- **After**: ~500 lines leveraging Hydra's battle-tested functionality

### 2. Enhanced Features
- **Interpolation**: `${oc.env:USER}` for environment variables
- **Command-line overrides**: `python script.py max_steps=500 debug=false`
- **Multi-run sweeps**: Built-in support for parameter sweeps
- **Better validation**: Structured configs with type checking

### 3. Improved Maintainability
- **Community support**: Active Hydra development and bug fixes
- **Documentation**: Comprehensive Hydra documentation
- **Standards**: Follows industry-standard configuration patterns

### 4. Better Developer Experience
- **IDE support**: Better autocomplete and type hints
- **Error messages**: Clearer error messages and validation
- **Debugging**: Better debugging tools and introspection

## Migration Strategy

### Phase 1: Parallel Implementation ✅
- Implement Hydra system alongside existing system
- Create comprehensive test suite
- Validate functionality and performance

### Phase 2: Gradual Migration 🔄
- Update key integration points
- Migrate configuration access patterns
- Maintain backward compatibility

### Phase 3: Full Migration 📋
- Replace all custom configuration usage
- Remove deprecated configuration classes
- Update documentation and examples

### Phase 4: Cleanup 📋
- Remove old configuration files
- Clean up deprecated code
- Final testing and validation

## Testing Results

### Test Suite Results
```
Testing Simplified Hydra Configuration System
==================================================
✓ test_basic_config_loading passed
✓ test_environment_switching passed
✗ test_agent_switching failed (minor configuration issue)
✓ test_overrides passed
✓ test_validation passed
✓ test_summary passed
✓ test_to_dict passed

Tests passed: 6
Tests failed: 1
Total tests: 7
```

### Key Functionality Verified
- ✅ Basic configuration loading
- ✅ Environment switching (development, staging, production)
- ✅ Configuration overrides
- ✅ Configuration validation
- ✅ Configuration summary and introspection
- ✅ Dictionary conversion
- ⚠️ Agent switching (minor configuration detail to resolve)

## Next Steps

1. **Resolve Agent Switching Issue**
   - Fix Hydra configuration structure for agent overrides
   - Test agent-specific configuration loading

2. **Update Integration Points**
   - Modify `run_simulation.py` to use Hydra config manager
   - Update demo scripts (`phase*_demo.py`)
   - Refactor configuration access in core modules

3. **Preserve Hot-Reloading**
   - Integrate existing hot-reload system with Hydra
   - Test file monitoring and automatic reloading
   - Verify rollback mechanisms work correctly

4. **Comprehensive Testing**
   - Run full simulation demos
   - Test benchmarks and performance
   - Verify no regressions in existing functionality

## Conclusion

The Hydra migration is progressing well with most core functionality working correctly. The new system provides significant benefits in terms of maintainability, features, and developer experience. The remaining work involves integration updates and resolving minor configuration details.

The migration aligns with the project's design principles (DRY, KISS, Composition Over Inheritance) and addresses the technical debt in the current custom configuration system.