# Hydra Configuration System Guide

This guide provides comprehensive documentation for the Hydra-based configuration system that replaces the custom hierarchical configuration management.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration Structure](#configuration-structure)
4. [Basic Usage](#basic-usage)
5. [Environment Management](#environment-management)
6. [Agent Configuration](#agent-configuration)
7. [Configuration Overrides](#configuration-overrides)
8. [Hot-Reloading](#hot-reloading)
9. [Validation](#validation)
10. [Migration Guide](#migration-guide)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## Overview

The Hydra configuration system provides a modern, maintainable approach to configuration management with the following benefits:

- **Hierarchical Configuration**: Base → Environment → Agent-specific overrides
- **Hot-Reloading**: Automatic configuration reloading when files change
- **Validation**: Built-in configuration validation and error reporting
- **Interpolation**: Support for environment variables and dynamic values
- **Command-Line Overrides**: Runtime configuration overrides
- **Multi-Run Sweeps**: Built-in support for parameter sweeps (future)

## Installation

The Hydra configuration system requires the following dependencies:

```bash
pip install hydra-core>=1.3.0 omegaconf>=2.3.0
```

These are automatically installed when you install the project dependencies.

## Configuration Structure

The Hydra configuration system uses a hierarchical directory structure:

```
config_hydra/conf/
├── config.yaml              # Main configuration with defaults
├── base/
│   └── base.yaml           # Base configuration (100+ parameters)
├── environments/           # Environment-specific overrides
│   ├── development.yaml    # Development settings
│   ├── staging.yaml        # Staging settings
│   └── production.yaml     # Production settings
└── agents/                 # Agent-specific behavior overrides
    ├── system_agent.yaml   # Cooperative behavior
    ├── independent_agent.yaml # Independent behavior
    └── control_agent.yaml  # Balanced behavior
```

### Main Configuration File

The `config.yaml` file defines the default configuration and composition:

```yaml
# @package _global_
# Main configuration file for Hydra-based configuration system

defaults:
  - base/base
  - environments: development
  - agents: system_agent
  - _self_

# Global configuration overrides can be added here
```

### Base Configuration

The `base/base.yaml` file contains all default configuration parameters:

```yaml
# @package _global_
# Base Configuration

# Simulation settings
simulation_id: "base-simulation"
max_steps: 1000
environment: "base"

# Environment dimensions
width: 100
height: 100

# Agent settings
system_agents: 10
independent_agents: 10
control_agents: 10

# ... (100+ more parameters)
```

### Environment Configurations

Environment-specific configurations override base settings:

**Development Environment** (`environments/development.yaml`):
```yaml
# @package _global_
# Development Environment Configuration

# Enable debug mode and verbose logging
debug: true
verbose_logging: true

# Reduce simulation complexity for faster development
max_steps: 100
max_population: 50

# Use in-memory database for faster development
use_in_memory_db: true
persist_db_on_completion: false
```

**Production Environment** (`environments/production.yaml`):
```yaml
# @package _global_
# Production Environment Configuration

# Disable debug mode for production
debug: false
verbose_logging: false

# Full simulation complexity
max_steps: 1000
max_population: 300

# Use persistent database for production
use_in_memory_db: false
persist_db_on_completion: true
```

### Agent Configurations

Agent-specific configurations override environment settings:

**System Agent** (`agents/system_agent.yaml`):
```yaml
# @package _global_
# System Agent Specific Configuration

# System agents are more cooperative and efficient
agent_parameters:
  SystemAgent:
    gather_efficiency_multiplier: 0.6
    gather_cost_multiplier: 0.3
    min_resource_threshold: 0.15
    share_weight: 0.4
    attack_weight: 0.02
```

## Basic Usage

### Creating a Configuration Manager

```python
from farm.core.config_hydra_simple import create_simple_hydra_config_manager

# Create config manager with default settings
config_manager = create_simple_hydra_config_manager(
    config_dir="/workspace/config_hydra/conf",
    environment="development",
    agent="system_agent"
)
```

### Accessing Configuration Values

```python
# Get simple values
max_steps = config_manager.get('max_steps')
debug_mode = config_manager.get('debug')

# Get nested values using dot notation
share_weight = config_manager.get('agent_parameters.SystemAgent.share_weight')
redis_host = config_manager.get('redis.host')

# Get values with defaults
timeout = config_manager.get('timeout', default=30)
```

### Converting to Dictionary

```python
# Convert entire configuration to dictionary
config_dict = config_manager.to_dict()

# Use with existing SimulationConfig
from farm.core.config import SimulationConfig
sim_config = SimulationConfig.from_dict(config_dict)
```

## Environment Management

### Switching Environments

```python
# Switch to production environment
config_manager.update_environment("production")

# Switch to staging environment
config_manager.update_environment("staging")

# Switch back to development
config_manager.update_environment("development")
```

### Environment Detection

The system automatically detects the environment from environment variables:

1. `FARM_ENVIRONMENT`
2. `ENVIRONMENT`
3. `ENV`
4. `NODE_ENV`
5. `PYTHON_ENV`

If none are set, defaults to `development`.

### Available Environments

```python
# Get list of available environments
environments = config_manager.get_available_environments()
print(environments)  # ['development', 'staging', 'production']
```

## Agent Configuration

### Switching Agent Types

```python
# Switch to independent agent
config_manager.update_agent("independent_agent")

# Switch to control agent
config_manager.update_agent("control_agent")

# Switch back to system agent
config_manager.update_agent("system_agent")
```

### Available Agents

```python
# Get list of available agent types
agents = config_manager.get_available_agents()
print(agents)  # ['control_agent', 'independent_agent', 'system_agent']
```

### Agent-Specific Parameters

```python
# Get agent-specific parameters
system_params = config_manager.get('agent_parameters.SystemAgent')
independent_params = config_manager.get('agent_parameters.IndependentAgent')
control_params = config_manager.get('agent_parameters.ControlAgent')
```

## Configuration Overrides

### Runtime Overrides

```python
# Create config manager with initial overrides
config_manager = create_simple_hydra_config_manager(
    config_dir="/workspace/config_hydra/conf",
    environment="development",
    overrides=["max_steps=200", "debug=false"]
)

# Add additional overrides
config_manager.add_override("max_population=100")

# Remove overrides
config_manager.remove_override("max_population=100")
```

### Command-Line Overrides

```bash
# Run simulation with overrides
python run_simulation_hydra.py \
    --environment production \
    --agent independent_agent \
    --override "max_steps=500" \
    --override "debug=false"
```

### Override Examples

```python
# Simple value overrides
"max_steps=500"
"debug=false"
"width=200"

# Nested value overrides
"agent_parameters.SystemAgent.share_weight=0.5"
"redis.host=localhost"
"visualization.canvas_size=[800,600]"
```

## Hot-Reloading

### Basic Hot-Reloading

```python
from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
from farm.core.config.hot_reload import ReloadConfig, ReloadStrategy

# Create hot-reloader
reload_config = ReloadConfig(
    strategy=ReloadStrategy.BATCHED,
    batch_delay=1.0,
    validate_on_reload=True,
    enable_rollback=True
)

hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)

# Start monitoring
hot_reloader.start_monitoring()

# Configuration will automatically reload when files change
# ... your application code ...

# Stop monitoring
hot_reloader.stop_monitoring()
```

### Reload Strategies

**Immediate Reload**:
```python
reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
```

**Batched Reload** (recommended):
```python
reload_config = ReloadConfig(
    strategy=ReloadStrategy.BATCHED,
    batch_delay=1.0,
    max_batch_size=10
)
```

**Scheduled Reload**:
```python
reload_config = ReloadConfig(
    strategy=ReloadStrategy.SCHEDULED,
    schedule_interval=5.0
)
```

**Manual Reload**:
```python
reload_config = ReloadConfig(strategy=ReloadStrategy.MANUAL)

# Manually trigger reload
hot_reloader.manual_reload()
```

### Notification System

```python
def on_config_reload(notification):
    print(f"Config reloaded: {notification.message}")
    if notification.error:
        print(f"Error: {notification.error}")

# Add notification callback
hot_reloader.add_notification_callback(on_config_reload)
```

## Validation

### Configuration Validation

```python
# Validate current configuration
errors = config_manager.validate_configuration()

if errors:
    print("Configuration validation failed:")
    for area, error_list in errors.items():
        print(f"  {area}:")
        for error in error_list:
            print(f"    - {error}")
else:
    print("Configuration validation passed!")
```

### Configuration Summary

```python
# Get configuration summary
summary = config_manager.get_configuration_summary()

print(f"Environment: {summary['environment']}")
print(f"Agent: {summary['agent']}")
print(f"Available environments: {summary['available_environments']}")
print(f"Available agents: {summary['available_agents']}")
print(f"Config keys: {len(summary['config_keys'])}")
```

## Migration Guide

### From Custom Configuration System

If you're migrating from the custom hierarchical configuration system:

1. **Replace EnvironmentConfigManager**:
   ```python
   # Old way
   from farm.core.config import EnvironmentConfigManager
   config_manager = EnvironmentConfigManager("config/base.yaml")
   
   # New way
   from farm.core.config_hydra_simple import create_simple_hydra_config_manager
   config_manager = create_simple_hydra_config_manager()
   ```

2. **Update Configuration Access**:
   ```python
   # Old way
   config = config_manager.get_effective_config()
   max_steps = config['max_steps']
   
   # New way
   max_steps = config_manager.get('max_steps')
   ```

3. **Update Hot-Reloading**:
   ```python
   # Old way
   from farm.core.config import ConfigurationHotReloader
   hot_reloader = ConfigurationHotReloader(config_manager)
   
   # New way
   from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
   hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
   ```

### Configuration File Migration

1. **Base Configuration**: Copy parameters from `config/base.yaml` to `config_hydra/conf/base/base.yaml`
2. **Environment Configs**: Copy environment-specific settings to `config_hydra/conf/environments/`
3. **Agent Configs**: Copy agent-specific settings to `config_hydra/conf/agents/`

## Best Practices

### Configuration Organization

1. **Keep base configuration minimal**: Only include truly default values
2. **Use environment-specific overrides**: Override only what's different
3. **Group related parameters**: Use nested structures for related settings
4. **Document configuration options**: Add comments explaining parameter purposes

### Performance Considerations

1. **Lazy loading**: Configuration is loaded only when needed
2. **Caching**: Configuration is cached until files change
3. **Validation**: Enable validation only when needed
4. **Hot-reloading**: Use batched strategy for better performance

### Security Considerations

1. **Environment variables**: Use `${oc.env:VAR}` for sensitive data
2. **File permissions**: Restrict access to configuration files
3. **Validation**: Always validate configuration before use
4. **Rollback**: Enable rollback for failed configurations

## Troubleshooting

### Common Issues

**Configuration not loading**:
```python
# Check if config directory exists
import os
config_dir = "/workspace/config_hydra/conf"
if not os.path.exists(config_dir):
    print(f"Config directory not found: {config_dir}")
```

**Environment switching not working**:
```python
# Check available environments
environments = config_manager.get_available_environments()
print(f"Available environments: {environments}")

# Check if environment file exists
env_file = f"{config_dir}/environments/{environment}.yaml"
if not os.path.exists(env_file):
    print(f"Environment file not found: {env_file}")
```

**Hot-reloading not working**:
```python
# Check if monitoring is active
if hot_reloader.is_monitoring():
    print("Monitoring is active")
else:
    print("Monitoring is not active")

# Check reload statistics
stats = hot_reloader.get_reload_stats()
print(f"Reload stats: {stats}")
```

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create config manager with debug logging
config_manager = create_simple_hydra_config_manager(
    config_dir="/workspace/config_hydra/conf",
    environment="development"
)
```

### Error Handling

```python
try:
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development"
    )
except Exception as e:
    print(f"Failed to create config manager: {e}")
    # Handle error appropriately
```

## Examples

### Complete Example

```python
#!/usr/bin/env python3
"""
Complete example of using the Hydra configuration system.
"""

import sys
sys.path.append('/workspace')

from farm.core.config_hydra_simple import create_simple_hydra_config_manager
from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
from farm.core.config.hot_reload import ReloadConfig, ReloadStrategy

def main():
    # Create configuration manager
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        agent="system_agent",
        overrides=["max_steps=200"]
    )
    
    # Print configuration summary
    summary = config_manager.get_configuration_summary()
    print(f"Environment: {summary['environment']}")
    print(f"Agent: {summary['agent']}")
    print(f"Max steps: {config_manager.get('max_steps')}")
    print(f"Debug mode: {config_manager.get('debug')}")
    
    # Set up hot-reloading
    reload_config = ReloadConfig(
        strategy=ReloadStrategy.BATCHED,
        batch_delay=1.0,
        validate_on_reload=True,
        enable_rollback=True
    )
    
    hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
    
    # Add notification callback
    def on_reload(notification):
        print(f"Config reloaded: {notification.message}")
    
    hot_reloader.add_notification_callback(on_reload)
    
    # Start monitoring
    hot_reloader.start_monitoring()
    print("Hot-reloading started. Modify configuration files to see changes.")
    
    try:
        # Your application code here
        import time
        time.sleep(10)  # Simulate application running
    finally:
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("Hot-reloading stopped.")

if __name__ == "__main__":
    main()
```

This guide provides comprehensive documentation for using the Hydra configuration system. For more advanced usage, refer to the [Hydra documentation](https://hydra.cc/) and the project's test files.