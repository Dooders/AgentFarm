# Configuration System API Reference

This document provides comprehensive API reference for the Agent Farm configuration system.

## Core Classes

### ConfigurationOrchestrator

The main orchestrator class that coordinates configuration loading, caching, and validation. This is the primary entry point for configuration management.

#### Constructor

##### `__init__(cache=None, loader=None, validator=None)`

Initialize the configuration orchestrator.

**Parameters:**
- `cache` (ConfigCache, optional): Cache instance to use (default: new ConfigCache)
- `loader` (OptimizedConfigLoader, optional): Loader instance to use (default: new OptimizedConfigLoader)
- `validator` (SafeConfigLoader, optional): Validator instance to use (default: new SafeConfigLoader)

**Example:**
```python
from farm.config import ConfigurationOrchestrator

# Use default components
orchestrator = ConfigurationOrchestrator()

# Use custom components
orchestrator = ConfigurationOrchestrator(
    cache=my_cache,
    loader=my_loader,
    validator=my_validator
)
```

#### Instance Methods

##### `load_config(environment="development", profile=None, validate=True, use_cache=True, strict_validation=False, auto_repair=False, config_dir="farm/config")`

Load configuration with full pipeline: cache → load → validate.

**Parameters:**
- `environment` (str): Environment name ("development", "production", "testing")
- `profile` (str, optional): Profile name ("benchmark", "simulation", "research")
- `validate` (bool): Whether to validate the configuration (default: True)
- `use_cache` (bool): Whether to use caching for performance (default: True)
- `strict_validation` (bool): Whether to treat warnings as errors (default: False)
- `auto_repair` (bool): Whether to attempt automatic repair of validation errors (default: False)
- `config_dir` (str): Base configuration directory (default: "farm/config")

**Returns:** `SimulationConfig` instance

**Raises:**
- `FileNotFoundError`: If required configuration files are missing
- `ValidationError`: If validation fails and auto_repair is False

**Example:**
```python
# Load development config
config = orchestrator.load_config()

# Load production with benchmark profile and validation
config = orchestrator.load_config(
    environment="production",
    profile="benchmark",
    validate=True,
    use_cache=True
)
```

##### `load_config_with_status(...)`

Load configuration with detailed status information.

**Parameters:** Same as `load_config()`

**Returns:** `Tuple[SimulationConfig, Dict[str, Any]]` - (config, status_dict)

**Example:**
```python
config, status = orchestrator.load_config_with_status(environment="production")
print(f"Success: {status['success']}")
print(f"Cached: {status['cached']}")
print(f"Errors: {len(status['errors'])}")
```

##### `invalidate_cache(environment=None)`

Invalidate cached configurations.

**Parameters:**
- `environment` (str, optional): Specific environment to invalidate (default: all)

##### `get_cache_stats()`

Get cache statistics and metrics.

**Returns:** Dictionary with cache statistics

##### `preload_common_configs(environments=None, profiles=None, config_dir="farm/config")`

Preload commonly used configurations into cache.

**Parameters:**
- `environments` (list, optional): List of environments to preload
- `profiles` (list, optional): List of profiles to preload
- `config_dir` (str): Configuration directory

### SimulationConfig

The main configuration class containing all simulation parameters.

#### Class Methods

##### `from_centralized_config(environment="development", profile=None, config_dir="config", use_cache=True)`

Load configuration from the centralized config structure.

**Parameters:**
- `environment` (str): Environment name ("development", "production", "testing")
- `profile` (str, optional): Profile name ("benchmark", "simulation", "research")
- `config_dir` (str): Base configuration directory (default: "config")
- `use_cache` (bool): Whether to use caching for performance (default: True)

**Returns:** `SimulationConfig` instance

**Example:**
```python
from farm.core.config import SimulationConfig

# Load development config
config = SimulationConfig.from_centralized_config(environment="development")

# Load production with benchmark profile
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)
```


##### `from_dict(data)`

Create configuration from a dictionary.

**Parameters:**
- `data` (dict): Configuration data dictionary

**Returns:** `SimulationConfig` instance

#### Instance Methods

##### `version_config(description=None)`

Create a versioned copy of the configuration with a unique hash.

**Parameters:**
- `description` (str, optional): Description of this configuration version

**Returns:** New `SimulationConfig` instance with versioning metadata

**Example:**
```python
versioned = config.version_config("Experiment with learning rate 0.001")
print(f"Version: {versioned.config_version}")
```

##### `save_versioned_config(directory, description=None)`

Save configuration as a versioned file.

**Parameters:**
- `directory` (str): Directory to save the configuration
- `description` (str, optional): Description of the configuration

**Returns:** Path to the saved configuration file

##### `diff_config(other)`

Compare this configuration with another and return differences.

**Parameters:**
- `other` (SimulationConfig): Configuration to compare against

**Returns:** Dictionary containing configuration differences

**Example:**
```python
diff = config1.diff_config(config2)
for key, change in diff.items():
    print(f"{key}: {change['self']} -> {change['other']}")
```

##### `to_dict()`

Convert configuration to a JSON-serializable dictionary.

**Returns:** Dictionary representation of the configuration

##### `to_yaml(file_path)`

Save configuration to a YAML file.

**Parameters:**
- `file_path` (str): Path to save the configuration

##### `copy()`

Create a deep copy of the configuration.

**Returns:** New `SimulationConfig` instance

#### Static Methods

##### `list_config_versions(directory)`

List all available configuration versions in a directory.

**Parameters:**
- `directory` (str): Directory containing versioned configs

**Returns:** List of version information dictionaries

##### `load_versioned_config(directory, version)`

Load a specific versioned configuration.

**Parameters:**
- `directory` (str): Directory containing versioned configs
- `version` (str): Version hash to load

**Returns:** `SimulationConfig` instance

##### `check_deprecated_config_files()`

Check for deprecated configuration files and warn about migration.

**Returns:** None (issues warnings if deprecated files are found)

## Template System

### ConfigTemplate

Parameterized configuration templates with variable substitution.

#### Class Methods

##### `from_yaml(file_path)`

Load a configuration template from a YAML file.

**Parameters:**
- `file_path` (str): Path to the template YAML file

**Returns:** `ConfigTemplate` instance

##### `from_config(config)`

Create a template from an existing configuration.

**Parameters:**
- `config` (SimulationConfig): Configuration to convert to template

**Returns:** `ConfigTemplate` instance

#### Instance Methods

##### `instantiate(variables)`

Instantiate the template with specific variable values.

**Parameters:**
- `variables` (dict): Dictionary mapping variable names to values

**Returns:** `SimulationConfig` instance

**Raises:** `ValueError` if required variables are missing

**Example:**
```python
template = ConfigTemplate.from_yaml("config/templates/my_template.yaml")
config = template.instantiate({
    "env_size": 200,
    "agent_count": 25
})
```

##### `get_required_variables()`

Get list of all required variables in the template.

**Returns:** List of variable names (strings)

##### `validate_variables(variables)`

Validate that all required variables are provided.

**Parameters:**
- `variables` (dict): Variables to validate

**Returns:** List of missing variables (empty if all present)

##### `to_yaml(file_path)`

Save the template to a YAML file.

**Parameters:**
- `file_path` (str): Path to save the template

### ConfigTemplateManager

Manager for configuration templates.

#### Constructor

##### `__init__(template_dir="config/templates")`

Initialize template manager.

**Parameters:**
- `template_dir` (str): Directory containing templates

#### Instance Methods

##### `save_template(name, template, description=None)`

Save a template to the template directory.

**Parameters:**
- `name` (str): Template name
- `template` (ConfigTemplate): Template to save
- `description` (str, optional): Template description

**Returns:** Path to saved template

##### `load_template(name)`

Load a template from the template directory.

**Parameters:**
- `name` (str): Template name

**Returns:** `ConfigTemplate` instance

##### `list_templates()`

List all available templates.

**Returns:** List of template information dictionaries

##### `create_experiment_configs(template_name, variable_sets, output_dir="config/experiments")`

Create multiple configuration files from a template with different variable sets.

**Parameters:**
- `template_name` (str): Name of template to use
- `variable_sets` (list): List of variable dictionaries
- `output_dir` (str): Directory to save generated configs

**Returns:** List of paths to generated config files

## Caching and Performance

### ConfigCache

Thread-safe configuration cache with automatic invalidation.

#### Constructor

##### `__init__(max_size=50, max_memory_mb=100.0)`

Initialize the configuration cache.

**Parameters:**
- `max_size` (int): Maximum number of cached configurations
- `max_memory_mb` (float): Maximum memory usage in MB

#### Instance Methods

##### `get(cache_key, filepath=None)`

Retrieve a configuration from cache.

**Parameters:**
- `cache_key` (str): Unique cache key
- `filepath` (str, optional): File path to check for modifications

**Returns:** Cached `SimulationConfig` or `None`

##### `put(cache_key, config, filepath=None)`

Store a configuration in cache.

**Parameters:**
- `cache_key` (str): Unique cache key
- `config` (SimulationConfig): Configuration to cache
- `filepath` (str, optional): Associated file path

##### `invalidate(cache_key)`

Invalidate a specific cache entry.

**Parameters:**
- `cache_key` (str): Cache key to invalidate

##### `clear()`

Clear all cached configurations.

##### `get_stats()`

Get cache statistics.

**Returns:** Dictionary with cache statistics

### OptimizedConfigLoader

Optimized configuration loader with caching.

#### Constructor

##### `__init__(cache=None)`

Initialize optimized loader.

**Parameters:**
- `cache` (ConfigCache, optional): Cache instance to use

#### Instance Methods

##### `load_centralized_config(environment="development", profile=None, config_dir="config", use_cache=True)`

Load centralized configuration with caching.

**Parameters:** Same as `SimulationConfig.from_centralized_config()`

**Returns:** `SimulationConfig` instance

##### `preload_common_configs(config_dir="config")`

Preload commonly used configurations into cache.

**Parameters:**
- `config_dir` (str): Configuration directory

### LazyConfigLoader

Lazy configuration loader that defers expensive operations.

#### Constructor

##### `__init__(loader=None)`

Initialize lazy loader.

**Parameters:**
- `loader` (OptimizedConfigLoader, optional): Underlying loader to use

#### Instance Methods

##### `configure(environment="development", profile=None, config_dir="config")`

Configure the loader parameters.

**Parameters:** Same as `load_centralized_config()`

**Returns:** Self for method chaining

##### `get_config()`

Get the configuration, loading it if necessary.

**Returns:** `SimulationConfig` instance

##### `reload()`

Force reload the configuration.

**Returns:** Fresh `SimulationConfig` instance

## Runtime Reloading

### ConfigWatcher

Watches configuration files for changes and triggers reload callbacks.

#### Constructor

##### `__init__(watch_interval=2.0)`

Initialize the configuration watcher.

**Parameters:**
- `watch_interval` (float): How often to check for file changes (seconds)

#### Instance Methods

##### `watch_file(filepath, callback)`

Start watching a configuration file for changes.

**Parameters:**
- `filepath` (str): Path to the configuration file
- `callback` (callable): Function called when config changes

##### `unwatch_file(filepath, callback=None)`

Stop watching a configuration file.

**Parameters:**
- `filepath` (str): Path to stop watching
- `callback` (callable, optional): Specific callback to remove

##### `start()`

Start the file watching thread.

##### `stop()`

Stop the file watching thread.

### ReloadableConfig

A configuration wrapper that supports runtime reloading.

#### Constructor

##### `__init__(config, watcher=None)`

Initialize a reloadable configuration.

**Parameters:**
- `config` (SimulationConfig): Initial configuration
- `watcher` (ConfigWatcher, optional): ConfigWatcher instance

#### Instance Methods

##### `watch_file(filepath)`

Watch a configuration file for changes.

**Parameters:**
- `filepath` (str): Path to watch

##### `add_change_callback(callback)`

Add a callback to be called when configuration changes.

**Parameters:**
- `callback` (callable): Function to call with new config

##### `remove_change_callback(callback)`

Remove a change callback.

**Parameters:**
- `callback` (callable): Callback to remove

##### `reload_from_file(filepath)`

Manually reload configuration from a file.

**Parameters:**
- `filepath` (str): Path to load from

## Monitoring and Observability

### ConfigMonitor

Configuration system monitor with metrics and logging.

#### Constructor

##### `__init__(logger=None)`

Initialize the configuration monitor.

**Parameters:**
- `logger` (logging.Logger, optional): Logger instance to use

#### Instance Methods

##### `log_config_operation(...)`

Log a configuration operation with metrics.

##### `get_metrics_summary()`

Get summary statistics of configuration operations.

**Returns:** Dictionary with metrics summary

##### `get_recent_errors(limit=10)`

Get recent configuration errors.

**Parameters:**
- `limit` (int): Maximum number of errors to return

**Returns:** List of error metrics

##### `get_performance_trends(operation=None)`

Get performance trends for operations.

**Parameters:**
- `operation` (str, optional): Specific operation to analyze

**Returns:** Dictionary with performance trend data

##### `export_metrics(filepath)`

Export metrics to a JSON file.

**Parameters:**
- `filepath` (str): Path to export metrics

### Utility Functions

##### `get_global_cache()`

Get the global configuration cache instance.

**Returns:** `ConfigCache` instance

##### `get_global_monitor()`

Get the global configuration monitor instance.

**Returns:** `ConfigMonitor` instance

##### `create_reloadable_config(config_or_path, watch_path=None)`

Create a reloadable configuration.

**Parameters:**
- `config_or_path` (SimulationConfig or str): Initial config or path
- `watch_path` (str, optional): Path to watch for changes

**Returns:** `ReloadableConfig` instance

## Global Orchestrator Functions

### `get_global_orchestrator()`

Get the global configuration orchestrator instance.

**Returns:** `ConfigurationOrchestrator` instance

**Example:**
```python
from farm.config import get_global_orchestrator

orchestrator = get_global_orchestrator()
config = orchestrator.load_config(environment="production")
```

### `load_config(environment="development", profile=None, **kwargs)`

Convenience function to load configuration using the global orchestrator.

**Parameters:** Same as `ConfigurationOrchestrator.load_config()`

**Returns:** `SimulationConfig` instance

**Example:**
```python
from farm.config import load_config

# Simple loading using global orchestrator
config = load_config()
config = load_config(environment="production", profile="benchmark")
```

##### `get_config_system_health()`

Get overall health status of the configuration system.

**Returns:** Dictionary with health metrics

##### `log_config_system_status()`

Log the current status of the configuration system.

## Command-Line Interface

### farm.config.cli

The configuration CLI provides command-line access to all configuration features.

#### Commands

##### `version`

Configuration versioning commands.

**Subcommands:**
- `create`: Create a versioned configuration
- `list`: List versioned configurations
- `load`: Load a specific version

##### `template`

Configuration templating commands.

**Subcommands:**
- `create`: Create a template from config
- `list`: List available templates
- `instantiate`: Instantiate a template
- `batch`: Create batch of configs from template

##### `diff`

Compare two configurations.

##### `watch`

Watch configuration files for changes.

**Subcommands:**
- `start`: Start watching a config file
- `status`: Show watch status

## Error Handling

The configuration system includes comprehensive error handling:

- **FileNotFoundError**: Raised when configuration files don't exist
- **ValueError**: Raised for invalid template variables or malformed data
- **DeprecationWarning**: Issued for deprecated API usage
- **ValidationError**: Raised for invalid configuration data

All errors include detailed error messages and context information.

## Best Practices

1. **Use the ConfigurationOrchestrator** for all configuration loading (coordinates caching, loading, and validation)
2. **Enable caching** for production deployments to improve performance
3. **Version important configurations** for reproducible experiments
4. **Use templates** for systematic parameter sweeps
5. **Monitor configuration health** in production systems
6. **Handle deprecation warnings** by migrating to new APIs

## Migration Guide

### From Legacy Config Loading

**Old approach:**
```python
config = SimulationConfig.from_yaml("config.yaml")
```

**New approach:**
```python
config = SimulationConfig.from_centralized_config(
    environment="development"
)
```

### From Manual Config Management

**Old approach:**
```python
# Manual config creation and modification
config = SimulationConfig()
config.width = 200
config.height = 200
```

**New approach:**
```python
# Use environment/profile system
config = SimulationConfig.from_centralized_config(
    environment="production"
)
```

## Performance Considerations

- **Caching**: Enable caching for frequently accessed configurations
- **Lazy loading**: Use `LazyConfigLoader` for configurations loaded on-demand
- **Monitoring**: Enable monitoring to track performance bottlenecks
- **Preloading**: Preload common configurations at startup

## Thread Safety

All configuration system components are thread-safe:
- `ConfigCache` uses locks for thread-safe access
- `ConfigMonitor` safely handles concurrent operations
- `ConfigWatcher` runs in a separate daemon thread

## Memory Management

The cache system includes automatic memory management:
- LRU eviction when cache size limits are reached
- Memory usage limits to prevent excessive memory consumption
- Automatic cleanup of invalid cached entries
