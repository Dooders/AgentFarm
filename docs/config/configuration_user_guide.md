# Configuration System User Guide

This guide provides practical instructions for using the Agent Farm configuration system effectively.

## Getting Started

### Basic Configuration Loading

The easiest way to load a configuration is using the orchestrator pattern:

```python
from farm.config import ConfigurationOrchestrator

# Create orchestrator for configuration management
orchestrator = ConfigurationOrchestrator()

# Load development configuration
config = orchestrator.load_config()

# Load production configuration
config = orchestrator.load_config(environment="production")

# Load with a specific profile
config = orchestrator.load_config(
    environment="production",
    profile="benchmark"
)

# Load with validation options
config = orchestrator.load_config(
    environment="development",
    validate=True,
    use_cache=True,
    auto_repair=False
)
```

### Legacy Loading (Still Supported)

For backward compatibility, the old methods still work:

```python
from farm.core.config import SimulationConfig

# These methods still work but use the orchestrator internally
config = SimulationConfig.from_centralized_config()
config = SimulationConfig.from_centralized_config(environment="production")
```

### Understanding the Configuration Hierarchy

Configurations are loaded with this precedence (highest to lowest):
1. **Profile overrides** (e.g., benchmark settings)
2. **Environment overrides** (e.g., production settings)
3. **Base configuration** (default.yaml)

## Environment Management

### Available Environments

- **`development`**: Optimized for development with smaller simulations and debugging enabled
- **`production`**: Optimized for performance with larger simulations and safety settings
- **`testing`**: Minimal configuration for fast unit tests and CI/CD

### Switching Environments

```python
# Development (default)
config = SimulationConfig.from_centralized_config()

# Production
config = SimulationConfig.from_centralized_config(environment="production")

# Testing
config = SimulationConfig.from_centralized_config(environment="testing")
```

### Custom Environments

To add a custom environment:

1. Create a new YAML file in `config/environments/`
2. Override only the settings that differ from the base configuration

Example `config/environments/staging.yaml`:
```yaml
width: 150
height: 150
debug: false
max_population: 200
```

## Profile Management

### Available Profiles

- **`benchmark`**: Performance benchmarking with optimized settings
- **`simulation`**: Standard simulation runs with balanced settings
- **`research`**: Research experiments with advanced features

### Using Profiles

```python
# Research profile for experiments
config = SimulationConfig.from_centralized_config(
    environment="development",
    profile="research"
)

# Benchmark profile for performance testing
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)
```

### Custom Profiles

To create a custom profile:

1. Create a new YAML file in `config/profiles/`
2. Define settings specific to your use case

Example `config/profiles/experiment.yaml`:
```yaml
learning_rate: 0.0005
gamma: 0.99
epsilon_decay: 0.999
dqn_hidden_size: 256
max_steps: 5000
```

## Configuration Versioning

### Why Version Configurations?

Versioning ensures that experiments are reproducible by creating immutable snapshots of configurations.

### Creating Versioned Configurations

```python
from farm.config import ConfigurationOrchestrator

# Load a configuration using orchestrator
orchestrator = ConfigurationOrchestrator()
config = orchestrator.load_config(
    environment="production",
    profile="research"
)

# Create a versioned snapshot
versioned_config = config.version_config(
    description="Experiment: testing new learning algorithm"
)

print(f"Version hash: {versioned_config.config_version}")
print(f"Created: {versioned_config.config_created_at}")
```

### Saving and Loading Versions

```python
# Save versioned configuration
filepath = versioned_config.save_versioned_config(
    directory="config/experiments",
)

# List all versions
versions = SimulationConfig.list_config_versions("config/experiments")
for v in versions:
    print(f"{v['version'][:8]}: {v['description']} ({v['created_at']})")

# Load a specific version
old_config = SimulationConfig.load_versioned_config(
    directory="config/experiments",
    version="a1b2c3d4..."  # Full version hash
)
```

### Best Practices for Versioning

1. **Version important experiments**: Always version configs for research experiments
2. **Use descriptive names**: Include clear descriptions of what the version contains
3. **Organize by project**: Use different directories for different research projects
4. **Keep version directories**: Don't delete old versions - they ensure reproducibility

## Configuration Templates

### What are Templates?

Templates allow you to create parameterized configurations that can be instantiated with different values, perfect for systematic parameter sweeps.

### Creating Templates

```python
from farm.config import ConfigurationOrchestrator
from farm.config.template import ConfigTemplate, ConfigTemplateManager

# Load a base configuration using orchestrator
orchestrator = ConfigurationOrchestrator()
base_config = orchestrator.load_config()

# Create a template from the configuration
template = ConfigTemplate.from_config(base_config)

# Modify the template to add variables
template_dict = template.template_dict
template_dict['width'] = '{{env_width}}'
template_dict['height'] = '{{env_width}}'  # Same variable for square environments
template_dict['system_agents'] = '{{agent_count}}'

# Recreate template with variables
template = ConfigTemplate(template_dict)

# Save the template
manager = ConfigTemplateManager()
manager.save_template(
    name="population_study",
    template=template,
    description="Template for studying agent population effects"
)
```

### Using Templates

```python
# Load a template
manager = ConfigTemplateManager()
template = manager.load_template("population_study")

# Check what variables are needed
required_vars = template.get_required_variables()
print(f"Required variables: {required_vars}")

# Instantiate with specific values
config = template.instantiate({
    'env_width': 200,
    'agent_count': 25
})

print(f"Created config: {config.width}x{config.height}, {config.system_agents} agents")
```

### Batch Template Instantiation

```python
# Create multiple configurations with different parameters
variable_sets = [
    {'env_width': 100, 'agent_count': 10},
    {'env_width': 150, 'agent_count': 15},
    {'env_width': 200, 'agent_count': 20},
    {'env_width': 250, 'agent_count': 25}
]

# Generate all configurations at once
config_files = manager.create_experiment_configs(
    template_name="population_study",
    variable_sets=variable_sets,
    output_dir="config/generated_experiments"
)

print(f"Generated {len(config_files)} experiment configurations")
for path in config_files:
    print(f"  {path}")
```

### Template File Format

Templates are stored as YAML files with this structure:

```yaml
_metadata:
  name: template_name
  description: "Template description"
  required_variables:
    - var1
    - var2

template:
  width: "{{var1}}"
  height: "{{var1}}"
  system_agents: "{{var2}}"
  # ... other config settings
```

## Runtime Configuration Reloading

### Why Runtime Reloading?

Runtime reloading allows you to modify configuration files while your simulation is running and have the changes applied automatically.

### Basic Usage

```python
from farm.config.watcher import create_reloadable_config

# Create a reloadable configuration
reloadable = create_reloadable_config("config.yaml")

# Watch the file for changes
reloadable.watch_file("config.yaml")

# Use the configuration
config = reloadable.config
print(f"Current width: {config.width}")

# The configuration will automatically update when config.yaml changes
# You can access it anytime with reloadable.config
```

### Advanced Usage with Callbacks

```python
from farm.config.watcher import create_reloadable_config

def on_config_change(new_config):
    print(f"Configuration updated! New width: {new_config.width}")
    # Update your simulation with new parameters
    simulation.update_parameters(new_config)

reloadable = create_reloadable_config("config.yaml")
reloadable.add_change_callback(on_config_change)
reloadable.watch_file("config.yaml")

# Run your simulation - it will automatically adapt to config changes
simulation.run(reloadable.config)
```

### Manual Reloading

```python
# Force a reload from disk
reloadable.reload_from_file("config.yaml")

# Or reload using the configured path
reloadable.reload()
```

## Configuration Comparison

### Comparing Configurations

```python
from farm.config import ConfigurationOrchestrator

# Load two configurations to compare
orchestrator = ConfigurationOrchestrator()
config1 = orchestrator.load_config(environment="development")
config2 = orchestrator.load_config(environment="production")

# Get differences
differences = config1.diff_config(config2)

print(f"Found {len(differences)} differences:")
for key, change in differences.items():
    print(f"  {key}:")
    print(f"    Config 1: {change['self']}")
    print(f"    Config 2: {change['other']}")
```

### Using the CLI for Comparison

```bash
# Compare two config files
python -m farm.config.cli diff farm/config/versions/config_abc123.yaml farm/config/versions/config_def456.yaml

# Compare a config file with a versioned config
python -m farm.config.cli diff farm/config/default.yaml abc123 --version-dir farm/config/versions
```

## Performance Optimization

### Enabling Caching

```python
# Caching is enabled by default
config = SimulationConfig.from_centralized_config(
    environment="production",
    use_cache=True  # Default is True
)

# Disable caching if needed
config = SimulationConfig.from_centralized_config(
    environment="production",
    use_cache=False
)
```

### Lazy Loading

```python
from farm.config.cache import LazyConfigLoader
from farm.config import ConfigurationOrchestrator

# Create a lazy loader with orchestrator
orchestrator = ConfigurationOrchestrator()
lazy_loader = LazyConfigLoader(loader=orchestrator._loader)
lazy_loader.configure(environment="production", profile="research")

# Configuration is loaded only when first accessed
config = lazy_loader.get_config()  # Loads here

# Later access uses cached version
config2 = lazy_loader.get_config()  # No loading

# Force reload
config3 = lazy_loader.reload()  # Reloads from disk
```

### Preloading Common Configurations

```python
from farm.config import ConfigurationOrchestrator

orchestrator = ConfigurationOrchestrator()

# Preload commonly used configurations at startup
orchestrator.preload_common_configs()

# These will now load instantly from cache
config = orchestrator.load_config(environment="production")
```

## Monitoring and Debugging

### Health Monitoring

```python
from farm.config.monitor import get_config_system_health, log_config_system_status

# Get current health status
health = get_config_system_health()
print(f"Status: {health['status']}")
print(f"Success rate: {health['success_rate']:.1%}")
print(f"Average operation time: {health['avg_operation_time']:.3f}s")

# Log status to configured logger
log_config_system_status()
```

### Performance Monitoring

```python
from farm.config.monitor import get_global_monitor

monitor = get_global_monitor()

# Get performance metrics
summary = monitor.get_metrics_summary()
print(f"Total operations: {summary['total_operations']}")
print(f"Cache hit rate: {summary['cache_hit_rate']:.1%}")

# Get recent errors
errors = monitor.get_recent_errors(limit=5)
for error in errors:
    print(f"Error: {error.operation} - {error.error_type}")

# Export metrics for analysis
monitor.export_metrics("config_metrics.json")
```

### CLI Monitoring

```bash
# Watch a configuration file
python -m farm.config.cli watch start farm/config/default.yaml --verbose

# Check watch status
python -m farm.config.cli watch status
```

## Command-Line Interface

The configuration CLI provides access to all features from the command line.

### Version Management

```bash
# Create a versioned configuration
python -m farm.config.cli version create \
  --environment production \
  --profile benchmark \
  --description "Production benchmark config" \
  --output-dir farm/config/versions

# List all versions
python -m farm.config.cli version list --directory farm/config/versions

# Load a specific version (prints to stdout)
python -m farm.config.cli version load a1b2c3d4 \
  --directory farm/config/versions
```

### Template Operations

```bash
# Create a template
python -m farm.config.cli template create my_template \
  --environment development \
  --description "Development template"

# List templates
python -m farm.config.cli template list

# Instantiate a template
python -m farm.config.cli template instantiate my_template output.yaml \
  --variables env_size=200 agent_count=25

# Batch generate configs
python -m farm.config.cli template batch my_template \
  --variable-file variables.json \
  --output-dir farm/config/experiments
```

### File Watching

```bash
# Start watching a config file
python -m farm.config.cli watch start farm/config/default.yaml --verbose

# Check watch status
python -m farm.config.cli watch status
```

## Troubleshooting

### Common Issues

**Configuration not found:**
```
FileNotFoundError: Base configuration not found: config/default.yaml
```
**Solution:** Ensure the config directory structure exists and files are in place.

**Template variables missing:**
```
ValueError: Missing required variable: env_size
```
**Solution:** Check that all required variables are provided when instantiating templates.

**Cache issues:**
```
# Configuration seems stale
```
**Solution:** Clear the cache or disable caching temporarily.

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check cache status:**
   ```python
   from farm.config.cache import get_global_cache
   cache = get_global_cache()
   stats = cache.get_stats()
   print(stats)
   ```

3. **Monitor operations:**
   ```python
   from farm.config.monitor import get_global_monitor
   monitor = get_global_monitor()
   summary = monitor.get_metrics_summary()
   print(summary)
   ```

4. **Validate configurations:**
   ```python
   # Check if config loads without errors
   try:
       config = SimulationConfig.from_centralized_config(...)
       print("Configuration loaded successfully")
   except Exception as e:
       print(f"Configuration error: {e}")
   ```

## Advanced Usage

### Custom Configuration Classes

```python
from farm.core.config import SimulationConfig

class CustomConfig(SimulationConfig):
    """Custom configuration with additional parameters."""

    def __init__(self, custom_param="default", **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def to_dict(self):
        result = super().to_dict()
        result['custom_param'] = self.custom_param
        return result

    @classmethod
    def from_dict(cls, data):
        custom_param = data.pop('custom_param', 'default')
        config = super().from_dict(data)
        config.custom_param = custom_param
        return config
```

### Integration with Experiment Frameworks

```python
# Example integration with a parameter sweep framework
def run_parameter_sweep():
    template = load_template("parameter_sweep_template")

    results = []
    for learning_rate in [0.001, 0.01, 0.1]:
        for gamma in [0.9, 0.95, 0.99]:
            # Instantiate config with parameters
            config = template.instantiate({
                'learning_rate': learning_rate,
                'gamma': gamma
            })

            # Version the config
            versioned_config = config.version_config(
                f"LR={learning_rate}, Gamma={gamma}"
            )

            # Save versioned config
            config_path = versioned_config.save_versioned_config("config/sweep")

            # Run experiment
            result = run_experiment(config)
            result['config_version'] = versioned_config.config_version
            results.append(result)

    return results
```

This guide covers the most common usage patterns. For complete API documentation, see `docs/configuration_api.md`.
