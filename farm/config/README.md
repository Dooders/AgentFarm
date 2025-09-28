# Centralized Configuration System

This directory contains the centralized configuration system for the Agent Farm Simulation Framework. The system provides a hierarchical configuration structure that supports environment-specific and profile-specific overrides.

## Directory Structure

```
config/
├── default.yaml              # Base configuration with all default values
├── environments/             # Environment-specific overrides
│   ├── development.yaml      # Development environment settings
│   ├── production.yaml       # Production environment settings
│   └── testing.yaml          # Testing/CI environment settings
├── profiles/                 # Use-case specific profiles
│   ├── benchmark.yaml        # Performance benchmarking settings
│   ├── simulation.yaml       # Standard simulation runs
│   └── research.yaml         # Research and experimentation settings
├── schema.json               # JSON schema for configuration validation
└── README.md                 # This file
```

## Configuration Hierarchy

Configurations are loaded with the following precedence (highest to lowest):

1. **Profile overrides** (highest precedence)
2. **Environment overrides**
3. **Base configuration** (default.yaml)

## Usage

### Loading Configuration in Code

```python
from farm.core.config import SimulationConfig

# Load development configuration
config = SimulationConfig.from_centralized_config(environment="development")

# Load production configuration with benchmark profile
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)

# Load testing configuration with research profile
config = SimulationConfig.from_centralized_config(
    environment="testing",
    profile="research"
)
```

### Environment Descriptions

#### Development
- Smaller simulation size for faster iteration
- In-memory database for quick testing
- Full debugging enabled
- Optimized for development workflow

#### Production
- Larger simulation size for comprehensive testing
- Persistent database with safety settings
- Performance optimizations
- Minimal debugging output

#### Testing
- Minimal simulation size for fast unit tests
- In-memory database
- Fixed seeds for reproducible tests
- Minimal visualization

### Profile Descriptions

#### Benchmark
- Medium simulation size for balanced benchmarking
- Performance-optimized settings
- Reduced learning parameters for faster convergence
- Minimal visualization for performance

#### Simulation
- Standard simulation settings
- Balanced agent distribution
- Persistent database for analysis
- Standard visualization

#### Research
- Large simulation size for comprehensive research
- Full debugging and logging
- Advanced learning parameters
- Enhanced curriculum learning
- Detailed visualization

## Configuration Validation

The `schema.json` file contains a JSON schema that can be used to validate configuration files. This ensures that all required fields are present and values are within acceptable ranges.

## Configuration Architecture

The centralized configuration system provides hierarchical configuration management with environment-specific and profile-specific overrides. All configuration files are organized in this directory structure for maximum maintainability and flexibility.

## Adding New Environments or Profiles

### New Environment
1. Create a new YAML file in `config/environments/`
2. Override only the settings that differ from the base configuration
3. Follow the same structure as existing environment files

### New Profile
1. Create a new YAML file in `config/profiles/`
2. Override settings specific to your use case
3. Profiles have the highest precedence and will override environment settings

## Advanced Features (Phase 3)

### Configuration Versioning

Create versioned snapshots of configurations for reproducible experiments:

```python
from farm.core.config import SimulationConfig

# Create a versioned configuration
config = SimulationConfig.from_centralized_config(environment="production")
versioned = config.version_config("Experiment with learning rate 0.001")

# Save versioned config
filepath = versioned.save_versioned_config("config/versions", "Learning rate experiment")

# List all versions
versions = SimulationConfig.list_config_versions("config/versions")

# Load specific version
old_config = SimulationConfig.load_versioned_config("config/versions", "a1b2c3d4")
```

### Configuration Templating

Create parameterized templates for systematic parameter sweeps:

```python
from farm.config.template import ConfigTemplate, ConfigTemplateManager

# Create a template with placeholders
template_dict = {
    "width": "{{env_size}}",
    "height": "{{env_size}}",
    "system_agents": "{{agent_count}}",
    # ... other config
}
template = ConfigTemplate(template_dict)

# Instantiate with specific values
config = template.instantiate({
    "env_size": 200,
    "agent_count": 25
})

# Batch generate multiple configs
manager = ConfigTemplateManager()
variable_sets = [
    {"env_size": 100, "agent_count": 10},
    {"env_size": 200, "agent_count": 20},
    {"env_size": 300, "agent_count": 30}
]
config_files = manager.create_experiment_configs("my_template", variable_sets)
```

### Configuration Diffing

Compare configurations to understand differences:

```python
from farm.core.config import SimulationConfig

config1 = SimulationConfig.from_centralized_config(environment="development")
config2 = SimulationConfig.from_centralized_config(environment="production")

differences = config1.diff_config(config2)
for key, change in differences.items():
    print(f"{key}: {change['self']} -> {change['other']}")
```

### Runtime Configuration Reloading

Automatically reload configurations when files change:

```python
from farm.config.watcher import create_reloadable_config

def on_config_change(new_config):
    print(f"Configuration updated! New width: {new_config.width}")
    # Update your simulation with new config

reloadable = create_reloadable_config("config/default.yaml")
reloadable.add_change_callback(on_config_change)
reloadable.watch_file("config/default.yaml")
```

### Command-Line Tools

Use the configuration CLI for advanced management:

```bash
# Version management
python -m farm.config.cli version create --environment production --description "Production config"
python -m farm.config.cli version list

# Template operations
python -m farm.config.cli template create my_template --environment development
python -m farm.config.cli template instantiate my_template output.yaml --variables size=200

# Configuration comparison
python -m farm.config.cli diff farm/config/versions/config_abc123.yaml farm/config/versions/config_def456.yaml

# File watching
python -m farm.config.cli watch start farm/config/default.yaml --verbose
```

## Best Practices

1. **Keep base configuration complete**: `default.yaml` should contain all possible configuration options with sensible defaults
2. **Minimize overrides**: Environment and profile files should only override the settings that need to be different
3. **Use descriptive names**: Environment and profile names should clearly indicate their purpose
4. **Version important configurations**: Use versioning for experiments that need to be reproducible
5. **Template systematic studies**: Use templating for parameter sweeps and comparative studies
6. **Document changes**: Update this README when adding new environments or profiles
7. **Validate configurations**: Use the schema to validate your configuration files

## Configuration Parameters

For detailed information about all available configuration parameters, see the generated `schema.json` file or refer to the `SimulationConfig` dataclass in `farm/core/config.py`.
