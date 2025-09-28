# Pydantic Validation Guide for Hydra Configuration

## Overview

The Hydra configuration system now includes comprehensive Pydantic validation for stronger type safety, better error messages, and improved developer experience. This guide covers how to use and extend the Pydantic validation system.

## Table of Contents

1. [Introduction](#introduction)
2. [Available Models](#available-models)
3. [Using Validation](#using-validation)
4. [Validation Features](#validation-features)
5. [Error Handling](#error-handling)
6. [Extending Models](#extending-models)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Introduction

Pydantic validation provides:

- **Type Safety**: Automatic type checking and conversion
- **Data Validation**: Range checks, format validation, and custom rules
- **Better Error Messages**: Clear, actionable validation errors
- **IDE Support**: Better autocomplete and type hints
- **Documentation**: Self-documenting configuration schemas

## Available Models

### Core Models

#### `HydraSimulationConfig`
The main configuration model for complete simulation settings.

```python
from farm.core.config_hydra_models import HydraSimulationConfig

# Create a validated configuration
config = HydraSimulationConfig(
    width=100,
    height=100,
    max_steps=1000,
    system_agents=10,
    independent_agents=10,
    control_agents=10
)
```

#### `HydraEnvironmentConfig`
Environment-specific configuration overrides.

```python
from farm.core.config_hydra_models import HydraEnvironmentConfig

env_config = HydraEnvironmentConfig(
    debug=True,
    max_steps=500,
    learning_rate=0.01
)
```

#### `HydraAgentConfig`
Agent-specific configuration overrides.

```python
from farm.core.config_hydra_models import HydraAgentConfig, AgentParameters

agent_config = HydraAgentConfig(
    agent_parameters={
        "SystemAgent": AgentParameters(
            share_weight=0.4,
            attack_weight=0.1
        )
    }
)
```

### Nested Models

#### `VisualizationConfig`
Visualization and rendering settings.

```python
from farm.core.config_hydra_models import VisualizationConfig

viz_config = VisualizationConfig(
    canvas_size=(800, 600),
    background_color="white",
    max_animation_frames=10
)
```

#### `RedisMemoryConfig`
Redis memory configuration.

```python
from farm.core.config_hydra_models import RedisMemoryConfig

redis_config = RedisMemoryConfig(
    host="localhost",
    port=6379,
    db=0
)
```

#### `AgentParameters`
Agent behavior parameters.

```python
from farm.core.config_hydra_models import AgentParameters

params = AgentParameters(
    gather_efficiency_multiplier=0.5,
    share_weight=0.3,
    attack_weight=0.1
)
```

#### `AgentTypeRatios`
Agent type distribution ratios.

```python
from farm.core.config_hydra_models import AgentTypeRatios

ratios = AgentTypeRatios(
    SystemAgent=0.4,
    IndependentAgent=0.3,
    ControlAgent=0.3
)
```

## Using Validation

### With Hydra Config Manager

The `SimpleHydraConfigManager` includes built-in Pydantic validation:

```python
from farm.core.config_hydra_simple import create_simple_hydra_config_manager

# Create config manager
config_manager = create_simple_hydra_config_manager(
    config_dir="/path/to/config",
    environment="development",
    agent="system_agent"
)

# Validate configuration
errors = config_manager.validate_configuration()
if errors:
    print("Validation errors:", errors)
else:
    print("Configuration is valid!")

# Get validated configuration
validated_config = config_manager.get_validated_config()
print(f"Environment: {validated_config.width}x{validated_config.height}")
```

### Direct Validation

You can also validate configuration dictionaries directly:

```python
from farm.core.config_hydra_models import validate_config_dict

config_dict = {
    "width": 100,
    "height": 100,
    "max_steps": 1000,
    # ... other fields
}

try:
    validated_config = validate_config_dict(config_dict)
    print("Configuration is valid!")
except ValidationError as e:
    print("Validation errors:", e.errors())
```

### Environment and Agent Validation

```python
from farm.core.config_hydra_models import validate_environment_config, validate_agent_config

# Validate environment config
env_config = {"debug": True, "max_steps": 500}
validated_env = validate_environment_config(env_config)

# Validate agent config
agent_config = {
    "agent_parameters": {
        "SystemAgent": {
            "share_weight": 0.4,
            "attack_weight": 0.1
        }
    }
}
validated_agent = validate_agent_config(agent_config)
```

## Validation Features

### Type Validation

Automatic type checking and conversion:

```python
# These will be automatically converted to the correct types
config = HydraSimulationConfig(
    width="100",  # String converted to int
    height=100.0,  # Float converted to int
    debug="true"   # String converted to bool
)
```

### Range Validation

Built-in range checks for numeric fields:

```python
# These will raise ValidationError
config = HydraSimulationConfig(
    width=-100,  # Must be >= 10
    height=20000,  # Must be <= 10000
    max_steps=0    # Must be >= 1
)
```

### Pattern Validation

Regex pattern validation for string fields:

```python
# These will raise ValidationError
config = HydraSimulationConfig(
    position_discretization_method="invalid",  # Must match ^(floor|round|ceil)$
    db_pragma_profile="wrong"  # Must match ^(safety|balanced|performance)$
)
```

### Custom Validation

Custom validation rules for complex constraints:

```python
# Agent ratios must sum to 1.0
ratios = AgentTypeRatios(
    SystemAgent=0.5,
    IndependentAgent=0.5,
    ControlAgent=0.5  # Total = 1.5, will raise ValidationError
)

# Agent population cannot exceed max population
config = HydraSimulationConfig(
    max_population=50,
    system_agents=30,
    independent_agents=30,  # Total = 60, exceeds max_population
    control_agents=0
)
```

### Nested Validation

Automatic validation of nested configuration objects:

```python
config = HydraSimulationConfig(
    width=100,
    height=100,
    visualization=VisualizationConfig(
        canvas_size=(400, 400),
        death_mark_color=[255, 0, 0]  # RGB validation
    ),
    redis=RedisMemoryConfig(
        host="localhost",
        port=6379
    )
)
```

## Error Handling

### ValidationError Structure

Pydantic validation errors provide detailed information:

```python
from pydantic import ValidationError

try:
    config = HydraSimulationConfig(width=-100)
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {'.'.join(str(x) for x in error['loc'])}")
        print(f"Message: {error['msg']}")
        print(f"Input: {error['input']}")
        print(f"Type: {error['type']}")
```

### Error Categories

The validation system categorizes errors:

```python
errors = config_manager.validate_configuration()

# Check for different types of errors
if 'pydantic' in errors:
    print("Pydantic validation errors:", errors['pydantic'])

if 'environment' in errors:
    print("Environment validation errors:", errors['environment'])

if 'agent' in errors:
    print("Agent validation errors:", errors['agent'])

if 'general' in errors:
    print("General errors:", errors['general'])
```

## Extending Models

### Adding New Fields

To add new configuration fields:

```python
from pydantic import BaseModel, Field
from farm.core.config_hydra_models import HydraSimulationConfig

class ExtendedSimulationConfig(HydraSimulationConfig):
    # Add new fields
    new_feature_enabled: bool = Field(default=False, description="Enable new feature")
    new_parameter: float = Field(default=1.0, ge=0.0, le=10.0, description="New parameter")
    
    @field_validator('new_parameter')
    @classmethod
    def validate_new_parameter(cls, v):
        if v > 5.0:
            logger.warning("New parameter is quite high")
        return v
```

### Custom Validation Rules

Add custom validation logic:

```python
from pydantic import field_validator, model_validator

class CustomConfig(BaseModel):
    start_time: int = Field(ge=0, le=23, description="Start hour (0-23)")
    end_time: int = Field(ge=0, le=23, description="End hour (0-23)")
    
    @model_validator(mode='after')
    def validate_time_range(self):
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        return self
    
    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_hour(cls, v):
        if not isinstance(v, int):
            raise ValueError("Hour must be an integer")
        return v
```

### Creating New Models

Create specialized models for specific use cases:

```python
from pydantic import BaseModel, Field

class ExperimentConfig(BaseModel):
    """Configuration for running experiments."""
    
    experiment_name: str = Field(min_length=1, max_length=100, description="Experiment name")
    num_runs: int = Field(ge=1, le=1000, description="Number of experiment runs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Experiment parameters")
    
    @field_validator('experiment_name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Experiment name must be alphanumeric with underscores or hyphens")
        return v
```

## Best Practices

### 1. Use Descriptive Field Names

```python
# Good
max_simulation_steps: int = Field(ge=1, description="Maximum number of simulation steps")

# Avoid
max_steps: int = Field(ge=1)  # Less descriptive
```

### 2. Provide Default Values

```python
# Good
learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Learning rate")

# Avoid
learning_rate: float = Field(ge=0.0, le=1.0)  # No default value
```

### 3. Use Appropriate Validation

```python
# Good - specific validation
port: int = Field(ge=1, le=65535, description="Port number")

# Avoid - too restrictive
port: int = Field(ge=1024, le=65535)  # Excludes common ports like 80, 443
```

### 4. Add Helpful Error Messages

```python
@field_validator('agent_ratios')
@classmethod
def validate_ratios(cls, v):
    total = sum(v.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Agent type ratios must sum to 1.0, got {total:.3f}. "
            f"Please adjust the ratios to sum to 1.0."
        )
    return v
```

### 5. Use Nested Models

```python
# Good - organized structure
class SimulationConfig(BaseModel):
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

# Avoid - flat structure
class SimulationConfig(BaseModel):
    env_width: int
    env_height: int
    agent_count: int
    viz_canvas_size: Tuple[int, int]
    # ... many more fields
```

## Examples

### Complete Configuration Example

```python
from farm.core.config_hydra_models import (
    HydraSimulationConfig,
    VisualizationConfig,
    RedisMemoryConfig,
    AgentTypeRatios,
    AgentParameters
)

# Create a complete validated configuration
config = HydraSimulationConfig(
    # Environment settings
    width=200,
    height=200,
    max_steps=2000,
    max_population=150,
    
    # Agent settings
    system_agents=20,
    independent_agents=20,
    control_agents=10,
    
    # Learning parameters
    learning_rate=0.01,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    
    # Debug settings
    debug=False,
    verbose_logging=True,
    
    # Nested configurations
    visualization=VisualizationConfig(
        canvas_size=(800, 600),
        background_color="black",
        max_animation_frames=10
    ),
    
    redis=RedisMemoryConfig(
        host="localhost",
        port=6379,
        db=0
    ),
    
    agent_type_ratios=AgentTypeRatios(
        SystemAgent=0.4,
        IndependentAgent=0.4,
        ControlAgent=0.2
    ),
    
    agent_parameters={
        "SystemAgent": AgentParameters(
            share_weight=0.4,
            attack_weight=0.1,
            cooperation_threshold=0.6
        ),
        "IndependentAgent": AgentParameters(
            share_weight=0.2,
            attack_weight=0.3,
            cooperation_threshold=0.4
        ),
        "ControlAgent": AgentParameters(
            share_weight=0.5,
            attack_weight=0.05,
            cooperation_threshold=0.8
        )
    }
)

print(f"Validated configuration: {config.simulation_id}")
print(f"Environment: {config.width}x{config.height}")
print(f"Total agents: {config.system_agents + config.independent_agents + config.control_agents}")
```

### Error Handling Example

```python
from pydantic import ValidationError

def validate_and_handle_errors(config_dict):
    """Validate configuration and handle errors gracefully."""
    try:
        config = validate_config_dict(config_dict)
        return config, None
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        return None, errors

# Usage
config_dict = {
    "width": -100,  # Invalid
    "height": 100,
    "max_steps": 1000
}

config, errors = validate_and_handle_errors(config_dict)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

### Integration with Hydra Example

```python
from farm.core.config_hydra_simple import create_simple_hydra_config_manager

def run_simulation_with_validation():
    """Run simulation with comprehensive validation."""
    
    # Create config manager
    config_manager = create_simple_hydra_config_manager(
        config_dir="/path/to/config",
        environment="production",
        agent="system_agent"
    )
    
    # Validate configuration
    errors = config_manager.validate_configuration()
    if errors:
        print("Configuration validation failed:")
        for category, error_list in errors.items():
            print(f"  {category}:")
            for error in error_list:
                print(f"    - {error}")
        return False
    
    # Get validated configuration
    try:
        validated_config = config_manager.get_validated_config()
        print(f"✅ Configuration validated successfully")
        print(f"   Environment: {validated_config.width}x{validated_config.height}")
        print(f"   Max steps: {validated_config.max_steps}")
        print(f"   Debug mode: {validated_config.debug}")
        
        # Run simulation with validated config
        # ... simulation code here ...
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting validated config: {e}")
        return False

# Run the simulation
success = run_simulation_with_validation()
if success:
    print("Simulation completed successfully!")
else:
    print("Simulation failed due to configuration issues.")
```

## Conclusion

Pydantic validation provides a robust foundation for configuration management in the Hydra system. It ensures data integrity, provides clear error messages, and improves the overall developer experience. By following the patterns and best practices outlined in this guide, you can create reliable, maintainable configuration systems.

For more information, see:
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Hydra Configuration Guide](hydra_configuration_guide.md)
- [Migration Guide](migration_to_hydra.md)