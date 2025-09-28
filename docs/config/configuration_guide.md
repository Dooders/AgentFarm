# Configuration Guide

This guide explains how to configure AgentFarm simulations using the YAML-based configuration system. Proper configuration is essential for controlling simulation behavior, agent properties, learning parameters, and analysis settings.

## Configuration File Structure

AgentFarm uses a flat YAML configuration structure with nested sections for complex configurations. The main configuration file contains all simulation parameters at the top level, with specialized nested configurations for visualization, Redis memory, and observation systems.

```yaml
# Core simulation parameters (flat structure)
width: 100                          # Environment width
height: 100                         # Environment height
system_agents: 10                   # Number of system agents
independent_agents: 10              # Number of independent agents
control_agents: 10                  # Number of control agents
learning_rate: 0.001               # Global learning rate
# ... many more parameters

# Specialized nested configurations
visualization:                      # Visualization settings
  canvas_size: [400, 400]
  background_color: "black"
  # ... visualization parameters

redis:                              # Redis memory configuration
  host: localhost
  port: 6379
  # ... Redis parameters

observation:                        # Observation system config (optional)
  # ... observation parameters
```

## Core Simulation Parameters

The configuration uses a flat structure with parameters organized by functional area:

### Environment Settings

```yaml
# World dimensions and discretization
width: 100                          # Grid width (cells)
height: 100                         # Grid height (cells)
position_discretization_method: "floor"  # Position rounding: "floor", "round", "ceil"
use_bilinear_interpolation: true    # Use bilinear interpolation for resources

# Resource system
initial_resources: 20               # Starting resources in environment
resource_regen_rate: 0.1            # Resource regeneration rate (0-1)
resource_regen_amount: 2            # Resources added per regeneration
max_resource_amount: 30             # Maximum resources per cell
```

### Agent Population and Basic Properties

```yaml
# Agent counts
system_agents: 10                   # Number of SystemAgent instances
independent_agents: 10              # Number of IndependentAgent instances
control_agents: 10                  # Number of ControlAgent instances

# Agent type ratios (must sum to 1.0)
agent_type_ratios:
  SystemAgent: 0.33
  IndependentAgent: 0.33
  ControlAgent: 0.34

# Basic agent properties
initial_resource_level: 5           # Starting resources per agent
max_population: 300                 # Maximum total agent population
starvation_threshold: 100           # Steps agent can survive without resources
offspring_cost: 3                   # Resources required to reproduce
min_reproduction_resources: 8       # Minimum resources needed for reproduction
offspring_initial_resources: 5      # Resources given to offspring
perception_radius: 2                # Cells visible in each direction
base_attack_strength: 2             # Base attack damage multiplier
base_defense_strength: 2            # Base defense multiplier
seed: 1234567890                    # Random seed (null for random)
```

### Agent Type Descriptions

- **SystemAgent**: Cooperative behavior with balanced resource gathering and sharing
- **IndependentAgent**: Self-interested, prioritizes individual survival with aggressive tendencies
- **ControlAgent**: Balanced approach between cooperation and self-interest

### Global Learning Parameters

```yaml
# Core reinforcement learning parameters
learning_rate: 0.001               # Neural network learning rate
gamma: 0.95                        # Discount factor (0-1)
epsilon_start: 1.0                  # Initial exploration rate
epsilon_min: 0.01                  # Minimum exploration rate
epsilon_decay: 0.995               # Exploration decay rate
memory_size: 2000                  # Experience replay buffer size
batch_size: 32                     # Training batch size
training_frequency: 4              # Steps between training updates
dqn_hidden_size: 24                # Hidden layer size for DQN
tau: 0.005                         # Target network soft update rate
```

### Agent Behavior Settings

```yaml
# Consumption and movement
base_consumption_rate: 0.15         # Resources consumed per step
max_movement: 8                     # Maximum cells per movement
gathering_range: 30                 # Maximum distance for gathering
max_gather_amount: 3                # Maximum resources gathered per action
territory_range: 30                 # Territorial range for behavior decisions

# Action probability modifiers
social_range: 30                    # Range for social interactions
move_mult_no_resources: 1.5         # Movement multiplier when no nearby resources
gather_mult_low_resources: 1.5      # Gathering multiplier when resources needed
share_mult_wealthy: 1.3             # Sharing multiplier when agent has excess
share_mult_poor: 0.5                # Sharing multiplier when agent needs resources
attack_starvation_threshold: 0.5    # Starvation threshold for desperate behavior
attack_mult_desperate: 1.4          # Attack multiplier when desperate
attack_mult_stable: 0.6             # Attack multiplier when stable
```

### Combat System

```yaml
# Combat parameters
starting_health: 100.0              # Initial agent health
attack_range: 20.0                  # Maximum attack distance
attack_base_damage: 10.0            # Base damage per attack
attack_kill_reward: 5.0             # Reward for killing another agent
```

### Agent-Specific Parameters

```yaml
# Agent type configurations
agent_parameters:
  SystemAgent:
    gather_efficiency_multiplier: 0.4
    gather_cost_multiplier: 0.4
    min_resource_threshold: 0.2
    share_weight: 0.3
    attack_weight: 0.05
  IndependentAgent:
    gather_efficiency_multiplier: 0.7
    gather_cost_multiplier: 0.2
    min_resource_threshold: 0.05
    share_weight: 0.05
    attack_weight: 0.25
  ControlAgent:
    gather_efficiency_multiplier: 0.55
    gather_cost_multiplier: 0.3
    min_resource_threshold: 0.125
    share_weight: 0.15
    attack_weight: 0.15
```

### Action-Specific Learning Modules

Each action type (move, gather, share, attack) has its own DQN learning parameters:

```yaml
# Movement learning parameters
move_target_update_freq: 100        # Target network update frequency
move_memory_size: 10000             # Experience replay size
move_learning_rate: 0.001           # Learning rate
move_gamma: 0.99                    # Discount factor
move_epsilon_start: 1.0             # Initial exploration
move_epsilon_min: 0.01              # Minimum exploration
move_epsilon_decay: 0.995           # Exploration decay
move_dqn_hidden_size: 64            # Network hidden size
move_batch_size: 32                 # Training batch size
move_tau: 0.005                     # Target update rate
move_base_cost: -0.1                # Base movement cost
move_resource_approach_reward: 0.3  # Reward for approaching resources
move_resource_retreat_penalty: -0.2 # Penalty for moving away from resources

# Similar parameters exist for gather_*, share_*, and attack_* actions
```

### Curriculum Learning

```yaml
# Phase-based learning curriculum
curriculum_phases:
  - steps: 100
    enabled_actions: ["move", "gather"]
  - steps: 200
    enabled_actions: ["move", "gather", "share", "attack"]
  - steps: -1  # -1 means remaining steps
    enabled_actions: ["move", "gather", "share", "attack", "reproduce"]
```

### Database Configuration

```yaml
# Database settings
use_in_memory_db: false              # Use in-memory database
persist_db_on_completion: true       # Save in-memory DB to disk
in_memory_db_memory_limit_mb: null   # Memory limit (null = unlimited)
in_memory_tables_to_persist: null    # Tables to persist (null = all)

# SQLite pragma settings
db_pragma_profile: "balanced"        # "balanced", "performance", "safety", "memory"
db_cache_size_mb: 200                # Cache size in MB
db_synchronous_mode: "NORMAL"        # "OFF", "NORMAL", "FULL"
db_journal_mode: "WAL"               # Journal mode
db_custom_pragmas: {}                # Additional pragmas
```

### Device and Performance Configuration

```yaml
# Neural network device configuration
device_preference: "auto"            # "auto", "cpu", "cuda", "cuda:X"
device_fallback: true                # Fallback to CPU if preferred device unavailable
device_memory_fraction: null         # GPU memory fraction (0.0-1.0)
device_validate_compatibility: true  # Validate tensor compatibility

# Simulation control
simulation_steps: 100                # Number of steps to simulate
max_steps: 1000                      # Maximum steps before termination
max_wait_steps: 10                   # Maximum wait steps between actions

# Debug settings
debug: false                         # Enable debug output
verbose_logging: false               # Enable verbose logging
```

## Visualization Configuration

The `visualization` section configures display and rendering options:

```yaml
visualization:
  # Canvas and display
  canvas_size: [400, 400]            # Rendering canvas size
  padding: 20                        # Padding around canvas
  background_color: "black"          # Background color

  # Animation settings
  max_animation_frames: 5            # Maximum animation frames
  animation_min_delay: 50            # Minimum delay between frames

  # Resource visualization
  max_resource_amount: 30            # Maximum resource amount for scaling
  resource_colors:
    glow_red: 150
    glow_green: 255
    glow_blue: 50
  resource_size: 2                   # Size of resource indicators

  # Agent visualization
  agent_radius_scale: 2              # Agent radius scaling factor
  birth_radius_scale: 4              # Birth event radius scale
  death_mark_scale: 1.5              # Death mark scale
  agent_colors:
    SystemAgent: "blue"
    IndependentAgent: "red"
    ControlAgent: "#DAA520"

  # Text and UI
  min_font_size: 10                  # Minimum font size
  font_scale_factor: 40              # Font scaling factor
  font_family: "arial"               # Font family

  # Event markers
  death_mark_color: [255, 0, 0]      # Death marker color (RGB)
  birth_mark_color: [255, 255, 255]  # Birth marker color (RGB)

  # Metric visualization colors
  metric_colors:
    total_agents: "#4a90e2"
    system_agents: "#50c878"
    independent_agents: "#e74c3c"
    control_agents: "#DAA520"
    total_resources: "#f39c12"
    average_agent_resources: "#9b59b6"
```

## Redis Memory Configuration

The `redis` section configures Redis memory backend settings:

```yaml
redis:
  host: localhost                     # Redis server host
  port: 6379                         # Redis server port
  db: 0                              # Redis database number
  password: null                     # Redis password (null for no auth)
  decode_responses: true             # Decode responses as strings
  environment: default               # Environment identifier
```

## Observation Configuration

The optional `observation` section configures the advanced observation system (Pydantic-based):

```yaml
observation:
  observation_channels: []           # List of observation channels
  observation_range: 5               # Observation radius
  storage_mode: "dense"              # Storage mode: "dense", "sparse", "hybrid"
  dtype: "float32"                   # Data type for observations
  # ... additional observation parameters
```

## Configuration Versioning

AgentFarm supports configuration versioning for reproducible experiments:

```yaml
# Versioning metadata (automatically managed)
config_version: "abc123def456..."     # Unique version hash
config_created_at: "2024-01-15T10:30:00Z"  # Creation timestamp
config_description: "Experiment configuration"  # Optional description
```

## Configuration Validation

The configuration system includes built-in validation and schema checking to ensure configurations are correct.

## Configuration Management

### Loading Configurations

```python
from farm.core.config import SimulationConfig

# Load from centralized config system
config = SimulationConfig.from_centralized_config()

# Load specific environment and profile
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)

# Load from YAML file directly
config = SimulationConfig.from_yaml("path/to/config.yaml")
```

### Programmatic Configuration

```python
from farm.core.config import SimulationConfig

# Create configuration programmatically
config = SimulationConfig(
    width=200,
    height=200,
    system_agents=15,
    learning_rate=0.001,
    # ... other parameters
)
```

### Configuration Templates

AgentFarm supports configuration templates for parameterized experiments:

```python
from farm.config.template import ConfigTemplate

# Load a template
template = ConfigTemplate.from_yaml("config/templates/my_template.yaml")

# Instantiate with parameters
config = template.instantiate({
    'env_size': 200,
    'agent_count': 25
})
```

## Best Practices

### Organization
- Use the centralized configuration system instead of direct YAML loading
- Leverage environments (development, production, testing) for different deployment scenarios
- Use profiles for different types of experiments (benchmark, research, simulation)
- Version important configurations for reproducible research

### Validation
- Always use the centralized config loading system which includes validation
- Test configurations with small simulations before running large experiments
- Use configuration versioning for important experiments

### Performance
- Configure database pragmas appropriately for your use case ("balanced" is usually best)
- Set appropriate device preferences for neural network computations
- Use in-memory databases for fast prototyping and development

### Maintenance
- Keep configurations under version control
- Document the purpose and expected outcomes of configuration changes
- Use descriptive names for configuration files and versions
- Archive configurations used for published results

This configuration system provides comprehensive control over all aspects of AgentFarm simulations while maintaining a clean, hierarchical structure for ease of use and maintenance.
