# Configuration Guide

This guide explains how to configure AgentFarm simulations using the YAML-based configuration system. Proper configuration is essential for controlling simulation behavior, agent properties, learning parameters, and analysis settings.

## Electron Config Explorer (GUI)

The Electron-based Config Explorer provides an interactive way to view and edit configuration files.

- Open the Explorer from the sidebar button in the Electron app.
- Use "Open…" to load a configuration, "Save"/"Save As…" to persist changes.

### Grayscale Mode and Theme

- The Explorer uses a professional greyscale theme by default via CSS variables in `src/styles/index.css` and Leva overrides in `src/styles/leva-theme.css`.
- `ThemeProvider` sets `data-theme="custom"` on the root element to activate the Leva greyscale.
- Toggle additional full-UI grayscale filter with:
  - In console: `localStorage.setItem('ui:grayscale', 'true'); location.reload()`
  - Remove: `localStorage.removeItem('ui:grayscale'); location.reload()`
- Controls are compact: 28px height, subtle borders, high-contrast monochrome focus rings.

### Keyboard Navigation & Accessibility

- Navigate the section list with Arrow keys; Home/End jump to first/last.
- Press Enter or Space to select a focused section.
- Focus indicators are visible when tabbing; disabled controls are clearly indicated.

### Multi-Config Compare

- Click "Compare…" to open a secondary configuration for side-by-side comparison.
- Differences are highlighted:
  - Form view: fields with differences show a purple highlight with a "Copy from compare" button to copy the value into the current config.
  - YAML view: a side-by-side grid lists key paths and values for current vs compare.
- Click "Clear Compare" to exit comparison mode.

### Preset Bundles

- Click "Apply Preset…" to select a preset file (a configuration containing overrides).
- The preset is deep-merged into the current configuration.
- Click "Undo Preset" to revert the last applied preset (stack-based undo).

### Validation & Unsaved Changes

- The Explorer runs server-side validation after edits and shows a validity badge.
- An "Unsaved" indicator appears when changes are pending and disables once saved.

Acceptance Criteria covered:
- View two configurations concurrently and highlight differences in both form and YAML views.
- Apply preset bundles with undo support.
- Maintain validation and unsaved indicators after modifications.
- Provide a grayscale theme toggle, synchronized between panels and persisted across sessions.

## Configuration File Structure

AgentFarm uses a hierarchical YAML configuration system with the following main sections:

```yaml
# Main configuration sections
environment:        # Simulation world settings
agents:            # Agent properties and behavior
learning:          # Reinforcement learning parameters
channels:          # Observation channel configuration
actions:           # Action-specific parameters
analysis:          # Analysis and visualization settings
database:          # Data persistence configuration
visualization:     # Display and plotting options
experiment:        # Experiment management settings
```

## Environment Configuration

The environment section controls the simulation world properties:

```yaml
environment:
  # World dimensions
  width: 50                    # Grid width (cells)
  height: 50                   # Grid height (cells)

  # Resource system
  resource_distribution: "uniform"  # "uniform", "clustered", "scattered"
  initial_resource_count: 200       # Starting resources
  resource_regeneration_rate: 0.02  # Regeneration per step (0-1)
  max_resources_per_cell: 10        # Maximum resources per cell
  resource_clusters: 3              # Number of clusters (for clustered distribution)

  # Time and simulation control
  max_steps: 1000                   # Maximum simulation steps
  random_seed: 42                   # Random seed for reproducibility

  # Environmental dynamics
  weather_enabled: false            # Enable weather system
  weather_intensity: 0.5            # Weather system intensity
  weather_update_rate: 0.1          # Weather change rate

  # Performance settings
  spatial_index_enabled: true       # Enable spatial indexing
  batch_size: 32                    # Batch processing size
  num_threads: 4                    # Parallel processing threads
```

### Resource Distribution Types

- **uniform**: Resources distributed evenly across the grid
- **clustered**: Resources concentrated in clusters
- **scattered**: Resources spread randomly with some clustering

## Agent Configuration

The agents section defines agent properties and population settings:

```yaml
agents:
  # Population settings
  initial_count: 10                 # Number of agents at start
  max_agents: 20                    # Maximum agent population
  agent_types:                      # Agent type distribution
    system: 0.7                     # 70% system agents
    independent: 0.2                # 20% independent agents
    control: 0.1                    # 10% control agents

  # Physical properties
  initial_resources: 100            # Starting resource level
  max_resources: 200                # Maximum resource capacity
  resource_consumption_rate: 1.0    # Resources consumed per step
  movement_speed: 1.0               # Cells per step

  # Perception and cognition
  observation_radius: 6             # Cells visible in each direction
  fov_radius: 5                     # Field-of-view radius
  memory_size: 1000                 # Experience replay memory
  attention_span: 0.95              # Information decay factor

  # Behavioral parameters
  exploration_rate: 0.1             # ε-greedy exploration (0-1)
  exploration_decay: 0.995          # Exploration rate decay
  cooperation_threshold: 150        # Resource level for cooperation
  aggression_threshold: 50          # Resource level triggering aggression

  # Reproduction settings
  reproduction_enabled: true        # Allow agent reproduction
  reproduction_cost: 50             # Resources required for reproduction
  reproduction_threshold: 180       # Minimum resources for reproduction
  inheritance_factor: 0.8           # Trait inheritance (0-1)

  # Death and survival
  starvation_threshold: 100         # Max steps agent can survive without resources
  max_age: 1000                     # Maximum agent age (steps)
  damage_multiplier: 1.0            # Combat damage scaling
```

### Agent Type Descriptions

- **system**: Balanced behavior with learning capabilities
- **independent**: Self-interested, prioritize individual survival
- **control**: Reduced autonomy for controlled experiments

## Learning Configuration

The learning section configures reinforcement learning parameters:

```yaml
learning:
  # Core learning parameters
  learning_rate: 0.001              # Neural network learning rate
  gamma: 0.99                       # Discount factor (0-1)
  epsilon: 0.1                      # Initial exploration rate
  epsilon_decay: 0.995              # Exploration decay rate
  epsilon_min: 0.01                 # Minimum exploration rate

  # Neural network architecture
  network_architecture:             # Hidden layer sizes
    - 128
    - 64
    - 32
  activation_function: "relu"       # "relu", "tanh", "sigmoid"
  dropout_rate: 0.1                 # Dropout regularization

  # Experience replay
  memory_size: 10000                # Replay buffer size
  batch_size: 32                    # Training batch size
  target_update_frequency: 100      # Target network update frequency

  # Training optimization
  optimizer: "adam"                 # "adam", "rmsprop", "sgd"
  loss_function: "mse"              # "mse", "huber", "mae"
  gradient_clipping: 1.0            # Maximum gradient norm

  # Advanced learning features
  double_dqn: true                  # Use Double DQN
  dueling_dqn: true                 # Use Dueling DQN architecture
  prioritized_replay: true          # Use prioritized experience replay
  curiosity_driven: false           # Enable curiosity-driven exploration

  # Multi-agent learning
  centralized_critic: false         # Use centralized value function
  opponent_modeling: true           # Model other agents' behavior
  communication_enabled: false      # Allow agent communication
```

## Channel Configuration

The channels section configures the observation system:

```yaml
channels:
  # Core observation settings
  observation_radius: 6             # Base observation radius
  fov_radius: 5                     # Field-of-view radius
  channel_stacking: true            # Stack multiple time steps

  # Channel-specific settings
  decay_factors:                    # Temporal decay rates (0-1)
    trails: 0.95                    # Movement trail decay
    damage_heat: 0.90               # Combat heat decay
    ally_signals: 0.85              # Communication signal decay
    known_empty: 0.98               # Memory of explored areas

  # Channel enablement
  enabled_channels:                 # Which channels to include
    - SELF_HP
    - ALLIES_HP
    - ENEMIES_HP
    - RESOURCES
    - OBSTACLES
    - VISIBILITY
    - KNOWN_EMPTY
    - DAMAGE_HEAT
    - TRAILS

  # Custom channels
  custom_channels:                  # User-defined channels
    - name: "WEATHER"
      type: "dynamic"
      gamma: 0.95
    - name: "RESOURCE_DENSITY"
      type: "instant"

  # Channel processing
  normalization: "layer"            # "none", "layer", "batch", "instance"
  preprocessing:                    # Data preprocessing
    gaussian_blur: false
    sigma: 0.5
    edge_detection: false
```

## Action Configuration

The actions section configures behavior-specific parameters:

```yaml
actions:
  # Movement action
  move:
    enabled: true
    cost_factor: 1.0                 # Resource cost multiplier
    diagonal_movement: true          # Allow diagonal movement
    obstacle_penalty: 5.0            # Cost of moving through obstacles
    pathfinding_algorithm: "a_star"  # "a_star", "greedy", "random"

  # Gathering action
  gather:
    enabled: true
    base_efficiency: 1.0             # Base gathering rate
    specialization_bonus: 1.5        # Bonus for repeated gathering
    tool_use_enabled: false          # Allow tool use
    max_carry_capacity: 50           # Maximum resources carried

  # Combat action
  attack:
    enabled: true
    base_damage: 20                  # Base attack damage
    defense_factor: 0.8              # Damage reduction
    range: 1                         # Attack range (cells)
    cooldown: 3                      # Steps between attacks

  # Sharing action
  share:
    enabled: true
    generosity_factor: 0.4           # Fraction of resources shared
    relationship_bonus: 1.2          # Bonus for repeated sharing
    max_sharing_distance: 3          # Maximum sharing distance

  # Reproduction action
  reproduce:
    enabled: true
    energy_cost: 50                  # Resource cost
    gestation_period: 10             # Steps to create offspring
    mutation_rate: 0.1               # Genetic mutation probability
    inheritance_weights:             # Trait inheritance weights
      learning_rate: 0.8
      cooperation_threshold: 0.9
      aggression_threshold: 0.7
```

## Analysis Configuration

The analysis section configures data collection and visualization:

```yaml
analysis:
  # Data collection
  metrics_enabled: true             # Collect performance metrics
  event_logging: true               # Log simulation events
  agent_tracking: true              # Track individual agents
  spatial_analysis: true            # Analyze spatial patterns

  # Metrics to collect
  tracked_metrics:
    - survival_rate
    - resource_distribution
    - action_frequencies
    - spatial_dispersion
    - cooperation_network
    - learning_progress
    - reproductive_success

  # Analysis parameters
  analysis_window: 50               # Moving average window
  statistical_tests:                # Statistical analysis
    - t_test
    - anova
    - correlation
    - regression

  # Visualization settings
  plot_style: "seaborn"             # Matplotlib style
  color_palette: "viridis"          # Color scheme
  figure_size: [12, 8]              # Default figure size
  dpi: 300                         # Resolution for saved figures

  # Real-time visualization
  real_time_plots: true             # Update plots during simulation
  plot_update_frequency: 10         # Steps between plot updates
  live_metrics_dashboard: true      # Display metrics dashboard
```

## Database Configuration

The database section configures data persistence:

```yaml
database:
  # Database settings
  engine: "sqlite"                  # "sqlite", "postgresql", "mysql"
  database_name: "simulation.db"    # Database file/location
  connection_pool_size: 5           # Connection pool size

  # Data retention
  retain_raw_data: true             # Keep raw simulation data
  retention_period: 30             # Days to retain data
  compression_enabled: true         # Compress old data

  # Schema configuration
  enable_indexes: true              # Create database indexes
  enable_foreign_keys: true         # Enforce referential integrity
  enable_triggers: true             # Use database triggers

  # Backup settings
  automatic_backup: true            # Enable automatic backups
  backup_frequency: "daily"         # "hourly", "daily", "weekly"
  backup_retention: 7               # Number of backups to keep

  # Performance optimization
  wal_mode: true                    # Write-ahead logging (SQLite)
  synchronous_mode: "normal"        # SQLite synchronization
  cache_size: 1000000               # Database cache size
```

## Visualization Configuration

The visualization section configures display and plotting:

```yaml
visualization:
  # Real-time display
  display_enabled: true             # Show simulation during run
  display_mode: "window"            # "window", "fullscreen"
  window_size: [1200, 800]          # Display window size
  frame_rate: 30                    # Target frame rate

  # Agent rendering
  agent_render_mode: "circle"       # "circle", "sprite", "text"
  agent_size: 8                     # Agent display size
  show_agent_ids: false             # Display agent identifiers
  color_by_type: true               # Color agents by type
  color_by_health: true             # Color agents by health

  # Environment rendering
  show_grid: true                   # Display grid lines
  show_resources: true              # Display resource levels
  resource_color_map: "plasma"      # Resource visualization colormap
  terrain_colormap: "terrain"       # Terrain visualization

  # Dynamic elements
  show_trails: true                 # Display movement trails
  trail_length: 20                  # Maximum trail length
  show_combat_effects: true         # Display combat animations
  effect_duration: 15               # Effect display duration

  # Information overlays
  show_overlay: true                # Display information overlay
  overlay_position: "top_right"     # Overlay position
  show_fps: true                    # Display frame rate
  show_step_count: true             # Display simulation step
  show_population_stats: true       # Display population statistics

  # Video recording
  record_video: false               # Enable video recording
  video_filename: "simulation.mp4"  # Output video file
  video_fps: 30                     # Video frame rate
  video_quality: "high"             # "low", "medium", "high"
```

## Experiment Configuration

The experiment section configures systematic parameter studies:

```yaml
experiment:
  # Experiment metadata
  name: "resource_study"            # Experiment identifier
  description: "Study resource distribution effects"
  author: "Research Team"
  date: "2024-01-15"

  # Parameter variations
  parameter_sweep:
    resource_distribution:          # Parameter to vary
      - uniform
      - clustered
      - scattered
    initial_resources:              # Another parameter
      - 200
      - 300
      - 400
    num_agents:                     # Third parameter
      - 10
      - 15
      - 20

  # Experimental design
  replications: 5                   # Runs per parameter combination
  steps_per_run: 1000               # Steps per simulation
  randomization: true               # Randomize parameter order

  # Statistical design
  statistical_tests:                # Tests to perform
    - anova                         # Analysis of variance
    - t_test                        # T-tests between conditions
    - correlation                   # Correlation analysis
  significance_level: 0.05          # Statistical significance threshold

  # Control conditions
  control_condition:                # Baseline configuration
    resource_distribution: "uniform"
    initial_resources: 300
    num_agents: 15

  # Stopping criteria
  early_stopping: true              # Stop if convergence reached
  convergence_threshold: 0.01       # Convergence criterion
  max_runtime_hours: 24             # Maximum runtime
```

## Configuration Validation

AgentFarm includes configuration validation to catch errors early:

```yaml
validation:
  # Enable validation
  validate_on_load: true            # Validate configuration on load
  strict_validation: false          # Fail on unknown parameters

  # Parameter constraints
  constraints:
    environment.width:
      min: 10
      max: 200
    agents.initial_count:
      min: 1
      max: 100
    learning.learning_rate:
      min: 0.0001
      max: 1.0

  # Required parameters
  required_parameters:
    - environment.width
    - environment.height
    - agents.initial_count

  # Parameter dependencies
  dependencies:
    weather.weather_intensity:
      requires: environment.weather_enabled
    learning.prioritized_replay:
      requires: learning.memory_size
```

## Configuration Management

### Loading Configurations

```python
from farm.core.config import load_config, merge_configs

# Load from file
config = load_config('config/my_simulation.yaml')

# Merge multiple configurations
base_config = load_config('config/base.yaml')
override_config = load_config('config/experiment.yaml')
final_config = merge_configs(base_config, override_config)
```

### Programmatic Configuration

```python
from farm.core.config import ConfigBuilder

# Build configuration programmatically
config = (ConfigBuilder()
    .set_environment(width=60, height=60)
    .set_agents(count=15, resources=100)
    .set_learning(rate=0.001, memory=5000)
    .enable_channels(['SELF_HP', 'ALLIES_HP', 'RESOURCES'])
    .build())
```

### Configuration Templates

AgentFarm provides configuration templates for common scenarios:

```bash
# List available templates
farm config list-templates

# Create new config from template
farm config create --template resource_study --output my_study.yaml

# Validate configuration
farm config validate my_simulation.yaml
```

## Best Practices

### Organization
- Use descriptive names for configuration files
- Group related parameters together
- Document parameter purposes with comments
- Version control your configurations

### Validation
- Always validate configurations before running
- Use parameter constraints to prevent invalid values
- Test configurations with small simulations first
- Document parameter ranges and valid values

### Maintenance
- Regularly review and update configurations
- Use configuration inheritance for related experiments
- Document configuration changes and their rationale
- Archive configurations used for published results

### Performance
- Adjust observation radii based on agent needs
- Use appropriate decay rates for dynamic channels
- Configure batch sizes based on available memory
- Enable spatial indexing for large environments

This configuration system provides flexible control over all aspects of AgentFarm simulations while maintaining ease of use and preventing common configuration errors.
