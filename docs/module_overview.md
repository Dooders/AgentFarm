# AgentFarm Module Overview

## Introduction

AgentFarm is a comprehensive multi-agent reinforcement learning simulation platform designed for researching complex adaptive systems, emergent behaviors, and agent interactions. Built on PettingZoo's AECEnv framework, it provides a flexible and extensible environment for studying how autonomous agents compete, cooperate, and evolve in dynamic environments.

## Architecture Overview

AgentFarm follows a modular architecture with clear separation of concerns:

```
AgentFarm/
├── Core Framework
│   ├── Environment (farm.core.environment)
│   ├── Observations (farm.core.observations)
│   ├── Perception Channels (farm.core.channels)
│   └── State Management (farm.core.state)
├── Agent System
│   ├── Base Agents (farm.core.agent)
│   └── Decision Making (farm.actions)
├── Data & Analysis
│   ├── Database (farm.database)
│   ├── Analysis Tools (farm.analysis)
│   └── Metrics (farm.core.metrics_tracker)
├── Configuration & Control
│   ├── Configuration (farm.core.config)
│   └── Simulation Runners (farm.runners)
└── Extensions & Tools
    ├── Visualization (farm.visualization)
    ├── GUI (farm.gui)
    └── Utilities (farm.utils)
```

## Core Components

### 1. Environment (`farm.core.environment`)

The Environment class is the heart of the simulation, managing:

- **Spatial World**: 2D grid where agents and resources exist
- **Agent Lifecycle**: Creation, movement, interaction, and death
- **Resource Management**: Distribution, consumption, and regeneration
- **Simulation Loop**: Time-step progression and event handling
- **Multi-Agent RL**: Compatible with PettingZoo's AECEnv interface

**Key Features:**
- Configurable grid size and resource patterns
- Real-time metrics tracking
- Database logging for analysis
- Support for different agent types

### 2. Agent System (`farm.core.agent`)

Agents are autonomous entities with learning capabilities:

**Basic Agent Types:**
- **BaseAgent**: Unified agent class that handles all agent behaviors including movement, resource gathering, sharing, and combat

**Capabilities:**
- Movement and navigation
- Resource gathering and sharing
- Combat and conflict resolution
- Reproduction and inheritance
- Reinforcement learning via Deep Q-Networks<sup>**</sup>

<sub>* Not implemented yet</sub>
<sub>** RL algorithm will be configurable in the future (e.g., SB3, custom algorithms)</sub>

### 3. Observation System (`farm.core.observations`)

Each agent maintains a **local observation** of the world:

**Channel Types:**
- `SELF_HP`: Agent's own health status
- `ALLIES_HP`: Health of nearby allies
- `ENEMIES_HP`: Health of nearby enemies
- `RESOURCES`: Resource locations and quantities
- `OBSTACLES`: Terrain and obstacle information
- `VISIBILITY`: Field-of-view mask
- `DAMAGE_HEAT`: Recent combat events (decays over time)
- `TRAILS`: Movement trails (decays over time)
- `ALLY_SIGNAL`: Communication signals (decays over time)
- `GOAL`: Current objectives or waypoints
- `LANDMARKS`: Permanent landmarks and waypoints (persistent)

**Dynamic Decay System:**
Certain channels implement temporal decay using configurable gamma factors to simulate fading information and memory.

### 4. Dynamic Channel System (`farm.core.channels`)

The channel system provides a flexible framework for extending observations:

**Channel Behaviors:**
- `INSTANT`: Immediate information (refreshed each step)
- `DYNAMIC`: Persistent information with temporal decay
- `PERSISTENT`: Persistent information that remains until explicitly cleared

**Usage:**
```python
from farm.core.channels import ChannelHandler, ChannelBehavior, register_channel

class CustomDynamicChannel(ChannelHandler):
    """Example DYNAMIC channel with temporal decay."""
    def __init__(self, name):
        super().__init__(name, ChannelBehavior.DYNAMIC, gamma=0.95)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        # Custom processing logic
        pass

class CustomPersistentChannel(ChannelHandler):
    """Example PERSISTENT channel that maintains information until cleared."""
    def __init__(self, name):
        super().__init__(name, ChannelBehavior.PERSISTENT)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        # Custom processing logic that accumulates information
        pass

# Register the channels
dynamic_idx = register_channel(CustomDynamicChannel("DYNAMIC_CHANNEL"))
persistent_idx = register_channel(CustomPersistentChannel("PERSISTENT_CHANNEL"))
```

### 5. Action System (`farm.actions`)

Agents can perform various actions through specialized modules:

- **Movement**: Navigate the environment with pathfinding
- **Gather**: Collect resources from the environment
- **Attack**: Engage in combat with other agents
- **Share**: Redistribute resources to allies
- **Reproduce**: Create offspring (with inheritance)
- **Defend**: Defend against enemies (damage-modifying)
- **Pass**: Pass turn, no action

Each action module uses reinforcement learning to optimize behavior.

## Configuration System

AgentFarm uses a comprehensive YAML-based configuration system:

```yaml
# config.yaml
environment:
  width: 50
  height: 50
  resource_distribution: clustered

agents:
  initial_count: 20
  perception_radius: 6
  initial_resources: 100

learning:
  learning_rate: 0.001
  memory_size: 10000
  gamma: 0.99

channels:
  observation_radius: 6
  fov_radius: 5
  decay_factors:
    trails: 0.95
    damage_heat: 0.90
```

## Usage Examples

### Running a Basic Simulation

```python
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.agent import BaseAgent
import torch

# Configure observations
obs_config = ObservationConfig(R=6, fov_radius=5)

# Create environment
env = Environment(
    width=50,
    height=50,
    resource_distribution="uniform",
    obs_config=obs_config
)

# Add agents
for i in range(10):
    agent = BaseAgent(
        agent_id=f"agent_{i}",
        position=(random.randint(0, 49), random.randint(0, 49)),
        resource_level=100,
        environment=env
    )
    env.add_agent(agent)

# Run simulation
for step in range(1000):
    env.step()
    # Access metrics, observations, etc.
```

### Custom Agent Implementation

```python
from farm.core.agent import BaseAgent
from farm.core.channels import ChannelHandler, ChannelBehavior

class CustomAgent(BaseAgent):
    def __init__(self, agent_id, position, resource_level, environment, **kwargs):
        super().__init__(agent_id, position, resource_level, environment, **kwargs)

        # Custom initialization
        self.personality_trait = kwargs.get('personality', 'neutral')

    def decide_action(self):
        # Custom decision logic based on personality
        if self.personality_trait == 'aggressive':
            # Prioritize combat
            return self._select_attack_action()
        elif self.personality_trait == 'cooperative':
            # Prioritize sharing
            return self._select_share_action()
        else:
            # Default behavior
            return super().decide_action()
```

### Extending Observations with Custom Channels

```python
from farm.core.channels import ChannelHandler, ChannelBehavior, register_channel
import torch

class EmotionChannel(ChannelHandler):
    """Channel representing agent emotional states with temporal decay."""

    def __init__(self, name):
        super().__init__(name, ChannelBehavior.DYNAMIC, gamma=0.9)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Update emotion channel based on agent state."""
        agent = kwargs.get('agent')
        if agent:
            # Encode emotional state (fear, aggression, contentment)
            fear_level = self._calculate_fear(agent)
            aggression_level = self._calculate_aggression(agent)
            contentment_level = self._calculate_contentment(agent)

            # Update observation tensor
            obs_size = observation.shape[-1]
            center = obs_size // 2

            observation[channel_idx, center, center] = torch.tensor([
                fear_level, aggression_level, contentment_level
            ]).mean()

    def _calculate_fear(self, agent):
        """Calculate fear based on nearby enemies and health."""
        nearby_enemies = agent.environment.get_nearby_enemies(agent.position)
        health_ratio = agent.resource_level / agent.initial_resources
        return min(1.0, (len(nearby_enemies) * 0.2) + (1 - health_ratio) * 0.5)

class TerritoryChannel(ChannelHandler):
    """Channel representing agent's claimed territory (persistent)."""

    def __init__(self, name):
        super().__init__(name, ChannelBehavior.PERSISTENT)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Mark territory cells in the observation."""
        territory_cells = kwargs.get('territory_cells', [])
        if not territory_cells:
            return

        R = config.R
        ay, ax = agent_world_pos

        for cell_y, cell_x in territory_cells:
            dy = cell_y - ay
            dx = cell_x - ax
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                # Mark territory (accumulates over time)
                observation[channel_idx, y, x] = 1.0

# Register the channels
emotion_channel_idx = register_channel(EmotionChannel("EMOTION"))
territory_channel_idx = register_channel(TerritoryChannel("TERRITORY"))
```

### Experiment Configuration

```python
from farm.core.config import ExperimentConfig
from farm.runners.experiment_runner import ExperimentRunner

# Define parameter variations
experiment = ExperimentConfig(
    name="resource_distribution_study",
    variations=[
        {"resource_distribution": "uniform", "resource_clusters": 1},
        {"resource_distribution": "clustered", "resource_clusters": 3},
        {"resource_distribution": "scattered", "resource_clusters": 10},
    ],
    num_iterations=5,  # Run 5 simulations per variation
    num_steps=2000     # Each simulation runs for 2000 steps
)

# Run experiment
runner = ExperimentRunner()
results = runner.run_experiment(experiment)

# Analyze results
runner.generate_comparison_report(results)
```

## Data Analysis and Visualization

AgentFarm provides comprehensive analysis tools:

### Built-in Metrics
- Agent survival rates and lifespans
- Resource consumption patterns
- Action distribution analysis
- Reproduction and genealogy tracking
- Spatial distribution analysis

### Analysis Scripts
```bash
# Generate comparative analysis
python -m farm.analysis.comparative_analysis --path simulations/experiment_001

# Create visualization videos
python scripts/animate_simulation.py --simulation simulation.db --output video.mp4

# Analyze agent behaviors
python scripts/social_analysis.py --simulation simulation.db
```

### Database Schema
Simulation data is stored in SQLite with tables for:
- Agent states and actions
- Resource states over time
- Reproduction events
- Interaction logs
- Performance metrics

## Performance and Scaling

### Optimization Features
- Spatial indexing for efficient proximity queries
- Batch processing for observation updates
- Memory-efficient channel management
- Configurable observation radii

### Benchmarking
AgentFarm includes benchmarking tools to measure performance:
```bash
python benchmarks/run_benchmarks.py --config benchmark_config.yaml
```

## Integration with Reinforcement Learning

AgentFarm is designed for reinforcement learning research:

### Observation Space
- Multi-channel tensor observations
- Egocentric agent-centered views
- Configurable observation radii
- Temporal decay for dynamic channels

### Action Space
- Discrete action types (move, gather, attack, share, reproduce)
- Continuous parameters for movement directions
- Hierarchical action selection

### Reward Design
- Resource acquisition rewards
- Combat outcome rewards
- Cooperation incentives
- Survival bonuses

## Extending AgentFarm

### Adding New Action Types
1. Create action function in `farm.actions`
2. Update Action enum in `environment.py`
3. Implement learning logic if needed
4. Add configuration parameters

### Creating Custom Environments
```python
from farm.core.environment import Environment

class CustomEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization

    def update_resources(self):
        # Custom resource regeneration logic
        super().update_resources()

    def handle_agent_interaction(self, agent1, agent2):
        # Custom interaction rules
        pass
```

### Implementing New Analysis Tools
```python
from farm.analysis.base_analyzer import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, simulation_data):
        # Custom analysis logic
        return analysis_results

    def generate_report(self, results):
        # Generate visualizations/reports
        pass
```

## Best Practices

### Configuration Management
- Use version control for configuration files
- Document parameter meanings and valid ranges
- Validate configurations before running simulations

### Experiment Design
- Start with small-scale experiments
- Use parameter sweeps for systematic exploration
- Implement proper statistical analysis
- Document experimental hypotheses and methods

### Performance Optimization
- Adjust observation radii based on agent needs
- Use appropriate decay rates for dynamic channels
- Monitor memory usage for long simulations
- Consider spatial partitioning for large environments

### Code Organization
- Extend existing classes rather than modifying core code
- Use the channel system for observation extensions
- Follow the established patterns for action modules
- Document custom components thoroughly

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce observation radius
- Decrease number of dynamic channels
- Use shorter simulation lengths

**Slow Performance:**
- Reduce agent count
- Optimize channel processing
- Use batch operations where possible

**Observation Mismatches:**
- Ensure channel indices match observation tensor size
- Check channel registration order
- Verify observation configuration consistency

**Database Issues:**
- Check file permissions
- Ensure database isn't locked by another process
- Verify schema compatibility

## API Reference

### Core Classes
- `Environment`: Main simulation environment
- `BaseAgent`: Abstract agent base class
- `AgentObservation`: Agent observation management
- `ChannelHandler`: Channel processing interface
- `ChannelRegistry`: Global channel management

### Key Functions
- `register_channel()`: Register new observation channels
- `get_channel_registry()`: Access global channel registry
- `setup_db()`: Initialize simulation database

### Configuration Classes
- `ObservationConfig`: Observation system configuration
- `ExperimentConfig`: Experiment parameter configuration

## Support and Resources

- **Documentation**: Comprehensive guides in `docs/` directory
- **Examples**: Working examples in `examples/` directory
- **Tests**: Extensive test suite in `tests/` directory
- **Community**: GitHub issues and discussions

## Conclusion

AgentFarm provides a powerful, flexible platform for multi-agent reinforcement learning research and complex systems modeling. Its modular architecture, comprehensive observation system, and extensive analysis tools make it suitable for a wide range of research applications, from studying emergent behaviors to developing novel reinforcement learning algorithms.

The platform's strength lies in its balance of accessibility for newcomers and extensibility for advanced researchers, enabling both educational use and cutting-edge research in artificial intelligence and complex systems.
