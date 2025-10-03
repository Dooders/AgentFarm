# Customization & Flexibility

## Overview

AgentFarm is designed with flexibility at its core, allowing users to customize virtually every aspect of their simulations. From defining custom parameters and rules to creating specialized agent behaviors, the platform provides comprehensive tools for tailoring simulations to specific research questions and scenarios.

## Key Capabilities

### Custom Parameters
- **Simulation Parameters**: Configure population size, duration, update frequency, and more
- **Environmental Parameters**: Define world size, resource distribution, terrain features
- **Agent Parameters**: Customize initial states, capabilities, attributes, and constraints
- **Behavioral Parameters**: Set learning rates, decision thresholds, and action probabilities

### Rule Definition
- **Interaction Rules**: Define how agents interact with each other
- **Environmental Rules**: Specify resource regeneration, decay, and distribution
- **Physics Rules**: Configure movement, collision detection, and spatial constraints
- **Evolution Rules**: Set mutation rates, selection pressures, and inheritance patterns

### Custom Environments
- **Topology**: Create grid-based, continuous, or network-based environments
- **Resources**: Define multiple resource types with custom properties
- **Obstacles**: Add terrain features, barriers, and restricted zones
- **Dynamic Conditions**: Implement changing environmental conditions over time

### Specialized Agent Behaviors
- **Decision Systems**: Implement custom decision-making algorithms
- **Learning Mechanisms**: Define reinforcement learning, evolutionary, or rule-based learning
- **Perception Systems**: Configure what agents can observe and how
- **Memory Systems**: Customize agent memory and information retention

## Configuration System

### Configuration Files

AgentFarm uses YAML configuration files for easy parameter management:

```yaml
simulation:
  num_agents: 100
  num_steps: 1000
  random_seed: 42

environment:
  width: 100
  height: 100
  resource_distribution: "uniform"
  regeneration_rate: 0.1

agents:
  initial_health: 100
  movement_speed: 1.5
  perception_radius: 10
  learning_rate: 0.01
```

### Programmatic Configuration

For dynamic configuration, use Python code:

```python
from farm.config import SimulationConfig, EnvironmentConfig, AgentConfig

# Create custom configuration
config = SimulationConfig(
    environment=EnvironmentConfig(
        size=(100, 100),
        resource_types=['food', 'water', 'shelter']
    ),
    agent=AgentConfig(
        initial_population=100,
        max_age=1000,
        reproduction_threshold=0.8
    )
)
```

## Custom Agent Types

### Creating Specialized Agents

Define custom agent classes with unique behaviors:

```python
from farm.agents import Agent

class ForagingAgent(Agent):
    """Agent specialized in resource gathering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inventory = []
        self.foraging_efficiency = 1.0
    
    def decide_action(self, observations):
        # Custom foraging logic
        nearest_resource = self.find_nearest_resource(observations)
        if nearest_resource:
            return self.move_towards(nearest_resource)
        return self.explore()
    
    def collect_resource(self, resource):
        # Custom collection behavior
        amount = resource.quantity * self.foraging_efficiency
        self.inventory.append((resource.type, amount))
```

### Behavior Composition

Combine multiple behaviors using composition:

```python
from farm.behaviors import Foraging, Social, Learning

class ComplexAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foraging = Foraging(self)
        self.social = Social(self)
        self.learning = Learning(self)
    
    def decide_action(self, observations):
        # Combine multiple behavior strategies
        if self.is_hungry():
            return self.foraging.find_food(observations)
        elif self.needs_social_interaction():
            return self.social.find_companions(observations)
        else:
            return self.learning.explore(observations)
```

## Custom Experiments

### Experiment Design

Create custom experiments with parameter sweeps:

```python
from farm.experiments import Experiment, ParameterSweep

class CustomExperiment(Experiment):
    def setup_parameters(self):
        return ParameterSweep({
            'num_agents': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'environment_size': [(50, 50), (100, 100)]
        })
    
    def run_trial(self, parameters):
        # Run simulation with specific parameters
        simulation = Simulation(parameters)
        return simulation.run()
    
    def analyze_results(self, results):
        # Custom analysis logic
        return self.compute_metrics(results)
```

## Custom Scenarios

### Scenario Definition

Define specific scenarios for testing:

```python
from farm.scenarios import Scenario

class PredatorPreyScenario(Scenario):
    """Predator-prey dynamics scenario."""
    
    def initialize_environment(self):
        # Set up environment
        self.add_resource_patches(num_patches=10)
        self.set_regeneration_rate(0.1)
    
    def initialize_agents(self):
        # Create agent populations
        self.add_agents('prey', count=80, initial_health=100)
        self.add_agents('predator', count=20, initial_health=150)
    
    def define_interactions(self):
        # Set up interaction rules
        self.add_rule('predator', 'prey', 'hunt')
        self.add_rule('prey', 'resource', 'forage')
```

## Extension Points

### Custom Modules

AgentFarm supports custom module integration:

- **Action Systems**: Define new action types and execution logic
- **Perception Systems**: Create custom observation channels
- **Memory Systems**: Implement specialized memory architectures
- **Analysis Tools**: Build custom analysis and visualization tools

### Plugin Architecture

Extend functionality through plugins:

```python
from farm.plugins import Plugin

class CustomAnalysisPlugin(Plugin):
    def on_simulation_start(self, simulation):
        # Setup custom tracking
        pass
    
    def on_step_complete(self, simulation, step):
        # Collect custom metrics
        pass
    
    def on_simulation_end(self, simulation):
        # Generate custom reports
        pass
```

## Configuration Management

### Configuration Validation

AgentFarm validates configurations to prevent errors:

```python
from farm.config import validate_config

# Validate before running
errors = validate_config(config)
if errors:
    print(f"Configuration errors: {errors}")
else:
    simulation = Simulation(config)
    simulation.run()
```

### Configuration Templates

Use templates for common scenarios:

```python
from farm.config import load_template

# Load predefined template
config = load_template('predator_prey')

# Customize as needed
config.num_agents = 200
config.environment.size = (150, 150)
```

## Best Practices

### Parameter Organization
- Group related parameters together
- Use meaningful names and documentation
- Set sensible defaults
- Provide validation ranges

### Modularity
- Keep components loosely coupled
- Use dependency injection
- Follow SOLID principles
- Design for extensibility

### Testing
- Test custom components in isolation
- Validate configurations before long runs
- Use small-scale tests for rapid iteration
- Profile performance with custom components

## Related Documentation

- [Configuration Guide](../config/configuration_guide.md)
- [User Guide](../user-guide.md)
- [Developer Guide](../developer-guide.md)
- [Experiments Documentation](../experiments.md)
- [Agent Documentation](../agents.md)

## Examples

For practical examples:
- [Configuration User Guide](../config/configuration_user_guide.md)
- [Service Usage Examples](../data/service_usage_examples.md)
- [Generic Simulation Scenario How-To](../generic_simulation_scenario_howto.md)
