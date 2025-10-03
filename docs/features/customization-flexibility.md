# Customization & Flexibility

## Overview

AgentFarm is designed from the ground up to be flexible and extensible. Every research question is unique, and the platform recognizes this by providing comprehensive tools for customizing virtually every aspect of your simulations. Whether you need to adjust high-level parameters or implement completely custom agent behaviors, the system gives you the freedom to tailor simulations to your specific needs.

The design follows the Open-Closed Principle: you can extend capabilities through inheritance, composition, and plugins without modifying the core codebase. This approach provides both stability and flexibility, letting you implement your unique requirements while benefiting from the platform's robust infrastructure.

## Comprehensive Parameter Configuration

At the foundation is a powerful configuration system that lets you define and manage all simulation parameters in a structured way. Simulation parameters control fundamentals like population size, duration, timestep size, and random seeds for reproducibility. You can adjust these to scale from small exploratory runs to large production experiments.

Environmental parameters define the world where agents exist. You specify dimensions, topology (bounded or wraparound), resource distributions (uniform, clustered, gradient-based), and environmental dynamics. Agent parameters control initial conditions and capabilities - starting health, resources, sensory ranges, movement speeds, and metabolic rates. Behavioral parameters fine-tune decision-making, including learning rates, exploration strategies, and action selection rules.

The flexibility extends to every level. You can create homogeneous populations where all agents share characteristics, or heterogeneous populations with diverse types and capabilities. You can design static environments or dynamic ones that change over time. Every parameter can be adjusted independently to create the exact scenario you need.

## Flexible Configuration System

AgentFarm supports multiple configuration approaches. For many users, YAML configuration files provide an intuitive, human-readable way to specify parameters. The hierarchical structure naturally maps to nested settings, and plain-text format makes configurations easy to read, edit, version control, and share.

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
```

For dynamic scenarios, programmatic configuration through Python offers maximum flexibility. You can construct config objects in code, compute parameter values based on formulas, generate configurations algorithmically for parameter sweeps, or determine settings at runtime based on previous results.

```python
from farm.config import SimulationConfig, EnvironmentConfig, AgentConfig

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

The system also supports configuration inheritance and composition. You can define base configurations and create variations by overriding specific parameters. This promotes reusability - define a baseline scenario once, then create treatment variants that modify only the experimental manipulation.

Configuration validation catches errors before simulations run. The validator checks that required parameters are present, values fall within valid ranges, parameter combinations make sense, and settings won't cause problems. This saves time by identifying issues early.

## Custom Agent Implementations

The real power of flexibility becomes apparent when implementing custom agent types. The base Agent class provides common functionality like state management and perception, while leaving decision-making logic open for customization. By subclassing and implementing the `decide_action` method, you can create agents with any cognitive architecture you imagine.

```python
from farm.agents import Agent

class ForagingAgent(Agent):
    """Agent specialized in resource gathering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inventory = []
        self.foraging_efficiency = 1.0
    
    def decide_action(self, observations):
        nearest_resource = self.find_nearest_resource(observations)
        if nearest_resource:
            return self.move_towards(nearest_resource)
        return self.explore()
    
    def collect_resource(self, resource):
        amount = resource.quantity * self.foraging_efficiency
        self.inventory.append((resource.type, amount))
```

Agents can range from simple reactive entities following if-then rules to sophisticated cognitive agents that build internal models, plan ahead, predict others' behaviors, and learn from experience. Memory systems can be customized - some agents might be purely reactive with no memory, while others maintain detailed episodic memories or learned policies. Perception systems determine what information agents access, from limited local perception to perfect global knowledge.

## Behavioral Composition

Rather than creating monolithic agent classes, AgentFarm encourages behavioral composition where complex behaviors emerge from combining simpler modules. This compositional approach makes it easier to create agents with multifaceted capabilities by mixing and matching components.

```python
from farm.behaviors import Foraging, Social, Learning

class ComplexAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foraging = Foraging(self)
        self.social = Social(self)
        self.learning = Learning(self)
    
    def decide_action(self, observations):
        if self.is_hungry():
            return self.foraging.find_food(observations)
        elif self.needs_social_interaction():
            return self.social.find_companions(observations)
        else:
            return self.learning.explore(observations)
```

Behavioral modules can be developed and tested independently, then combined in different configurations. Different agent types can mix these modules in different proportions, creating behavioral diversity without code duplication. This modularity also makes debugging easier since you can isolate and test individual components.

## Custom Experiments

Beyond individual simulations, AgentFarm provides extensive support for designing complete experiments. The Experiment class provides a framework for parameter sweeps, multiple replications, systematic result collection, and comparative analysis.

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
        simulation = Simulation(parameters)
        return simulation.run()
    
    def analyze_results(self, results):
        return self.compute_metrics(results)
```

Parameter sweeps vary one or more parameters across ranges to understand their effects. AgentFarm handles generating parameter combinations, dispatching simulation runs, and organizing results. Experimental designs can specify replication strategies to account for stochasticity, automatically running multiple replications with different random seeds and computing summary statistics.

## Custom Scenarios

Scenarios represent complete simulation setups that combine environment configuration, agent initialization, interaction rules, and measurement protocols into reusable packages. Well-defined scenarios completely specify simulation setups, making it easy for others to reproduce your results.

```python
from farm.scenarios import Scenario

class PredatorPreyScenario(Scenario):
    """Predator-prey dynamics scenario."""
    
    def initialize_environment(self):
        self.add_resource_patches(num_patches=10)
        self.set_regeneration_rate(0.1)
    
    def initialize_agents(self):
        self.add_agents('prey', count=80, initial_health=100)
        self.add_agents('predator', count=20, initial_health=150)
    
    def define_interactions(self):
        self.add_rule('predator', 'prey', 'hunt')
        self.add_rule('prey', 'resource', 'forage')
```

Scenarios can be composed, where complex scenarios build from simpler components. You might define a base environment and add different agent populations to create variants. This reduces redundancy and makes systematic comparisons easier.

## Extension Points and Plugins

For advanced needs, AgentFarm provides well-defined extension points. The action system is extensible, letting you define new action types. Perception systems can be extended with custom observation channels. Memory systems can be replaced with alternative implementations. Analysis tools can be extended with custom analyzers for domain-specific metrics.

The plugin architecture allows functionality to be packaged as reusable modules. Plugins can hook into various simulation lifecycle stages - initialization, per-step updates, decision-making, and finalization. This makes it possible to add cross-cutting concerns like custom logging or specialized analysis without modifying core code.

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

Effective configuration management ensures reproducibility. AgentFarm validates configurations before running, catching errors early. Configuration templates provide starting points for common scenarios, encoding best practices and sensible defaults. Rather than starting from scratch, load an appropriate template and customize it.

Version control integration is facilitated by plain-text configuration files. Configurations can be committed to git repositories alongside code, making it easy to track how experimental designs evolve, document parameters used for specific analyses, and roll back to previous configurations if needed.

## Best Practices

When customizing AgentFarm, follow best practices to ensure maintainability and efficiency. Organize parameters logically, use meaningful names, provide documentation, and set sensible defaults. Keep components modular and loosely coupled, following the Single Responsibility Principle where each class has one clear purpose.

Test custom components thoroughly. Write unit tests for individual components and integration tests to verify they work together. Use small-scale tests for rapid iteration. Profile performance with custom components, especially for code called frequently like agent decision-making.

## Related Documentation

For detailed information, see the [Configuration Guide](../config/configuration_guide.md) for comprehensive parameter documentation, [User Guide](../user-guide.md) for common customization patterns, [Developer Guide](../developer-guide.md) for architecture and advanced extensions, [Experiments Documentation](../experiments.md) for systematic experiments, and [Agent Documentation](../agents.md) for the agent architecture.

## Examples

Practical examples can be found in the [Configuration User Guide](../config/configuration_user_guide.md), [Service Usage Examples](../data/service_usage_examples.md), and [Generic Simulation Scenario How-To](../generic_simulation_scenario_howto.md), which walks through implementing a complete custom scenario.
