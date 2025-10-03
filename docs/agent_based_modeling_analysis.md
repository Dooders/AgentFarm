# Agent-Based Modeling & Analysis

![Project Status](https://img.shields.io/badge/feature-agent%20modeling-blue)

## Overview

Agent-Based Modeling (ABM) is a core feature of AgentFarm that enables researchers and developers to explore complex systems through computational simulations. This powerful framework allows you to create, execute, and analyze simulations where autonomous agents interact with each other and their environment, producing emergent behaviors and revealing system dynamics that would be difficult to study otherwise.

### What is Agent-Based Modeling?

Agent-Based Modeling is a computational approach that simulates the actions and interactions of autonomous agents to understand the behavior of complex systems. Unlike traditional analytical approaches, ABM focuses on individual entities (agents) and their behaviors, allowing emergent patterns to arise naturally from agent interactions.

**Key Characteristics:**
- **Individual-Level Focus**: Model each agent's unique attributes, behaviors, and decision-making
- **Emergent Behavior**: System-level patterns emerge from local interactions
- **Heterogeneity**: Agents can have diverse properties and behaviors
- **Adaptive Learning**: Agents can learn and adapt based on experience
- **Environmental Context**: Agents interact with a shared environment and each other

---

## Core Capabilities

### 1. Run Complex Simulations with Interacting, Adaptive Agents

AgentFarm enables you to create sophisticated multi-agent simulations where agents autonomously interact, adapt, and evolve over time.

#### Agent Types

AgentFarm supports multiple agent types, each with distinct behaviors:

- **System Agents**: Cooperative agents that prioritize collective goals and resource sharing
- **Independent Agents**: Self-oriented agents focused on individual survival and resource acquisition
- **Control Agents**: Baseline agents for experimental comparison
- **Custom Agents**: Define your own agent types with specialized behaviors

#### Agent Capabilities

Each agent in AgentFarm possesses a rich set of capabilities:

```python
# Agent Core Attributes
- Autonomous decision-making
- Spatial awareness and navigation
- Resource gathering and management
- Health and survival mechanics
- Combat and defense capabilities
- Communication with nearby agents
- Learning through reinforcement
- Memory and experience tracking
```

#### Creating Agents

```python
from farm.core.agent import BaseAgent
from farm.core.environment import Environment

# Create an adaptive agent
agent = BaseAgent(
    agent_id="agent_001",
    position=(25, 25),
    resource_level=100,
    agent_type="IndependentAgent",
    environment=environment,
    spatial_service=environment.spatial_service,
    generation=0,
    starting_health=100.0
)

# Agent automatically handles:
# - Perception of nearby agents and resources
# - Decision-making based on current state
# - Action execution and learning
# - Memory storage and retrieval
```

#### Agent Interactions

Agents can interact in multiple ways:

1. **Resource Sharing**: Agents can share resources with nearby agents
2. **Combat**: Agents can engage in offensive or defensive actions
3. **Communication**: Agents can exchange information within interaction zones
4. **Cooperation**: System agents can coordinate for collective benefit
5. **Competition**: Independent agents compete for limited resources

**Example Interaction:**

```python
# During simulation step
for agent in environment.agents.values():
    if agent.alive:
        # Agent perceives its environment
        perception = agent.perceive()
        
        # Agent decides on action based on observation
        action = agent.decide_action()
        
        # Agent executes action (move, gather, share, attack, etc.)
        result = agent.execute_action(action)
        
        # Agent learns from experience
        agent.update_learning(result)
```

#### Adaptive Behaviors

Agents adapt through multiple mechanisms:

**Reinforcement Learning:**
```python
from farm.core.decision.config import DecisionConfig

# Configure learning parameters
decision_config = DecisionConfig(
    algorithm="dqn",           # Deep Q-Network
    learning_rate=0.001,
    discount_factor=0.99,
    exploration_rate=0.1
)

# Agent learns optimal policies through experience
```

**Evolutionary Adaptation:**
```python
# Agents reproduce when conditions are met
if agent.resource_level > reproduction_threshold:
    offspring = agent.reproduce()
    # Offspring inherits parent's genome with mutations
    offspring.genome = mutate(agent.genome)
```

---

### 2. Study Emergent Behaviors and System Dynamics

One of the most powerful aspects of agent-based modeling is observing emergent behaviors—patterns that arise from agent interactions without being explicitly programmed.

#### What are Emergent Behaviors?

Emergent behaviors are system-level patterns that result from local agent interactions. In AgentFarm simulations, you might observe:

- **Resource Clustering**: Agents naturally form groups around resource-rich areas
- **Cooperation Networks**: System agents develop sharing relationships
- **Territorial Behavior**: Agents establish and defend resource territories
- **Migration Patterns**: Population movements in response to resource depletion
- **Social Hierarchies**: Dominance structures emerge through repeated interactions
- **Adaptive Strategies**: Population-level strategy shifts in response to environmental pressures

#### System Dynamics Analysis

AgentFarm provides comprehensive tools to analyze system-level dynamics:

**Population Dynamics:**
```python
from farm.analysis.simulation_analysis import SimulationAnalyzer

analyzer = SimulationAnalyzer(simulation_db_path)

# Analyze population trends
population_stats = analyzer.get_population_over_time()
# Returns: time series of agent counts by type

# Study survival rates
survival_analysis = analyzer.calculate_survival_rates()
# Returns: survival probabilities by agent type and time period
```

**Resource Dynamics:**
```python
# Track resource distribution and depletion
resource_stats = analyzer.get_resource_statistics()
# Returns: resource availability, consumption rates, spatial distribution

# Analyze resource sustainability
sustainability_metrics = analyzer.assess_resource_sustainability()
# Returns: depletion rates, recovery patterns, critical thresholds
```

**Behavioral Dynamics:**
```python
# Identify action patterns
action_analysis = analyzer.analyze_action_distributions()
# Returns: frequency of each action type over time

# Study decision-making patterns
decision_patterns = analyzer.get_decision_patterns()
# Returns: state-action mappings, strategy evolution
```

#### Example: Observing Emergence

```python
# Run simulation with mixed agent population
config = SimulationConfig(
    width=100,
    height=100,
    system_agents=50,      # Cooperative agents
    independent_agents=50,  # Competitive agents
    initial_resources=500,
    num_steps=1000
)

results = run_simulation(config)

# Analyze emergent cooperation
cooperation_metrics = analyzer.measure_cooperation_levels()
# Observe: Do system agents cluster together?
# Do they share resources more frequently?

# Analyze competition dynamics
competition_metrics = analyzer.measure_competition_intensity()
# Observe: Do combat rates increase with resource scarcity?
# Do territorial patterns emerge?

# Study resource utilization patterns
utilization = analyzer.analyze_resource_utilization()
# Observe: Do different agent types develop different foraging strategies?
# Does resource clustering emerge?
```

---

### 3. Track Agent Interactions and Environmental Influences

Understanding how agents interact with each other and respond to environmental conditions is crucial for ABM research.

#### Agent Interaction Tracking

AgentFarm automatically logs and tracks all agent interactions:

**Combat Interactions:**
```python
# Query combat events
combat_events = analyzer.get_combat_interactions(
    start_step=0,
    end_step=1000,
    agent_id="agent_001"  # Optional: filter by specific agent
)

# Returns detailed combat logs:
# - Timestamp and location
# - Attacker and defender identities
# - Damage dealt and health changes
# - Outcome (success/failure/retreat)
```

**Resource Sharing:**
```python
# Track resource exchange networks
sharing_network = analyzer.get_resource_sharing_network()

# Returns graph structure:
# - Nodes: agents
# - Edges: sharing events
# - Edge weights: amount of resources shared

# Analyze sharing patterns
sharing_stats = analyzer.analyze_sharing_behavior()
# Returns: sharing frequency, reciprocity rates, network centrality
```

**Proximity and Encounters:**
```python
# Track agent encounters
encounter_data = analyzer.get_encounter_history(
    distance_threshold=5.0,  # Within 5 units
    min_duration=3           # At least 3 steps
)

# Returns: who met whom, when, and for how long
```

#### Environmental Influence Analysis

Environment shapes agent behavior in multiple ways:

**Resource Availability:**
```python
# Correlate agent behavior with local resource density
behavior_resource_correlation = analyzer.correlate_behavior_with_resources()

# Returns: statistical relationships between:
# - Resource availability and agent movement patterns
# - Resource density and combat frequency
# - Resource depletion and migration behaviors
```

**Spatial Distribution:**
```python
# Analyze how spatial position influences outcomes
spatial_analysis = analyzer.analyze_spatial_influences()

# Returns:
# - Survival probability by location
# - Resource gathering efficiency by region
# - Combat frequency heat maps
```

**Environmental Stressors:**
```python
# Track agent responses to environmental changes
response_analysis = analyzer.analyze_environmental_responses(
    stressor_type="resource_depletion",
    threshold=0.3  # 30% resource reduction
)

# Returns:
# - Behavioral changes before/after stressor
# - Adaptation strategies employed
# - Population-level impacts
```

#### Interaction Networks

Visualize and analyze agent interaction networks:

```python
from farm.analysis.network_analysis import InteractionNetworkAnalyzer

network_analyzer = InteractionNetworkAnalyzer(simulation_db_path)

# Build interaction graph
interaction_graph = network_analyzer.build_interaction_graph(
    interaction_types=["combat", "sharing", "communication"],
    time_window=(0, 1000)
)

# Analyze network properties
network_metrics = network_analyzer.calculate_network_metrics()
# Returns:
# - Degree distribution
# - Clustering coefficient
# - Betweenness centrality
# - Community structure

# Identify influential agents
influential_agents = network_analyzer.identify_key_agents()
# Returns agents with high centrality scores
```

---

### 4. Analyze Trends and Patterns Over Time

Temporal analysis reveals how simulations evolve and helps identify long-term trends and cyclical patterns.

#### Time Series Analysis

AgentFarm provides comprehensive time series analysis tools:

**Population Trends:**
```python
from farm.analysis.time_series_analysis import TimeSeriesAnalyzer

ts_analyzer = TimeSeriesAnalyzer(simulation_db_path)

# Analyze population trajectories
population_trends = ts_analyzer.analyze_population_trends()

# Returns:
# - Growth/decline rates by agent type
# - Turning points and inflection points
# - Statistical trend tests (Mann-Kendall, etc.)
# - Forecast future population levels
```

**Resource Dynamics:**
```python
# Track resource availability over time
resource_trends = ts_analyzer.analyze_resource_trends()

# Returns:
# - Resource depletion/recovery cycles
# - Seasonal patterns (if configured)
# - Sustainability indicators
# - Critical thresholds and tipping points
```

**Behavioral Evolution:**
```python
# Study how agent behaviors change over time
behavior_evolution = ts_analyzer.analyze_behavior_evolution()

# Returns:
# - Action frequency changes
# - Strategy shifts
# - Learning curves
# - Adaptation timescales
```

#### Pattern Recognition

Identify recurring patterns and anomalies:

**Cyclical Patterns:**
```python
# Detect periodic behaviors
cycles = ts_analyzer.detect_cycles(
    metric="population_size",
    agent_type="SystemAgent"
)

# Returns:
# - Cycle period and amplitude
# - Phase relationships
# - Cycle stability measures
```

**Phase Transitions:**
```python
# Identify critical transitions
transitions = ts_analyzer.detect_phase_transitions()

# Returns:
# - Transition points (step numbers)
# - Regime characteristics (before/after)
# - Transition indicators (variance, autocorrelation)
```

**Anomaly Detection:**
```python
# Identify unusual events
anomalies = ts_analyzer.detect_anomalies(
    metric="combat_frequency",
    threshold=3.0  # 3 standard deviations
)

# Returns: timestamps and descriptions of anomalous events
```

#### Comparative Analysis

Compare multiple simulation runs to identify robust patterns:

```python
from farm.analysis.simulation_comparison import SimulationComparison

# Compare multiple simulations
comparison = SimulationComparison([
    "simulation_1.db",
    "simulation_2.db",
    "simulation_3.db"
])

# Identify consistent patterns across runs
common_patterns = comparison.find_common_patterns()

# Statistical comparison
statistical_tests = comparison.run_comparative_tests()
# Returns: ANOVA, t-tests, effect sizes

# Variance decomposition
variance_analysis = comparison.decompose_variance()
# Returns: within-run vs between-run variance
```

---

## Practical Examples

### Example 1: Basic Agent-Based Simulation

```python
#!/usr/bin/env python3
"""
Basic agent-based modeling example.
Demonstrates fundamental ABM concepts in AgentFarm.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.analysis.simulation_analysis import SimulationAnalyzer

def run_basic_abm():
    """Run a basic agent-based model."""
    
    # Configure simulation
    config = SimulationConfig(
        width=50,
        height=50,
        initial_resources=200,
        resource_regen_rate=0.02,
        system_agents=10,
        independent_agents=10,
        num_steps=500,
        seed=42  # For reproducibility
    )
    
    # Run simulation
    print("Running agent-based simulation...")
    results = run_simulation(config)
    
    print(f"Simulation ID: {results['simulation_id']}")
    print(f"Final step: {results['final_step']}")
    print(f"Surviving agents: {results['surviving_agents']}")
    
    # Analyze results
    analyzer = SimulationAnalyzer(results['db_path'])
    
    # Population dynamics
    pop_stats = analyzer.get_population_over_time()
    print(f"\nPopulation dynamics:")
    print(f"  Peak population: {pop_stats['peak_count']}")
    print(f"  Final population: {pop_stats['final_count']}")
    
    # Resource dynamics
    resource_stats = analyzer.get_resource_statistics()
    print(f"\nResource dynamics:")
    print(f"  Total resources consumed: {resource_stats['total_consumed']}")
    print(f"  Average resources per agent: {resource_stats['avg_per_agent']:.2f}")
    
    # Behavioral analysis
    action_dist = analyzer.analyze_action_distributions()
    print(f"\nAction distribution:")
    for action, frequency in action_dist.items():
        print(f"  {action}: {frequency:.2%}")

if __name__ == "__main__":
    run_basic_abm()
```

### Example 2: Studying Emergent Cooperation

```python
#!/usr/bin/env python3
"""
Studying emergent cooperation in agent populations.
Demonstrates how to analyze cooperative behaviors.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.analysis.simulation_analysis import SimulationAnalyzer

def study_cooperation():
    """Study emergent cooperation patterns."""
    
    # Run simulation with system agents (cooperative)
    config = SimulationConfig(
        width=100,
        height=100,
        system_agents=50,      # High proportion of cooperative agents
        independent_agents=10, # Few competitive agents
        initial_resources=300,
        resource_regen_rate=0.015,
        num_steps=1000,
        seed=42
    )
    
    print("Running cooperation study...")
    results = run_simulation(config)
    
    # Analyze cooperation
    analyzer = SimulationAnalyzer(results['db_path'])
    
    # Resource sharing analysis
    sharing_network = analyzer.get_resource_sharing_network()
    print(f"\nResource Sharing:")
    print(f"  Total sharing events: {sharing_network['total_events']}")
    print(f"  Average shares per agent: {sharing_network['avg_per_agent']:.2f}")
    
    # Compare system vs independent agents
    survival_by_type = analyzer.calculate_survival_rates_by_type()
    print(f"\nSurvival Rates:")
    for agent_type, rate in survival_by_type.items():
        print(f"  {agent_type}: {rate:.2%}")
    
    # Spatial clustering
    clustering = analyzer.analyze_spatial_clustering()
    print(f"\nSpatial Clustering:")
    print(f"  System agents clustering coefficient: {clustering['system']:.3f}")
    print(f"  Independent agents clustering: {clustering['independent']:.3f}")
    
    # Cooperative advantage
    advantage = analyzer.calculate_cooperative_advantage()
    print(f"\nCooperative Advantage:")
    print(f"  Average resources (system): {advantage['system_avg']:.2f}")
    print(f"  Average resources (independent): {advantage['independent_avg']:.2f}")
    print(f"  Advantage ratio: {advantage['ratio']:.2f}x")

if __name__ == "__main__":
    study_cooperation()
```

### Example 3: Environmental Impact Study

```python
#!/usr/bin/env python3
"""
Studying environmental influences on agent behavior.
Demonstrates environmental analysis capabilities.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation_batch
from farm.analysis.simulation_comparison import SimulationComparison

def study_environmental_impact():
    """Study how resource availability affects agent behavior."""
    
    # Create configurations with varying resource levels
    configs = []
    resource_levels = [100, 200, 400, 800]  # Low to high
    
    for resources in resource_levels:
        config = SimulationConfig(
            width=100,
            height=100,
            system_agents=25,
            independent_agents=25,
            initial_resources=resources,
            resource_regen_rate=0.01,
            num_steps=1000,
            seed=42
        )
        configs.append(config)
    
    print("Running environmental impact study...")
    batch_results = run_simulation_batch(configs)
    
    # Compare results across resource levels
    comparison = SimulationComparison([r['db_path'] for r in batch_results])
    
    # Analyze impact on survival
    survival_impact = comparison.compare_survival_rates()
    print("\nSurvival Rate by Resource Level:")
    for i, resources in enumerate(resource_levels):
        rate = survival_impact[i]
        print(f"  {resources} resources: {rate:.2%}")
    
    # Analyze impact on behavior
    behavior_impact = comparison.compare_action_distributions()
    print("\nBehavioral Changes:")
    print("  Combat frequency vs resources:", 
          behavior_impact['combat_correlation'])
    print("  Sharing frequency vs resources:", 
          behavior_impact['sharing_correlation'])
    
    # Analyze resource efficiency
    efficiency = comparison.compare_resource_efficiency()
    print("\nResource Efficiency:")
    for i, resources in enumerate(resource_levels):
        eff = efficiency[i]
        print(f"  {resources} resources: {eff['agents_per_resource']:.3f} agents/resource")

if __name__ == "__main__":
    study_environmental_impact()
```

### Example 4: Parameter Sweep Experiment

```python
#!/usr/bin/env python3
"""
Parameter sweep for agent-based model exploration.
Demonstrates systematic parameter space exploration.
"""

from farm.config import SimulationConfig
from farm.core.experiment_tracker import ExperimentTracker
import itertools

def run_parameter_sweep():
    """Run systematic parameter sweep experiment."""
    
    # Define parameter ranges
    agent_ratios = [0.2, 0.5, 0.8]  # Proportion of system agents
    resource_rates = [0.01, 0.02, 0.03]  # Regeneration rates
    total_agents = 50
    
    # Create experiment tracker
    tracker = ExperimentTracker("agent_ratio_experiment")
    
    print("Running parameter sweep...")
    results_matrix = []
    
    # Iterate over parameter combinations
    for ratio, regen_rate in itertools.product(agent_ratios, resource_rates):
        num_system = int(total_agents * ratio)
        num_independent = total_agents - num_system
        
        config = SimulationConfig(
            width=100,
            height=100,
            system_agents=num_system,
            independent_agents=num_independent,
            initial_resources=300,
            resource_regen_rate=regen_rate,
            num_steps=1000,
            seed=42
        )
        
        # Run simulation
        from farm.core.simulation import run_simulation
        results = run_simulation(config)
        
        # Store results
        tracker.log_run(
            parameters={
                "system_ratio": ratio,
                "regen_rate": regen_rate,
                "total_agents": total_agents
            },
            metrics={
                "final_population": results['surviving_agents'],
                "total_steps": results['final_step']
            }
        )
        
        results_matrix.append({
            "ratio": ratio,
            "regen_rate": regen_rate,
            "survival_rate": results['surviving_agents'] / total_agents
        })
    
    # Analyze results
    print("\n=== Parameter Sweep Results ===")
    print(f"{'System Ratio':<15} {'Regen Rate':<12} {'Survival Rate'}")
    print("-" * 50)
    
    for result in results_matrix:
        print(f"{result['ratio']:<15.2f} {result['regen_rate']:<12.3f} "
              f"{result['survival_rate']:.2%}")
    
    # Generate report
    tracker.generate_report()
    print(f"\nFull report saved to: {tracker.report_path}")

if __name__ == "__main__":
    run_parameter_sweep()
```

---

## Advanced Features

### Multi-Simulation Experiments

Run and compare multiple simulations systematically:

```python
from farm.core.experiment_tracker import ExperimentTracker

# Create experiment
experiment = ExperimentTracker("resource_competition_study")

# Define variations
variations = [
    {"system_agents": 40, "independent_agents": 10},
    {"system_agents": 25, "independent_agents": 25},
    {"system_agents": 10, "independent_agents": 40}
]

# Run experiment
experiment.run_variations(base_config, variations, iterations=5)

# Analyze results
experiment.generate_comparative_report()
```

### Custom Analysis Pipelines

Create custom analysis workflows:

```python
from farm.analysis.pipeline import AnalysisPipeline

# Define custom analysis pipeline
pipeline = AnalysisPipeline()

# Add analysis steps
pipeline.add_step("population_dynamics", analyze_population)
pipeline.add_step("behavioral_clustering", cluster_behaviors)
pipeline.add_step("network_analysis", analyze_interactions)
pipeline.add_step("report_generation", generate_report)

# Run pipeline
results = pipeline.run(simulation_db_path)
```

### Machine Learning Integration

Use ML for pattern recognition:

```python
from farm.research.behavioral_clustering import BehavioralClusterAnalyzer

# Cluster agents by behavior
clusterer = BehavioralClusterAnalyzer(simulation_db_path)

# Extract behavioral features
features = clusterer.extract_behavioral_features()

# Perform clustering
clusters = clusterer.fit_clusters(n_clusters=5)

# Analyze cluster characteristics
cluster_profiles = clusterer.profile_clusters()
```

---

## Performance Optimization

### Efficient Spatial Queries

AgentFarm uses advanced spatial indexing for efficient agent queries:

```python
from farm.core.spatial import SpatialIndex

# Spatial index automatically handles:
# - KD-tree for radial queries
# - Quadtree for range queries
# - Hash grid for neighbor queries

# Efficient nearby agent queries
nearby = spatial_service.get_agents_in_radius(
    position=(50, 50),
    radius=10.0
)
```

### Batch Processing

Process multiple simulations efficiently:

```python
from farm.core.simulation import run_simulation_batch

# Run multiple simulations in parallel
configs = [create_config(seed=i) for i in range(10)]
results = run_simulation_batch(configs, num_workers=4)
```

### Memory Management

Optimize memory usage for large simulations:

```python
# Use memory-mapped arrays for large state data
config = SimulationConfig(
    use_memmap=True,  # Enable memory mapping
    memmap_dir="/tmp/simulation_data"
)

# Use Redis for distributed agent memory
from farm.memory.redis_memory import AgentMemoryManager

memory_manager = AgentMemoryManager(
    host="localhost",
    port=6379
)
```

---

## Best Practices

### 1. Simulation Design

**Define Clear Objectives:**
- What research questions are you addressing?
- What outcomes are you measuring?
- What constitutes success or failure?

**Start Simple:**
- Begin with minimal agent types
- Use small populations for initial testing
- Add complexity incrementally

**Control Randomness:**
- Always use seeds for reproducibility
- Document random parameters
- Run multiple replications

### 2. Analysis Workflow

**Systematic Data Collection:**
```python
# Always track:
# - Initial conditions
# - Parameter values
# - Random seeds
# - Outcome metrics

results_log = {
    "config": config.to_dict(),
    "seed": 42,
    "outcomes": analyzer.get_summary_statistics()
}
```

**Validation:**
```python
# Verify simulation behavior
assert results['final_step'] == config.num_steps
assert all(agent.alive or agent.death_recorded for agent in agents)
```

**Reproducibility:**
```python
# Document everything
experiment_metadata = {
    "date": datetime.now(),
    "version": "0.1.0",
    "config": config,
    "hardware": get_system_info(),
    "results": results
}
```

### 3. Performance Considerations

**Profile Before Optimizing:**
```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()

run_simulation(config)

profiler.disable()
profiler.print_stats(sort='cumtime')
```

**Monitor Resource Usage:**
```python
from farm.utils.monitoring import SimulationMonitor

monitor = SimulationMonitor()
monitor.start()

run_simulation(config)

stats = monitor.get_statistics()
print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
print(f"Average CPU: {stats['avg_cpu_percent']:.1f}%")
```

---

## Troubleshooting

### Common Issues

**Issue: Simulation runs too slowly**

```python
# Solutions:
# 1. Reduce spatial query frequency
config.observation_update_frequency = 5  # Update every 5 steps

# 2. Use smaller observation radius
obs_config.R = 4  # Smaller radius

# 3. Enable batch updates
config.use_batch_updates = True
```

**Issue: Memory usage too high**

```python
# Solutions:
# 1. Enable memory mapping
config.use_memmap = True

# 2. Reduce observation storage
obs_config.storage_mode = "HYBRID"  # Use sparse storage

# 3. Limit history retention
config.max_history_steps = 100
```

**Issue: Agents die too quickly**

```python
# Solutions:
# 1. Increase initial resources
config.initial_resources = 500  # More resources

# 2. Increase regeneration rate
config.resource_regen_rate = 0.03  # Faster regeneration

# 3. Adjust agent starting conditions
config.agent_starting_resources = 150  # More starting resources
```

---

## Additional Resources

### Documentation
- [Core Architecture](core_architecture.md) - System design and components
- [Agents](agents.md) - Detailed agent system documentation
- [Experiments](experiments.md) - Running experiments guide
- [Usage Examples](usage_examples.md) - More practical examples
- [API Reference](api_reference.md) - Complete API documentation

### Tutorials
- [Basic Simulation Setup](usage_examples.md#tutorial-1-basic-simulation-setup)
- [Custom Agent Implementation](usage_examples.md#tutorial-2-custom-agent-behaviors)
- [Experiment Design](ExperimentQuickStart.md)

### Research Examples
- [Experiment Case Studies](experiments/)
- [Analysis Techniques](analysis/)
- [Benchmark Studies](benchmarks/reports/)

---

## Research Applications

Agent-Based Modeling in AgentFarm is suitable for various research domains:

### Ecology & Biology
- Population dynamics
- Predator-prey relationships
- Resource competition
- Evolutionary adaptation
- Disease spread models

### Social Sciences
- Cooperation emergence
- Social network formation
- Collective decision-making
- Cultural evolution
- Market dynamics

### Computer Science
- Multi-agent systems
- Reinforcement learning
- Swarm intelligence
- Distributed systems
- Emergent computation

### Economics
- Market simulations
- Resource allocation
- Agent-based macroeconomics
- Trading strategies
- Network effects

---

## Contributing

We welcome contributions to improve AgentFarm's agent-based modeling capabilities:

- **Bug Reports**: Report issues with simulations or analysis
- **Feature Requests**: Suggest new agent types or analysis methods
- **Documentation**: Improve examples and tutorials
- **Research**: Share your research findings and use cases

See [Contributing Guidelines](../CONTRIBUTING.md) for more information.

---

## Citation

If you use AgentFarm for your research, please cite:

```bibtex
@software{agentfarm2024,
  title={AgentFarm: A Platform for Agent-Based Modeling and Analysis},
  author={Dooders Research Team},
  year={2024},
  url={https://github.com/Dooders/AgentFarm}
}
```

---

## Support

For questions and support:
- **GitHub Issues**: [https://github.com/Dooders/AgentFarm/issues](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [https://github.com/Dooders/AgentFarm/docs](https://github.com/Dooders/AgentFarm/docs)
- **Discussions**: Use GitHub Discussions for questions and community interaction

---

**Ready to explore complex systems?** Start with the [Basic Simulation Example](#example-1-basic-agent-based-simulation) or check out our [Quick Start Guide](../README.md#quick-start) to begin your agent-based modeling journey!
