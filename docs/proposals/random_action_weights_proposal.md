# Random Action Weight Distribution Experiments

## Overview

This proposal outlines a framework for conducting experiments with random action weight distributions in the AgentFarm simulation. By enabling agents to start with randomized action preferences, we can study how different behavioral tendencies evolve over time and identify optimal strategies that emerge through natural selection.

## Background

Currently, agents in the simulation have predefined action weights that determine their likelihood of choosing different actions (move, gather, share, attack, reproduce). These weights are part of the agent's genomic code and can evolve through mutation and crossover during reproduction.

The existing implementation already supports:
- Storing action weights in the database
- Including action weights in the agent's genome
- Mutating and crossing over action weights during reproduction
- Tracking action weights across generations

## Proposed Implementation

### 1. Random Weight Generation

Add a new static method to the `Genome` class for generating random action weights:

```python
@staticmethod
def random_weights(action_names=None, distribution="uniform"):
    """Generate a genome with random action weights.
    
    Args:
        action_names: List of action names to include (defaults to standard set)
        distribution: Type of random distribution to use ("uniform", "gaussian", "exponential")
        
    Returns:
        A genome dictionary with randomized action weights
    """
    if action_names is None:
        action_names = ["move", "gather", "share", "attack", "reproduce"]
        
    # Generate random weights based on specified distribution
    weights = []
    for _ in range(len(action_names)):
        if distribution == "uniform":
            weights.append(random.random())
        elif distribution == "gaussian":
            # Mean 0.5, std dev 0.2, clamped to positive values
            weights.append(max(0.01, random.gauss(0.5, 0.2)))
        elif distribution == "exponential":
            weights.append(random.expovariate(2.0))
        else:
            weights.append(random.random())  # Default to uniform
    
    # Normalize weights
    total = sum(weights)
    normalized_weights = [w/total for w in weights]
    
    # Create action set
    action_set = list(zip(action_names, normalized_weights))
    
    # Create minimal genome
    return {
        "action_set": action_set,
        "module_states": {},
        "agent_type": "IndependentAgent",  # Can be overridden later
        "resource_level": 100,  # Default value
        "current_health": 100,  # Default value
    }
```

### 2. Population Generation Utility

Create a utility function for generating populations with random action weights:

```python
def create_population_with_random_weights(
    environment, 
    population_size, 
    agent_types=None,
    weight_distribution="uniform"
):
    """Create a population of agents with random action weights.
    
    Args:
        environment: The simulation environment
        population_size: Number of agents to create
        agent_types: List of agent classes to use (defaults to all types)
        weight_distribution: Type of random distribution for weights
        
    Returns:
        List of created agents
    """
    from farm.agents import SystemAgent, IndependentAgent, ControlAgent
    
    if agent_types is None:
        agent_types = [SystemAgent, IndependentAgent, ControlAgent]
    
    agents = []
    
    for i in range(population_size):
        # Select random position
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height)
        )
        
        # Select random agent type
        agent_class = random.choice(agent_types)
        
        # Generate random genome
        genome = Genome.random_weights(distribution=weight_distribution)
        genome["agent_type"] = agent_class.__name__
        
        # Create agent from genome
        agent_id = environment.get_next_agent_id()
        agent = Genome.to_agent(genome, agent_id, position, environment)
        
        # Add to environment
        environment.add_agent(agent)
        agents.append(agent)
    
    return agents
```

### 3. Experiment Configuration

Extend the simulation configuration to support random weight experiments:

```python
class RandomWeightExperimentConfig:
    """Configuration for random weight distribution experiments."""
    
    # Whether to use random weights
    use_random_weights = True
    
    # Distribution type for random weights
    # Options: "uniform", "gaussian", "exponential"
    weight_distribution = "uniform"
    
    # Whether to track weight evolution
    track_weight_evolution = True
    
    # Whether to use different distributions for different agent types
    agent_specific_distributions = False
    
    # Agent-specific distributions if enabled
    agent_distributions = {
        "SystemAgent": "gaussian",
        "IndependentAgent": "uniform",
        "ControlAgent": "exponential"
    }
```

### 4. Analysis Tools

Add utilities for analyzing the evolution of action weights:

```python
def analyze_weight_evolution(db_path, num_generations=10):
    """Analyze how action weights evolve over generations.
    
    Args:
        db_path: Path to simulation database
        num_generations: Number of generations to analyze
        
    Returns:
        DataFrame with weight statistics by generation
    """
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    query = text("""
        SELECT 
            generation,
            COUNT(*) as agent_count,
            AVG(json_extract(action_weights, '$.move')) as avg_move_weight,
            AVG(json_extract(action_weights, '$.gather')) as avg_gather_weight,
            AVG(json_extract(action_weights, '$.share')) as avg_share_weight,
            AVG(json_extract(action_weights, '$.attack')) as avg_attack_weight,
            AVG(json_extract(action_weights, '$.reproduce')) as avg_reproduce_weight,
            agent_type
        FROM agents
        GROUP BY generation, agent_type
        ORDER BY generation
        LIMIT :num_generations
    """)
    
    df = pd.read_sql(query, engine, params={"num_generations": num_generations})
    return df
```

## Experimental Design

With these tools in place, we can conduct several types of experiments:

1. **Baseline Comparison**: Compare the performance of agents with random weights to those with predefined weights.

2. **Distribution Comparison**: Test which random distribution (uniform, gaussian, exponential) produces the most successful agents.

3. **Evolution Tracking**: Observe how action weights evolve over generations and identify convergence patterns.

4. **Agent Type Analysis**: Determine if different agent types (System, Independent, Control) evolve different optimal weight distributions.

5. **Environmental Adaptation**: Test how different environmental conditions affect the evolution of action weights.

## Implementation Plan

1. Add the `random_weights` method to the `Genome` class
2. Create the population generation utility
3. Extend the simulation configuration
4. Implement analysis tools
5. Create example experiments

## Expected Outcomes

This implementation will allow us to:

1. Discover emergent behavioral strategies through evolution
2. Identify optimal action weight distributions for different scenarios
3. Better understand the relationship between agent behavior and survival
4. Create more diverse and realistic agent populations

## Conclusion

Implementing random action weight distributions will significantly enhance the experimental capabilities of the AgentFarm simulation. By allowing agents to start with diverse behavioral tendencies and tracking how these evolve, we can gain valuable insights into emergent strategies and optimal behavior patterns in multi-agent systems. 