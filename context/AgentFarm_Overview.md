# AgentFarm: A Comprehensive Guide

## Introduction

AgentFarm is a sophisticated agent-based simulation platform designed for researching complex systems, emergent behaviors, and multi-agent interactions. The platform allows researchers to create, run, and analyze simulations where autonomous agents interact with each other and their environment, competing for resources, reproducing, and evolving over time.

## Core Concepts

### Agents

Agents are autonomous entities that can:
- Move around the environment
- Gather resources
- Share resources with other agents
- Attack other agents
- Reproduce to create offspring

The codebase uses a unified agent system:
- **BaseAgent**: The unified agent class that handles all agent behaviors including movement, resource gathering, sharing, and combat

The `BaseAgent` class provides core functionality including:
- Decision-making mechanisms
- Resource management
- Reproduction capabilities
- Combat systems
- Perception of the environment

### Environment

The simulation environment (`Environment` class) manages:
- A 2D grid where agents and resources exist
- Resource distribution and regeneration
- Agent interactions and collisions
- Metrics collection and analysis
- Simulation state persistence

### Resources

Resources are the primary commodity in the simulation:
- Agents need resources to survive
- Resources regenerate over time
- Resource distribution can be configured
- Agents compete for limited resources

## Learning and Decision Making

Agents use reinforcement learning to make decisions:
- Each agent has specialized modules for different actions (move, gather, attack, share)
- Agents learn from experience using Deep Q-Networks (DQN)
- Experiences are stored in replay memory for training
- Exploration vs. exploitation is managed through epsilon-greedy policies

## Configuration System

The simulation is highly configurable through a YAML-based configuration system:
- Environment parameters (size, resource distribution)
- Agent parameters (initial resources, perception radius)
- Learning parameters (learning rates, memory sizes)
- Action-specific parameters for each module
- Visualization settings

## Running Simulations

There are multiple ways to run simulations:
1. **Single Simulation**: Using `run_simulation.py` for individual runs
2. **Experiments**: Using `run_experiment.py` for parameter sweeps and comparative analysis
3. **GUI Interface**: Through the Tkinter-based GUI for visual interaction

### Single Simulation

```bash
python run_simulation.py --config config.yaml --steps 1000
```

### Experiments

Experiments allow running multiple simulations with different parameter configurations to compare outcomes:

```bash
python run_experiment.py --experiment dominance_factors
```

## Analysis Tools

AgentFarm includes comprehensive analysis tools:
- **Comparative Analysis**: Compare outcomes across different simulation configurations
- **Agent Analysis**: Analyze individual agent behaviors and performance
- **Health/Resource Dynamics**: Study how resources affect agent health and survival
- **Action Distribution**: Analyze what actions agents take over time
- **Reproduction Diagnosis**: Study reproduction patterns and genealogy
- **Learning Experience**: Analyze how agents learn over time

## Data Persistence

Simulation data is stored in SQLite databases:
- Agent states and actions
- Resource states
- Reproduction events
- Metrics over time
- Learning experiences

## Visualization

The platform includes multiple visualization options:
- Real-time visualization during simulation
- Post-simulation analysis charts and graphs
- Comparative visualizations for experiments
- Video generation of simulation runs

## Project Structure

```
AgentFarm/
├── farm/                   # Core simulation framework
│   ├── actions/            # Agent action modules
│   ├── agents/             # Agent implementations
│   ├── analysis/           # Analysis tools
│   ├── core/               # Core simulation components
│   ├── database/           # Data persistence
│   ├── environments/       # Environment implementations
│   ├── gui/                # GUI components
│   ├── runners/            # Simulation runners
│   └── visualization/      # Visualization tools
├── scripts/                # Utility scripts
├── simulations/            # Simulation output directory
├── experiments/            # Experiment configurations
├── results/                # Analysis results
├── logs/                   # Log files
├── config.yaml             # Default configuration
├── run_simulation.py       # Script to run a single simulation
└── run_experiment.py       # Script to run experiments
```

## Getting Started

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a basic simulation**:
   ```bash
   python run_simulation.py --config config.yaml --steps 1000
   ```

3. **View the results**:
   - Check the `simulations` directory for output data
   - Use analysis tools to generate visualizations:
     ```bash
     python -m farm.analysis.comparative_analysis --path simulations/latest
     ```

## Advanced Usage

### Custom Agent Types

You can create custom agent types by extending the `BaseAgent` class:

```python
from farm.core.agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_id, position, resource_level, environment, **kwargs):
        super().__init__(agent_id, position, resource_level, environment, **kwargs)
        # Custom initialization
        
    def decide_action(self):
        # Custom decision logic
        return super().decide_action()
```

### Custom Environments

Create specialized environments by extending the `Environment` class:

```python
from farm.core.environment import Environment

class MyCustomEnvironment(Environment):
    def __init__(self, width, height, resource_distribution, **kwargs):
        super().__init__(width, height, resource_distribution, **kwargs)
        # Custom initialization
        
    def update(self):
        # Custom update logic
        super().update()
```

### Custom Experiments

Define custom experiments in `run_experiment.py`:

```python
experiments = [
    ExperimentConfig(
        name="my_experiment",
        variations=[
            {"parameter_name": value1},
            {"parameter_name": value2},
        ],
        num_iterations=5,
        num_steps=1000
    )
]
```

## Key Files

- **config.yaml**: Main configuration file
- **run_simulation.py**: Script to run individual simulations
- **run_experiment.py**: Script to run experiments with parameter variations
- **farm/core/environment.py**: Core environment implementation
- **farm/agents/base_agent.py**: Base agent implementation
- **farm/core/simulation.py**: Simulation controller

## Conclusion

AgentFarm provides a powerful platform for researching complex systems through agent-based simulations. With its flexible configuration, comprehensive analysis tools, and extensible architecture, it enables researchers to explore a wide range of scenarios and hypotheses about agent behavior, resource competition, and emergent phenomena. 