# Unified AgentFarm API

A clean, intuitive API for agentic systems to control simulations and experiments through a unified interface.

## Overview

The Unified AgentFarm API provides a simplified interface that abstracts away the complexity of the underlying simulation and experiment systems. It's designed specifically for agents to easily understand and use for research and experimentation.

## Key Features

- **Session-Based Organization**: Organize work into logical sessions with persistent state
- **Unified Interface**: Single API for both simulations and experiments
- **Configuration Templates**: Pre-built configurations for common scenarios
- **Real-time Monitoring**: Track progress and status of running simulations/experiments
- **Event System**: Subscribe to and monitor events during execution
- **Analysis Integration**: Built-in analysis and comparison capabilities
- **Error Handling**: Comprehensive error handling and validation

## Quick Start

```python
from farm.api import AgentFarmController

# Initialize controller
controller = AgentFarmController()

# Create a session
session_id = controller.create_session("My Research", "Testing agent behaviors")

# Create and run a simulation
simulation_id = controller.create_simulation(session_id, {
    "name": "Basic Test",
    "steps": 1000,
    "agents": {"system_agents": 10, "independent_agents": 10}
})

# Start simulation
controller.start_simulation(session_id, simulation_id)

# Monitor progress
status = controller.get_simulation_status(session_id, simulation_id)
print(f"Progress: {status.progress_percentage:.1f}%")

# Get results
results = controller.get_simulation_results(session_id, simulation_id)
print(f"Final agents: {results.final_agent_count}")
```

## Core Components

### AgentFarmController

The main controller class that provides the unified API:

```python
controller = AgentFarmController(workspace_path="my_workspace")
```

**Key Methods:**
- `create_session(name, description)` - Create a new research session
- `create_simulation(session_id, config)` - Create a simulation
- `start_simulation(session_id, simulation_id)` - Start a simulation
- `get_simulation_status(session_id, simulation_id)` - Get current status
- `get_simulation_results(session_id, simulation_id)` - Get results

### Session Management

Sessions organize your work and provide persistent storage:

```python
# Create session
session_id = controller.create_session("My Research", "Description")

# List sessions
sessions = controller.list_sessions()

# Get session info
session = controller.get_session(session_id)
```

### Configuration Templates

Pre-built configurations for common scenarios:

```python
# List available templates
templates = controller.get_available_configs()

# Create config from template
config = controller.create_config_from_template("basic_simulation", {
    "steps": 2000,
    "agents": {"system_agents": 20}
})

# Validate configuration
validation = controller.validate_config(config)
```

## Configuration Examples

### Basic Simulation

```python
config = {
    "name": "Basic Test",
    "steps": 1000,
    "environment": {
        "width": 100,
        "height": 100,
        "resources": 50
    },
    "agents": {
        "system_agents": 10,
        "independent_agents": 10,
        "control_agents": 0
    },
    "learning": {
        "enabled": True,
        "algorithm": "dqn"
    }
}
```

### Combat Simulation

```python
config = {
    "name": "Combat Test",
    "steps": 2000,
    "environment": {
        "width": 150,
        "height": 150,
        "resources": 30
    },
    "agents": {
        "system_agents": 15,
        "independent_agents": 15
    },
    "combat": {
        "enabled": True,
        "damage_multiplier": 1.5
    }
}
```

### Experiment with Variations

```python
config = {
    "name": "Parameter Study",
    "description": "Compare different agent populations",
    "iterations": 5,
    "base_config": {
        "steps": 1000,
        "environment": {"width": 100, "height": 100, "resources": 50}
    },
    "variations": [
        {"agents": {"system_agents": 5, "independent_agents": 15}},
        {"agents": {"system_agents": 10, "independent_agents": 10}},
        {"agents": {"system_agents": 15, "independent_agents": 5}}
    ]
}
```

## Available Templates

- **basic_simulation**: Simple simulation with default parameters
- **combat_simulation**: Combat-focused simulation with fighting mechanics
- **research_simulation**: Research-focused with detailed logging
- **basic_experiment**: Basic experiment with parameter variations
- **parameter_sweep**: Systematic parameter variation study

## Event Monitoring

Subscribe to events for real-time monitoring:

```python
# Subscribe to events
subscription_id = controller.subscribe_to_events(
    session_id,
    ["simulation_started", "simulation_completed"],
    simulation_id=simulation_id
)

# Get event history
events = controller.get_event_history(session_id, subscription_id)
```

## Analysis and Comparison

Built-in analysis capabilities:

```python
# Analyze single simulation
analysis = controller.analyze_simulation(session_id, simulation_id)

# Compare multiple simulations
comparison = controller.compare_simulations(session_id, [sim_id1, sim_id2, sim_id3])
```

## Error Handling

The API provides comprehensive error handling:

```python
try:
    simulation_id = controller.create_simulation(session_id, config)
    controller.start_simulation(session_id, simulation_id)
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## CLI Interface

Test the API using the command-line interface:

```bash
# Run all demos
python -m farm.api.cli --all

# Run specific demo
python -m farm.api.cli --demo simulation

# Run with custom workspace
python -m farm.api.cli --workspace my_workspace --demo experiment
```

## Examples

Comprehensive examples are available in `examples.py`:

```bash
python -m farm.api.examples
```

Examples include:
- Basic simulation workflow
- Parameter variation experiments
- Configuration template usage
- Analysis and comparison
- Event monitoring

## Data Models

The API uses structured data models for all responses:

- **SessionInfo**: Session information and metadata
- **SimulationStatus**: Current simulation status and progress
- **SimulationResults**: Results from completed simulations
- **ExperimentStatus**: Current experiment status and progress
- **ExperimentResults**: Results from completed experiments
- **ConfigTemplate**: Configuration template definitions
- **ValidationResult**: Configuration validation results

## Best Practices

1. **Use Sessions**: Organize related work into sessions
2. **Validate Configs**: Always validate configurations before use
3. **Monitor Progress**: Use status methods to track execution
4. **Handle Errors**: Implement proper error handling
5. **Clean Up**: Use context managers or call cleanup() when done

## Integration

The API is designed to be easily integrated with:
- MCP (Model Context Protocol) servers
- Web APIs
- Command-line tools
- Jupyter notebooks
- Research workflows

## Support

For questions, issues, or contributions, please refer to the main AgentFarm documentation or create an issue in the project repository.
