# AgentFarm MCP Simulation Server

A Model Context Protocol (MCP) server that provides programmatic access to the AgentFarm simulation system for LLM agents and other automated tools.

## Overview

This MCP server exposes the core AgentFarm simulation capabilities through a standardized interface, allowing LLM agents to:

- **Create Simulations**: Configure and run individual agent-based simulations
- **Manage Experiments**: Run multi-iteration experiments with parameter variations
- **Analyze Results**: Perform comprehensive analysis on simulation outcomes
- **Export Data**: Extract simulation data in various formats
- **Compare Simulations**: Analyze differences between simulation runs

## Installation

1. Install the MCP dependencies:
```bash
pip install -r mcp_requirements.txt
```

2. Ensure the AgentFarm system is properly installed with its dependencies:
```bash
pip install -r requirements.txt
```

## Tools Available

### Core Simulation Tools

#### `create_simulation`
Creates and runs a single simulation with specified parameters.

**Parameters:**
- `config` (object): Simulation configuration
  - `width`, `height` (int): Environment dimensions
  - `system_agents`, `independent_agents`, `control_agents` (int): Agent counts
  - `simulation_steps` (int): Number of simulation steps
  - `initial_resources` (int): Starting resource amount
  - `seed` (int, optional): Random seed for reproducibility
- `output_path` (string, optional): Path to save results

**Example:**
```json
{
  "config": {
    "width": 100,
    "height": 100,
    "system_agents": 15,
    "independent_agents": 10,
    "control_agents": 5,
    "simulation_steps": 1000,
    "initial_resources": 25,
    "seed": 42
  },
  "output_path": "my_simulation"
}
```

#### `create_experiment`
Sets up a multi-iteration experiment with parameter variations.

**Parameters:**
- `name` (string): Experiment name
- `description` (string): Experiment description
- `base_config` (object): Base configuration for all iterations
- `variations` (array, optional): Parameter variations per iteration
- `num_iterations` (int): Number of iterations to run
- `steps_per_iteration` (int): Steps per iteration

#### `run_experiment`
Executes a created experiment.

**Parameters:**
- `experiment_id` (string): ID of experiment to run
- `run_analysis` (boolean): Whether to run analysis after completion

### Analysis Tools

#### `analyze_simulation`
Runs comprehensive analysis on simulation results.

**Parameters:**
- `simulation_path` (string): Path to simulation database or experiment directory
- `analysis_types` (array): Types of analysis to run
  - Available: `"dominance"`, `"advantage"`, `"genesis"`, `"social_behavior"`, `"all"`
- `output_path` (string, optional): Path to save analysis results

#### `batch_analyze`
Runs batch analysis across multiple simulations in an experiment.

**Parameters:**
- `experiment_path` (string): Path to experiment directory
- `analysis_modules` (array): Analysis modules to run
- `save_to_db` (boolean): Whether to save results to database
- `output_path` (string, optional): Path to save consolidated analysis

#### `compare_simulations`
Compares results between multiple simulations.

**Parameters:**
- `simulation_paths` (array): Paths to simulation databases
- `metrics` (array): Metrics to compare
- `output_path` (string, optional): Path to save comparison results

### Data Management Tools

#### `export_simulation_data`
Exports simulation data in various formats.

**Parameters:**
- `simulation_path` (string): Path to simulation database
- `format` (string): Export format (`"csv"`, `"json"`, `"parquet"`)
- `data_types` (array): Data types to export
  - Available: `"agents"`, `"actions"`, `"states"`, `"resources"`, `"steps"`, `"all"`
- `output_path` (string): Path to save exported data

#### `list_simulations`
Lists all available simulations and their metadata.

**Parameters:**
- `search_path` (string): Directory to search for simulations

#### `get_simulation_status`
Gets current status and basic metrics of a simulation.

**Parameters:**
- `simulation_id` (string): ID of simulation to check

#### `get_simulation_summary`
Gets detailed summary of simulation results.

**Parameters:**
- `simulation_path` (string): Path to simulation database
- `include_charts` (boolean): Whether to generate summary charts

### Research Project Tools

#### `create_research_project`
Creates a structured research project with multiple experiments.

**Parameters:**
- `name` (string): Research project name
- `description` (string): Research description and goals
- `base_path` (string): Base directory for research
- `tags` (array): Tags for categorizing research

## Usage Examples

### Basic Simulation

```python
# Create a simple simulation
result = await session.call_tool("create_simulation", {
    "config": {
        "width": 50,
        "height": 50,
        "system_agents": 10,
        "independent_agents": 10,
        "control_agents": 10,
        "simulation_steps": 500,
        "seed": 123
    }
})
```

### Parameter Study Experiment

```python
# Create an experiment studying agent population effects
result = await session.call_tool("create_experiment", {
    "name": "population_density_study",
    "description": "Effects of population density on survival and cooperation",
    "base_config": {
        "width": 100,
        "height": 100,
        "simulation_steps": 1000,
        "initial_resources": 20
    },
    "variations": [
        {"system_agents": 5, "independent_agents": 5, "control_agents": 5},
        {"system_agents": 10, "independent_agents": 10, "control_agents": 10},
        {"system_agents": 20, "independent_agents": 20, "control_agents": 20},
        {"system_agents": 30, "independent_agents": 30, "control_agents": 30}
    ],
    "num_iterations": 4
})

# Extract experiment ID and run it
experiment_id = extract_experiment_id(result)
await session.call_tool("run_experiment", {
    "experiment_id": experiment_id,
    "run_analysis": True
})
```

### Comprehensive Analysis

```python
# Analyze simulation results
result = await session.call_tool("analyze_simulation", {
    "simulation_path": "experiments/population_study/",
    "analysis_types": ["dominance", "advantage", "social_behavior"],
    "output_path": "analysis/population_study_results"
})

# Export data for external analysis
await session.call_tool("export_simulation_data", {
    "simulation_path": "experiments/population_study/iteration_1/simulation.db",
    "format": "csv",
    "data_types": ["all"],
    "output_path": "exports/population_study_data"
})
```

## Analysis Capabilities

The server provides access to AgentFarm's comprehensive analysis suite:

### Dominance Analysis
- Population dominance patterns
- Agent type survival rates
- Dominance switching dynamics
- Resource proximity effects

### Advantage Analysis
- Resource acquisition advantages
- Reproduction efficiency comparisons
- Survival rate analysis
- Composite advantage scoring

### Genesis Analysis
- Initial condition effects
- Critical period identification
- Predictive modeling of outcomes
- Cross-simulation pattern detection

### Social Behavior Analysis
- Cooperation and competition patterns
- Resource sharing networks
- Spatial clustering behavior
- Social interaction metrics

## Configuration

The server uses the standard AgentFarm configuration system. Key parameters include:

### Environment Settings
- `width`, `height`: Environment dimensions
- `initial_resources`: Starting resource distribution
- `resource_regen_rate`: Resource regeneration speed

### Agent Settings
- `system_agents`, `independent_agents`, `control_agents`: Agent populations
- `initial_resource_level`: Starting agent resources
- `max_population`: Population capacity limits

### Learning Parameters
- `learning_rate`: DQN learning rate
- `epsilon_start`, `epsilon_min`, `epsilon_decay`: Exploration parameters
- `batch_size`, `memory_size`: Training parameters

### Combat Parameters
- `starting_health`: Agent health values
- `attack_range`, `attack_base_damage`: Combat mechanics

## Error Handling

The server provides detailed error messages and logging for debugging:

- Configuration validation errors
- Simulation runtime errors
- Analysis processing errors
- File I/O and database errors

All errors include full stack traces for troubleshooting.

## Running the Server

### As a standalone server:
```bash
python mcp_simulation_server.py
```

### With an MCP-compatible LLM client:
Configure your LLM client to connect to the server using stdio transport.

### Example client integration:
```python
from mcp_client_example import main
await main()
```

## File Structure

The server creates the following directory structure:

```
simulations/          # Individual simulation runs
experiments/          # Multi-iteration experiments
research/            # Research projects
comparisons/         # Simulation comparisons
exports/             # Exported data
analysis/            # Analysis results
```

## Advanced Usage

### Custom Analysis Pipelines

The server can be extended to run custom analysis pipelines:

```python
# Batch analysis across multiple experiments
await session.call_tool("batch_analyze", {
    "experiment_path": "experiments/agent_behavior_study/",
    "analysis_modules": ["dominance", "advantage", "social_behavior"],
    "save_to_db": True,
    "output_path": "analysis/comprehensive_study"
})
```

### Research Project Management

```python
# Create structured research project
await session.call_tool("create_research_project", {
    "name": "emergence_of_cooperation",
    "description": "Investigating how cooperative behaviors emerge in multi-agent systems",
    "tags": ["cooperation", "emergence", "social_behavior"]
})
```

## Integration Notes

This MCP server is designed to integrate seamlessly with:

- **Claude and other LLM agents**: Via the MCP protocol
- **Jupyter notebooks**: For interactive research
- **CI/CD pipelines**: For automated testing and validation
- **Research workflows**: For systematic experimentation

## Troubleshooting

### Common Issues

1. **Module import errors**: Ensure the AgentFarm package is in your Python path
2. **Database connection errors**: Check file permissions and disk space
3. **Memory issues**: Use smaller simulation parameters for large experiments
4. **Analysis failures**: Verify simulation completed successfully before analysis

### Logging

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements:
- Real-time simulation monitoring
- Interactive parameter optimization
- Advanced visualization generation
- Integration with external ML platforms
- Distributed experiment execution