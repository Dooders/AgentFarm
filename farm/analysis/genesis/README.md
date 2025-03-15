# Genesis Analysis Module

The Genesis module provides comprehensive analysis of how initial states and conditions impact simulation outcomes. It examines the starting configuration of agents, resources, and their spatial relationships to understand how these factors determine the trajectory and eventual dominance patterns in simulations.

## Module Structure

The Genesis module consists of three main components:

1. **compute.py** - Contains functions for computing metrics related to initial states and conditions
2. **analyze.py** - Provides analysis functions that interpret the computed metrics
3. **plot.py** - Implements visualization capabilities for the analysis results

## Installation

The Genesis module is part of the AgentFarm package and requires the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn sqlalchemy
```

## Usage

### Using the Genesis Analysis Script

The easiest way to use the Genesis module is through the provided script:

```bash
python scripts/genesis_analysis.py --experiment_path /path/to/experiment --output_path /path/to/output
```

#### Script Arguments

- `--experiment_path`: Path to the experiment directory containing simulation data (required)
- `--output_path`: Path where analysis results and visualizations will be saved (default: ./genesis_analysis_results_[timestamp])
- `--critical_period`: Number of steps to consider as the critical period (default: 100)
- `--single_sim`: Path to a specific simulation to analyze (optional)

### Using the Module Programmatically

You can also use the Genesis module directly in your Python code:

```python
from farm.analysis.genesis.analyze import analyze_genesis_factors, analyze_genesis_across_simulations
from farm.analysis.genesis.plot import plot_genesis_analysis_results

# Analyze a single simulation
engine = create_engine(f"sqlite:///path/to/simulation.db")
Session = sessionmaker(bind=engine)
session = Session()

# Run analysis
results = analyze_genesis_factors(session)

# Generate visualizations
plot_genesis_analysis_results(results, "output_directory")

# Analyze across multiple simulations
cross_sim_results = analyze_genesis_across_simulations("path/to/experiment")
```

## Analysis Capabilities

The Genesis module provides the following analysis capabilities:

1. **Initial State Metrics** - Analyzes the starting configuration of agents and resources
2. **Resource Proximity Analysis** - Measures initial distances between agents and resources
3. **Agent Positioning Analysis** - Examines spatial distribution of agents at simulation start
4. **Starting Attribute Analysis** - Compares initial resources allocated to different agent types
5. **Initial Relative Advantages** - Calculates resource proximity advantages between agent types
6. **Critical Period Analysis** - Examines the crucial early phase of simulations
7. **Genesis Impact Scoring** - Quantifies how initial conditions affect outcomes

## Visualization Capabilities

The Genesis module provides the following visualization capabilities:

1. **Initial State Visualization** - Displays the starting configuration of agents and resources
2. **Resource Proximity Heatmaps** - Shows resource accessibility across the simulation space
3. **Initial Advantage Comparison** - Compares starting advantages across agent types
4. **Critical Period Timelines** - Visualizes key events during the formative early phase
5. **Genesis Impact Charts** - Shows which initial factors most strongly influence outcomes
6. **Feature Importance Visualization** - Displays which initial factors matter most for each outcome
7. **Cross-Simulation Comparison** - Compares initial conditions across multiple simulations

## Example Output

When you run the Genesis analysis script, it will create a directory with the following structure:

```
output_directory/
├── cross_simulation_analysis.json
├── initial_state_comparison.png
├── critical_period_analysis.png
├── iteration_1/
│   ├── genesis_analysis_results.json
│   ├── initial_state.png
│   ├── resource_proximity_heatmap.png
│   ├── initial_advantages.png
│   ├── critical_period_timeline.png
│   ├── genesis_impact_scores.png
│   └── feature_importance.png
├── iteration_2/
│   └── ...
└── ...
```

## Applications

The Genesis analysis module enables:

1. **Simulation Design Optimization** - Improve simulation configurations for more balanced outcomes
2. **Strategy Development** - Identify optimal starting conditions for different agent types
3. **Outcome Prediction** - Forecast simulation results based on initial configurations
4. **Fairness Analysis** - Evaluate whether initial conditions create inherent advantages
5. **Sensitivity Testing** - Determine how sensitive outcomes are to variations in initial conditions 