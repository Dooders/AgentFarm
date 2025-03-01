# Generational Fitness Analysis

This module analyzes the fitness and adaptation of agents across generations in the AgentFarm simulation environment. It compares metrics between earlier and later generations to measure evolutionary progress.

## Overview

The generational fitness analysis module provides tools to:

1. Extract performance metrics for agents grouped by generation
2. Analyze resource acquisition and usage efficiency across generations
3. Evaluate action effectiveness and strategy evolution
4. Compare first generation metrics to later generations
5. Visualize fitness trends with charts and reports

## Key Metrics Analyzed

- **Survival Metrics**: Average survival time, reproduction success rate
- **Resource Metrics**: Average resource level, maximum resources acquired, resource efficiency
- **Action Effectiveness**: Reward per action, action distribution, strategy adaptation
- **Reproductive Success**: Reproduction attempts, success rate, offspring quality

## Usage

The module is automatically integrated with the `simple_research_analysis.py` script and will run when analyzing experiments. The analysis results are saved in the experiment's analysis directory under a `generational_fitness` folder.

### Manual Usage

You can also use the module directly:

```python
import generational_fitness_analysis

# Analyze a single database
db_path = "path/to/simulation.db"
output_dir = "path/to/output"
generational_fitness_analysis.analyze_generational_fitness(db_path, output_dir)

# Process an entire experiment
experiment_name = "my_experiment"
results = generational_fitness_analysis.process_experiment_generational_fitness(experiment_name)
```

## Output Files

For each simulation database, the module generates:

1. **Survival Trends Chart**: Shows how survival time and reproduction rate change across generations
2. **Resource Acquisition Chart**: Visualizes resource gathering efficiency across generations
3. **Reward Optimization Chart**: Tracks how rewards change across generations
4. **Action Evolution Chart**: Shows how action distribution and effectiveness evolve
5. **Generational Fitness Report**: Text report comparing first generation to later generations

## Interpreting Results

The generational fitness report provides a detailed comparison between the first generation and later generations, including:

- Absolute differences in key metrics
- Percentage changes from baseline
- Interpretations of whether metrics show improvement or decline
- Overall assessment of evolutionary progress

Positive percentage changes indicate that later generations have improved in that metric compared to the first generation, suggesting successful adaptation and evolution.

## Example Report

```
Generational Fitness Analysis Report
===================================

Comparing first generation to latest generations

Performance Metrics:
-----------------
avg_survival_time:
  First generation: 45.2500
  Latest generations (avg): 78.6667
  Absolute difference: 33.4167
  Percent change: 73.85%
  Interpretation: Significant improvement

avg_resources:
  First generation: 12.3400
  Latest generations (avg): 18.7500
  Absolute difference: 6.4100
  Percent change: 51.94%
  Interpretation: Significant improvement

...

Overall Fitness Assessment:
-------------------------
Later generations show improvement in 7/9 metrics (77.8%).
Overall, later generations demonstrate better fitness and adaptation.
```

## Integration

This module is designed to work seamlessly with the existing AgentFarm analysis pipeline. It's automatically called during experiment analysis and results are stored alongside other analysis outputs. 