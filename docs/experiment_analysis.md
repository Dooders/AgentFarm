# FARM Experiment Analysis

This document outlines the available experiment analysis capabilities in the FARM system, focusing on comparative analysis across multiple simulations.

## Available Analysis Tools

### 1. Comparative Analysis (`farm/analysis/comparative_analysis.py`)

Compares metrics across multiple simulations including:

- Population dynamics
- Resource utilization 
- Health and reward distributions
- Birth/death rates
- Agent lifespans
- Population vs simulation duration
- Population vs agent age

Key functions:
```python
compare_simulations(search_path: str, analysis_path: str)
plot_comparative_metrics(df: pd.DataFrame, output_dir: str)
```

### 2. Action Type Distribution (`farm/analysis/action_type_distribution.py`)

Analyzes action patterns across simulations:

- Action frequency distribution
- Action-reward correlations
- Chi-square tests for action-success relationships
- Temporal action patterns

Key functions:
```python
calculate_action_frequencies(actions_df: pd.DataFrame)
calculate_action_correlations(actions_df: pd.DataFrame)
```

### 3. Health Resource Dynamics (`farm/analysis/health_resource_dynamics.py`)

Studies health and resource relationships:

- Cross-correlation analysis
- Fourier analysis of cycles
- Health prediction modeling
- Strategy clustering

Key functions:
```python
analyze_health_resource_dynamics(db_path: str)
analyze_health_strategies(data: pd.DataFrame)
```

### 4. Learning Experience Analysis (`farm/analysis/learning_experience.py`)

Evaluates learning outcomes:

- Reward vs loss relationships
- State change impact
- Action selection patterns
- Learning efficiency over time

Key functions:
```python
analyze_learning_experiences(db_path: str)
plot_learning_metrics(rl_data: pd.DataFrame)
```

### 5. Reproduction Analysis (`farm/analysis/reproduction_diagnosis.py`)

Examines reproduction patterns:

- Success/failure rates
- Resource level impacts
- Generational analysis
- Population sustainability

Key functions:
```python
analyze_reproduction_patterns() -> Dict
plot_diagnostics(metrics: Dict)
```

### 6. Reward Efficiency Analysis (`farm/analysis/reward_efficiency.py`)

Studies reward optimization:

- Action-specific rewards
- Agent type efficiency
- Resource-reward relationships
- Strategy effectiveness

Key functions:
```python
analyze_reward_efficiency(data: pd.DataFrame)
reward_efficiency_pipeline(db_path: str)
```

### 7. Evolutionary / Genetics Analysis (`farm/analysis/genetics/`)

Analyzes agent genomes, chromosomes, lineage, and population-level genetic
statistics from simulation databases and evolution experiments.  This module
integrates with the `AnalysisService` and can be invoked directly:

See also the dedicated guide: [Genetics Analysis Module](genetics_analysis.md).

> **Note:** The path-based `AnalysisService` workflow (shown below) loads
> genetics data from a `simulation.db` SQLite file inside `experiment_path`.
> Evolution-experiment results that are not backed by a simulation DB must be
> passed as an in-memory `pd.DataFrame` directly to the lower-level APIs
> (e.g. `analyze_genetics`, `generate_genetics_report`) rather than via the
> path-based service call.

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

# Full genetics analysis — experiment_path must contain simulation.db
result = service.run(AnalysisRequest(
    module_name="genetics",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/genetics"),
))

# Only plots
result = service.run(AnalysisRequest(
    module_name="genetics",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/genetics"),
    group="plots",   # Options: "all", "analysis", "plots", "basic",
                     #          "report", "fitness_landscape",
                     #          "population_genetics", "adaptation_signatures"
))
```

Key capabilities:

- **Genetic diversity** – expected heterozygosity, Shannon entropy per locus
- **Allele-frequency trajectories** – per-locus frequency timeseries
- **Wright-Fisher overlay** – observed trajectories vs. neutral drift baseline
- **Phylogenetic trees** – lineage graphs (full + sampled) from parent-child IDs
- **Fitness landscapes** – single-locus correlations, pairwise epistasis, 2-D heatmap
- **Conserved-run timeline** – detects loci fixed across consecutive generations
- **Adaptation signatures** – realized mutation rates, sweep candidates
- **Population genetics** – F_ST differentiation, migration counts, gene-flow timeseries
- **Summary report** – Markdown (+ optional HTML) report via `generate_genetics_report`

Key functions:
```python
from farm.analysis.genetics import (
    analyze_genetics,              # population-level summary statistics
    generate_genetics_report,     # Markdown/HTML summary report
)
from farm.analysis.genetics.plot import (
    plot_allele_frequency_trajectories,
    plot_diversity_over_time,
    plot_wright_fisher_overlay,
    plot_phylogenetic_tree_basic,
    plot_phylogenetic_tree_sampled,
    plot_conserved_run_timeline,
    plot_fitness_landscape_2d,
)
```

## Usage Examples

### Basic Comparative Analysis
```python
from farm.analysis.comparative_analysis import compare_simulations

# Compare multiple simulations
compare_simulations(
    search_path="experiments/initial_experiments/databases",
    analysis_path="experiments/analysis_results"
)
```

### Action Pattern Analysis
```python
from farm.analysis.action_type_distribution import main as analyze_actions

# Analyze action patterns
analyze_actions(engine)  # SQLAlchemy engine connected to database
```

### Health-Resource Analysis
```python
from farm.analysis.health_resource_dynamics import analyze_health_resource_dynamics

# Analyze health and resource dynamics
analyze_health_resource_dynamics("simulations/simulation.db")
```

## Output Formats

Analysis results are provided in multiple formats:

1. **Visualizations**: PNG files with plots and charts
2. **CSV Data**: Raw numerical data for further analysis
3. **JSON Reports**: Structured analysis results
4. **Markdown Reports**: Human-readable summaries
5. **HTML Dashboards**: Interactive visualization dashboards

## Key Metrics Tracked

1. **Population Dynamics**
   - Mean/median/mode population
   - Population stability
   - Growth/decline rates

2. **Resource Management**
   - Resource efficiency
   - Distribution patterns
   - Sustainability metrics

3. **Agent Performance**
   - Health trends
   - Reward accumulation
   - Action effectiveness

4. **Learning Outcomes**
   - Strategy adaptation
   - Reward optimization
   - Knowledge transfer

5. **System Stability**
   - Population sustainability
   - Resource balance
   - Agent type distribution

## Future Enhancements

Planned analysis capabilities:

1. **Network Analysis**
   - Agent interaction patterns
   - Resource flow networks
   - Information propagation

2. **Environmental Impact**
   - Resource depletion patterns
   - Carrying capacity analysis
   - Sustainability metrics

3. **Agent Behavior**
   - Decision tree analysis
   - Strategy classification
   - Behavioral clustering

4. **Performance Optimization**
   - Bottleneck identification
   - Resource utilization
   - System efficiency metrics
