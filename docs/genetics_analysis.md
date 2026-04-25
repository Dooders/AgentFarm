# Genetics Analysis Module

The genetics analysis module lives under `farm/analysis/genetics/` and is exposed through `AnalysisService` as `module_name="genetics"`.

## Input Sources

The module supports two distinct input sources.  It is important to understand which path applies to your use case:

### 1. Simulation database (`simulation.db`)

When an `experiment_path` containing a `simulation.db` SQLite file is passed to `AnalysisService`, the data processor loads agent records (generation, lineage, action weights) from that database automatically.

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

# Requires experiment_path / "simulation.db" to exist
result = service.run(
    AnalysisRequest(
        module_name="genetics",
        experiment_path=Path("data/experiment_001"),  # must contain simulation.db
        output_path=Path("results/genetics"),
    )
)
```

### 2. Evolution-experiment data (in-memory)

Evolution-experiment results (`EvolutionExperimentResult` objects or plain `pd.DataFrame` values) do **not** reside on disk in a format the path-based loader recognises.  Pass them directly to the lower-level analysis APIs instead of going through `AnalysisService` with a path:

```python
import pandas as pd
from farm.analysis.common.context import AnalysisContext
from farm.analysis.genetics.analyze import analyze_genetics, generate_genetics_report

# df is a DataFrame produced by build_evolution_experiment_dataframe()
# or assembled from EvolutionExperimentResult data.
ctx = AnalysisContext(output_path=Path("results/genetics"))
stats = analyze_genetics(df)
report_path = generate_genetics_report(df, ctx)
```

## Quick Start (simulation.db path)

Run specific function groups:

```python
# Plots only
service.run(AnalysisRequest(
    module_name="genetics",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/genetics_plots"),
    group="plots",
))

# Report pipeline
service.run(AnalysisRequest(
    module_name="genetics",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/genetics_report"),
    group="report",
))
```

## Function Groups

The module currently exposes these groups:

- `all`
- `analysis`
- `plots`
- `basic`
- `report`
- `fitness_landscape`
- `population_genetics`
- `adaptation_signatures`

## Main Capabilities

- Population-level genetics summary via `analyze_genetics`
- Report generation via `generate_genetics_report`:
  - `genetics_report.md`
  - `genetics_report.html` (optional)
  - `genetics_summary.json`
- Plot outputs:
  - generation distribution
  - fitness over generations
  - allele-frequency trajectories
  - diversity over time
  - Wright-Fisher neutral-drift overlay
  - phylogenetic tree (full + sampled)
  - conserved-run timeline
  - marginal fitness effect and 2D fitness landscape
- Population genetics compute functions:
  - F_ST pairwise differentiation
  - migration counts
  - gene-flow timeseries
- Adaptation signatures:
  - realized mutation rate (per generation and per locus)
  - conserved runs and run/fitness correlation
  - sweep candidate detection

## Expected Outputs

Typical run outputs are written under the request `output_path` and include:

- Report files (`.md`, optional `.html`, `.json`)
- Plot files (`.png`)
- Analysis summary metadata from the service

## API Pointers

- Module registration: `farm/analysis/genetics/module.py`
- Compute functions: `farm/analysis/genetics/compute.py`
- Plot functions: `farm/analysis/genetics/plot.py`
- Report/summary functions: `farm/analysis/genetics/analyze.py`
- High-level package exports: `farm/analysis/genetics/__init__.py`
