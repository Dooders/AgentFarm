# Genetics Analysis Module

The genetics analysis module lives under `farm/analysis/genetics/` and is exposed through `AnalysisService` as `module_name="genetics"`.

It supports two input sources:

- Simulation database data (agent generation, lineage, action weights)
- Evolution experiment data (candidate lineage, chromosome values, fitness)

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

# Default run (group="all")
result = service.run(
    AnalysisRequest(
        module_name="genetics",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/genetics"),
    )
)
```

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
