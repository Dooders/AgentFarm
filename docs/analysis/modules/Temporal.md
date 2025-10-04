# Temporal Analysis Module

**Module Name**: `temporal`

Analyze temporal patterns, time series trends, periodic behaviors, and temporal efficiency.

---

## Overview

The Temporal module examines how metrics change over time, detecting trends, cycles, and temporal patterns in simulation data.

### Key Features

- Time series analysis
- Trend detection
- Seasonality/periodicity detection
- Autocorrelation analysis
- Efficiency over time
- Change point detection

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="temporal",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/temporal")
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Time step/iteration
- At least one metric column (numeric)

### Optional Columns

- Any metric columns to analyze
- `phase` (str): Simulation phase
- `event` (str): Special events

---

## Analysis Functions

### analyze_trends

Analyze temporal trends in metrics.

**Outputs:**
- `temporal_trends.csv`: Trend statistics
- Trend coefficients, R², significance

### analyze_periodicity

Detect periodic patterns and cycles.

**Outputs:**
- `periodicity_analysis.csv`: Cycle detection
- Period lengths, amplitudes, phases

### analyze_autocorrelation

Analyze temporal autocorrelation.

**Outputs:**
- `autocorrelation.csv`: ACF statistics
- Lag correlations, persistence

### analyze_change_points

Detect significant changes in time series.

**Outputs:**
- `change_points.csv`: Detected changes
- Change locations, magnitudes, directions

---

## Visualization Functions

### plot_time_series

Plot metrics over time.

**Output:** `time_series.png`

### plot_trends

Visualize temporal trends.

**Output:** `temporal_trends.png`

### plot_periodicity

Plot periodic patterns.

**Output:** `periodicity.png`

### plot_autocorrelation

Plot ACF/PACF.

**Output:** `autocorrelation.png`

---

## Examples

### Trend Analysis

```python
result = service.run(AnalysisRequest(
    module_name="temporal",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/trends"),
    analysis_kwargs={
        "analyze_trends": {
            "metrics": ["population", "resources", "efficiency"],
            "method": "linear"
        }
    }
))
```

### Cycle Detection

```python
result = service.run(AnalysisRequest(
    module_name="temporal",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/cycles"),
    analysis_kwargs={
        "analyze_periodicity": {
            "method": "fft",
            "min_period": 10,
            "max_period": 500
        }
    }
))
```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Population Module](./Population.md)

---

**Module Version**: 2.0.0
