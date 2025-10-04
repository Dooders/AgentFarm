# Analysis Module Refactoring Plan
**Date:** 2025-10-03  
**Status:** DRAFT - Ready for Implementation  
**Goal:** Consolidate all analysis code into the protocol-based `farm.analysis` module

---

## Executive Summary

This plan outlines the systematic migration of analysis code from multiple locations into the unified `farm.analysis` module, leveraging the existing protocol-based architecture. The refactoring will eliminate duplication, improve maintainability, and provide a consistent API for all analysis operations.

**Timeline:** 4-6 weeks  
**Risk Level:** LOW (existing architecture proven)  
**Breaking Changes:** Minimal (backward compatibility maintained)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Migration Strategy](#migration-strategy)
3. [Implementation Phases](#implementation-phases)
4. [Module Templates](#module-templates)
5. [Testing Strategy](#testing-strategy)
6. [Backward Compatibility](#backward-compatibility)
7. [Rollout Plan](#rollout-plan)

---

## Architecture Overview

### Current State
Your existing `farm.analysis` module provides:
- ✅ Protocol-based architecture (type-safe interfaces)
- ✅ `BaseAnalysisModule` for implementation
- ✅ `AnalysisService` for orchestration
- ✅ Registry system for module discovery
- ✅ Validation framework
- ✅ Progress tracking and caching
- ✅ Proven submodules: dominance, genesis, advantage, social_behavior

### Target Architecture

```
farm/analysis/
├── core.py                    # Base classes (existing)
├── protocols.py               # Interfaces (existing)
├── service.py                 # Service layer (existing)
├── registry.py                # Module registry (existing)
├── validation.py              # Validators (existing)
├── exceptions.py              # Exceptions (existing)
│
├── common/                    # Shared utilities
│   ├── context.py            # Analysis context (existing)
│   ├── metrics.py            # Common metrics (existing)
│   └── utils.py              # NEW: Shared analysis utilities
│
├── population/               # NEW: Population analysis
│   ├── __init__.py
│   ├── module.py            # Module implementation
│   ├── compute.py           # Statistical computations
│   ├── analyze.py           # Analysis functions
│   ├── plot.py              # Visualizations
│   └── data.py              # Data processing
│
├── resources/                # NEW: Resource analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   ├── plot.py
│   └── data.py
│
├── actions/                  # NEW: Action analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   ├── plot.py
│   └── data.py
│
├── agents/                   # NEW: Agent analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   ├── plot.py
│   ├── lifespan.py          # Lifespan analysis
│   └── behavior.py          # Behavior clustering
│
├── learning/                 # NEW: Learning analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   └── plot.py
│
├── spatial/                  # NEW: Spatial analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   ├── plot.py
│   ├── movement.py          # Movement patterns
│   └── location.py          # Location analysis
│
├── temporal/                 # NEW: Temporal analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   └── plot.py
│
├── reproduction/             # MIGRATE: From existing code
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   └── plot.py
│
├── combat/                   # NEW: Combat analysis
│   ├── __init__.py
│   ├── module.py
│   ├── compute.py
│   ├── analyze.py
│   └── plot.py
│
├── dominance/               # EXISTING: Keep as-is
├── genesis/                 # EXISTING: Keep as-is
├── advantage/               # EXISTING: Keep as-is
└── social_behavior/         # EXISTING: Keep as-is
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
**Goal:** Prepare infrastructure and utilities

**Tasks:**
1. Create shared utilities module
2. Extract common patterns from existing modules
3. Create migration templates
4. Set up testing infrastructure
5. Document migration process

### Phase 2: Core Analyzers (Weeks 2-3)
**Goal:** Migrate high-priority database analyzers

**Priority Order:**
1. **Population** (most used, clear boundaries)
2. **Resources** (high usage, well-defined)
3. **Actions** (moderate complexity)
4. **Agents** (includes lifespan, behavior)

### Phase 3: Specialized Analyzers (Week 4)
**Goal:** Migrate specialized analysis modules

**Modules:**
1. **Learning** (learning curves, experience)
2. **Spatial** (movement, location, clustering)
3. **Temporal** (time series, patterns)
4. **Combat** (combat metrics)

### Phase 4: Script Consolidation (Week 5)
**Goal:** Consolidate and deprecate scripts

**Tasks:**
1. Migrate useful script functionality
2. Update orchestration scripts to use new modules
3. Add deprecation warnings
4. Update documentation

### Phase 5: Testing & Documentation (Week 6)
**Goal:** Ensure quality and usability

**Tasks:**
1. Integration testing
2. Performance testing
3. Documentation updates
4. Migration guides
5. Example updates

---

## Implementation Phases

### Phase 1: Foundation

#### Task 1.1: Create Common Utilities

**File:** `farm/analysis/common/utils.py`

```python
"""
Common utility functions for analysis modules.

Extracted from database analyzers and scripts for reuse.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for a data array.
    
    Args:
        data: Numpy array of numeric values
        
    Returns:
        Dictionary containing mean, median, std, min, max, percentiles
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
    }


def calculate_trend(data: np.ndarray) -> float:
    """Calculate linear trend slope.
    
    Args:
        data: Time series data
        
    Returns:
        Slope of linear regression line
    """
    if len(data) < 2:
        return 0.0
    x = np.arange(len(data))
    return float(np.polyfit(x, data, 1)[0])


def calculate_rolling_mean(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Calculate rolling mean with specified window.
    
    Args:
        data: Input data array
        window: Rolling window size
        
    Returns:
        Array of rolling means
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')


def normalize_dict(d: Dict[str, int]) -> Dict[str, float]:
    """Normalize dictionary values to proportions summing to 1.0.
    
    Args:
        d: Dictionary with numeric values
        
    Returns:
        Dictionary with normalized values
    """
    total = sum(d.values())
    return {k: v / total if total > 0 else 0 for k, v in d.items()}


def create_output_subdirs(output_path: Path, subdirs: List[str]) -> Dict[str, Path]:
    """Create output subdirectories for organized results.
    
    Args:
        output_path: Base output path
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary mapping subdir names to paths
    """
    paths = {}
    for subdir in subdirs:
        path = output_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    return paths


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required: List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def align_time_series(
    data_list: List[np.ndarray], 
    max_length: Optional[int] = None
) -> np.ndarray:
    """Align multiple time series to same length by padding.
    
    Args:
        data_list: List of arrays to align
        max_length: Target length (uses max if None)
        
    Returns:
        2D array with aligned series
    """
    if not data_list:
        return np.array([])
    
    if max_length is None:
        max_length = max(len(arr) for arr in data_list)
    
    aligned = []
    for arr in data_list:
        if len(arr) < max_length:
            padded = np.pad(arr, (0, max_length - len(arr)), mode='edge')
        else:
            padded = arr[:max_length]
        aligned.append(padded)
    
    return np.array(aligned)
```

#### Task 1.2: Create Module Template

**File:** `farm/analysis/template/standard_module.py`

```python
"""
Template for creating new analysis modules.

Copy this template to create a new analysis module.
Follow the existing pattern from dominance, genesis, etc.
"""

from typing import Optional, Callable
from pathlib import Path
import pandas as pd

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
from farm.analysis.common.context import AnalysisContext


# ============================================================================
# Data Processing
# ============================================================================

def process_MODULE_data(experiment_path: Path, **kwargs) -> pd.DataFrame:
    """Process raw experiment data for MODULE analysis.
    
    Args:
        experiment_path: Path to experiment directory
        **kwargs: Additional processing options
        
    Returns:
        Processed DataFrame ready for analysis
    """
    # TODO: Implement data processing
    # 1. Load data from experiment_path
    # 2. Transform and clean data
    # 3. Compute derived metrics
    # 4. Return DataFrame
    
    raise NotImplementedError("Implement data processing")


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_MODULE_metrics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze MODULE metrics and save results.
    
    Args:
        df: Processed data
        ctx: Analysis context with output_path, logger, etc.
        **kwargs: Additional analysis options
    """
    ctx.logger.info("Analyzing MODULE metrics...")
    
    # TODO: Implement analysis
    # 1. Calculate metrics
    # 2. Save results to ctx.get_output_file()
    # 3. Report progress via ctx.report_progress()
    
    raise NotImplementedError("Implement analysis")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_MODULE_overview(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create overview visualization of MODULE data.
    
    Args:
        df: Processed data
        ctx: Analysis context
        **kwargs: Plot options
    """
    import matplotlib.pyplot as plt
    
    ctx.logger.info("Creating MODULE overview plot...")
    
    # TODO: Implement visualization
    # 1. Create figure
    # 2. Plot data
    # 3. Save to ctx.get_output_file()
    
    raise NotImplementedError("Implement visualization")


# ============================================================================
# Module Definition
# ============================================================================

class MODULEModule(BaseAnalysisModule):
    """Module for analyzing MODULE in simulations."""
    
    def __init__(self):
        super().__init__(
            name="MODULE",
            description="Analysis of MODULE patterns in simulations"
        )
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['iteration'],  # TODO: Adjust columns
                column_types={'iteration': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)
    
    def register_functions(self) -> None:
        """Register all analysis functions."""
        self._functions = {
            "analyze_metrics": make_analysis_function(analyze_MODULE_metrics),
            "plot_overview": make_analysis_function(plot_MODULE_overview),
            # TODO: Add more functions
        }
        
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [self._functions["analyze_metrics"]],
            "plots": [self._functions["plot_overview"]],
            # TODO: Add more groups
        }
    
    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for MODULE."""
        return SimpleDataProcessor(process_MODULE_data)
    
    # Optional: Database support
    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return False  # TODO: Set to True if using database
    
    def get_db_filename(self) -> Optional[str]:
        """Get database filename if using database."""
        return None  # TODO: Return filename if supports_database=True
    
    def get_db_loader(self) -> Optional[Callable]:
        """Get database loader if using database."""
        return None  # TODO: Return loader function if supports_database=True


# Create singleton instance
MODULE_module = MODULEModule()
```

---

### Phase 2: Core Analyzer Migration

#### Migration 2.1: Population Module

**Source:** `farm/database/analyzers/population_analyzer.py`  
**Target:** `farm/analysis/population/`

##### Step 1: Create Module Structure

```bash
mkdir -p farm/analysis/population
touch farm/analysis/population/__init__.py
touch farm/analysis/population/module.py
touch farm/analysis/population/compute.py
touch farm/analysis/population/analyze.py
touch farm/analysis/population/plot.py
touch farm/analysis/population/data.py
```

##### Step 2: Create Data Processor

**File:** `farm/analysis/population/data.py`

```python
"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.analyzers.population_analyzer import PopulationAnalyzer


def process_population_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process population data from experiment.
    
    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options
        
    Returns:
        DataFrame with population metrics over time
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        # Try alternative locations
        db_path = experiment_path / "data" / "simulation.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")
    
    # Load data using existing infrastructure
    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = PopulationRepository(db.session_manager)
    analyzer = PopulationAnalyzer(repository)
    
    # Get comprehensive statistics
    stats = analyzer.analyze_comprehensive_statistics()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame({
        'step': range(len(stats.metrics)),
        'total_agents': [m.total_agents for m in stats.metrics],
        'system_agents': [m.system_agents for m in stats.metrics],
        'independent_agents': [m.independent_agents for m in stats.metrics],
        'control_agents': [m.control_agents for m in stats.metrics],
        'avg_resources': [m.avg_resources for m in stats.metrics],
        # Add more metrics as needed
    })
    
    return df
```

##### Step 3: Create Compute Functions

**File:** `farm/analysis/population/compute.py`

```python
"""
Population statistical computations.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_population_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive population statistics.
    
    Args:
        df: Population data with columns: step, total_agents, etc.
        
    Returns:
        Dictionary of computed statistics
    """
    total = df['total_agents'].values
    
    stats = {
        'total': calculate_statistics(total),
        'peak_step': int(np.argmax(total)),
        'peak_value': int(np.max(total)),
        'final_value': int(total[-1]),
        'trend': calculate_trend(total),
        'survival_rate': float(np.mean(total > 0)),
    }
    
    # Per-type statistics
    for agent_type in ['system_agents', 'independent_agents', 'control_agents']:
        if agent_type in df.columns:
            stats[agent_type] = calculate_statistics(df[agent_type].values)
    
    return stats


def compute_birth_death_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Compute birth and death rates.
    
    Args:
        df: Population data with births, deaths columns
        
    Returns:
        Dictionary of rate metrics
    """
    if 'births' not in df.columns or 'deaths' not in df.columns:
        return {}
    
    total_births = df['births'].sum()
    total_deaths = df['deaths'].sum()
    n_steps = len(df)
    
    return {
        'total_births': int(total_births),
        'total_deaths': int(total_deaths),
        'birth_rate': float(total_births / n_steps),
        'death_rate': float(total_deaths / n_steps),
        'net_growth': int(total_births - total_deaths),
        'growth_rate': float((total_births - total_deaths) / n_steps),
    }


def compute_population_stability(df: pd.DataFrame, window: int = 50) -> Dict[str, float]:
    """Compute population stability metrics.
    
    Args:
        df: Population data
        window: Window size for stability calculation
        
    Returns:
        Stability metrics
    """
    total = df['total_agents'].values
    
    if len(total) < window:
        window = len(total) // 2
    
    # Calculate coefficient of variation in windows
    cv_list = []
    for i in range(len(total) - window):
        window_data = total[i:i+window]
        if np.mean(window_data) > 0:
            cv = np.std(window_data) / np.mean(window_data)
            cv_list.append(cv)
    
    return {
        'mean_cv': float(np.mean(cv_list)) if cv_list else 0.0,
        'stability_score': float(1.0 / (1.0 + np.mean(cv_list))) if cv_list else 1.0,
    }
```

##### Step 4: Create Analysis Functions

**File:** `farm/analysis/population/analyze.py`

```python
"""
Population analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.population.compute import (
    compute_population_statistics,
    compute_birth_death_rates,
    compute_population_stability,
)


def analyze_population_dynamics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze population dynamics and save results.
    
    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing population dynamics...")
    
    # Compute statistics
    stats = compute_population_statistics(df)
    rates = compute_birth_death_rates(df)
    stability = compute_population_stability(df)
    
    # Combine results
    results = {
        'statistics': stats,
        'rates': rates,
        'stability': stability,
    }
    
    # Save to file
    output_file = ctx.get_output_file("population_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    ctx.logger.info(f"Saved statistics to {output_file}")
    ctx.report_progress("Population analysis complete", 0.5)


def analyze_agent_composition(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent type composition over time.
    
    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent composition...")
    
    # Calculate proportions
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    composition_df = df.copy()
    
    for agent_type in agent_types:
        if agent_type in df.columns:
            composition_df[f'{agent_type}_pct'] = (
                df[agent_type] / df['total_agents']
            ) * 100
    
    # Save composition data
    output_file = ctx.get_output_file("agent_composition.csv")
    composition_df.to_csv(output_file, index=False)
    
    ctx.logger.info(f"Saved composition to {output_file}")
    ctx.report_progress("Composition analysis complete", 0.7)
```

##### Step 5: Create Visualization Functions

**File:** `farm/analysis/population/plot.py`

```python
"""
Population visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_population_over_time(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot population dynamics over time.
    
    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating population over time plot...")
    
    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot total population
    ax.plot(df['step'], df['total_agents'], 
            label='Total Population', linewidth=2, color='black')
    
    # Plot by type
    colors = {'system_agents': 'blue', 'independent_agents': 'green', 'control_agents': 'red'}
    for agent_type, color in colors.items():
        if agent_type in df.columns:
            ax.plot(df['step'], df[agent_type], 
                   label=agent_type.replace('_', ' ').title(),
                   linewidth=1.5, color=color, alpha=0.7)
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Population Count')
    ax.set_title('Population Dynamics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_file = ctx.get_output_file("population_over_time.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    ctx.logger.info(f"Saved plot to {output_file}")


def plot_birth_death_rates(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot birth and death rates over time.
    
    Args:
        df: Population data with births, deaths
        ctx: Analysis context
        **kwargs: Plot options
    """
    if 'births' not in df.columns or 'deaths' not in df.columns:
        ctx.logger.warning("Births/deaths data not available, skipping plot")
        return
    
    ctx.logger.info("Creating birth/death rates plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['step'], df['births'], label='Births', color='green', linewidth=1.5)
    ax.plot(df['step'], df['deaths'], label='Deaths', color='red', linewidth=1.5)
    ax.fill_between(df['step'], df['births'], alpha=0.3, color='green')
    ax.fill_between(df['step'], df['deaths'], alpha=0.3, color='red')
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Count')
    ax.set_title('Birth and Death Rates Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_file = ctx.get_output_file("birth_death_rates.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)
    
    ctx.logger.info(f"Saved plot to {output_file}")


def plot_agent_composition(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent type composition as stacked area chart.
    
    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating agent composition plot...")
    
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    available_types = [t for t in agent_types if t in df.columns]
    
    if not available_types:
        ctx.logger.warning("No agent type data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked area
    ax.stackplot(
        df['step'],
        *[df[t] for t in available_types],
        labels=[t.replace('_', ' ').title() for t in available_types],
        alpha=0.7
    )
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Population Count')
    ax.set_title('Agent Type Composition Over Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    output_file = ctx.get_output_file("agent_composition.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)
    
    ctx.logger.info(f"Saved plot to {output_file}")
```

##### Step 6: Create Module Definition

**File:** `farm/analysis/population/module.py`

```python
"""
Population analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.population.data import process_population_data
from farm.analysis.population.analyze import (
    analyze_population_dynamics,
    analyze_agent_composition,
)
from farm.analysis.population.plot import (
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
)


class PopulationModule(BaseAnalysisModule):
    """Module for analyzing population dynamics in simulations."""
    
    def __init__(self):
        super().__init__(
            name="population",
            description="Analysis of population dynamics, births, deaths, and agent composition"
        )
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step', 'total_agents'],
                column_types={'step': int, 'total_agents': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)
    
    def register_functions(self) -> None:
        """Register all population analysis functions."""
        
        # Analysis functions
        self._functions = {
            "analyze_dynamics": make_analysis_function(analyze_population_dynamics),
            "analyze_composition": make_analysis_function(analyze_agent_composition),
            "plot_population": make_analysis_function(plot_population_over_time),
            "plot_births_deaths": make_analysis_function(plot_birth_death_rates),
            "plot_composition": make_analysis_function(plot_agent_composition),
        }
        
        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_dynamics"],
                self._functions["analyze_composition"],
            ],
            "plots": [
                self._functions["plot_population"],
                self._functions["plot_births_deaths"],
                self._functions["plot_composition"],
            ],
            "basic": [
                self._functions["analyze_dynamics"],
                self._functions["plot_population"],
            ],
        }
    
    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for population analysis."""
        return SimpleDataProcessor(process_population_data)


# Create singleton instance
population_module = PopulationModule()
```

##### Step 7: Create Package Init

**File:** `farm/analysis/population/__init__.py`

```python
"""
Population analysis module.

Provides comprehensive analysis of population dynamics including:
- Total population trends
- Birth and death rates
- Agent type composition
- Population stability
- Survival rates
"""

from farm.analysis.population.module import population_module, PopulationModule
from farm.analysis.population.compute import (
    compute_population_statistics,
    compute_birth_death_rates,
    compute_population_stability,
)
from farm.analysis.population.analyze import (
    analyze_population_dynamics,
    analyze_agent_composition,
)
from farm.analysis.population.plot import (
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
)

__all__ = [
    "population_module",
    "PopulationModule",
    "compute_population_statistics",
    "compute_birth_death_rates",
    "compute_population_stability",
    "analyze_population_dynamics",
    "analyze_agent_composition",
    "plot_population_over_time",
    "plot_birth_death_rates",
    "plot_agent_composition",
]
```

##### Step 8: Register Module

**Update:** `farm/analysis/__init__.py`

```python
# Add to imports
from farm.analysis.population import population_module

# Module will auto-register via registry system
# Or manually register:
from farm.analysis.registry import registry
registry.register(population_module)
```

---

#### Migration 2.2: Resources Module

Follow the same pattern as Population:

1. Create `farm/analysis/resources/` directory structure
2. Implement `data.py` - process resource data from `ResourceRepository`
3. Implement `compute.py` - resource statistics, efficiency, distribution
4. Implement `analyze.py` - consumption patterns, hotspots, efficiency
5. Implement `plot.py` - resource over time, distribution, efficiency charts
6. Implement `module.py` - `ResourcesModule` class
7. Update `__init__.py` for clean imports
8. Register module in `farm/analysis/__init__.py`

**Key Functions to Migrate:**
- `analyze_resource_distribution()` → `analyze_resource_patterns()`
- `analyze_consumption_patterns()` → `analyze_consumption()`
- `analyze_efficiency_metrics()` → `analyze_efficiency()`
- `find_resource_hotspots()` → `analyze_hotspots()`

---

#### Migration 2.3: Actions Module

Similar pattern for actions:

1. Create `farm/analysis/actions/` structure
2. Process action data from database
3. Compute action frequencies, success rates, sequences
4. Analyze action patterns, distributions, rewards
5. Visualize action types, frequencies, success rates
6. Create `ActionsModule`
7. Register module

**Key Sources:**
- `farm/database/analyzers/action_stats_analyzer.py`
- `farm/database/analyzers/sequence_pattern_analyzer.py`
- `farm/database/analyzers/decision_pattern_analyzer.py`
- `farm/analysis/action_type_distribution.py`

---

#### Migration 2.4: Agents Module

Agent module with submodules:

1. Create `farm/analysis/agents/` structure
2. Add `lifespan.py` for lifespan analysis
3. Add `behavior.py` for behavior clustering
4. Implement core agent metrics
5. Create `AgentsModule`
6. Register module

**Key Sources:**
- `farm/database/analyzers/agent_analyzer.py`
- `farm/database/analyzers/lifespan_analysis.py`
- `farm/database/analyzers/behavior_clustering_analyzer.py`

---

### Phase 3: Specialized Analyzers

Apply the same migration pattern to:

1. **Learning Module** - from `learning_analyzer.py` and `learning_experience.py`
2. **Spatial Module** - from `spatial_analysis.py`, `movement_analysis.py`, `location_analysis.py`
3. **Temporal Module** - from `temporal_pattern_analyzer.py`
4. **Combat Module** - extract combat metrics from simulation analysis

---

### Phase 4: Script Consolidation

#### Task 4.1: Update Orchestration Scripts

Scripts like `advantage_analysis.py` become thin wrappers:

**Before:**
```python
# scripts/advantage_analysis.py - 500+ lines
# Lots of custom logic mixed with orchestration
```

**After:**
```python
#!/usr/bin/env python
"""Advantage analysis orchestration script."""

from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

def main():
    # Initialize service
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    # Find latest experiment
    experiment_path = find_latest_experiment()
    
    # Run advantage analysis
    request = AnalysisRequest(
        module_name="advantage",
        experiment_path=experiment_path,
        output_path=Path("results/advantage"),
        group="all"
    )
    
    result = service.run(request)
    
    if result.success:
        print(f"Analysis complete: {result.output_path}")
    else:
        print(f"Analysis failed: {result.error}")

if __name__ == "__main__":
    main()
```

#### Task 4.2: Migrate Utility Functions

Move reusable utilities from scripts to `farm.analysis.common.utils`:

- `scripts/data_extraction.py` → `farm.analysis.common.data_extraction`
- `scripts/database_utils.py` → `farm.database.utils` (already in database package)
- `scripts/visualization_utils.py` → `farm.analysis.common.visualization`

#### Task 4.3: Deprecate Redundant Scripts

Add deprecation warnings to scripts that are fully replaced:

```python
# scripts/OLD_SCRIPT.py
import warnings

warnings.warn(
    "This script is deprecated. Use 'from farm.analysis import MODULE' instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

### Phase 5: Testing & Documentation

#### Task 5.1: Create Module Tests

For each new module, create comprehensive tests:

**File:** `tests/analysis/test_population.py`

```python
"""Tests for population analysis module."""

import pytest
import pandas as pd
from pathlib import Path

from farm.analysis.population import (
    population_module,
    compute_population_statistics,
    analyze_population_dynamics,
)
from farm.analysis.common.context import AnalysisContext


def test_population_module_registration():
    """Test module is properly registered."""
    assert population_module.name == "population"
    assert len(population_module.get_function_names()) > 0


def test_compute_population_statistics():
    """Test population statistics computation."""
    df = pd.DataFrame({
        'step': range(100),
        'total_agents': [100 + i for i in range(100)],
        'system_agents': [50 + i//2 for i in range(100)],
    })
    
    stats = compute_population_statistics(df)
    
    assert 'total' in stats
    assert 'peak_step' in stats
    assert stats['peak_value'] == 199


def test_analyze_population_dynamics(tmp_path):
    """Test population dynamics analysis."""
    df = pd.DataFrame({
        'step': range(100),
        'total_agents': [100 + i for i in range(100)],
    })
    
    ctx = AnalysisContext(output_path=tmp_path)
    analyze_population_dynamics(df, ctx)
    
    # Check output file created
    output_file = tmp_path / "population_statistics.json"
    assert output_file.exists()


def test_population_module_integration(tmp_path, sample_experiment_path):
    """Test full module execution."""
    from farm.analysis.service import AnalysisService, AnalysisRequest
    from farm.core.services import EnvConfigService
    
    service = AnalysisService(EnvConfigService())
    
    request = AnalysisRequest(
        module_name="population",
        experiment_path=sample_experiment_path,
        output_path=tmp_path,
        group="basic"
    )
    
    result = service.run(request)
    
    assert result.success
    assert result.output_path.exists()
```

#### Task 5.2: Update Documentation

Update all documentation files:

1. **Module README** - Update `farm/analysis/README.md` with new modules
2. **Quick Reference** - Update `QUICK_REFERENCE.md`
3. **API Docs** - Generate API documentation
4. **Migration Guide** - Create migration guide for users
5. **Examples** - Update examples to use new modules

#### Task 5.3: Create Migration Guide

**File:** `farm/analysis/MIGRATION_GUIDE.md`

```markdown
# Migration Guide: Database Analyzers → Analysis Modules

## Overview

Database analyzers have been migrated to the protocol-based `farm.analysis` module system.

## Quick Migration

### Before (Old Way)
```python
from farm.database.database import SimulationDatabase
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

db = SimulationDatabase("sqlite:///simulation.db")
repository = PopulationRepository(db.session_manager)
analyzer = PopulationAnalyzer(repository)
stats = analyzer.analyze_comprehensive_statistics()
```

### After (New Way)
```python
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService
from pathlib import Path

service = AnalysisService(EnvConfigService())
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/population")
)
result = service.run(request)
```

## Module Mapping

| Old Analyzer | New Module | Notes |
|-------------|------------|-------|
| `PopulationAnalyzer` | `population` | Full feature parity |
| `ResourceAnalyzer` | `resources` | Enhanced with efficiency metrics |
| `ActionStatsAnalyzer` | `actions` | Merged with sequence patterns |
| `AgentAnalyzer` | `agents` | Includes lifespan & behavior |
| `LearningAnalyzer` | `learning` | Same functionality |
| `SpatialAnalysis` | `spatial` | Enhanced visualization |

## Backward Compatibility

Old analyzers remain available but are deprecated:

```python
# Still works but shows deprecation warning
from farm.database.analyzers.population_analyzer import PopulationAnalyzer
```

## Benefits of Migration

1. ✅ Consistent API across all analysis types
2. ✅ Automatic result caching
3. ✅ Progress tracking
4. ✅ Batch processing support
5. ✅ Better error handling
6. ✅ Type safety with protocols
```

---

## Testing Strategy

### Unit Tests
- Test each compute function independently
- Test analysis functions with mock data
- Test plot functions (check file creation, not visual output)
- Test module registration and discovery

### Integration Tests
- Test full analysis workflow end-to-end
- Test with real experiment data
- Test batch processing
- Test error conditions

### Performance Tests
- Benchmark analysis speed
- Test with large datasets
- Test memory usage
- Compare with old implementation

### Regression Tests
- Ensure results match old implementation
- Test backward compatibility
- Verify deprecation warnings work

---

## Backward Compatibility

### Strategy 1: Wrapper Classes (Recommended)

Keep old analyzer classes as thin wrappers:

**File:** `farm/database/analyzers/population_analyzer.py`

```python
"""
Population analyzer - DEPRECATED

Use farm.analysis.population module instead.
"""

import warnings
from typing import List, Optional

from farm.database.repositories.population_repository import PopulationRepository
from farm.database.data_types import PopulationStatistics


class PopulationAnalyzer:
    """DEPRECATED: Use farm.analysis.population instead."""
    
    def __init__(self, repository: PopulationRepository):
        warnings.warn(
            "PopulationAnalyzer is deprecated. "
            "Use farm.analysis.population module with AnalysisService instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.repository = repository
        # ... keep existing implementation for now
    
    def analyze_comprehensive_statistics(self, *args, **kwargs):
        """DEPRECATED: Use population module."""
        warnings.warn(
            "Use AnalysisService with 'population' module instead",
            DeprecationWarning,
            stacklevel=2
        )
        # Existing implementation continues to work
        # ...
```

### Strategy 2: Deprecation Timeline

- **Phase 1 (Weeks 1-6):** Add deprecation warnings
- **Phase 2 (Month 2):** Update all internal usage
- **Phase 3 (Month 3):** Mark as deprecated in docs
- **Phase 4 (Month 6):** Remove old implementations

---

## Rollout Plan

### Week 1: Foundation
- ✅ Create common utilities
- ✅ Update templates
- ✅ Set up testing infrastructure
- ✅ Document patterns

### Week 2: Population Module
- ✅ Implement population module
- ✅ Write tests
- ✅ Integration test
- ✅ Update documentation

### Week 3: Resources & Actions
- ✅ Implement resources module
- ✅ Implement actions module
- ✅ Write tests
- ✅ Integration tests

### Week 4: Agents & Specialized
- ✅ Implement agents module
- ✅ Implement learning module
- ✅ Implement spatial module
- ✅ Write tests

### Week 5: Scripts & Consolidation
- ✅ Update orchestration scripts
- ✅ Migrate utilities
- ✅ Add deprecation warnings
- ✅ Update examples

### Week 6: Testing & Docs
- ✅ Full integration testing
- ✅ Performance testing
- ✅ Documentation updates
- ✅ Migration guides
- ✅ Release preparation

---

## Success Criteria

### Functionality
- ✅ All old functionality available in new modules
- ✅ Results match old implementations (regression tests pass)
- ✅ No loss of features

### Quality
- ✅ Test coverage > 80%
- ✅ All tests passing
- ✅ No performance regression
- ✅ Type hints complete

### Documentation
- ✅ All modules documented
- ✅ Migration guide complete
- ✅ Examples updated
- ✅ API docs generated

### Usability
- ✅ Consistent API across modules
- ✅ Clear error messages
- ✅ Progress tracking works
- ✅ Easy to use

---

## Risk Mitigation

### Risk: Breaking Existing Code
**Mitigation:** Keep old analyzers with deprecation warnings

### Risk: Performance Regression
**Mitigation:** Benchmark before/after, optimize if needed

### Risk: Feature Gaps
**Mitigation:** Comprehensive feature checklist, user feedback

### Risk: Incomplete Migration
**Mitigation:** Phased approach, testing at each phase

---

## Next Steps

1. ✅ Review and approve this plan
2. ✅ Create detailed task breakdown
3. ✅ Set up project tracking
4. ✅ Begin Phase 1 implementation
5. ✅ Regular check-ins and adjustments

---

## Appendix: Code Checklist

### Per Module Checklist

- [ ] Create directory structure
- [ ] Implement `data.py` (data processing)
- [ ] Implement `compute.py` (computations)
- [ ] Implement `analyze.py` (analysis functions)
- [ ] Implement `plot.py` (visualizations)
- [ ] Implement `module.py` (module class)
- [ ] Update `__init__.py` (exports)
- [ ] Register module in `farm/analysis/__init__.py`
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Add examples

### Modules to Create

- [ ] Population
- [ ] Resources
- [ ] Actions
- [ ] Agents
- [ ] Learning
- [ ] Spatial
- [ ] Temporal
- [ ] Combat
- [ ] Reproduction (migrate existing)

---

**Plan Version:** 1.0  
**Created:** 2025-10-03  
**Status:** Ready for Implementation
