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
    compute_growth_rate_analysis,
    compute_demographic_metrics,
)
from farm.analysis.population.analyze import (
    analyze_population_dynamics,
    analyze_agent_composition,
    analyze_comprehensive_population,
)
from farm.analysis.population.plot import (
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
    plot_population_dashboard,
)

__all__ = [
    "population_module",
    "PopulationModule",
    "compute_population_statistics",
    "compute_birth_death_rates",
    "compute_population_stability",
    "compute_growth_rate_analysis",
    "compute_demographic_metrics",
    "analyze_population_dynamics",
    "analyze_agent_composition",
    "analyze_comprehensive_population",
    "plot_population_over_time",
    "plot_birth_death_rates",
    "plot_agent_composition",
    "plot_population_dashboard",
]
