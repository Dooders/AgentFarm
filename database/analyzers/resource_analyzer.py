from typing import List, Optional, Tuple, Union

from database.data_types import (
    ConsumptionStats,
    ResourceAnalysis,
    ResourceDistributionStep,
    ResourceEfficiencyMetrics,
    ResourceHotspot,
)
from database.enums import AnalysisScope
from database.repositories.resource_repository import ResourceRepository


class ResourceAnalyzer:
    """
    Analyzes resource dynamics and distribution patterns in the simulation.

    This class provides comprehensive analysis of resource-related data including
    distribution patterns, consumption statistics, concentration hotspots, and
    efficiency metrics across simulation timesteps.

    Attributes:
        repository (ResourceRepository): Repository instance for accessing resource data.

    Example:
        >>> analyzer = ResourceAnalyzer(resource_repository)
        >>> stats = analyzer.analyze_comprehensive_statistics()
        >>> print(f"Total consumed: {stats.consumption.total_consumed:.2f}")
    """

    def __init__(self, repository: ResourceRepository):
        """Initialize the ResourceAnalyzer with a ResourceRepository."""
        self.repository = repository

    def analyze_resource_distribution(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceDistributionStep]:
        """
        Analyze spatial resource distribution patterns over time.

        Retrieves and processes resource distribution metrics across simulation
        timesteps, including total quantities, densities, and distribution entropy.

        Args:
            scope: Analysis scope (e.g., 'simulation', 'episode')
            agent_id: Filter results for specific agent
            step: Filter results for specific step
            step_range: Filter results for step range

        Returns:
            List[ResourceDistributionStep]: Time series of distribution metrics containing:
                - step: Simulation timestep number
                - total_resources: Total quantity of resources present
                - average_per_cell: Mean resource density per grid cell
                - distribution_entropy: Shannon entropy of resource distribution

        Example:
            >>> distribution = analyzer.analyze_resource_distribution()
            >>> for step in distribution[:3]:
            ...     print(f"Step {step.step}: {step.total_resources:.2f} resources")
        """
        return self.repository.get_resource_distribution(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_consumption_patterns(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> ConsumptionStats:
        """
        Analyze resource consumption patterns and statistics.

        Calculates aggregate consumption metrics including totals, averages,
        peaks, and variability measures across the simulation timeline.

        Args:
            scope: Analysis scope for filtering
            agent_id: Filter results for specific agent
            step: Filter results for specific step
            step_range: Filter results for step range

        Returns:
            ConsumptionStats: Consumption statistics containing:
                - total_consumed: Total resources consumed across all steps
                - avg_consumption_rate: Mean consumption rate per timestep
                - peak_consumption: Maximum single-step consumption
                - consumption_variance: Variance in consumption rates

        Example:
            >>> stats = analyzer.analyze_consumption_patterns()
            >>> print(f"Average consumption: {stats.avg_consumption_rate:.2f}/step")
            >>> print(f"Peak consumption: {stats.peak_consumption:.2f}")
        """
        return self.repository.get_consumption_patterns(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_resource_hotspots(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceHotspot]:
        """
        Identify and analyze areas of high resource concentration.

        Examines spatial resource distribution to locate and rank areas with
        above-average resource concentrations.

        Args:
            scope: Analysis scope for filtering
            agent_id: Filter results for specific agent
            step: Filter results for specific step
            step_range: Filter results for step range

        Returns:
            List[ResourceHotspot]: Resource hotspots sorted by concentration (highest first):
                - position_x: X coordinate of hotspot
                - position_y: Y coordinate of hotspot
                - concentration: Average resource amount at location

        Example:
            >>> hotspots = analyzer.analyze_resource_hotspots()
            >>> for spot in hotspots[:3]:
            ...     print(f"Hotspot at ({spot.position_x}, {spot.position_y}): "
            ...           f"{spot.concentration:.2f}")
        """
        return self.repository.get_resource_hotspots(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_efficiency_metrics(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> ResourceEfficiencyMetrics:
        """
        Calculate resource utilization and efficiency metrics.

        Computes various efficiency measures related to resource distribution,
        consumption, and regeneration patterns.

        Args:
            scope: Analysis scope for filtering
            agent_id: Filter results for specific agent
            step: Filter results for specific step
            step_range: Filter results for step range

        Returns:
            ResourceEfficiencyMetrics: Efficiency metrics containing:
                - utilization_rate: Resource usage efficiency (0-1)
                - distribution_efficiency: Spatial distribution effectiveness (0-1)
                - consumption_efficiency: Resource consumption optimization (0-1)
                - regeneration_rate: Resource replenishment speed (units/step)

        Example:
            >>> efficiency = analyzer.analyze_efficiency_metrics()
            >>> print(f"Utilization rate: {efficiency.utilization_rate:.1%}")
            >>> print(f"Distribution efficiency: {efficiency.distribution_efficiency:.1%}")
        """
        return self.repository.get_efficiency_metrics(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_comprehensive_statistics(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> ResourceAnalysis:
        """
        Generate comprehensive resource analysis statistics.

        Combines distribution, consumption, hotspot and efficiency analyses
        into a complete resource behavior assessment.

        Args:
            scope: Analysis scope for filtering
            agent_id: Filter results for specific agent
            step: Filter results for specific step
            step_range: Filter results for step range

        Returns:
            ResourceAnalysis: Complete analysis containing:
                - distribution: Time series of distribution metrics
                - consumption: Aggregate consumption statistics
                - hotspots: Areas of high concentration
                - efficiency: Resource efficiency measures

        Example:
            >>> analysis = analyzer.analyze_comprehensive_statistics()
            >>> print(f"Total resources: {analysis.distribution[0].total_resources}")
            >>> print(f"Consumption rate: {analysis.consumption.avg_consumption_rate}")
        """
        return ResourceAnalysis(
            distribution=self.analyze_resource_distribution(
                scope, agent_id, step, step_range
            ),
            consumption=self.analyze_consumption_patterns(
                scope, agent_id, step, step_range
            ),
            hotspots=self.analyze_resource_hotspots(scope, agent_id, step, step_range),
            efficiency=self.analyze_efficiency_metrics(
                scope, agent_id, step, step_range
            ),
        )
