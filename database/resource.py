from typing import List

from sqlalchemy import func

from database.data_types import (
    ConsumptionStats,
    ResourceAnalysis,
    ResourceDistributionStep,
    ResourceEfficiencyMetrics,
    ResourceHotspot,
)
from database.models import ResourceState, SimulationStep
from database.utilities import execute_query


class ResourceRetriever:
    """Handles retrieval and analysis of resource-related data from simulation database.

    Provides methods for analyzing resource dynamics including distribution patterns,
    consumption statistics, concentration hotspots, and efficiency metrics across
    simulation timesteps.

    Methods
    -------
    resource_distribution()
        Retrieves time series of resource distribution metrics
    consumption_patterns()
        Calculates aggregate resource consumption statistics
    resource_hotspots()
        Identifies areas of high resource concentration
    efficiency_metrics()
        Computes resource utilization and efficiency measures
    execute()
        Performs comprehensive resource analysis
    """

    def __init__(self, database):
        """Initialize the ResourceRetriever.

        Parameters
        ----------
        database : SimulationDatabase
            Database connection instance used to execute queries
        """
        self.db = database

    @execute_query
    def resource_distribution(self, session) -> List[ResourceDistributionStep]:
        """Retrieve time series of resource distribution metrics.

        Queries database for spatial resource distribution patterns across simulation
        timesteps, including total quantities, densities, and distribution entropy.

        Returns
        -------
        List[ResourceDistributionStep]
            Sequence of distribution metrics per timestep:
            - step: Simulation timestep number
            - total_resources: Total quantity of resources present
            - average_per_cell: Mean resource density per grid cell
            - distribution_entropy: Shannon entropy of resource distribution
        """
        distribution_data = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.total_resources,
                SimulationStep.average_agent_resources,
                SimulationStep.resource_distribution_entropy,
            )
            .order_by(SimulationStep.step_number)
            .all()
        )

        return [
            ResourceDistributionStep(
                step=step,
                total_resources=total,
                average_per_cell=density,
                distribution_entropy=entropy,
            )
            for step, total, density, entropy in distribution_data
        ]

    @execute_query
    def consumption_patterns(self, session) -> ConsumptionStats:
        """Calculate aggregate resource consumption statistics.

        Analyzes consumption rates and variability across the entire simulation
        timeline, including totals, averages, peaks and variance measures.

        Returns
        -------
        ConsumptionStats
            Statistical measures of resource consumption:
            - total_consumed: Total resources consumed across all steps
            - avg_consumption_rate: Mean consumption rate per timestep
            - peak_consumption: Maximum single-step consumption
            - consumption_variance: Variance in consumption rates
        """
        # First get basic stats
        basic_stats = session.query(
            func.sum(SimulationStep.resources_consumed).label("total"),
            func.avg(SimulationStep.resources_consumed).label("average"),
            func.max(SimulationStep.resources_consumed).label("peak"),
        ).first()

        # Calculate variance manually: VAR = E[(X - μ)²]
        avg_consumption = basic_stats[1] or 0
        variance_calc = session.query(
            func.avg(
                (SimulationStep.resources_consumed - avg_consumption)
                * (SimulationStep.resources_consumed - avg_consumption)
            ).label("variance")
        ).first()

        return ConsumptionStats(
            total_consumed=float(basic_stats[0] or 0),
            avg_consumption_rate=float(basic_stats[1] or 0),
            peak_consumption=float(basic_stats[2] or 0),
            consumption_variance=float(variance_calc[0] or 0),
        )

    @execute_query
    def resource_hotspots(self, session) -> List[ResourceHotspot]:
        """Identify areas of high resource concentration.

        Analyzes spatial resource distribution to locate and rank areas with
        above-average resource concentrations.

        Returns
        -------
        List[ResourceHotspot]
            Resource hotspots sorted by concentration (highest first):
            - position_x: X coordinate of hotspot
            - position_y: Y coordinate of hotspot
            - concentration: Average resource amount at location
        """
        hotspot_data = (
            session.query(
                ResourceState.position_x,
                ResourceState.position_y,
                func.avg(ResourceState.amount).label("concentration"),
            )
            .group_by(ResourceState.position_x, ResourceState.position_y)
            .having(func.avg(ResourceState.amount) > 0)
            .order_by(func.avg(ResourceState.amount).desc())
            .all()
        )

        return [
            ResourceHotspot(
                position_x=x,
                position_y=y,
                concentration=concentration,
            )
            for x, y, concentration in hotspot_data
        ]

    @execute_query
    def efficiency_metrics(self, session) -> ResourceEfficiencyMetrics:
        #! Currently not working
        """Calculate resource utilization and efficiency metrics.

        Computes various efficiency measures related to resource distribution,
        consumption, and regeneration patterns.

        Returns
        -------
        ResourceEfficiencyMetrics
            Collection of efficiency metrics:
            - utilization_rate: Resource usage efficiency
            - distribution_efficiency: Spatial distribution effectiveness
            - consumption_efficiency: Resource consumption optimization
            - regeneration_rate: Resource replenishment speed
        """
        metrics = session.query(
            func.avg(SimulationStep.resource_efficiency).label("utilization"),
            func.avg(SimulationStep.distribution_efficiency).label("distribution"),
            func.avg(SimulationStep.consumption_efficiency).label("consumption"),
            func.avg(SimulationStep.regeneration_rate).label("regeneration"),
        ).first()

        return ResourceEfficiencyMetrics(
            utilization_rate=float(metrics[0] or 0),
            distribution_efficiency=float(metrics[1] or 0),
            consumption_efficiency=float(metrics[2] or 0),
            regeneration_rate=float(metrics[3] or 0),
        )

    @execute_query
    def execute(self, session) -> ResourceAnalysis:
        """Perform comprehensive resource analysis.

        Combines distribution, consumption, hotspot and efficiency analyses
        into a complete resource behavior assessment.

        Returns
        -------
        ResourceAnalysis
            Complete resource analysis including:
            - distribution: Time series of distribution metrics
            - consumption: Aggregate consumption statistics
            - hotspots: Areas of high concentration
            - efficiency: Resource efficiency measures
        """
        return ResourceAnalysis(
            distribution=self.resource_distribution(),
            consumption=self.consumption_patterns(),
            hotspots=self.resource_hotspots(),
            efficiency={},
        )
