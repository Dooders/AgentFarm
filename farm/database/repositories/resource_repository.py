from typing import List, Dict, Any

from sqlalchemy import func

from farm.database.data_types import (
    ConsumptionStats,
    ResourceAnalysis,
    ResourceDistributionStep,
    ResourceEfficiencyMetrics,
    ResourceHotspot,
)
from farm.database.models import ResourceModel, SimulationStepModel
from farm.database.repositories.base_repository import BaseRepository
from farm.database.session_manager import SessionManager


class ResourceRepository(BaseRepository[ResourceModel]):
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

    def __init__(self, session_manager: SessionManager):
        """Initialize the ResourceRepository.

        Parameters
        ----------
        database : SimulationDatabase
            Database connection instance used to execute queries
        """
        super().__init__(session_manager, ResourceModel)

    def resource_distribution(self) -> List[ResourceDistributionStep]:
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
        def query_distribution(session):
            distribution_data = (
                session.query(
                    SimulationStepModel.step_number,
                    SimulationStepModel.total_resources,
                    SimulationStepModel.average_agent_resources,
                    SimulationStepModel.resource_distribution_entropy,
                )
                .order_by(SimulationStepModel.step_number)
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

        return self.session_manager.execute_with_retry(query_distribution)

    def consumption_patterns(self) -> ConsumptionStats:
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
        def query_consumption(session):
            # First get basic stats
            basic_stats = session.query(
                func.sum(SimulationStepModel.resources_consumed).label("total"),
                func.avg(SimulationStepModel.resources_consumed).label("average"),
                func.max(SimulationStepModel.resources_consumed).label("peak"),
            ).first()

            # Calculate variance manually: VAR = E[(X - μ)²]
            avg_consumption = basic_stats[1] or 0
            variance_calc = session.query(
                func.avg(
                    (SimulationStepModel.resources_consumed - avg_consumption)
                    * (SimulationStepModel.resources_consumed - avg_consumption)
                ).label("variance")
            ).first()

            return ConsumptionStats(
                total_consumed=float(basic_stats[0] or 0),
                avg_consumption_rate=float(basic_stats[1] or 0),
                peak_consumption=float(basic_stats[2] or 0),
                consumption_variance=float(variance_calc[0] or 0),
            )

        return self.session_manager.execute_with_retry(query_consumption)

    def resource_hotspots(self) -> List[ResourceHotspot]:
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
        def query_hotspots(session):
            hotspot_data = (
                session.query(
                    ResourceModel.position_x,
                    ResourceModel.position_y,
                    func.avg(ResourceModel.amount).label("concentration"),
                )
                .group_by(ResourceModel.position_x, ResourceModel.position_y)
                .having(func.avg(ResourceModel.amount) > 0)
                .order_by(func.avg(ResourceModel.amount).desc())
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

        return self.session_manager.execute_with_retry(query_hotspots)

    def efficiency_metrics(self) -> ResourceEfficiencyMetrics:
        """Calculate resource utilization and efficiency metrics.

        Computes various efficiency measures related to resource distribution,
        consumption, and regeneration patterns.

        Returns
        -------
        ResourceEfficiencyMetrics
            Collection of efficiency metrics:
            - utilization_rate: Resource usage efficiency (resource_efficiency from db)
            - distribution_efficiency: Spatial distribution effectiveness (resource_distribution_entropy)
            - consumption_efficiency: Resource consumption optimization (calculated)
            - regeneration_rate: Resource replenishment speed (placeholder)
        """
        def query_efficiency(session):
            # Use available columns from the database
            metrics = session.query(
                func.avg(SimulationStepModel.resource_efficiency).label("utilization"),
                func.avg(SimulationStepModel.resource_distribution_entropy).label("distribution_entropy"),
            ).first()

            # Calculate consumption efficiency as resources consumed per total resources
            consumption_metrics = session.query(
                func.sum(SimulationStepModel.resources_consumed).label("total_consumed"),
                func.avg(SimulationStepModel.total_resources).label("avg_resources"),
            ).first()

            consumption_efficiency = 0.0
            if consumption_metrics[1] and consumption_metrics[1] > 0:
                consumption_efficiency = (consumption_metrics[0] or 0) / consumption_metrics[1]

            return ResourceEfficiencyMetrics(
                utilization_rate=float(metrics[0] or 0),
                distribution_efficiency=float(metrics[1] or 0),  # Using entropy as distribution efficiency
                consumption_efficiency=float(consumption_efficiency),
                regeneration_rate=0.0,  # Placeholder - not available in current schema
            )

        return self.session_manager.execute_with_retry(query_efficiency)

    def execute(self) -> ResourceAnalysis:
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
            efficiency=self.efficiency_metrics(),
        )

    def get_resource_positions_over_time(self) -> List[Dict[str, Any]]:
        """Get resource positions over time for spatial analysis.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing resource position data
        """

        def query_positions(session):
            # Query resource positions with amounts over time
            result = session.query(
                ResourceModel.step_number,
                ResourceModel.position_x,
                ResourceModel.position_y,
                ResourceModel.amount,
                ResourceModel.resource_id
            ).filter(
                ResourceModel.position_x.isnot(None),
                ResourceModel.position_y.isnot(None)
            ).order_by(
                ResourceModel.step_number,
                ResourceModel.position_x,
                ResourceModel.position_y
            ).all()

            return [
                {
                    'step': row.step_number,
                    'position_x': row.position_x,
                    'position_y': row.position_y,
                    'position_z': 0.0,  # Resources don't have z coordinate
                    'amount': row.amount,
                    'resource_type': str(row.resource_id)  # Use resource_id as type
                }
                for row in result
            ]

        return self.session_manager.execute_with_retry(query_positions)

    def get_resource_distribution_data(self) -> List[Dict[str, Any]]:
        """Get resource distribution data for spatial analysis.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing resource distribution data
        """

        def query_distribution(session):
            # Get aggregated resource distribution by position
            result = session.query(
                ResourceModel.position_x,
                ResourceModel.position_y,
                ResourceModel.position_z,
                func.sum(ResourceModel.amount).label('total_amount'),
                func.count(ResourceModel.id).label('resource_count'),
                func.avg(ResourceModel.amount).label('average_amount')
            ).filter(
                ResourceModel.position_x.isnot(None),
                ResourceModel.position_y.isnot(None)
            ).group_by(
                ResourceModel.position_x,
                ResourceModel.position_y,
                ResourceModel.position_z
            ).order_by(
                func.sum(ResourceModel.amount).desc()
            ).all()

            return [
                {
                    'position_x': row.position_x,
                    'position_y': row.position_y,
                    'position_z': row.position_z or 0.0,
                    'total_amount': row.total_amount,
                    'resource_count': row.resource_count,
                    'average_amount': row.average_amount
                }
                for row in result
            ]

        return self.session_manager.execute_with_retry(query_distribution)
