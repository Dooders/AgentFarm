from typing import List, Optional, Tuple, Union

from farm.database.data_types import (
    AgentDistribution,
    BasicPopulationStatistics,
    Population,
    PopulationMetrics,
    PopulationStatistics,
    PopulationVariance,
)
from farm.database.enums import AnalysisScope
from farm.database.repositories.population_repository import PopulationRepository


class PopulationAnalyzer:
    """
    Analyzes population data using methods from PopulationRepository.

    This class provides comprehensive analysis of population dynamics, resource utilization,
    and agent distributions across simulation steps. Calculates statistics about agent
    populations, resource consumption, and survival metrics.

    Attributes:
        repository (PopulationRepository): Repository instance for accessing population data.

    Example:
        >>> analyzer = PopulationAnalyzer(population_repository)
        >>> stats = analyzer.analyze_comprehensive_statistics()
        >>> print(f"Peak population: {stats.population_metrics.total_agents}")
    """

    def __init__(self, repository: PopulationRepository):
        """Initialize the PopulationAnalyzer with a PopulationRepository."""
        self.repository = repository

    def analyze_population_data(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[Population]:
        """
        Retrieve and analyze basic population data.

        Args:
            scope (Union[str, AnalysisScope]): Analysis scope (e.g., 'simulation', 'generation')
            agent_id (Optional[int]): Filter results for specific agent ID
            step (Optional[int]): Filter results for specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            List[Population]: List of Population objects containing raw population data

        Example:
            >>> pop_data = analyzer.analyze_population_data(scope=AnalysisScope.SIMULATION)
            >>> print(f"Data points: {len(pop_data)}")
        """
        return self.repository.get_population_data(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_basic_statistics(
        self,
        pop_data: Optional[List[Population]] = None,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> BasicPopulationStatistics:
        """
        Calculate basic population statistics from step data.

        Processes raw population data to compute fundamental statistics about
        the population and resource usage. If no population data is provided,
        it will be fetched using the specified filters.

        Args:
            pop_data (Optional[List[Population]]): Pre-fetched population data to analyze.
                If None, data will be fetched using other parameters.
            scope (Union[str, AnalysisScope]): Analysis scope for data filtering
                ('simulation' or 'generation')
            agent_id (Optional[int]): Filter results for a specific agent ID
            step (Optional[int]): Filter results for a specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            BasicPopulationStatistics: Statistics object containing:
                - avg_population (float): Mean population across all steps
                - death_step (int): Final step number where agents existed
                - peak_population (int): Maximum population reached
                - lowest_population (int): Minimum population reached
                - resources_consumed (float): Total resources used by agents
                - resources_available (float): Total resources available
                - sum_squared (float): Sum of squared population counts (for variance)
                - initial_population (int): Population count at first step
                - final_population (int): Population count at last step
                - step_count (int): Total number of steps analyzed

        Example:
            >>> stats = analyzer.analyze_basic_statistics()
            >>> print(f"Average population: {stats.avg_population:.2f}")
            >>> print(f"Peak population: {stats.peak_population}")
            >>> print(f"Resource efficiency: {stats.resources_consumed/stats.resources_available:.2%}")
        """
        if not pop_data:
            pop_data = self.analyze_population_data(scope, agent_id, step, step_range)

        if not pop_data:
            return BasicPopulationStatistics(
                avg_population=0.0,
                death_step=0,
                peak_population=0,
                lowest_population=0,
                resources_consumed=0.0,
                resources_available=0.0,
                sum_squared=0.0,
                initial_population=0,
                final_population=0,
                step_count=1,
            )

        stats = {
            "avg_population": sum(p.total_agents for p in pop_data) / len(pop_data),
            "death_step": max(p.step_number for p in pop_data),
            "peak_population": max(p.total_agents for p in pop_data),
            "lowest_population": min(p.total_agents for p in pop_data),
            "resources_consumed": sum(p.resources_consumed for p in pop_data),
            "resources_available": sum(p.total_resources for p in pop_data),
            "sum_squared": sum(p.total_agents * p.total_agents for p in pop_data),
            "initial_population": pop_data[0].total_agents,
            "final_population": pop_data[-1].total_agents,
            "step_count": len(pop_data),
        }

        return BasicPopulationStatistics(**stats)

    def analyze_agent_distribution(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> AgentDistribution:
        """
        Analyze the distribution of different agent types in the population.

        Calculates the breakdown of agents by their type (system, independent, control)
        across the specified scope and filters.

        Args:
            scope (Union[str, AnalysisScope]): Analysis scope for data filtering
                ('simulation' or 'generation')
            agent_id (Optional[int]): Filter results for a specific agent ID
            step (Optional[int]): Filter results for a specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            AgentDistribution: Distribution object containing:
                - system_agents (int): Count of system-controlled agents
                - independent_agents (int): Count of independently operating agents
                - control_agents (int): Count of control group agents
                - total_agents (int): Total number of all agent types

        Example:
            >>> dist = analyzer.analyze_agent_distribution()
            >>> print(f"System agents: {dist.system_agents}")
            >>> print(f"Independent agents: {dist.independent_agents}")
            >>> print(f"Control agents: {dist.control_agents}")
            >>> print(f"Distribution ratio: {dist.system_agents/dist.total_agents:.2%}")
        """
        return self.repository.get_agent_type_distribution(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_population_momentum(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Calculate population momentum metric.

        Population momentum captures the relationship between population growth
        and simulation duration. Higher values indicate better population sustainability
        and growth efficiency.

        Args:
            scope (Union[str, AnalysisScope]): Analysis scope (e.g., 'simulation', 'generation')
            agent_id (Optional[int]): Filter results for specific agent ID
            step (Optional[int]): Filter results for specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            float: Population momentum metric calculated as (final_step * max_population) / initial_population.
                  Returns 0.0 if initial population is 0 or no data is available.

        Example:
            >>> momentum = analyzer.analyze_population_momentum()
            >>> print(f"Population momentum: {momentum:.2f}")
        """
        pop_data = self.analyze_population_data(scope, agent_id, step, step_range)
        if not pop_data:
            return 0.0

        initial_population = pop_data[0].total_agents
        if initial_population == 0:
            return 0.0

        max_population = max(p.total_agents for p in pop_data)
        final_step = max(p.step_number for p in pop_data)

        return (final_step * max_population) / initial_population

    def analyze_population_metrics(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> PopulationMetrics:
        """
        Calculate comprehensive population metrics including total agents and type distribution.

        Combines basic population statistics with agent distribution analysis to provide
        a complete overview of the population composition at the specified scope.

        Args:
            scope (Union[str, AnalysisScope]): Analysis scope for data filtering
                ('simulation' or 'generation')
            agent_id (Optional[int]): Filter results for a specific agent ID
            step (Optional[int]): Filter results for a specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            PopulationMetrics: Metrics object containing:
                - total_agents (int): Peak number of agents across all steps
                - system_agents (int): Number of system-controlled agents
                - independent_agents (int): Number of independently operating agents
                - control_agents (int): Number of control group agents

        Example:
            >>> metrics = analyzer.analyze_population_metrics()
            >>> print(f"Total population: {metrics.total_agents}")
            >>> print(f"System agent ratio: {metrics.system_agents/metrics.total_agents:.2%}")
            >>> print(f"Independent agent ratio: {metrics.independent_agents/metrics.total_agents:.2%}")
        """
        pop_data = self.analyze_population_data(scope, agent_id, step, step_range)
        basic_stats = self.analyze_basic_statistics(pop_data)
        type_stats = self.analyze_agent_distribution(scope, agent_id, step, step_range)

        return PopulationMetrics(
            total_agents=basic_stats.peak_population,
            system_agents=int(type_stats.system_agents),
            independent_agents=int(type_stats.independent_agents),
            control_agents=int(type_stats.control_agents),
        )

    def analyze_population_variance(
        self,
        basic_stats: Optional[BasicPopulationStatistics] = None,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> PopulationVariance:
        """
        Calculate statistical measures of population variation over time.

        Computes variance, standard deviation, and coefficient of variation to measure
        population stability and fluctuation patterns. Uses pre-calculated basic statistics
        if provided, otherwise fetches and calculates them using the specified filters.

        Args:
            basic_stats (Optional[BasicPopulationStatistics]): Pre-calculated basic statistics.
                If None, stats will be calculated using other parameters.
            scope (Union[str, AnalysisScope]): Analysis scope for data filtering
                ('simulation' or 'generation')
            agent_id (Optional[int]): Filter results for a specific agent ID
            step (Optional[int]): Filter results for a specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            PopulationVariance: Variance metrics containing:
                - variance (float): Statistical variance of population size
                - standard_deviation (float): Square root of variance
                - coefficient_variation (float): Standard deviation normalized by mean
                    (0 if mean population is 0)

        Example:
            >>> variance = analyzer.analyze_population_variance()
            >>> print(f"Population variance: {variance.variance:.2f}")
            >>> print(f"Standard deviation: {variance.standard_deviation:.2f}")
            >>> print(f"Coefficient of variation: {variance.coefficient_variation:.2%}")
        """
        if not basic_stats:
            pop_data = self.analyze_population_data(scope, agent_id, step, step_range)
            basic_stats = self.analyze_basic_statistics(pop_data)

        variance = (basic_stats.sum_squared / basic_stats.step_count) - (
            basic_stats.avg_population**2
        )
        std_dev = variance**0.5
        cv = (
            std_dev / basic_stats.avg_population
            if basic_stats.avg_population > 0
            else 0
        )

        return PopulationVariance(
            variance=variance, standard_deviation=std_dev, coefficient_variation=cv
        )

    def analyze_comprehensive_statistics(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> PopulationStatistics:
        """
        Execute comprehensive population statistics calculation.

        Provides a complete statistical overview of the population including metrics,
        variance measures, momentum, and agent type distribution.

        Args:
            scope (Union[str, AnalysisScope]): Analysis scope (e.g., 'simulation', 'generation')
            agent_id (Optional[int]): Filter results for specific agent ID
            step (Optional[int]): Filter results for specific simulation step
            step_range (Optional[Tuple[int, int]]): Filter results for step range (start, end)

        Returns:
            PopulationStatistics: Comprehensive statistics containing:
                - population_metrics: Total agents and breakdown by type
                - population_variance: Statistical measures of population variation
                - population_momentum: Growth sustainability metric
                - agent_distribution: Distribution of different agent types

        Example:
            >>> stats = analyzer.analyze_comprehensive_statistics()
            >>> print(f"Total agents: {stats.population_metrics.total_agents}")
            >>> print(f"Population variance: {stats.population_variance.variance:.2f}")
        """
        pop_data = self.analyze_population_data(scope, agent_id, step, step_range)
        basic_stats = self.analyze_basic_statistics(pop_data)

        return PopulationStatistics(
            population_metrics=self.analyze_population_metrics(
                scope, agent_id, step, step_range
            ),
            population_variance=self.analyze_population_variance(basic_stats),
        )
