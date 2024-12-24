"""Data retrieval module for simulation database.

This module provides a comprehensive interface for querying and analyzing simulation data,
including agent states, resource distributions, learning patterns, and system metrics.

The DataRetriever class handles all database operations with optimized queries and
efficient data aggregation methods.

Features
--------
- Agent statistics and lifecycle analysis
- Population dynamics and demographics
- Resource distribution and consumption patterns
- Learning and adaptation metrics
- Behavioral pattern analysis
- Historical trend analysis
- Performance and efficiency metrics

Classes
-------
DataRetriever
    Main class handling all data retrieval and analysis operations

Dependencies
-----------
- sqlalchemy: Database ORM and query building
- pandas: Data processing and statistical analysis
- json: JSON data handling
- logging: Error and operation logging
- typing: Type hints and annotations

Notes
-----
All database operations are executed within transactions to ensure data consistency.
Query optimization is implemented for handling large datasets efficiently.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import and_, exists, func, not_
from sqlalchemy.orm import aliased

from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.analyzers.agent_analyzer import AgentAnalysis
from farm.database.analyzers.lifespan_analysis import AgentLifespanAnalysis
from farm.database.simulation import SimulationStateRetriever
from farm.database.utilities import execute_query

from .data_types import (
    ActionMetrics,
    AdvancedStatistics,
    AgentBehaviorMetrics,
    AgentDistribution,
    AgentLifespanStats,
    DecisionPatterns,
    DecisionPatternStats,
    DecisionSummary,
    InteractionPattern,
    InteractionStats,
    LearningStatistics,
    PopulationMetrics,
    PopulationStatistics,
    PopulationVariance,
    ResourceImpact,
    ResourceMetrics,
    SequencePattern,
    SimulationResults,
    StepActionData,
    SurvivalMetrics,
    TimePattern,
)
from .models import AgentModel, ActionModel

logger = logging.getLogger(__name__)


class DataRetriever:
    """Handles data retrieval operations for the simulation database."""

    def __init__(self, database):
        """Initialize data retriever with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database
        self._retrievers = {
            "simulation": SimulationStateRetriever(database),
            "agent_lifespan": AgentLifespanAnalysis(database),
            # "population": PopulationStatisticsRetriever(database),
            # "resource": ResourceRetriever(database),
            # "learning": LearningRetriever(database),
            "actions": ActionStatsAnalyzer(database),
            "agent": AgentAnalysis(database),
        }

    def __getattr__(self, name):
        """Dynamically search for methods in sub-retrievers.

        Parameters
        ----------
        name : str
            Name of the attribute/method being accessed

        Returns
        -------
        Any
            The found method/attribute from a sub-retriever

        Raises
        ------
        AttributeError
            If the attribute/method is not found in any sub-retriever
        """
        # Search through all sub-retrievers for the requested method
        for retriever in self._retrievers.values():
            if hasattr(retriever, name):
                return getattr(retriever, name)

        # If method not found in any retriever, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object and its sub-retrievers "
            f"have no attribute '{name}'"
        )

    def simulation_results(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step.

        Gets the full simulation state including agent states, resource states,
        and overall simulation metrics for the specified step number.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        SimulationResults
            Dictionary containing:
            - agent_states: List[AgentStates]
                States of all agents at this step
            - resource_states: List[ResourceStates]
                States of all resources at this step
            - simulation_state: SimulationState
                Overall simulation metrics and configuration

        Notes
        -----
        This is a convenience method that combines the results of agent_states(),
        resource_states(), and simulation_state() into a single response.
        Returns None for any components that are not found for the given step.
        """
        return self._retrievers["simulation"].execute(step_number)

    def agent_lifespan_statistics(self) -> AgentLifespanStats:
        """Calculate comprehensive statistics about agent lifespans.

        Returns
        -------
        AgentLifespanStats
            Data containing:
            - average_lifespan: float, mean lifespan across all agents
            - lifespan_by_type: Dict[str, float], mean lifespan per agent type
            - lifespan_by_generation: Dict[int, float], mean lifespan per generation
            - survival_rates: Dict[int, float], survival rate per generation

        Notes
        -----
        This method encapsulates the logic for calculating and returning
        comprehensive agent lifespan statistics.
        """
        return self._retrievers["agent_lifespan"].execute()

    def population_statistics(self) -> PopulationStatistics:
        """Calculate comprehensive population statistics for the simulation.

        Returns
        -------
        PopulationStatistics
            Dictionary containing:
            - basic_stats: Dict[str, float]
                - average_population: Mean population across all steps
                - peak_population: Maximum population reached
                - death_step: Final simulation step
                - total_steps: Total number of simulation steps
            - resource_metrics: Dict[str, float]
                - resource_utilization: Resource usage efficiency
                - resources_consumed: Total resources consumed
                - resources_available: Total resources available
                - utilization_per_agent: Average resource usage per agent
            - population_variance: Dict[str, float]
                - variance: Population variance
                - standard_deviation: Population standard deviation
                - coefficient_variation: Coefficient of variation
            - agent_distribution: Dict[str, float]
                - system_agents: Average number of system agents
                - independent_agents: Average number of independent agents
                - control_agents: Average number of control agents
            - survival_metrics: Dict[str, float]
                - survival_rate: Population survival rate
                - average_lifespan: Mean agent lifespan

        Notes
        -----
        This method aggregates data from the PopulationStatisticsRetriever to provide
        a comprehensive view of population dynamics throughout the simulation.
        """
        return self._retrievers["population"].execute()

    @execute_query
    def resource_statistics(self, session) -> Dict[str, Any]:
        """Get statistics about resource distribution and consumption.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - resource_distribution: Dict[str, List[float]]
                Time series data of resource distribution
            - efficiency_metrics: Dict[str, float]
                Resource efficiency and utilization metrics
            - consumption_patterns: Dict[str, float]
                Resource consumption statistics
            - hotspots: List[Tuple[float, float, float]]
                Resource concentration points
        """
        return self._retrievers["resource"].execute()

    @execute_query
    def learning_statistics(self, session) -> LearningStatistics:
        """Get statistics about agent learning and adaptation.

        Returns
        -------
        LearningStatistics
            Data containing:
            - learning_progress: Dict[str, float]
                Time series data of learning progress
            - module_performance: Dict[str, Dict[str, float]]
                Performance metrics for each learning module
        """
        return self._retrievers["learning"].execute()

    def step_actions(self, step_number: int) -> StepActionData:
        """Get all actions performed during a specific simulation step.

        Parameters
        ----------
        step_number : int
            The simulation step number to query

        Returns
        -------
        StepActionData
            Data containing step action information. See ActionsRetriever.step_actions
            for full documentation of return fields.
        """
        return self._retrievers["actions"].step_actions(step_number)

    def get_agent_behaviors(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> AgentBehaviorMetrics:
        """Get comprehensive analysis of agent behaviors across the simulation.

        Parameters
        ----------
        start_step : int, optional
            Starting step number for analysis window
        end_step : int, optional
            Ending step number for analysis window

        Returns
        -------
        AgentBehaviorMetrics
            Data containing:
            - temporal_patterns: Dict[str, Any]
                - step_data: Dict[int, Dict[str, Dict[str, float]]]
                    Per step action type statistics including:
                    - count: Number of occurrences
                    - average_reward: Average reward for the action
                - time_range: Dict[str, int]
                    - start: Starting step number
                    - end: Ending step number
            - type_behaviors: Dict[str, Dict[str, Any]]
                Per agent type behavior statistics including:
                - actions: Dict[str, Dict[str, float]]
                    Per action type statistics including:
                    - count: Number of occurrences
                    - average_reward: Average reward for the action
                    - reward_stddev: Standard deviation of rewards
                - most_common_action: Most common action type
                - most_rewarding_action: Most rewarding action type
            - interaction_patterns: Dict[str, Dict[str, float]]
                Per action type interaction statistics including:
                - interaction_count: Number of interactions
                - average_reward: Average reward for the interactions
            - resource_behaviors: Dict[str, Dict[str, float]]
                Per action type resource impact statistics including:
                - average_resource_change: Average resource change
                - action_count: Number of occurrences
            - behavior_summary: Dict[str, Any]
                - total_actions: Total number of actions
                - unique_action_types: Number of unique action types
                - most_common_behaviors: List of most common action types
        """

        def _query(session):
            # Build base query with time window
            base_query = session.query(ActionModel)
            if start_step is not None:
                base_query = base_query.filter(ActionModel.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(ActionModel.step_number <= end_step)

            # Get temporal behavior patterns
            temporal_patterns = (
                base_query.with_entities(
                    ActionModel.step_number,
                    ActionModel.action_type,
                    func.count().label("action_count"),
                    func.avg(ActionModel.reward).label("avg_reward"),
                )
                .group_by(ActionModel.step_number, ActionModel.action_type)
                .order_by(ActionModel.step_number)
                .all()
            )

            # Get behavior patterns by agent type
            type_patterns = (
                base_query.join(AgentModel)
                .with_entities(
                    AgentModel.agent_type,
                    ActionModel.action_type,
                    func.count().label("action_count"),
                    func.avg(ActionModel.reward).label("avg_reward"),
                    func.stddev(ActionModel.reward).label("reward_stddev"),
                )
                .group_by(AgentModel.agent_type, ActionModel.action_type)
                .all()
            )

            # Get interaction patterns
            interaction_patterns = (
                base_query.filter(ActionModel.action_target_id.isnot(None))
                .with_entities(
                    ActionModel.action_type,
                    func.count().label("interaction_count"),
                    func.avg(ActionModel.reward).label("avg_interaction_reward"),
                )
                .group_by(ActionModel.action_type)
                .all()
            )

            # Get resource-related behaviors
            resource_behaviors = (
                base_query.filter(
                    ActionModel.resources_before.isnot(None),
                    ActionModel.resources_after.isnot(None),
                )
                .with_entities(
                    ActionModel.action_type,
                    func.avg(
                        ActionModel.resources_after - ActionModel.resources_before
                    ).label("avg_resource_change"),
                    func.count().label("resource_action_count"),
                )
                .group_by(ActionModel.action_type)
                .all()
            )

            # Format temporal patterns
            temporal_data = {}
            for step, action_type, count, avg_reward in temporal_patterns:
                if step not in temporal_data:
                    temporal_data[step] = {}
                temporal_data[step][action_type] = {
                    "count": int(count),
                    "average_reward": float(avg_reward or 0),
                }

            # Format type-specific patterns
            type_data = {}
            for agent_type, action_type, count, avg_reward, stddev in type_patterns:
                if agent_type not in type_data:
                    type_data[agent_type] = {}
                type_data[agent_type][action_type] = {
                    "count": int(count),
                    "average_reward": float(avg_reward or 0),
                    "reward_stddev": float(stddev or 0),
                }

            return {
                "temporal_patterns": {
                    "step_data": temporal_data,
                    "time_range": {
                        "start": min(temporal_data.keys()) if temporal_data else None,
                        "end": max(temporal_data.keys()) if temporal_data else None,
                    },
                },
                "type_behaviors": {
                    agent_type: {
                        "actions": actions,
                        "most_common_action": (
                            max(actions.items(), key=lambda x: x[1]["count"])[0]
                            if actions
                            else None
                        ),
                        "most_rewarding_action": (
                            max(actions.items(), key=lambda x: x[1]["average_reward"])[
                                0
                            ]
                            if actions
                            else None
                        ),
                    }
                    for agent_type, actions in type_data.items()
                },
                "interaction_patterns": {
                    action_type: {
                        "interaction_count": int(count),
                        "average_reward": float(avg_reward or 0),
                    }
                    for action_type, count, avg_reward in interaction_patterns
                },
                "resource_behaviors": {
                    action_type: {
                        "average_resource_change": float(avg_change or 0),
                        "action_count": int(count),
                    }
                    for action_type, avg_change, count in resource_behaviors
                },
                "behavior_summary": {
                    "total_actions": sum(pattern[2] for pattern in type_patterns),
                    "unique_action_types": len(
                        set(pattern[1] for pattern in type_patterns)
                    ),
                    "most_common_behaviors": sorted(
                        [
                            (
                                action_type,
                                sum(1 for p in type_patterns if p[1] == action_type),
                            )
                            for action_type in set(p[1] for p in type_patterns)
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5],
                },
            }

        return self.db._execute_in_transaction(_query)

    def advanced_statistics(self) -> AdvancedStatistics:
        """Calculate advanced simulation statistics using optimized queries.

        Returns
        -------
        AdvancedStatistics: Dict[str, Any]
            Data containing:
            - Population metrics (peak, average, diversity)
            - Interaction statistics (conflict/cooperation ratios)
            - Resource efficiency metrics
            - Agent type distribution
            - Survival and adaptation metrics
        """

        def _query(session):
            # Get component statistics using helper methods
            pop_stats = self.population_stats(session)
            type_ratios = self.agent_type_ratios(session)
            interaction_metrics = self.interaction_metrics(session)
            resource_metrics = self.resource_metrics(session)

            # Calculate diversity index
            diversity = self._calculate_diversity_index(list(type_ratios.values()))

            return AdvancedStatistics(
                population_metrics=PopulationStatistics(
                    population_metrics=PopulationMetrics(**pop_stats),
                    population_variance=PopulationVariance(
                        total_agents=pop_stats["total_agents"],
                        system_agents=pop_stats["system_agents"],
                        independent_agents=pop_stats["independent_agents"],
                        control_agents=pop_stats["control_agents"],
                    ),
                ),
                interaction_metrics=InteractionPattern(
                    **interaction_metrics,
                    conflict_cooperation_ratio=(
                        interaction_metrics["conflict_rate"]
                        / interaction_metrics["cooperation_rate"]
                        if interaction_metrics["cooperation_rate"] > 0
                        else float("inf")
                    ),
                ),
                resource_metrics=ResourceMetrics(**resource_metrics),
                agent_distribution=AgentDistribution(
                    **type_ratios,
                    type_entropy=diversity,
                ),
                survival_metrics=SurvivalMetrics(
                    population_stability=(
                        pop_stats["minimum_population"] / pop_stats["peak_population"]
                    ),
                    health_maintenance=(pop_stats["average_health"] / 100.0),
                    interaction_rate=(
                        interaction_metrics["total_actions"]
                        / pop_stats["total_steps"]
                        / pop_stats["average_population"]
                    ),
                ),
            )

        return self.db._execute_in_transaction(_query)

    def _calculate_diversity_index(self, ratios: List[float]) -> float:
        """Calculate Shannon entropy for agent type diversity.

        Parameters
        ----------
        ratios : List[float]
            List of agent type proportions

        Returns
        -------
        float
            Shannon entropy diversity index
        """
        import math

        return sum(-ratio * math.log(ratio) if ratio > 0 else 0 for ratio in ratios)

    def _format_interaction_patterns(
        self, interaction_patterns
    ) -> Dict[str, Dict[str, float]]:
        """Format raw interaction pattern data into analysis dictionary.

        Parameters
        ----------
        interaction_patterns : List[Tuple]
            List of tuples containing (action_type, is_interaction, count, avg_reward)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping action types to their interaction statistics:
            - interaction_rate: Rate of interactive vs solo actions
            - solo_performance: Average reward for solo actions
            - interaction_performance: Average reward for interactive actions
        """
        interaction_analysis = {}

        for action_type, is_interaction, count, avg_reward in interaction_patterns:
            if action_type not in interaction_analysis:
                interaction_analysis[action_type] = {
                    "interaction_rate": 0,
                    "solo_performance": 0,
                    "interaction_performance": 0,
                }

            total = sum(i[2] for i in interaction_patterns if i[0] == action_type)
            if is_interaction:
                interaction_analysis[action_type]["interaction_rate"] = (
                    count / total if total > 0 else 0
                )
                interaction_analysis[action_type]["interaction_performance"] = float(
                    avg_reward or 0
                )
            else:
                interaction_analysis[action_type]["solo_performance"] = float(
                    avg_reward or 0
                )

        return interaction_analysis

    def _format_temporal_patterns(self, time_patterns) -> Dict[str, Dict[str, List]]:
        """Format raw time-based pattern data into analysis dictionary.

        Parameters
        ----------
        time_patterns : List[Tuple]
            List of tuples containing (action_type, count, avg_reward)

        Returns
        -------
        Dict[str, Dict[str, List]]
            Dictionary mapping action types to their temporal statistics:
            - time_distribution: List of action counts over time
            - reward_progression: List of average rewards over time
        """
        temporal_patterns = {}
        for action_type, count, avg_reward in time_patterns:
            if action_type not in temporal_patterns:
                temporal_patterns[action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                }
            temporal_patterns[action_type]["time_distribution"].append(int(count))
            temporal_patterns[action_type]["reward_progression"].append(
                float(avg_reward or 0)
            )
        return temporal_patterns

    def _format_resource_patterns(
        self, resource_patterns
    ) -> Dict[str, Dict[str, float]]:
        """Format raw resource pattern data into analysis dictionary.

        Parameters
        ----------
        resource_patterns : List[Tuple]
            List of tuples containing (action_type, avg_before, avg_change, count)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping action types to their resource statistics:
            - avg_resources_before: Average resources before action
            - avg_resource_change: Average change in resources
            - resource_efficiency: Resource change per action
        """
        return {
            action_type: {
                "avg_resources_before": float(avg_before or 0),
                "avg_resource_change": float(avg_change or 0),
                "resource_efficiency": (
                    float(avg_change or 0) / count if count > 0 else 0
                ),
            }
            for action_type, avg_before, avg_change, count in resource_patterns
        }

    def _format_sequence_analysis(
        self, sequential_patterns: List[Tuple], decision_patterns: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Format raw sequential pattern data into analysis dictionary.

        Parameters
        ----------
        sequential_patterns : List[Tuple]
            List of tuples containing (action, next_action, count)
        decision_patterns : Dict[str, Any]
            Dictionary containing decision pattern statistics

        Returns
        -------
        Dict[str, Dict[str, Union[int, float]]]
            Dictionary mapping action sequences to their statistics:
            - count: Number of occurrences of the sequence
            - probability: Probability of next_action following action
        """
        return {
            f"{action}->{next_action}": {
                "count": count,
                "probability": (
                    count / decision_patterns[action]["count"]
                    if action in decision_patterns
                    else 0
                ),
            }
            for action, next_action, count in sequential_patterns
            if action is not None and next_action is not None
        }

    def _format_decision_patterns(
        self, metrics: List[ActionMetrics], rewards_by_type: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Format basic decision patterns from metrics and rewards.

        Parameters
        ----------
        metrics : List[ActionMetrics]
            List of action metrics containing counts and basic stats
        rewards_by_type : Dict[str, List[float]]
            Dictionary mapping action types to lists of reward values

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing formatted decision patterns with:
            - count: Number of decisions
            - frequency: Relative frequency of decisions
            - reward_stats: Dictionary of reward statistics
        """
        import numpy as np

        decision_patterns = {}
        total_decisions = sum(m.decision_count for m in metrics)

        for metric in metrics:
            rewards = rewards_by_type.get(metric.action_type, [])
            reward_stddev = float(np.std(rewards)) if len(rewards) > 1 else 0.0

            decision_patterns[metric.action_type] = {
                "count": metric.decision_count,
                "frequency": (
                    float(metric.decision_count) / total_decisions
                    if total_decisions > 0
                    else 0
                ),
                "reward_stats": {
                    "average": metric.avg_reward,
                    "stddev": reward_stddev,
                    "min": metric.min_reward,
                    "max": metric.max_reward,
                },
            }

        return decision_patterns

    def _get_interaction_patterns(self, base_query) -> List[Tuple]:
        """Get interaction patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - is_interaction: bool
            - count: int
            - avg_reward: float
        """
        return (
            base_query.with_entities(
                ActionModel.action_type,
                ActionModel.action_target_id.isnot(None).label("is_interaction"),
                func.count().label("count"),
                func.avg(ActionModel.reward).label("avg_reward"),
            )
            .group_by(
                ActionModel.action_type,
                ActionModel.action_target_id.isnot(None),
            )
            .all()
        )

    def _get_time_patterns(self, base_query) -> List[Tuple]:
        """Get time-based patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - count: int
            - avg_reward: float
            Grouped by action type and time periods (100 steps)
        """
        return (
            base_query.with_entities(
                ActionModel.action_type,
                func.count().label("count"),
                func.avg(ActionModel.reward).label("avg_reward"),
            )
            .group_by(
                ActionModel.action_type,
                func.round(ActionModel.step_number / 100),  # Group by time periods
            )
            .all()
        )

    def _get_resource_patterns(self, base_query) -> List[Tuple]:
        """Get resource-driven patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - avg_resources_before: float
            - avg_resource_change: float
            - count: int
        """
        return (
            base_query.with_entities(
                ActionModel.action_type,
                func.avg(ActionModel.resources_before).label("avg_resources_before"),
                func.avg(
                    ActionModel.resources_after - ActionModel.resources_before
                ).label("avg_resource_change"),
                func.count().label("count"),
            )
            .filter(
                ActionModel.resources_before.isnot(None),
                ActionModel.resources_after.isnot(None),
            )
            .group_by(ActionModel.action_type)
            .all()
        )

    def _get_sequential_patterns(self, base_query) -> List[Tuple]:
        """Get sequential patterns of agent actions.

        Parameters
        ----------
        base_query : Query
            Base query to build upon

        Returns
        -------
        List[Tuple]
            Representing sequences of actions and their frequencies
        """
        # First, get the actions with their next actions using a subquery
        subq = (
            base_query.add_columns(
                ActionModel.step_number,
                ActionModel.action_type,
            )
            .order_by(ActionModel.step_number)
            .subquery()
        )

        # Create an alias for self-join
        next_action = aliased(subq)

        # Join to get action pairs
        return (
            base_query.session.query(
                subq.c.action_type,
                next_action.c.action_type.label("next_action"),
                func.count().label("sequence_count"),
            )
            .join(
                next_action,
                and_(
                    subq.c.step_number < next_action.c.step_number,
                    not_(
                        exists().where(
                            and_(
                                ActionModel.step_number > subq.c.step_number,
                                ActionModel.step_number < next_action.c.step_number,
                                ActionModel.agent_id
                                == base_query.whereclause.right.value,
                            )
                        )
                    ),
                ),
            )
            .group_by(subq.c.action_type, next_action.c.action_type)
            .all()
        )

    def _get_all_patterns(
        self, base_query
    ) -> Tuple[SequencePattern, ResourceImpact, TimePattern, InteractionStats]:
        """Get all pattern types from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        Tuple[SequencePattern, ResourceImpact, TimePattern, InteractionStats]
            Tuple containing:
            - sequential_patterns: Sequences of actions and their frequencies
            - resource_patterns: Resource-driven patterns and impacts
            - time_patterns: Time-based patterns and trends
            - interaction_patterns: Interaction-based patterns and statistics
        """
        return (
            self._get_sequential_patterns(base_query),
            self._get_resource_patterns(base_query),
            self._get_time_patterns(base_query),
            self._get_interaction_patterns(base_query),
        )

    def agent_decision_patterns(
        self,
        agent_id: Optional[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> DecisionPatterns:
        """Get comprehensive analysis of agent decision-making patterns.

        Parameters
        ----------
        agent_id : Optional[str]
            Specific agent ID to analyze. If None, analyzes all agents.
        start_step : Optional[int]
            Starting step for analysis window
        end_step : Optional[int]
            Ending step for analysis window

        Returns
        -------
        DecisionPatterns
            Comprehensive metrics about agent decision patterns including:
            - Basic decision statistics
            - Sequential patterns
            - Context-based decisions
            - Resource-driven patterns
            - Interaction patterns
            - Learning progression
        """

        def _query(session):
            # Base query for additional analysis
            base_query = session.query(ActionModel)
            if agent_id is not None:
                base_query = base_query.filter(ActionModel.agent_id == agent_id)
            if start_step is not None:
                base_query = base_query.filter(ActionModel.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(ActionModel.step_number <= end_step)

            # Get basic action metrics
            action_metrics: List[ActionMetrics] = self.action_metrics(
                session, agent_id, start_step, end_step
            )

            # Get rewards by action type
            rewards_by_type: Dict[str, List[float]] = self.rewards_by_type(
                session,
                [(m.action_type, m.decision_count) for m in action_metrics],
                agent_id,
            )

            # Get all patterns
            (
                sequential_patterns,
                resource_patterns,
                time_patterns,
                interaction_patterns,
            ) = self._get_all_patterns(base_query)

            # Format decision patterns
            decision_patterns: DecisionPatternStats = self._format_decision_patterns(
                action_metrics, rewards_by_type
            )

            # Format sequential patterns
            sequence_analysis: SequencePattern = self._format_sequence_analysis(
                sequential_patterns, decision_patterns
            )

            # Format resource patterns
            resource_impact: ResourceImpact = self._format_resource_patterns(
                resource_patterns
            )

            # Format time-based patterns
            temporal_patterns: TimePattern = self._format_temporal_patterns(
                time_patterns
            )

            # Format interaction patterns
            interaction_analysis: InteractionStats = self._format_interaction_patterns(
                interaction_patterns
            )

            # Create decision summary
            decision_summary: DecisionSummary = self._create_decision_summary(
                decision_patterns
            )

            # Return with the created decision summary
            return {
                "decision_patterns": decision_patterns,
                "sequence_analysis": sequence_analysis,
                "resource_impact": resource_impact,
                "temporal_patterns": temporal_patterns,
                "interaction_analysis": interaction_analysis,
                "decision_summary": decision_summary,
            }

        return self.db._execute_in_transaction(_query)

    def _create_decision_summary(
        self, decision_patterns: Dict[str, Any]
    ) -> DecisionSummary:
        """Create a summary of decision patterns.

        Parameters
        ----------
        decision_patterns: Dict[str, Any]
            Dictionary containing decision pattern statistics

        Returns
        -------
        DecisionSummary
            Summary containing:
            - total_decisions: Total number of decisions made
            - unique_actions: Number of unique action types
            - most_frequent: Most frequently used action type
            - most_rewarding: Action type with highest average reward
            - action_diversity: Shannon entropy of action distribution
        """
        total_decisions = sum(p["count"] for p in decision_patterns.values())

        return {
            "total_decisions": total_decisions,
            "unique_actions": len(decision_patterns),
            "most_frequent": (
                max(
                    decision_patterns.items(),
                    key=lambda x: x[1]["frequency"],
                )[0]
                if decision_patterns
                else None
            ),
            "most_rewarding": (
                max(
                    decision_patterns.items(),
                    key=lambda x: x[1]["reward_stats"]["average"],
                )[0]
                if decision_patterns
                else None
            ),
            "action_diversity": -sum(
                p["frequency"] * math.log(p["frequency"])
                for p in decision_patterns.values()
                if p["frequency"] > 0
            ),  # Shannon entropy
        }
