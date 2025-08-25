from typing import Dict, List, Optional, Tuple, Union

from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.analyzers.behavior_clustering_analyzer import (
    BehaviorClusteringAnalyzer,
)
from farm.database.analyzers.causal_analyzer import CausalAnalyzer
from farm.database.analyzers.decision_pattern_analyzer import DecisionPatternAnalyzer
from farm.database.analyzers.resource_impact_analyzer import ResourceImpactAnalyzer
from farm.database.analyzers.sequence_pattern_analyzer import SequencePatternAnalyzer
from farm.database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from farm.database.data_types import (
    ActionMetrics,
    BehaviorClustering,
    CausalAnalysis,
    DecisionPatterns,
    ResourceImpact,
    SequencePattern,
    TimePattern,
)
from farm.database.enums import AnalysisScope
from farm.database.repositories.action_repository import ActionRepository


class ActionsService:
    """
    High-level service for analyzing agent actions using various analyzers.

    This service orchestrates different types of analysis on agent actions including:
    - Basic action statistics and metrics
    - Behavioral patterns and clustering
    - Causal relationships
    - Decision patterns
    - Resource impacts
    - Action sequences
    - Temporal patterns

    Attributes:
        action_repository (AgentActionRepository): Repository for accessing agent action data
        stats_analyzer (ActionStatsAnalyzer): Analyzer for basic action statistics
        behavior_analyzer (BehaviorClusteringAnalyzer): Analyzer for behavioral patterns
        causal_analyzer (CausalAnalyzer): Analyzer for causal relationships
        decision_analyzer (DecisionPatternAnalyzer): Analyzer for decision patterns
        resource_analyzer (ResourceImpactAnalyzer): Analyzer for resource impacts
        sequence_analyzer (SequencePatternAnalyzer): Analyzer for action sequences
        temporal_analyzer (TemporalPatternAnalyzer): Analyzer for temporal patterns
    """

    def __init__(self, action_repository: ActionRepository):
        """
        Initialize the ActionsService with required analyzers.

        Args:
            action_repository (AgentActionRepository): Repository for accessing agent action data
        """
        self.action_repository = action_repository

        # Initialize analyzers
        self.stats_analyzer = ActionStatsAnalyzer(action_repository)
        self.behavior_analyzer = BehaviorClusteringAnalyzer(action_repository)
        self.causal_analyzer = CausalAnalyzer(action_repository)
        self.decision_analyzer = DecisionPatternAnalyzer(action_repository)
        self.resource_analyzer = ResourceImpactAnalyzer(action_repository)
        self.sequence_analyzer = SequencePatternAnalyzer(action_repository)
        self.temporal_analyzer = TemporalPatternAnalyzer(action_repository)

    def analyze_actions(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[
        str,
        Union[
            List[ActionMetrics],
            BehaviorClustering,
            List[CausalAnalysis],
            DecisionPatterns,
            List[ResourceImpact],
            List[SequencePattern],
            List[TimePattern],
        ],
    ]:
        """
        Perform comprehensive analysis of agent actions.

        Args:
            scope (Union[str, AnalysisScope]): Scope of analysis. Defaults to SIMULATION.
            agent_id (Optional[int]): Specific agent to analyze. Defaults to None.
            step (Optional[int]): Specific step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze. Defaults to None.
            analysis_types (Optional[List[str]]): List of analysis types to perform.
                Available types: ['stats', 'behavior', 'causal', 'decision', 'resource',
                'sequence', 'temporal']. Defaults to None (all types).

        Returns:
            Dict containing results from each requested analysis type
        """
        if analysis_types is None:
            analysis_types = [
                "stats",
                "behavior",
                "causal",
                "decision",
                "resource",
                "sequence",
                "temporal",
            ]

        results = {}

        # Basic action statistics
        if "stats" in analysis_types:
            results["action_stats"] = self.stats_analyzer.analyze(
                scope, str(agent_id) if agent_id is not None else None, step, step_range
            )

        # Behavioral clustering
        if "behavior" in analysis_types:
            behavior_agent_id = str(agent_id) if agent_id is not None else None
            behavior_step = step if step is not None else 0
            behavior_step_range = step_range if step_range is not None else ()
            # Type: ignore because analyzer has incorrect type annotation (should be Optional[str])
            results["behavior_clusters"] = self.behavior_analyzer.analyze(scope, behavior_agent_id, behavior_step, behavior_step_range)  # type: ignore

        # Causal analysis
        if "causal" in analysis_types:
            action_types = self._get_unique_action_types(
                scope, agent_id, step, step_range
            )
            results["causal_analysis"] = [
                self.causal_analyzer.analyze(action_type, scope, agent_id, step_range)
                for action_type in action_types
            ]

        # Decision patterns
        if "decision" in analysis_types:
            results["decision_patterns"] = self.decision_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Resource impacts
        if "resource" in analysis_types:
            results["resource_impacts"] = self.resource_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Sequence patterns
        if "sequence" in analysis_types:
            results["sequence_patterns"] = self.sequence_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Temporal patterns
        if "temporal" in analysis_types:
            results["temporal_patterns"] = self.temporal_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        return results

    def _get_unique_action_types(
        self,
        scope: Union[str, AnalysisScope],
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        """
        Get unique action types from the repository based on specified criteria.

        Args:
            scope (Union[str, AnalysisScope]): Scope of analysis
            agent_id (Optional[int]): Specific agent ID. Defaults to None.
            step (Optional[int]): Specific step. Defaults to None.
            step_range (Optional[Tuple[int, int]]): Range of steps. Defaults to None.

        Returns:
            List[str]: List of unique action types
        """
        actions = self.action_repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        return list(set(action.action_type for action in actions))

    def get_action_summary(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get a high-level summary of agent actions and their effectiveness.

        Args:
            scope (Union[str, AnalysisScope]): Scope of analysis. Defaults to SIMULATION.
            agent_id (Optional[int]): Specific agent to analyze. Defaults to None.

        Returns:
            Dict containing summary metrics for each action type:
                - success_rate: Percentage of actions with positive rewards
                - avg_reward: Average reward per action
                - frequency: Relative frequency of the action
                - resource_efficiency: Average resource gain/loss per action
        """
        action_stats = self.stats_analyzer.analyze(
            scope, str(agent_id) if agent_id is not None else None
        )
        resource_impacts = self.resource_analyzer.analyze(scope, agent_id)

        summary = {}
        for stat in action_stats:
            action_type = stat.action_type
            resource_impact = next(
                (r for r in resource_impacts if r.action_type == action_type), None
            )

            summary[action_type] = {
                "success_rate": (
                    len([r for r in stat.rewards if r > 0]) / stat.count
                    if stat.count > 0
                    else 0
                ),
                "avg_reward": stat.avg_reward,
                "frequency": stat.frequency,
                "resource_efficiency": (
                    resource_impact.resource_efficiency if resource_impact else 0
                ),
            }

        return summary
