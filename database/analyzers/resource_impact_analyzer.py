from typing import List, Optional, Tuple, Union

from database.data_types import ResourceImpact
from database.enums import AnalysisScope
from database.repositories.action_repository import ActionRepository


class ResourceImpactAnalyzer:
    """
    Analyzes the resource impact of agent actions in a simulation.

    This class processes agent actions to calculate various resource-related metrics
    such as average resource changes and efficiency for different action types.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data.
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the ResourceImpactAnalyzer.

        Args:
            repository (AgentActionRepository): Repository instance for querying agent actions.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceImpact]:
        """
        Analyze resource impacts of agent actions based on specified criteria.

        This method aggregates and analyzes resource changes across different action types,
        calculating average resource states and changes.

        Args:
            scope (Union[str, AnalysisScope]): The scope of analysis (e.g., simulation, episode).
                Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int]): ID of the specific agent to analyze. If None, analyzes all agents.
            step (Optional[int]): Specific simulation step to analyze. If None, analyzes all steps.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze (inclusive).
                If None, analyzes all steps.

        Returns:
            List[ResourceImpact]: List of ResourceImpact objects containing analysis results for each
            action type, including:
                - Average resources before the action
                - Average resource change
                - Resource efficiency metrics
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        impacts = {}

        for action in actions:
            if action.action_type not in impacts:
                impacts[action.action_type] = {
                    "total_resources_before": 0,
                    "total_resource_change": 0,
                    "count": 0,
                }

            impact = impacts[action.action_type]
            impact["total_resources_before"] += action.resources_before or 0
            impact["total_resource_change"] += (action.resources_after or 0) - (
                action.resources_before or 0
            )
            impact["count"] += 1

        return [
            ResourceImpact(
                action_type=action_type,
                avg_resources_before=(
                    impact["total_resources_before"] / impact["count"]
                    if impact["count"] > 0
                    else 0
                ),
                avg_resource_change=(
                    impact["total_resource_change"] / impact["count"]
                    if impact["count"] > 0
                    else 0
                ),
                resource_efficiency=(
                    impact["total_resource_change"] / impact["count"]
                    if impact["count"] > 0
                    else 0
                ),
            )
            for action_type, impact in impacts.items()
        ]
