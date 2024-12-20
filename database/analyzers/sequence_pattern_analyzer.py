from typing import List, Optional, Tuple, Union

from database.data_types import SequencePattern
from database.enums import AnalysisScope
from database.repositories.action_repository import ActionRepository


class SequencePatternAnalyzer:
    """
    A class that analyzes sequences of agent actions to identify patterns and their probabilities.

    This analyzer looks for consecutive pairs of actions performed by the same agent and
    calculates the frequency and probability of these action sequences occurring.
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the SequencePatternAnalyzer with a repository.

        Args:
            repository (AgentActionRepository): The repository containing agent actions data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[SequencePattern]:
        """
        Analyze agent actions to identify sequence patterns within the specified scope and constraints.

        Args:
            scope (Union[str, AnalysisScope]): The scope of analysis (e.g., SIMULATION, EPISODE).
                Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int]): If provided, limit analysis to actions of this specific agent.
                Defaults to None.
            step (Optional[int]): If provided, limit analysis to actions at this specific step.
                Defaults to None.
            step_range (Optional[Tuple[int, int]]): If provided, limit analysis to actions within
                this step range (inclusive). Defaults to None.

        Returns:
            List[SequencePattern]: A list of identified sequence patterns, each containing:
                - sequence: The action sequence (e.g., "action1->action2")
                - count: Number of times this sequence occurred
                - probability: Probability of the second action following the first action

        Example:
            analyzer = SequencePatternAnalyzer(repository)
            patterns = analyzer.analyze(scope="EPISODE", agent_id=1)
            # Returns patterns like: [SequencePattern(sequence="MOVE->ATTACK", count=5, probability=0.25)]
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        sequences = {}
        action_counts = {}

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            if current.agent_id == next_action.agent_id:
                sequence_key = f"{current.action_type}->{next_action.action_type}"
                sequences[sequence_key] = sequences.get(sequence_key, 0) + 1
                action_counts[current.action_type] = (
                    action_counts.get(current.action_type, 0) + 1
                )

        return [
            SequencePattern(
                sequence=sequence,
                count=count,
                probability=(
                    count / action_counts[sequence.split("->")[0]]
                    if sequence.split("->")[0] in action_counts
                    else 0
                ),
            )
            for sequence, count in sequences.items()
        ]
