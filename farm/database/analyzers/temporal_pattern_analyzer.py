from typing import List, Optional, Tuple, Union

import numpy as np

from farm.database.analyzers.analysis_utils import calculate_rolling_mean
from farm.database.data_types import EventSegment, TimePattern
from farm.database.enums import AnalysisScope
from farm.database.repositories.action_repository import ActionRepository


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in agent actions over time, including rolling averages and event segmentation.

    This class processes agent actions to identify patterns in their occurrence
    and associated rewards across different time periods. It provides functionality
    for analyzing action frequencies, reward distributions, and temporal segmentation
    of agent behaviors.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data.
            Handles all database interactions for retrieving agent actions.

    Example:
        >>> repo = AgentActionRepository()
        >>> analyzer = TemporalPatternAnalyzer(repo)
        >>> patterns = analyzer.analyze(scope='simulation', time_period_size=100)
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the TemporalPatternAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[str] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        time_period_size: int = 100,
        rolling_window_size: int = 10,
    ) -> List[TimePattern]:
        """
        Analyze temporal patterns in agent actions, including rolling averages.

        This method processes agent actions to:
        1. Identify patterns in action occurrence over time
        2. Calculate reward progression across time periods
        3. Generate rolling averages over specified window sizes

        The analysis divides the timeline into periods of specified size and
        calculates various metrics for each period, including action frequencies
        and average rewards.

        Args:
            scope (Union[str, AnalysisScope]): Scope of analysis (e.g., 'simulation', 'episode').
                Determines the context in which actions are analyzed.
            agent_id (Optional[str]): ID of specific agent to analyze. If None,
                analyzes actions from all agents.
            step (Optional[int]): Specific step to analyze. If provided, only
                analyzes actions at this step.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze,
                specified as (start_step, end_step). If None, analyzes all steps.
            time_period_size (int): Size of each time period in steps. Determines
                the granularity of the analysis. Larger values provide more
                aggregated results.
            rolling_window_size (int): Size of window for calculating rolling
                averages. Larger windows produce smoother trends but may miss
                short-term variations.

        Returns:
            List[TimePattern]: List of temporal patterns for each action type.
            Each TimePattern contains:
                - action_type: The type of action analyzed
                - time_distribution: Action frequencies per time period
                - reward_progression: Average rewards per time period
                - rolling_average_rewards: Smoothed reward progression
                - rolling_average_counts: Smoothed action frequencies

        Example:
            >>> patterns = analyzer.analyze(
            ...     scope='episode',
            ...     agent_id=1,
            ...     time_period_size=50,
            ...     rolling_window_size=5
            ... )
            >>> for pattern in patterns:
            ...     print(f"Action: {pattern.action_type}")
            ...     print(f"Average reward: {np.mean(pattern.reward_progression)}")
        """
        actions = self.repository.get_actions_by_scope(
            scope,
            str(agent_id) if agent_id is not None else None,
            step_range=step_range,
        )
        patterns = {}
        max_time_period = 0

        for action in actions:
            if action.action_type not in patterns:
                patterns[action.action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                    "steps": [],
                }

            time_period = action.step_number // time_period_size
            while len(patterns[action.action_type]["time_distribution"]) <= time_period:
                patterns[action.action_type]["time_distribution"].append(0)
                patterns[action.action_type]["reward_progression"].append(0)
                max_time_period = max(max_time_period, time_period)

            patterns[action.action_type]["time_distribution"][time_period] += 1
            patterns[action.action_type]["reward_progression"][time_period] += (
                action.reward or 0
            )
            patterns[action.action_type]["steps"].append(action.step_number)

        # Calculate average rewards per time period
        for action_type, data in patterns.items():
            for i in range(len(data["reward_progression"])):
                if data["time_distribution"][i] > 0:
                    data["reward_progression"][i] /= data["time_distribution"][i]

        # Calculate rolling averages
        for action_type, data in patterns.items():
            rewards = data["reward_progression"]
            counts = data["time_distribution"]
            # Extend counts and rewards to match max_time_period
            counts += [0] * (max_time_period + 1 - len(counts))
            rewards += [0] * (max_time_period + 1 - len(rewards))
            # Use utility functions for rolling average calculation
            rewards_array = np.array(rewards)
            counts_array = np.array(counts)
            rolling_rewards = calculate_rolling_mean(rewards_array, rolling_window_size)
            rolling_counts = calculate_rolling_mean(counts_array, rolling_window_size)
            data["rolling_average_rewards"] = rolling_rewards.tolist()
            data["rolling_average_counts"] = rolling_counts.tolist()

        return [
            TimePattern(
                action_type=action_type,
                time_distribution=data["time_distribution"],
                reward_progression=data["reward_progression"],
                rolling_average_rewards=data.get("rolling_average_rewards", []),
                rolling_average_counts=data.get("rolling_average_counts", []),
            )
            for action_type, data in patterns.items()
        ]

    def segment_events(
        self,
        event_steps: List[int],
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[EventSegment]:
        """
        Segment metrics before and after key events.

        Divides the timeline into segments based on specified event steps and
        calculates metrics for each segment. This is useful for analyzing how
        agent behavior changes around significant events.

        Args:
            event_steps (List[int]): List of step numbers where events occur.
                These steps mark the boundaries between segments.
            scope (Union[str, AnalysisScope]): Scope of analysis (e.g., 'simulation', 'episode').
                Determines the context in which actions are analyzed.
            agent_id (Optional[int]): ID of specific agent to analyze. If None,
                analyzes actions from all agents.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze,
                specified as (start_step, end_step). If None, analyzes all steps.

        Returns:
            List[EventSegment]: List of metrics segmented by events. Each EventSegment contains:
                - start_step: Beginning of the segment
                - end_step: End of the segment
                - action_counts: Dictionary of action frequencies in the segment
                - average_rewards: Dictionary of mean rewards per action type

        Example:
            >>> event_steps = [100, 200, 300]
            >>> segments = analyzer.segment_events(
            ...     event_steps=event_steps,
            ...     scope='simulation'
            ... )
            >>> for segment in segments:
            ...     print(f"Segment: {segment.start_step} to {segment.end_step}")
            ...     print(f"Action counts: {segment.action_counts}")
        """
        actions = self.repository.get_actions_by_scope(
            scope,
            str(agent_id) if agent_id is not None else None,
            step_range=step_range,
        )
        # Sort events to ensure correct segmentation
        event_steps = sorted(event_steps)
        segments = []
        last_event_step: int = 0
        for event_step in event_steps + [None]:  # Add None to capture last segment
            segment_actions = [
                a
                for a in actions
                if last_event_step
                <= int(getattr(a, "step_number", 0))
                < (event_step if event_step is not None else float("inf"))
            ]
            # Calculate metrics for this segment
            action_counts = {}
            total_rewards = {}
            for action in segment_actions:
                action_type = action.action_type
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                total_rewards[action_type] = total_rewards.get(action_type, 0) + (
                    action.reward or 0
                )
            average_rewards = {
                action_type: total_rewards[action_type] / action_counts[action_type]
                for action_type in action_counts
            }
            segments.append(
                EventSegment(
                    start_step=last_event_step,
                    end_step=event_step,
                    action_counts=action_counts,
                    average_rewards=average_rewards,
                )
            )
            last_event_step = event_step if event_step is not None else -1

        return segments
