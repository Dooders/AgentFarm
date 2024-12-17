from typing import List, Optional, Tuple, Union
import numpy as np

from analysis.decision_pattern_analyzer import DecisionPatternAnalyzer
from analysis.resource_impact_analyzer import ResourceImpactAnalyzer
from analysis.temporal_pattern_analyzer import TemporalPatternAnalyzer
from database.data_types import ActionMetrics
from database.enums import AnalysisScope
from database.repositories.action_repository import ActionRepository


class ActionStatsAnalyzer:
    """
    Analyzes statistics and patterns of agent actions in a simulation.

    This class processes action data to generate metrics including frequency, rewards,
    interaction rates, and various patterns of agent behavior. It now includes deeper
    statistical analysis of rewards including variance, standard deviation, median,
    quartiles, and confidence intervals.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data.
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the ActionStatsAnalyzer.

        Args:
            repository (AgentActionRepository): Repository instance for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """
        Analyze action statistics based on specified scope and filters.

        This method processes action data to:
        1. Calculate frequency and reward statistics for each action type
        2. Determine interaction rates and performance metrics
        3. Analyze temporal, resource, and decision-making patterns
        4. Compute detailed statistical measures of reward distributions

        Args:
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): Specific agent to analyze. Defaults to None.
            step (Optional[int], optional): Specific step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.

        Returns:
            List[ActionMetrics]: List of metrics for each action type containing:
                - action_type: Type of the analyzed action
                - count: Total occurrences of the action
                - frequency: Relative frequency (e.g., 0.3 means 30% of all actions)
                - avg_reward: Mean reward received (e.g., 2.5 means average reward of +2.5)
                - min_reward: Minimum reward received
                - max_reward: Maximum reward received
                - variance_reward: Variance of rewards
                - std_dev_reward: Standard deviation of rewards
                - median_reward: Median value of rewards
                - quartiles_reward: First and third quartiles [Q1, Q3]
                - confidence_interval: 95% confidence interval for avg_reward
                - interaction_rate: Proportion of actions involving other agents
                - solo_performance: Average reward for non-interactive actions
                - interaction_performance: Average reward for interactive actions
                - temporal_patterns: Timing and sequence patterns
                - resource_impacts: Resource utilization effects
                - decision_patterns: Decision-making patterns

        Example:
            For a complete analysis of "gather" actions, the result might look like:

            ActionMetrics(
                action_type="gather",
                count=100,
                frequency=0.4,                # 40% of all actions were gather
                avg_reward=2.5,               # Average reward of +2.5
                min_reward=0.0,               # Minimum reward received
                max_reward=5.0,               # Maximum reward received
                variance_reward=0.2,          # Variance of rewards
                std_dev_reward=0.447,         # Standard deviation of rewards
                median_reward=2.5,            # Median value of rewards
                quartiles_reward=[2.0, 3.0],  # First and third quartiles
                confidence_interval=0.087,     # 95% confidence interval
                interaction_rate=0.1,         # 10% of gather actions involved other agents
                solo_performance=2.7,         # Average reward when gathering alone
                interaction_performance=1.2,   # Average reward when gathering with others
                temporal_patterns=[...],      # See TemporalPatternAnalyzer
                resource_impacts=[...],       # See ResourceImpactAnalyzer
                decision_patterns=[...]       # See DecisionPatternAnalyzer
            )

        Note:
            - Frequency and rates are expressed as decimals between 0 and 1
            - Performance metrics are calculated only for actions with valid rewards
            - Patterns include detailed analysis of behavior sequences and context
            - Statistical measures require at least 2 data points for meaningful calculation
            - Confidence intervals are calculated at 95% confidence level
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        total_actions = len(actions)

        action_metrics = {}
        for action in actions:
            if action.action_type not in action_metrics:
                action_metrics[action.action_type] = {
                    "count": 0,
                    "total_reward": 0,
                    "min_reward": float("inf"),
                    "max_reward": float("-inf"),
                    "interaction_count": 0,
                    "interaction_reward": 0,
                    "solo_reward": 0,
                }
            metrics = action_metrics[action.action_type]
            metrics["count"] += 1
            metrics["total_reward"] += action.reward or 0
            metrics["min_reward"] = min(metrics["min_reward"], action.reward or 0)
            metrics["max_reward"] = max(metrics["max_reward"], action.reward or 0)
            if action.action_target_id:
                metrics["interaction_count"] += 1
                metrics["interaction_reward"] += action.reward or 0
            else:
                metrics["solo_reward"] += action.reward or 0

        temporal_patterns = TemporalPatternAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )
        resource_impacts = ResourceImpactAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )
        decision_patterns = DecisionPatternAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )

        # Collect rewards for each action type
        rewards_by_action = {}
        for action in actions:
            if action.action_type not in rewards_by_action:
                rewards_by_action[action.action_type] = []
            rewards_by_action[action.action_type].append(action.reward or 0)

        action_metrics_list = []
        for action_type, metrics in action_metrics.items():
            rewards = rewards_by_action[action_type]
            count = metrics["count"]
            avg_reward = metrics["total_reward"] / count if count > 0 else 0
            # New statistical calculations
            variance_reward = np.var(rewards) if count > 1 else 0
            std_dev_reward = np.std(rewards) if count > 1 else 0
            median_reward = np.median(rewards)
            quartiles_reward = (
                np.percentile(rewards, [25, 75]).tolist() if count > 1 else [0, 0]
            )
            # Confidence interval calculation (95% confidence level)
            confidence_interval = (
                1.96 * (std_dev_reward / np.sqrt(count)) if count > 1 else 0
            )

            action_metrics_list.append(
                ActionMetrics(
                    action_type=action_type,
                    count=count,
                    frequency=(
                        metrics["count"] / total_actions if total_actions > 0 else 0
                    ),
                    avg_reward=avg_reward,
                    min_reward=metrics["min_reward"],
                    max_reward=metrics["max_reward"],
                    variance_reward=variance_reward,
                    std_dev_reward=std_dev_reward,
                    median_reward=median_reward,
                    quartiles_reward=quartiles_reward,
                    confidence_interval=confidence_interval,
                    interaction_rate=(
                        metrics["interaction_count"] / count if count > 0 else 0
                    ),
                    solo_performance=(
                        metrics["solo_reward"] / (count - metrics["interaction_count"])
                        if (count - metrics["interaction_count"]) > 0
                        else 0
                    ),
                    interaction_performance=(
                        metrics["interaction_reward"] / metrics["interaction_count"]
                        if metrics["interaction_count"] > 0
                        else 0
                    ),
                    temporal_patterns=[
                        p for p in temporal_patterns if p.action_type == action_type
                    ],
                    resource_impacts=[
                        r for r in resource_impacts if r.action_type == action_type
                    ],
                    decision_patterns=[
                        d
                        for d in decision_patterns.decision_patterns
                        if d.action_type == action_type
                    ],
                )
            )

        return action_metrics_list
