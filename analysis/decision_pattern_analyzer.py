from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from database.data_types import DecisionPatterns, DecisionPatternStats, DecisionSummary
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository


class DecisionPatternAnalyzer:
    """
    Analyzes decision patterns from agent actions to identify behavioral trends and statistics.

    This class processes agent actions to extract meaningful patterns, frequencies, and reward statistics,
    providing insights into agent decision-making behavior. It can analyze patterns across different
    scopes (simulation, episode) and calculate various metrics including:
    - Action frequencies and reward statistics
    - Temporal trends and patterns
    - Decision diversity metrics
    - Co-occurrence patterns between different action types
    - Contribution metrics for each action type

    The analyzer helps understand:
    - How agents make decisions over time
    - Which actions are most frequent/rewarding
    - How diverse the decision-making process is
    - Whether there are temporal patterns or correlations between actions
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the DecisionPatternAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> DecisionPatterns:
        """
        Analyze decision patterns within the specified scope and parameters.

        This method processes agent actions to extract comprehensive pattern statistics including:
        - Frequency analysis of different action types
        - Reward statistics (mean, median, variance, etc.)
        - Temporal trends and patterns
        - Action diversity metrics
        - Co-occurrence patterns between actions

        Args:
            scope (Union[str, AnalysisScope]): The scope of analysis (e.g., SIMULATION, EPISODE).
                Determines the context in which patterns are analyzed.
            agent_id (Optional[int]): Specific agent ID to analyze. If None, analyzes all agents.
            step (Optional[int]): Specific step to analyze. If provided, only analyzes that step.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze (inclusive).
                Format: (start_step, end_step).

        Returns:
            DecisionPatterns: Object containing:
                - decision_patterns: List of DecisionPatternStats for each action type
                - decision_summary: Overall statistics including diversity metrics and co-occurrence patterns

        Example:
            >>> analyzer = DecisionPatternAnalyzer(repository)
            >>> patterns = analyzer.analyze(scope=AnalysisScope.EPISODE, agent_id=1)
            >>> print(patterns.decision_summary.total_decisions)
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        total_decisions = len(actions)

        # Track temporal patterns and sequences
        temporal_metrics = defaultdict(lambda: defaultdict(list))
        first_occurrences = {}
        action_sequences = []
        current_sequence = []
        current_step = None
        window_size = 10  # For rolling statistics

        decision_metrics = {}
        for action in sorted(actions, key=lambda x: x.step_number):
            action_type = action.action_type

            # Handle action sequences for co-occurrence
            if current_step != action.step_number:
                if current_sequence:
                    action_sequences.append(current_sequence)
                current_sequence = []
                current_step = action.step_number
            current_sequence.append(action_type)

            # Track first occurrence of each action type
            if action_type not in first_occurrences:
                first_occurrences[action_type] = {
                    "step": action.step_number,
                    "reward": action.reward or 0,
                }

            # Track temporal metrics
            temporal_metrics[action_type]["steps"].append(action.step_number)
            temporal_metrics[action_type]["rewards"].append(action.reward or 0)

            # Existing metrics tracking
            if action_type not in decision_metrics:
                decision_metrics[action_type] = {
                    "count": 0,
                    "rewards": [],
                }
            metrics = decision_metrics[action_type]
            metrics["count"] += 1
            metrics["rewards"].append(action.reward or 0)

        # Add final sequence if exists
        if current_sequence:
            action_sequences.append(current_sequence)

        # Calculate temporal trends for each action type
        temporal_trends = self._calculate_temporal_trends(temporal_metrics, window_size)

        patterns = [
            DecisionPatternStats(
                action_type=action_type,
                count=metrics["count"],
                frequency=(
                    metrics["count"] / total_decisions if total_decisions > 0 else 0
                ),
                reward_stats=self._calculate_reward_stats(metrics["rewards"]),
                contribution_metrics=self._calculate_contribution_metrics(
                    metrics["rewards"],
                    sum(m["count"] for m in decision_metrics.values()),
                    sum(sum(m["rewards"]) for m in decision_metrics.values()),
                ),
                temporal_stats=temporal_trends.get(action_type, {}),
                first_occurrence=first_occurrences.get(action_type, {}),
            )
            for action_type, metrics in decision_metrics.items()
        ]

        # Calculate co-occurrence matrix
        co_occurrence = self._calculate_co_occurrence(action_sequences)

        summary = DecisionSummary(
            total_decisions=total_decisions,
            unique_actions=len(decision_metrics),
            most_frequent=(
                max(patterns, key=lambda x: x.count).action_type if patterns else None
            ),
            most_rewarding=(
                max(patterns, key=lambda x: x.reward_stats["average"]).action_type
                if patterns
                else None
            ),
            action_diversity=self._calculate_diversity(patterns),
            normalized_diversity=self._calculate_normalized_diversity(patterns),
            co_occurrence_patterns=co_occurrence,
        )

        return DecisionPatterns(
            decision_patterns=patterns,
            decision_summary=summary,
        )

    def _calculate_reward_stats(self, rewards: List[float]) -> dict:
        """
        Calculate comprehensive reward statistics.

        Args:
            rewards (List[float]): List of reward values for a specific action type.

        Returns:
            dict: Dictionary containing various reward statistics.
        """
        if not rewards:
            return {
                "average": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "variance": 0,
                "std_dev": 0,
                "percentile_25": 0,
                "percentile_50": 0,
                "percentile_75": 0,
            }

        rewards_array = np.array(rewards)
        return {
            "average": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "variance": float(np.var(rewards_array)),
            "std_dev": float(np.std(rewards_array)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_50": float(np.percentile(rewards_array, 50)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
        }

    def _calculate_diversity(self, patterns: List[DecisionPatternStats]) -> float:
        """
        Calculate the diversity of decision patterns using Shannon entropy.

        This method quantifies how varied and balanced the agent's decision-making is.
        A higher value indicates more diverse and evenly distributed decision-making patterns,
        while a lower value suggests more focused or repetitive behavior.

        Args:
            patterns (List[DecisionPatternStats]): List of decision pattern statistics,
                each containing frequency information for different action types.

        Returns:
            float: Shannon entropy value representing decision diversity.
                - 0.0 indicates completely uniform decision-making
                - Higher values indicate more diverse decision-making
        """
        import math

        return -sum(
            p.frequency * math.log(p.frequency) if p.frequency > 0 else 0
            for p in patterns
        )

    def _calculate_contribution_metrics(
        self, rewards: List[float], total_actions: int, total_rewards: float
    ) -> dict:
        """
        Calculate metrics showing how this action contributes to overall diversity and rewards.

        Args:
            rewards (List[float]): List of rewards for this action type
            total_actions (int): Total number of actions across all types
            total_rewards (float): Sum of all rewards across all action types

        Returns:
            dict: Dictionary containing contribution metrics
        """
        if not rewards or total_actions == 0 or total_rewards == 0:
            return {"action_share": 0.0, "reward_share": 0.0, "reward_efficiency": 0.0}

        action_share = len(rewards) / total_actions
        reward_share = sum(rewards) / total_rewards
        reward_efficiency = reward_share / action_share if action_share > 0 else 0

        return {
            "action_share": action_share,
            "reward_share": reward_share,
            "reward_efficiency": reward_efficiency,
        }

    def _calculate_normalized_diversity(
        self, patterns: List[DecisionPatternStats]
    ) -> float:
        """
        Calculate normalized Shannon diversity index (0-1 scale).

        Args:
            patterns (List[DecisionPatternStats]): List of decision pattern statistics

        Returns:
            float: Normalized diversity index between 0 and 1
        """
        if not patterns:
            return 0.0

        raw_diversity = self._calculate_diversity(patterns)
        max_diversity = np.log(len(patterns)) if patterns else 0

        return raw_diversity / max_diversity if max_diversity > 0 else 0

    def _calculate_co_occurrence(
        self, action_sequences: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate the co-occurrence matrix for action types.

        Args:
            action_sequences (List[List[str]]): List of action sequences by step

        Returns:
            Dict[str, Dict[str, float]]: Co-occurrence frequencies and correlations
        """
        if not action_sequences:
            return {}

        # Count co-occurrences
        co_occurrences = defaultdict(lambda: defaultdict(int))
        action_counts = defaultdict(int)

        for sequence in action_sequences:
            # Count individual actions
            for action in sequence:
                action_counts[action] += 1

            # Count co-occurrences
            for i, action1 in enumerate(sequence):
                for action2 in sequence[i + 1 :]:
                    co_occurrences[action1][action2] += 1
                    co_occurrences[action2][action1] += 1

        # Calculate correlation coefficients
        total_steps = len(action_sequences)
        correlations = {}

        for action1 in action_counts:
            correlations[action1] = {}
            for action2 in action_counts:
                if action1 == action2:
                    continue

                # Calculate correlation coefficient
                p_a1 = action_counts[action1] / total_steps
                p_a2 = action_counts[action2] / total_steps
                p_both = co_occurrences[action1][action2] / total_steps

                # Calculate correlation (phi coefficient)
                numerator = p_both - (p_a1 * p_a2)
                denominator = np.sqrt(p_a1 * p_a2 * (1 - p_a1) * (1 - p_a2))
                correlation = numerator / denominator if denominator != 0 else 0

                correlations[action1][action2] = {
                    "count": co_occurrences[action1][action2],
                    "frequency": p_both,
                    "correlation": correlation,
                }

        return correlations

    def _calculate_temporal_trends(
        self, temporal_metrics: Dict[str, Dict[str, List]], window_size: int
    ) -> Dict[str, Dict[str, List]]:
        """
        Calculate temporal trends and patterns for each action type.

        Analyzes how action frequencies and rewards change over time, including:
        - Overall trends (increasing/decreasing)
        - Rolling statistics for smoothed analysis
        - Consistency of action selection
        - Periodicity in action patterns
        - Recent trend directions

        Args:
            temporal_metrics: Dictionary containing temporal data for each action type:
                {action_type: {'steps': [...], 'rewards': [...]}}
            window_size: Size of the rolling window for trend calculations

        Returns:
            Dictionary containing temporal analysis for each action type:
                {action_type: {
                    'frequency_trend': float,
                    'reward_trend': float,
                    'rolling_frequencies': List[float],
                    'rolling_rewards': List[float],
                    'consistency': float,
                    'periodicity': float,
                    'recent_trend': str
                }}
        """
        trends = {}

        for action_type, metrics in temporal_metrics.items():
            steps = metrics["steps"]
            rewards = metrics["rewards"]

            if not steps:
                continue

            # Convert to numpy arrays for calculations
            steps_array = np.array(steps)
            rewards_array = np.array(rewards)

            # Calculate frequency over time
            total_steps = max(steps) - min(steps) + 1
            step_counts = np.bincount(steps_array - min(steps_array))
            frequencies = step_counts / total_steps

            # Calculate rolling statistics
            rolling_freq = self._calculate_rolling_mean(frequencies, window_size)
            rolling_rewards = self._calculate_rolling_mean(rewards_array, window_size)

            # Calculate trends
            freq_trend = self._calculate_trend(frequencies)
            reward_trend = self._calculate_trend(rewards_array)

            trends[action_type] = {
                "frequency_trend": float(freq_trend),
                "reward_trend": float(reward_trend),
                "rolling_frequencies": rolling_freq.tolist(),
                "rolling_rewards": rolling_rewards.tolist(),
                "consistency": float(self._calculate_consistency(frequencies)),
                "periodicity": float(self._calculate_periodicity(frequencies)),
                "recent_trend": self._get_recent_trend(frequencies, window_size),
            }

        return trends

    def _calculate_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean with the specified window size."""
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate the overall trend using linear regression."""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope

    def _calculate_consistency(self, frequencies: np.ndarray) -> float:
        """Calculate the consistency of action frequency."""
        if len(frequencies) < 2:
            return 1.0
        return 1.0 - np.std(frequencies) / (np.mean(frequencies) + 1e-10)

    def _calculate_periodicity(self, frequencies: np.ndarray) -> float:
        """Calculate the periodicity of action frequency using autocorrelation."""
        if len(frequencies) < 2:
            return 0.0

        # Calculate autocorrelation
        mean = np.mean(frequencies)
        var = np.var(frequencies)
        normalized = frequencies - mean
        correlation = np.correlate(normalized, normalized, mode="full")
        correlation = correlation[len(correlation) // 2 :]

        # Find peaks in autocorrelation
        if var == 0:
            return 0.0

        correlation = correlation / (var * len(frequencies))
        peaks = self._find_peaks(correlation)

        return float(np.mean(peaks)) if peaks else 0.0

    def _find_peaks(self, data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Find peaks in the data above the threshold."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(data[i])
        return np.array(peaks)

    def _get_recent_trend(self, data: np.ndarray, window: int) -> str:
        """Determine the trend direction in the recent window."""
        if len(data) < window:
            return "stable"

        recent_data = data[-window:]
        trend = self._calculate_trend(recent_data)

        if trend > 0.1:
            return "increasing"
        elif trend < -0.1:
            return "decreasing"
        return "stable"
