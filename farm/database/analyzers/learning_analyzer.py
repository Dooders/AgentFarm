from typing import Dict, List, Optional, Union

import pandas as pd

from database.data_types import (
    AgentLearningStats,
    LearningEfficiencyMetrics,
    LearningProgress,
    LearningStatistics,
    ModulePerformance,
)
from database.enums import AnalysisScope
from database.repositories.learning_repository import LearningRepository


class LearningAnalyzer:
    """
    Analyzes learning-related data and patterns in agent behavior.

    This class provides comprehensive analysis of learning experiences, module performance,
    and adaptation patterns throughout the simulation. It processes learning metrics to
    understand agent improvement and behavioral adaptation.

    Attributes:
        repository (LearningRepository): Repository instance for accessing learning data.

    Example:
        >>> analyzer = LearningAnalyzer(learning_repository)
        >>> stats = analyzer.analyze_comprehensive_statistics()
        >>> print(f"Average reward: {stats.learning_efficiency.reward_efficiency:.2f}")
    """

    def __init__(self, repository: LearningRepository):
        """Initialize the LearningAnalyzer with a LearningRepository."""
        self.repository = repository

    def analyze_learning_progress(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[tuple[int, int]] = None,
    ) -> List[LearningProgress]:
        """
        Calculate aggregated learning progress metrics over time.

        Retrieves and aggregates learning metrics for each simulation step, including
        rewards earned and action patterns.

        Args:
            scope: Analysis scope (e.g., 'simulation', 'episode')
            agent_id: Filter results for specific agent ID
            step: Filter results for specific simulation step
            step_range: Filter results for step range (start, end)

        Returns:
            List[LearningProgress]: List of learning progress metrics per step, containing:
                - step: Step number in the simulation
                - reward: Average reward achieved in this step
                - action_count: Total number of actions taken
                - unique_actions: Number of distinct actions used

        Example:
            >>> progress = analyzer.analyze_learning_progress()
            >>> for p in progress[:3]:
            ...     print(f"Step {p.step}: {p.reward:.2f} reward, {p.unique_actions} actions")
        """
        return self.repository.get_learning_progress(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_module_performance(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[tuple[int, int]] = None,
    ) -> Dict[str, ModulePerformance]:
        """
        Calculate performance metrics for each learning module type.

        Aggregates and analyzes performance data for each unique learning module,
        including rewards, action counts, and action diversity metrics.

        Args:
            scope: Analysis scope for data filtering
            agent_id: Filter results for specific agent ID
            step: Filter results for specific simulation step
            step_range: Filter results for step range (start, end)

        Returns:
            Dict[str, ModulePerformance]: Dictionary mapping module identifiers to their
            performance metrics, containing:
                - module_type: Type of learning module
                - module_id: Unique identifier for the module
                - avg_reward: Average reward achieved by the module
                - total_actions: Total number of actions taken
                - unique_actions: Number of distinct actions used

        Example:
            >>> performance = analyzer.analyze_module_performance()
            >>> for module, stats in performance.items():
            ...     print(f"{module}: {stats.avg_reward:.2f} avg reward")
        """
        return self.repository.get_module_performance(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

    def analyze_agent_learning_stats(
        self,
        agent_id: Optional[int] = None,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        step: Optional[int] = None,
        step_range: Optional[tuple[int, int]] = None,
    ) -> Dict[str, AgentLearningStats]:
        """
        Get learning statistics for specific agent or all agents.

        Retrieves and analyzes learning performance metrics either for a specific
        agent or aggregated across all agents.

        Args:
            agent_id: If provided, limits analysis to specific agent
            scope: Analysis scope for data filtering
            step: Filter results for specific simulation step
            step_range: Filter results for step range (start, end)

        Returns:
            Dict[str, AgentLearningStats]: Dictionary mapping agent/module combinations
            to their statistics:
                - reward_mean: Average reward achieved
                - total_actions: Total number of actions taken
                - actions_used: List of unique actions performed

        Example:
            >>> stats = analyzer.analyze_agent_learning_stats(agent_id=1)
            >>> for module, agent_stats in stats.items():
            ...     print(f"{module}: {agent_stats.reward_mean:.2f} mean reward")
        """
        return self.repository.get_agent_learning_stats(
            self.repository.session_manager.create_session(),
            agent_id=agent_id,
            scope=scope,
            step=step,
            step_range=step_range,
        )

    def analyze_learning_efficiency(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[tuple[int, int]] = None,
    ) -> LearningEfficiencyMetrics:
        """
        Calculate learning efficiency metrics.

        Computes various efficiency metrics to evaluate the overall learning
        performance and stability of the system.

        Args:
            scope: Analysis scope for data filtering
            agent_id: Filter results for specific agent ID
            step: Filter results for specific simulation step
            step_range: Filter results for step range (start, end)

        Returns:
            LearningEfficiencyMetrics: Efficiency metrics containing:
                - reward_efficiency: Average reward across all experiences (0.0 to 1.0)
                - action_diversity: Ratio of unique actions to total actions (0.0 to 1.0)
                - learning_stability: Measure of consistency in learning performance (0.0 to 1.0)

        Example:
            >>> efficiency = analyzer.analyze_learning_efficiency()
            >>> print(f"Reward efficiency: {efficiency.reward_efficiency:.2%}")
            >>> print(f"Action diversity: {efficiency.action_diversity:.2%}")
        """
        experiences = self.repository.get_learning_experiences(
            self.repository.session_manager.create_session(),
            scope=scope,
            agent_id=agent_id,
            step=step,
            step_range=step_range,
        )

        if not experiences:
            return LearningEfficiencyMetrics(
                reward_efficiency=0.0,
                action_diversity=0.0,
                learning_stability=0.0,
            )

        # Convert to DataFrame for analysis
        df = pd.DataFrame(experiences)

        # Calculate metrics
        reward_efficiency = df["reward"].mean()

        # Calculate action diversity (unique actions / total actions)
        total_actions = len(df)
        unique_actions = df["action_taken_mapped"].nunique()
        action_diversity = unique_actions / total_actions if total_actions > 0 else 0

        # Calculate learning stability (inverse of reward variance)
        reward_variance = df.groupby("module_type")["reward"].var().mean()
        learning_stability = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0

        return LearningEfficiencyMetrics(
            reward_efficiency=float(reward_efficiency or 0),
            action_diversity=float(action_diversity or 0),
            learning_stability=float(learning_stability or 0),
        )

    def analyze_comprehensive_statistics(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[tuple[int, int]] = None,
    ) -> LearningStatistics:
        """
        Generate a comprehensive learning statistics report.

        Combines multiple analysis methods to create a complete picture of
        learning performance, including progress over time, module-specific
        metrics, and efficiency measures.

        Args:
            scope: Analysis scope for data filtering
            agent_id: Filter results for specific agent ID
            step: Filter results for specific simulation step
            step_range: Filter results for step range (start, end)

        Returns:
            LearningStatistics: Complete learning statistics including:
                - learning_progress: Time series of rewards and losses
                - module_performance: Per-module performance metrics
                - agent_learning_stats: Per-agent learning statistics
                - learning_efficiency: Overall efficiency metrics

        Example:
            >>> stats = analyzer.analyze_comprehensive_statistics()
            >>> print(f"Total modules: {len(stats.module_performance)}")
            >>> print(f"Learning stability: {stats.learning_efficiency.learning_stability:.2%}")
        """
        return LearningStatistics(
            learning_progress=self.analyze_learning_progress(
                scope, agent_id, step, step_range
            ),
            module_performance=self.analyze_module_performance(
                scope, agent_id, step, step_range
            ),
            agent_learning_stats=self.analyze_agent_learning_stats(
                agent_id, scope, step, step_range
            ),
            learning_efficiency=self.analyze_learning_efficiency(
                scope, agent_id, step, step_range
            ),
        )
