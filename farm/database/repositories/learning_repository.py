from typing import Dict, List, Optional, Tuple

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from farm.database.data_types import AgentLearningStats, LearningProgress, ModulePerformance
from farm.database.models import LearningExperienceModel
from farm.database.repositories.base_repository import BaseRepository
from farm.database.scope_utils import filter_scope
from farm.database.session_manager import SessionManager


class LearningRepository(BaseRepository[LearningExperienceModel]):
    """
    Repository for managing and querying learning experience data.

    This class provides methods to access and aggregate learning-related data including
    progress metrics, module performance, and agent learning statistics. It handles the
    persistence and retrieval of learning experiences from the database.

    Attributes:
        session_manager (SessionManager): Manager for database sessions and transactions

    Examples:
        >>> repo = LearningRepository(session_manager)
        >>> progress = repo.get_learning_progress(session, scope="episode")
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize repository with session manager."""
        super().__init__(session_manager, LearningExperienceModel)

    def get_learning_progress(
        self,
        session: Session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[LearningProgress]:
        """
        Retrieve learning progress metrics aggregated over time steps.

        Calculates average rewards, action counts, and unique action usage for each step
        in the learning process. Results can be filtered by agent, specific steps, or
        step ranges.

        Args:
            session (Session): Active database session for executing queries
            scope (str): Analysis scope identifier ('simulation', 'episode', etc.)
            agent_id (Optional[int]): ID of specific agent to analyze. If None, includes all agents
            step (Optional[int]): Specific step number to analyze. If None, includes all steps
            step_range (Optional[Tuple[int, int]]): Start and end step numbers to analyze

        Returns:
            List[LearningProgress]: Ordered list of progress metrics per step, containing:
                - step: Step number
                - reward: Average reward for the step
                - action_count: Total number of actions taken
                - unique_actions: Number of distinct actions used

        Examples:
            >>> progress = repo.get_learning_progress(session, "episode", agent_id=1)
            >>> print(f"Step {progress[0].step}: Reward={progress[0].reward}")
        """
        query = session.query(
            LearningExperienceModel.step_number,
            func.avg(LearningExperienceModel.reward).label("avg_reward"),
            func.count(LearningExperienceModel.action_taken).label("action_count"),
            func.count(distinct(LearningExperienceModel.action_taken_mapped)).label(
                "unique_actions"
            ),
        ).group_by(LearningExperienceModel.step_number)

        query = filter_scope(query, scope, agent_id, step, step_range)
        results = query.order_by(LearningExperienceModel.step_number).all()

        return [
            LearningProgress(
                step=step,
                reward=float(reward or 0),
                action_count=int(count or 0),
                unique_actions=int(unique or 0),
            )
            for step, reward, count, unique in results
        ]

    def get_module_performance(
        self,
        session: Session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, ModulePerformance]:
        """
        Calculate aggregate performance metrics for each learning module.

        Computes module-specific statistics including average rewards and action usage patterns.
        Results are grouped by module type and ID to allow comparison across different
        modules.

        Args:
            session (Session): Active database session for executing queries
            scope (str): Analysis scope identifier ('simulation', 'episode', etc.)
            agent_id (Optional[int]): Filter results for specific agent
            step (Optional[int]): Filter results for specific step
            step_range (Optional[Tuple[int, int]]): Filter results for step range

        Returns:
            Dict[str, ModulePerformance]: Dictionary mapping module types to their performance metrics:
                - module_type: Type identifier of the module
                - module_id: Unique identifier of the module instance
                - avg_reward: Mean reward achieved by the module
                - total_actions: Total number of actions taken
                - unique_actions: Number of distinct actions used

        Examples:
            >>> perf = repo.get_module_performance(session, "simulation")
            >>> for module, stats in perf.items():
            ...     print(f"{module}: avg_reward={stats.avg_reward:.2f}")
        """
        query = session.query(
            LearningExperienceModel.module_type,
            LearningExperienceModel.module_id,
            func.avg(LearningExperienceModel.reward).label("avg_reward"),
            func.count(LearningExperienceModel.action_taken).label("total_actions"),
            func.count(distinct(LearningExperienceModel.action_taken_mapped)).label(
                "unique_actions"
            ),
        ).group_by(
            LearningExperienceModel.module_type,
            LearningExperienceModel.module_id,
        )

        query = filter_scope(query, scope, agent_id, step, step_range)
        results = query.all()

        return {
            f"{module_type}": ModulePerformance(
                module_type=module_type,
                module_id=module_id,
                avg_reward=float(avg_reward or 0),
                total_actions=int(total_actions or 0),
                unique_actions=int(unique_actions or 0),
            )
            for module_type, module_id, avg_reward, total_actions, unique_actions in results
        }

    def get_agent_learning_stats(
        self,
        session: Session,
        agent_id: Optional[int] = None,
        scope: str = "simulation",
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, AgentLearningStats]:
        """
        Get comprehensive learning statistics for individual agents.

        Aggregates performance metrics and action usage patterns per agent and module type.
        Useful for analyzing agent behavior and learning effectiveness over time.

        Args:
            session (Session): Active database session for executing queries
            agent_id (Optional[int]): Filter results for specific agent. If None, includes all agents
            scope (str): Analysis scope, defaults to "simulation"
            step (Optional[int]): Filter results for specific step
            step_range (Optional[Tuple[int, int]]): Filter results for step range

        Returns:
            Dict[str, AgentLearningStats]: Dictionary mapping module types to agent statistics:
                - agent_id: Identifier of the agent
                - reward_mean: Average reward achieved
                - total_actions: Total number of actions taken
                - actions_used: List of unique action identifiers used

        Examples:
            >>> stats = repo.get_agent_learning_stats(session, agent_id=1)
            >>> for module, agent_stats in stats.items():
            ...     print(f"{module}: {len(agent_stats.actions_used)} unique actions")
        """
        query = session.query(
            LearningExperienceModel.agent_id,
            LearningExperienceModel.module_type,
            func.avg(LearningExperienceModel.reward).label("reward_mean"),
            func.count(LearningExperienceModel.action_taken).label("total_actions"),
            func.group_concat(
                distinct(LearningExperienceModel.action_taken_mapped)
            ).label("actions_used"),
        ).group_by(
            LearningExperienceModel.agent_id,
            LearningExperienceModel.module_type,
        )

        query = filter_scope(query, scope, agent_id, step, step_range)
        results = query.all()

        return {
            f"{module_type}": AgentLearningStats(
                agent_id=agent_id,
                reward_mean=float(reward_mean or 0),
                total_actions=int(total_actions or 0),
                actions_used=actions_used.split(",") if actions_used else [],
            )
            for agent_id, module_type, reward_mean, total_actions, actions_used in results
        }

    def get_learning_experiences(
        self,
        session: Session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[LearningExperienceModel]:
        """
        Retrieve learning experiences with filtering.

        Args:
            session: Database session
            scope: Analysis scope
            agent_id: Filter for specific agent
            step: Filter for specific step
            step_range: Filter for step range

        Returns:
            List[LearningExperienceModel]: Filtered learning experiences
        """
        query = session.query(LearningExperienceModel)
        query = filter_scope(query, scope, agent_id, step, step_range)
        return query.all()
