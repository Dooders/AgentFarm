from typing import Any, Dict, List, Optional, Tuple

from farm.database.analyzers.analysis_utils import (
    _calculate_correlation,
    _normalize_dict,
)
from farm.database.analyzers.spatial_analysis import SpatialAnalyzer
from farm.database.data_types import (
    AdversarialInteractionAnalysis,
    BasicAgentInfo,
    CollaborativeInteractionAnalysis,
    ConflictAnalysis,
    CounterfactualAnalysis,
    EnvironmentalImpactAnalysis,
    ExplorationExploitation,
    LearningCurveAnalysis,
    ResilienceAnalysis,
    RiskRewardAnalysis,
)
from farm.database.repositories.agent_repository import AgentRepository


class AgentAnalysis:
    def __init__(self, repository: AgentRepository) -> None:
        """Initialize the AgentAnalysis with a repository interface.

        Args:
            repository (AgentRepository): Repository interface for accessing and querying agent data.
        """
        self.repository = repository

    def analyze(self, agent_id: str) -> BasicAgentInfo:
        """Analyze and retrieve basic information for a specific agent.

        Retrieves and compiles fundamental agent information including identification,
        temporal data, resource metrics, and genealogical information.

        Args:
            agent_id (str): Unique identifier of the agent to analyze.

        Returns:
            BasicAgentInfo: Data object containing:
                - agent_id: Unique identifier
                - agent_type: Classification of the agent
                - birth_time: Timestamp of agent creation
                - death_time: Timestamp of agent termination (if applicable)
                - lifespan: Duration between birth and death
                - initial_resources: Starting resource allocation
                - starting_health: Initial health value
                - starvation_threshold: Minimum resource threshold
                - generation: Genealogical generation number
                - genome_id: Unique identifier of agent's genome
        """
        agent = self.repository.get_agent_by_id(agent_id)
        #! add final metrics like reward, health, reproduced, consumed, etc.
        return BasicAgentInfo(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            birth_time=agent.birth_time,
            death_time=agent.death_time,
            lifespan=(
                (agent.death_time - agent.birth_time) if agent.death_time else None
            ),
            initial_resources=agent.initial_resources,
            starting_health=agent.starting_health,
            starvation_threshold=agent.starvation_threshold,
            # Genealogy
            generation=agent.generation,
            genome_id=agent.genome_id,
        )

    def analyze_exploration_exploitation(
        self, agent_id: Optional[str] = None
    ) -> ExplorationExploitation:
        """Analyze how agents balance exploration versus exploitation behaviors.

        Examines action patterns to determine how agents distribute their efforts between
        exploring new actions and exploiting known successful behaviors.

        Args:
            agent_id (Optional[str]): Specific agent to analyze. If None, analyzes all agents.

        Returns:
            ExplorationExploitation: Analysis results containing:
                - exploration_rate: Ratio of new action attempts
                - exploitation_rate: Ratio of repeated action attempts
                - reward_comparison: Dict comparing rewards for new vs. known actions:
                    - new_actions_avg: Mean reward for exploration
                    - known_actions_avg: Mean reward for exploitation
        """
        # Get actions through repository
        actions = self.repository.get_actions_by_agent_id(agent_id) if agent_id else []

        # Track unique and repeated actions
        action_history = {}
        exploration_count = 0
        exploitation_count = 0
        new_action_rewards = []
        known_action_rewards = []

        for action in actions:
            action_key = (action.agent_id, action.action_type)

            if action_key not in action_history:
                exploration_count += 1
                if action.reward is not None:
                    new_action_rewards.append(action.reward)
                action_history[action_key] = action.reward
            else:
                exploitation_count += 1
                if action.reward is not None:
                    known_action_rewards.append(action.reward)

        total_actions = exploration_count + exploitation_count

        return ExplorationExploitation(
            exploration_rate=(
                exploration_count / total_actions if total_actions > 0 else 0
            ),
            exploitation_rate=(
                exploitation_count / total_actions if total_actions > 0 else 0
            ),
            reward_comparison={
                "new_actions_avg": (
                    sum(new_action_rewards) / len(new_action_rewards)
                    if new_action_rewards
                    else 0
                ),
                "known_actions_avg": (
                    sum(known_action_rewards) / len(known_action_rewards)
                    if known_action_rewards
                    else 0
                ),
            },
        )

    def analyze_adversarial_interactions(
        self, agent_id: Optional[str] = None
    ) -> AdversarialInteractionAnalysis:
        """Analyze agent performance in competitive scenarios and conflicts.

        Evaluates effectiveness in competitive situations by examining attack patterns,
        defense mechanisms, and strategic adaptations to opponent behaviors.

        Args:
            agent_id (Optional[str]): Specific agent to analyze. If None, analyzes all agents.

        Returns:
            AdversarialInteractionAnalysis: Analysis results containing:
                - win_rate: Ratio of successful competitive actions
                - damage_efficiency: Average reward from successful actions
                - counter_strategies: Distribution of opponent response patterns
        """
        # Get actions through repository
        actions = self.repository.get_actions_by_agent_id(agent_id) if agent_id else []

        # Filter competitive actions
        competitive_actions = [
            a for a in actions if a.action_type in ["attack", "defend", "compete"]
        ]

        successful = [a for a in competitive_actions if a.reward and a.reward > 0]

        # Calculate counter-strategies
        counter_actions = {}
        for action in competitive_actions:
            if action.action_target_id:
                # Get all actions for the target
                target_actions = self.repository.get_actions_by_agent_id(
                    action.action_target_id
                )
                # Find first response after this action
                target_response = next(
                    (a for a in target_actions if a.step_number > action.step_number),
                    None,
                )
                if target_response:
                    if target_response.action_type not in counter_actions:
                        counter_actions[target_response.action_type] = 0
                    counter_actions[target_response.action_type] += 1

        total_counters = sum(counter_actions.values())
        counter_frequencies = (
            {
                action: count / total_counters
                for action, count in counter_actions.items()
            }
            if total_counters > 0
            else {}
        )

        return AdversarialInteractionAnalysis(
            win_rate=(
                len(successful) / len(competitive_actions) if competitive_actions else 0
            ),
            damage_efficiency=(
                sum(a.reward for a in successful) / len(successful) if successful else 0
            ),
            counter_strategies=counter_frequencies,
        )

    def analyze_collaboration(
        self, agent_id: Optional[str] = None
    ) -> CollaborativeInteractionAnalysis:
        """Analyze patterns and outcomes of cooperative behaviors between agents.

        Evaluates collaborative effectiveness by comparing rewards from cooperative
        actions versus individual actions and measuring overall collaboration frequency.

        Args:
            agent_id (Optional[str]): Specific agent to analyze. If None, analyzes all agents.

        Returns:
            CollaborativeInteractionAnalysis: Analysis results containing:
                - collaboration_rate: Frequency of cooperative actions
                - group_reward_impact: Average reward from collaborative actions
                - synergy_metrics: Difference between collaborative and solo action rewards
        """
        # Get actions through repository
        actions = self.repository.get_actions_by_agent_id(agent_id) if agent_id else []

        # Filter collaborative actions
        collaborative_actions = [
            a for a in actions if a.action_type in ["share", "help", "cooperate"]
        ]

        # Filter solo actions
        solo_actions = [a for a in actions if not a.action_target_id]

        collaborative_rewards = [
            a.reward for a in collaborative_actions if a.reward is not None
        ]
        solo_rewards = [a.reward for a in solo_actions if a.reward is not None]

        return CollaborativeInteractionAnalysis(
            collaboration_rate=(
                len(collaborative_actions) / len(actions) if actions else 0
            ),
            group_reward_impact=(
                sum(collaborative_rewards) / len(collaborative_rewards)
                if collaborative_rewards
                else 0
            ),
            synergy_metrics=(
                (
                    sum(collaborative_rewards) / len(collaborative_rewards)
                    - sum(solo_rewards) / len(solo_rewards)
                )
                if collaborative_rewards and solo_rewards
                else 0
            ),
        )

    def analyze_learning_curve(
        self, agent_id: Optional[str] = None
    ) -> LearningCurveAnalysis:
        """Analyze agent learning progress and performance improvements over time.

        Tracks performance metrics across time periods to measure learning effectiveness,
        success rates, and reduction in suboptimal decisions.

        Args:
            agent_id (Optional[str]): Specific agent to analyze. If None, analyzes all agents.

        Returns:
            LearningCurveAnalysis: Analysis results containing:
                - action_success_over_time: List of success rates per time period
                - reward_progression: List of average rewards per time period
                - mistake_reduction: Decrease in error rate between first and last periods
        """
        # Get actions through repository
        actions = self.repository.get_actions_by_agent_id(agent_id) if agent_id else []

        # Group actions by time periods (every 100 steps)
        time_periods = {}
        for action in actions:
            period = action.step_number // 100
            if period not in time_periods:
                time_periods[period] = {"rewards": [], "successes": 0, "total": 0}
            if action.reward is not None:
                time_periods[period]["rewards"].append(action.reward)
                if action.reward > 0:
                    time_periods[period]["successes"] += 1
            time_periods[period]["total"] += 1

        # Sort periods and calculate metrics
        sorted_periods = sorted(time_periods.items())

        return LearningCurveAnalysis(
            action_success_over_time=[
                period["successes"] / period["total"] if period["total"] > 0 else 0
                for _, period in sorted_periods
            ],
            reward_progression=[
                (
                    sum(period["rewards"]) / len(period["rewards"])
                    if period["rewards"]
                    else 0
                )
                for _, period in sorted_periods
            ],
            mistake_reduction=self._calculate_mistake_reduction(sorted_periods),
        )

    def _calculate_mistake_reduction(self, periods) -> float:
        """Calculate the reduction in mistake rate between first and last time periods.

        Args:
            periods (List[Tuple]): Sorted list of time periods with performance metrics.

        Returns:
            float: Difference between early and late mistake rates (0 to 1).
                  Higher values indicate greater improvement.
        """
        if not periods:
            return 0

        first_period = periods[0][1]
        last_period = periods[-1][1]

        early_mistakes = (
            1 - first_period["successes"] / first_period["total"]
            if first_period["total"] > 0
            else 0
        )
        late_mistakes = (
            1 - last_period["successes"] / last_period["total"]
            if last_period["total"] > 0
            else 0
        )

        return max(0, early_mistakes - late_mistakes)

    def analyze_conflicts(self, agent_id: Optional[str] = None) -> ConflictAnalysis:
        """Analyze patterns of conflict and conflict resolution strategies.

        Examines sequences of actions leading to and resolving conflicts, including
        trigger patterns, resolution strategies, and outcome analysis.

        Args:
            agent_id (Optional[str]): Specific agent ID to analyze. If None, analyzes all agents.

        Returns:
            ConflictAnalysis: Analysis results containing:
                - conflict_trigger_actions (Dict[str, float]): Actions that commonly lead to conflicts,
                  normalized as proportions (0-1)
                - conflict_resolution_actions (Dict[str, float]): Actions used to resolve conflicts,
                  normalized as proportions (0-1)
                - conflict_outcome_metrics (Dict[str, float]): Average rewards for different
                  resolution strategies

        Examples:
            >>> analyzer = AgentAnalysis(repository)
            >>> conflicts = analyzer.analyze_conflicts(agent_id=1)
            >>> print("Common triggers:", conflicts.conflict_trigger_actions)
            >>> print("Best resolution:", max(conflicts.conflict_outcome_metrics.items(),
            ...       key=lambda x: x[1])[0])
        """
        # Get actions through repository
        actions = self.repository.get_ordered_actions(agent_id) if agent_id else []

        conflict_triggers = {}
        conflict_resolutions = {}
        conflict_outcomes = {}

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            if next_action.action_type in ["attack", "defend"]:
                if current.action_type not in conflict_triggers:
                    conflict_triggers[current.action_type] = 0
                conflict_triggers[current.action_type] += 1

            if current.action_type in ["attack", "defend"]:
                if next_action.action_type not in conflict_resolutions:
                    conflict_resolutions[next_action.action_type] = 0
                    conflict_outcomes[next_action.action_type] = []
                conflict_resolutions[next_action.action_type] += 1
                if next_action.reward is not None:
                    conflict_outcomes[next_action.action_type].append(
                        next_action.reward
                    )

        return ConflictAnalysis(
            conflict_trigger_actions=_normalize_dict(conflict_triggers),
            conflict_resolution_actions=_normalize_dict(conflict_resolutions),
            conflict_outcome_metrics={
                action: sum(outcomes) / len(outcomes) if outcomes else 0
                for action, outcomes in conflict_outcomes.items()
            },
        )

    def analyze_risk_reward(self, agent_id: Optional[str] = None) -> RiskRewardAnalysis:
        """Analyze risk-taking behavior and associated outcomes.

        Examines the relationship between action risk levels and rewards, including risk
        appetite assessment and outcome analysis. Risk level is determined by reward
        variance relative to mean reward.

        Args:
            agent_id (Optional[str]): Specific agent ID to analyze. If None, analyzes all agents.

        Returns:
            RiskRewardAnalysis: Analysis results containing:
                - high_risk_actions (Dict[str, float]): Actions with high reward variance
                  (>mean/2) and their average returns
                - low_risk_actions (Dict[str, float]): Actions with low reward variance
                  (≤mean/2) and their average returns
                - risk_appetite (float): Proportion of high-risk actions taken (0.0 to 1.0)

        Examples:
            >>> analyzer = AgentAnalysis(repository)
            >>> risk = analyzer.analyze_risk_reward(agent_id=1)
            >>> print(f"Risk appetite: {risk.risk_appetite:.2%}")
            >>> print("High-risk actions:", risk.high_risk_actions)
        """
        # Get action statistics through repository
        action_stats = self.repository.get_action_statistics(agent_id)

        high_risk = {}
        low_risk = {}
        total_actions = 0
        high_risk_actions = 0

        for stat in action_stats:
            if stat.reward_std > stat.reward_avg / 2:  # High variability
                high_risk[stat.action_type] = float(stat.reward_avg or 0)
                high_risk_actions += stat.count
            else:
                low_risk[stat.action_type] = float(stat.reward_avg or 0)
            total_actions += stat.count

        return RiskRewardAnalysis(
            high_risk_actions=high_risk,
            low_risk_actions=low_risk,
            risk_appetite=high_risk_actions / total_actions if total_actions > 0 else 0,
        )

    def analyze_counterfactuals(
        self, agent_id: Optional[str] = None
    ) -> CounterfactualAnalysis:
        """Analyze potential alternative outcomes and missed opportunities.

        Examines what-if scenarios by analyzing unused or underutilized actions and their
        potential impacts based on observed outcomes. Identifies high-value actions that
        were underutilized and compares actual strategy performance against optimal.

        Args:
            agent_id (Optional[str]): Specific agent ID to analyze. If None, analyzes all agents.

        Returns:
            CounterfactualAnalysis: Analysis results containing:
                - counterfactual_rewards (Dict[str, float]): Average rewards for each action type
                - missed_opportunities (Dict[str, float]): High-value actions that were used
                  less than half the median usage rate
                - strategy_comparison (Dict[str, float]): Performance delta between each
                  strategy and the average performance

        Examples:
            >>> analyzer = AgentAnalysis(repository)
            >>> analysis = analyzer.analyze_counterfactuals(agent_id=1)
            >>> print("Missed opportunities:", analysis.missed_opportunities)
            >>> print("Best strategy:", max(analysis.strategy_comparison.items(),
            ...       key=lambda x: x[1])[0])
        """
        # Get actions through repository
        actions = self.repository.get_actions_by_agent_id(agent_id) if agent_id else []

        action_rewards = {}
        for action in actions:
            if action.action_type not in action_rewards:
                action_rewards[action.action_type] = []
            if action.reward is not None:
                action_rewards[action.action_type].append(action.reward)

        avg_rewards = {
            action: sum(rewards) / len(rewards)
            for action, rewards in action_rewards.items()
            if rewards
        }

        action_counts = {
            action: len(rewards) for action, rewards in action_rewards.items()
        }
        median_usage = sorted(action_counts.values())[len(action_counts) // 2]
        underused = {
            action: count
            for action, count in action_counts.items()
            if count < median_usage / 2
        }

        return CounterfactualAnalysis(
            counterfactual_rewards=avg_rewards,
            missed_opportunities={
                action: avg_rewards.get(action, 0) for action in underused
            },
            strategy_comparison={
                action: reward - sum(avg_rewards.values()) / len(avg_rewards)
                for action, reward in avg_rewards.items()
            },
        )

    def analyze_resilience(self, agent_id: Optional[str] = None) -> ResilienceAnalysis:
        """Analyze agent recovery patterns and adaptation to failures.

        Examines how agents respond to and recover from negative outcomes, including
        recovery speed, adaptation strategies, and impact assessment. A failure is
        defined as any action resulting in negative reward.

        Args:
            agent_id (Optional[str]): Specific agent ID to analyze. If None, analyzes all agents.

        Returns:
            ResilienceAnalysis: Analysis results containing:
                - recovery_rate (float): Average number of steps needed to return to
                  positive rewards after a failure
                - adaptation_rate (float): Proportion of actions that differ from the
                  failing action during recovery periods (0.0 to 1.0)
                - failure_impact (float): Average magnitude of performance drop during
                  failure periods (pre-failure reward - minimum reward)

        Examples:
            >>> analyzer = AgentAnalysis(repository)
            >>> resilience = analyzer.analyze_resilience(agent_id=1)
            >>> print(f"Recovery time: {resilience.recovery_rate:.1f} steps")
            >>> print(f"Adaptation rate: {resilience.adaptation_rate:.1%}")
        """
        # Get ordered actions through repository
        actions = self.repository.get_ordered_actions(agent_id) if agent_id else []

        recovery_times = []
        adaptation_speeds = []
        failure_impacts = []

        current_failure = False
        failure_start = 0
        pre_failure_reward = 0

        for i, action in enumerate(actions):
            if action.reward is not None and action.reward < 0:
                if not current_failure:
                    current_failure = True
                    failure_start = i
                    pre_failure_reward = (
                        sum(
                            a.reward
                            for a in actions[max(0, i - 5) : i]
                            if a.reward is not None
                        )
                        / 5
                    )
            elif current_failure and action.reward is not None and action.reward > 0:
                recovery_times.append(i - failure_start)

                post_failure_actions = actions[failure_start:i]
                if post_failure_actions:
                    adaptation = sum(
                        1
                        for a in post_failure_actions
                        if a.action_type != actions[failure_start].action_type
                    ) / len(post_failure_actions)
                    adaptation_speeds.append(adaptation)

                failure_impacts.append(
                    pre_failure_reward
                    - min(
                        a.reward for a in post_failure_actions if a.reward is not None
                    )
                )

                current_failure = False

        return ResilienceAnalysis(
            recovery_rate=(
                sum(recovery_times) / len(recovery_times) if recovery_times else 0
            ),
            adaptation_rate=(
                sum(adaptation_speeds) / len(adaptation_speeds)
                if adaptation_speeds
                else 0
            ),
            failure_impact=(
                sum(failure_impacts) / len(failure_impacts) if failure_impacts else 0
            ),
        )

    def analyze_environmental_impact(
        self, agent_id: Optional[str] = None
    ) -> EnvironmentalImpactAnalysis:
        """Analyze how environment affects agent action outcomes.

        Examines the relationship between environmental states (like resource levels)
        and action outcomes, including adaptation patterns and spatial effects.

        Args:
            agent_id (Optional[str]): Specific agent ID to analyze. If None, analyzes all agents.

        Returns:
            EnvironmentalImpactAnalysis: Analysis results containing:
                - environmental_state_impact : Dict[str, float]
                    Correlation between resource levels and action outcomes
                - adaptive_behavior : Dict[str, float]
                    Measures of agent adaptation to environmental changes
                - spatial_analysis : Dict[str, Any]
                    Analysis of location-based patterns and effects
        """
        # Get actions and states through repository
        results = self.repository.get_actions_with_states(agent_id)

        # Analyze resource levels impact
        resource_impacts = {}
        for action, state in results:
            resource_level = state.resource_level
            if action.action_type not in resource_impacts:
                resource_impacts[action.action_type] = []
            if action.reward is not None:
                resource_impacts[action.action_type].append(
                    (resource_level, action.reward)
                )

        environmental_state_impact = {}
        for action, rewards in resource_impacts.items():
            if rewards:
                environmental_state_impact[action] = _calculate_correlation(
                    [r[0] for r in rewards], [r[1] for r in rewards]
                )

        # Adaptive behavior
        adaptive_behavior = self._analyze_adaptation(results)

        # Spatial analysis using the new SpatialAnalyzer
        spatial_analysis = SpatialAnalyzer.analyze_spatial_patterns(results)

        return EnvironmentalImpactAnalysis(
            environmental_state_impact=environmental_state_impact,
            adaptive_behavior=adaptive_behavior,
            spatial_analysis=spatial_analysis,
        )

    def _analyze_adaptation(self, results: List[Tuple[Any, Any]]) -> Dict[str, float]:
        """Analyze how agents adapt to changing conditions.

        Args:
            results: List of action-state pairs to analyze

        Returns:
            Dict[str, float]: Adaptation metrics including:
                - adaptation_rate: How quickly agents modify behavior
                - success_rate: Success rate of adapted behaviors
                - stability: Consistency of adapted behaviors
        """
        agent_actions = {}
        for action, state in results:
            if action.agent_id not in agent_actions:
                agent_actions[action.agent_id] = []
            agent_actions[action.agent_id].append((action, state))

        total_adaptations = 0
        successful_adaptations = 0
        total_behavior_changes = 0
        consistent_behavior_count = 0
        total_behavior_count = 0

        for actions in agent_actions.values():
            actions.sort(key=lambda x: x[0].step_number)

            last_action_type = None
            last_reward = None
            consecutive_same_behavior = 0

            for action, state in actions:
                if last_action_type is not None:
                    # Check if the agent adapted (changed action type)
                    if action.action_type != last_action_type:
                        total_behavior_changes += 1
                        # Consider it an adaptation if the new action's reward is higher
                        if (
                            action.reward
                            and last_reward
                            and action.reward > last_reward
                        ):
                            total_adaptations += 1
                            successful_adaptations += 1

                    # Count consistent behaviors
                    if action.action_type == last_action_type:
                        consecutive_same_behavior += 1
                    else:
                        consecutive_same_behavior = 1

                    total_behavior_count += 1

                last_action_type = action.action_type
                last_reward = action.reward

            # Add stability metric
            consistent_behavior_count += consecutive_same_behavior

        adaptation_rate = (
            total_adaptations / total_behavior_changes
            if total_behavior_changes > 0
            else 0.0
        )
        success_rate = (
            successful_adaptations / total_adaptations if total_adaptations > 0 else 0.0
        )
        stability = (
            consistent_behavior_count / total_behavior_count
            if total_behavior_count > 0
            else 0.0
        )

        return {
            "adaptation_rate": adaptation_rate,
            "success_rate": success_rate,
            "stability": stability,
        }
