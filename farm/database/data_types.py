"""Data types for simulation metrics and statistics.

This module defines dataclasses that represent the structured return types
for various simulation metrics and statistics queries.

Classes
-------
SimulationStateData
    Data for a specific simulation step
AgentLifespanStats
    Statistics about agent lifespans
PopulationStatistics
    Comprehensive population statistics
ResourceMetrics
    Resource distribution and efficiency metrics
PopulationVariance
    Population variance statistics
AgentDistribution
    Distribution of different agent types
SurvivalMetrics
    Population survival metrics
ResourceDistribution
    Resource distribution statistics over time
EfficiencyMetrics
    Resource efficiency metrics
LearningProgress
    Learning and adaptation statistics
ModulePerformance
    Performance metrics for learning modules
HistoricalMetrics
    Historical simulation metrics
InteractionPattern
    Statistics about agent interactions
ResourceBehavior
    Resource-related behavior statistics
RewardStats
    Statistics about action rewards
DecisionPattern
    Pattern of agent decisions
AgentBehaviorMetrics
    Comprehensive analysis of agent behaviors
AgentDecisionMetrics
    Analysis of agent decision-making patterns
StepActionData
    Data about actions in a specific simulation step
AgentStateData
    Comprehensive data about an agent's state
LifespanStatistics
    Statistics about agent lifespans by type and generation
SurvivalRatesByGeneration
    Statistics about survival rates by generation
AgentLifespanResults
    Combined agent lifespan statistics and survival rates.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


@dataclass
class SimulationState:
    """Current state metrics for the simulation.

    Attributes
    ----------
    total_agents : int
        Total number of agents in the simulation
    system_agents : int
        Number of system-controlled agents
    independent_agents : int
        Number of independent agents
    control_agents : int
        Number of control group agents
    total_resources : float
        Total resources available in the environment
    average_agent_resources : float
        Mean resources per agent
    births : int
        Number of new agents created
    deaths : int
        Number of agent deaths
    current_max_generation : int
        Highest generation number among living agents
    resource_efficiency : float
        Measure of resource utilization efficiency
    resource_distribution_entropy : float
        Entropy measure of resource distribution
    average_agent_health : float
        Mean health level across all agents
    average_agent_age : float
        Mean age of all agents
    average_reward : float
        Mean reward across all agents
    combat_encounters : int
        Number of combat interactions
    successful_attacks : int
        Number of successful attack actions
    resources_shared : float
        Amount of resources shared between agents
    genetic_diversity : float
        Measure of genetic variation in population
    dominant_genome_ratio : float
        Ratio of agents sharing the most common genome
    resources_consumed : float
        Total resources consumed by the simulation
    """

    total_agents: int
    system_agents: int
    independent_agents: int
    control_agents: int
    total_resources: float
    average_agent_resources: float
    births: int
    deaths: int
    current_max_generation: int
    resource_efficiency: float
    resource_distribution_entropy: float
    average_agent_health: float
    average_agent_age: float
    average_reward: float
    combat_encounters: int
    successful_attacks: int
    resources_shared: float
    genetic_diversity: float
    dominant_genome_ratio: float
    resources_consumed: float


@dataclass
class SimulationResults:
    """Data for a specific simulation step.

    Attributes
    ----------
    agent_states : List[Tuple]
        List of agent state records containing:
        (step_number, agent_id, agent_type, position_x, position_y,
         resource_level, current_health, is_defending)
    resource_states : List[Tuple]
        List of resource state records containing:
        (resource_id, amount, position_x, position_y)
    simulation_state : Dict[str, Any]
        Dictionary of simulation metrics for the step including:
        - total_agents: Total number of agents
        - total_resources: Total available resources
        - average_agent_health: Mean health across agents
        - resource_efficiency: Resource utilization efficiency
    """

    agent_states: List[Tuple]
    resource_states: List[Tuple]
    simulation_state: Dict[str, Any]


@dataclass
class AgentLifespanStats:
    """Statistics about agent lifespans.

    Attributes
    ----------
    average_lifespan : float
        Mean lifespan across all agents
    lifespan_by_type : Dict[str, float]
        Mean lifespan per agent type
    lifespan_by_generation : Dict[int, float]
        Mean lifespan per generation
    survival_rates : Dict[int, float]
        Survival rate per generation
    """

    average_lifespan: float
    lifespan_by_type: Dict[str, float]
    lifespan_by_generation: Dict[int, float]
    survival_rates: Dict[int, float]


@dataclass
class ResourceMetrics:
    """Resource distribution and efficiency metrics.

    Attributes
    ----------
    resource_utilization : float
        Resource usage efficiency
    resources_consumed : float
        Total resources consumed
    resources_available : float
        Total resources available
    utilization_per_agent : float
        Average resource usage per agent
    """

    resource_utilization: float
    resources_consumed: float
    resources_available: float
    utilization_per_agent: float


@dataclass
class PopulationVariance:
    """Population variance statistics.

    Attributes
    ----------
    variance : float
        Population size variance
    standard_deviation : float
        Population size standard deviation
    coefficient_variation : float
        Coefficient of variation
    """

    variance: float
    standard_deviation: float
    coefficient_variation: float


@dataclass
class AgentDistribution:
    """Distribution of different agent types.

    Attributes
    ----------
    system_agents : float
        Average number of system agents
    independent_agents : float
        Average number of independent agents
    control_agents : float
        Average number of control agents
    """

    system_agents: float
    independent_agents: float
    control_agents: float


@dataclass
class SurvivalMetrics:
    """Population survival metrics.

    Attributes
    ----------
    survival_rate : float
        Population survival rate
    average_lifespan : float
        Average agent lifespan
    """

    survival_rate: float
    average_lifespan: float


@dataclass
class ResourceDistributionStep:
    """Data for a single step's resource distribution.

    Attributes
    ----------
    step : int
        Step number
    total_resources : float
        Total resources at this step
    average_per_cell : float
        Average resources per grid cell
    distribution_entropy : float
        Resource distribution entropy
    """

    step: int
    total_resources: float
    average_per_cell: float
    distribution_entropy: float


@dataclass
class ResourceDistributionData:
    """Resource distribution data over time.

    Attributes
    ----------
    steps : List[ResourceDistributionStep]
        List of resource distribution data for each step
    """

    steps: List[ResourceDistributionStep]


@dataclass
class EfficiencyMetrics:
    """Resource efficiency metrics.

    Attributes
    ----------
    average_efficiency : float
        Mean resource utilization efficiency
    efficiency_trend : List[float]
        Resource efficiency over time
    distribution_entropy : List[float]
        Resource distribution entropy over time
    """

    average_efficiency: float
    efficiency_trend: List[float]
    distribution_entropy: List[float]


@dataclass
class LearningProgress:
    """Learning progress metrics for a single simulation step.

    Attributes
    ----------
    step : int
        Step number
    reward : float
        Average reward for this step
    action_count : int
        Number of actions taken in this step
    unique_actions : int
        Number of unique actions used in this step
    """

    step: int
    reward: float
    action_count: int
    unique_actions: int


@dataclass
class ModulePerformance:
    """Performance metrics for a learning module.

    Attributes
    ----------
    module_type : str
        Type of learning module
    module_id : str
        Unique identifier for module instance
    avg_reward : float
        Average reward achieved
    total_actions : int
        Total number of actions taken
    unique_actions : int
        Number of unique actions used
    """

    module_type: str
    module_id: str
    avg_reward: float
    total_actions: int
    unique_actions: int


@dataclass
class HistoricalMetrics:
    """Historical simulation metrics.

    Attributes
    ----------
    steps : List[int]
        List of step numbers
    metrics : Dict[str, List[float]]
        Dictionary of metric lists including:
        - total_agents
        - system_agents
        - independent_agents
        - control_agents
        - total_resources
        - average_agent_resources
        - births
        - deaths
    """

    steps: List[int]
    metrics: Dict[str, List[float]]


@dataclass
class InteractionPattern:
    """Statistics about agent interactions.

    Attributes
    ----------
    interaction_count : int
        Number of interactions
    average_reward : float
        Average reward for interactions
    """

    interaction_count: int
    average_reward: float


@dataclass
class ResourceBehavior:
    """Resource-related behavior statistics.

    Attributes
    ----------
    average_resource_change : float
        Average change in resources
    action_count : int
        Number of resource-affecting actions
    """

    average_resource_change: float
    action_count: int


@dataclass
class RewardStats:
    """Statistics about action rewards.

    Attributes
    ----------
    average : float
        Mean reward value
    stddev : float
        Standard deviation of rewards
    min : float
        Minimum reward value
    max : float
        Maximum reward value
    """

    average: float
    stddev: float
    min: float
    max: float


@dataclass
class DecisionPattern:
    """Pattern of agent decisions.

    Attributes
    ----------
    count : int
        Number of times this decision was made
    frequency : float
        Frequency of this decision type
    reward_stats : RewardStats
        Statistics about rewards for this decision
    """

    count: int
    frequency: float
    reward_stats: RewardStats


@dataclass
class AgentBehaviorMetrics:
    """Comprehensive analysis of agent behaviors.

    Attributes
    ----------
    temporal_patterns : Dict[int, Dict[str, Dict[str, float]]]
        Behavior patterns over time, indexed by step
    type_behaviors : Dict[str, Dict[str, Any]]
        Behavior patterns by agent type
    interaction_patterns : Dict[str, InteractionPattern]
        Statistics about different types of interactions
    resource_behaviors : Dict[str, ResourceBehavior]
        Resource-related behavior statistics
    behavior_summary : Dict[str, Any]
        Summary statistics about behaviors
    """

    temporal_patterns: Dict[int, Dict[str, Dict[str, float]]]
    type_behaviors: Dict[str, Dict[str, Any]]
    interaction_patterns: Dict[str, InteractionPattern]
    resource_behaviors: Dict[str, ResourceBehavior]
    behavior_summary: Dict[str, Any]


@dataclass
class AgentDecisionMetrics:
    """Analysis of agent decision-making patterns.

    Attributes
    ----------
    decision_patterns : Dict[str, DecisionPattern]
        Statistics about different decision types
    temporal_trends : Dict[int, Dict[str, float]]
        Decision patterns over time
    context_influence : Dict[str, Dict[str, float]]
        Impact of context on decisions
    decision_outcomes : Dict[str, RewardStats]
        Outcome statistics for each decision type
    """

    decision_patterns: Dict[str, DecisionPattern]
    temporal_trends: Dict[int, Dict[str, float]]
    context_influence: Dict[str, Dict[str, float]]
    decision_outcomes: Dict[str, RewardStats]


@dataclass
class StepSummary:
    """Summary statistics for a simulation step.

    Attributes
    ----------
    total_actions : int
        Total number of actions taken in this step
    unique_agents : int
        Number of unique agents that took actions
    action_types : int
        Number of different action types used
    total_reward : float
        Sum of all rewards received in this step
    """

    total_actions: int
    unique_agents: int
    action_types: int
    total_reward: float


@dataclass
class ActionTypeStats:
    """Statistics for a specific action type in a step.

    Attributes
    ----------
    count : int
        Number of times this action was taken
    frequency : float
        Proportion of total actions this type represents
    avg_reward : float
        Average reward received for this action type
    total_reward : float
        Total reward accumulated for this action type
    """

    count: int
    frequency: float
    avg_reward: float
    total_reward: float


@dataclass
class DecisionPatternStats:
    """Statistics for a single decision/action type.

    Attributes
    ----------
    action_type : str
        The type of action
    count : int
        Number of times this action was taken
    frequency : float
        Frequency of this action relative to total decisions
    reward_stats : dict
        Statistics about rewards for this action type including:
        - average: Mean reward
        - median: Median reward
        - min: Minimum reward received
        - max: Maximum reward received
        - variance: Variance in rewards
        - std_dev: Standard deviation of rewards
        - percentile_25: 25th percentile of rewards
        - percentile_50: 50th percentile of rewards
        - percentile_75: 75th percentile of rewards
    """

    action_type: str
    count: int
    frequency: float
    reward_stats: dict = field(default_factory=dict)
    contribution_metrics: dict = field(default_factory=dict)
    temporal_stats: dict = field(default_factory=dict)
    first_occurrence: dict = field(default_factory=dict)


@dataclass
class DecisionSummary:
    """Summary statistics about decision making.

    Attributes
    ----------
    total_decisions : int
        Total number of decisions made
    unique_actions : int
        Number of unique action types used
    most_frequent : Optional[str]
        Most frequently taken action
    most_rewarding : Optional[str]
        Action with highest average reward
    action_diversity : float
        Shannon entropy of action distribution
    """

    total_decisions: int
    unique_actions: int
    most_frequent: Optional[str]
    most_rewarding: Optional[str]
    action_diversity: float


@dataclass
class ResourceMetricsStep:
    """Resource-related metrics for a simulation step.

    Attributes
    ----------
    net_resource_change : float
        Total change in resources across all agents
    average_resource_change : float
        Average change in resources per agent
    resource_transactions : int
        Number of actions that affected resources
    """

    net_resource_change: float
    average_resource_change: float
    resource_transactions: int


@dataclass
class InteractionNetwork:
    """Network of agent interactions in a step.

    Attributes
    ----------
    interactions : List[Dict[str, Any]]
        List of interaction records containing:
        - source: ID of initiating agent
        - target: ID of target agent
        - action_type: Type of interaction
        - reward: Reward received
    unique_interacting_agents : int
        Number of unique agents involved in interactions
    """

    interactions: List[Dict[str, Any]]
    unique_interacting_agents: int


@dataclass
class InteractionSummary:
    """Summary of agent interactions.

    Attributes
    ----------
    interaction_count : int
        Number of interactions
    average_reward : float
        Average reward for interactions
    """

    interaction_count: int
    average_reward: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for a simulation step.

    Attributes
    ----------
    success_rate : float
        Proportion of successful actions
    average_reward : float
        Average reward for all actions
    action_efficiency : float
        Efficiency of action usage
    """

    success_rate: float
    average_reward: float
    action_efficiency: float


@dataclass
class StepActionData:
    """Data about actions in a specific simulation step.

    Attributes
    ----------
    step_summary : StepSummary
        Summary statistics for the step
    action_statistics : Dict[str, ActionTypeStats]
        Statistics about different action types
    resource_metrics : ResourceMetricsStep
        Resource-related metrics
    detailed_actions : List[Dict[str, Any]]
        Detailed list of all actions
    """

    step_summary: StepSummary
    action_statistics: Dict[str, ActionTypeStats]
    detailed_actions: List[Dict[str, Any]]


@dataclass
class AgentStateData:
    """Comprehensive data about an agent's state.

    Attributes
    ----------
    basic_info : Dict[str, Any]
        Basic agent information
    genetic_info : Dict[str, Any]
        Genetic and hereditary information
    current_state : Dict[str, Any]
        Current agent state
    historical_metrics : Dict[str, float]
        Historical performance metrics
    action_history : Dict[str, Dict[str, float]]
        History of agent actions
    health_incidents : List[Dict[str, Any]]
        Record of health-related incidents
    """

    basic_info: Dict[str, Any]
    genetic_info: Dict[str, Any]
    current_state: Dict[str, Any]
    historical_metrics: Dict[str, float]
    action_history: Dict[str, Dict[str, float]]
    health_incidents: List[Dict[str, Any]]


@dataclass
class AgentStatesData:
    """Data representing agent states from database queries.

    Attributes
    ----------
    agent_states : List[Tuple]
        List of tuples containing:
        - step_number: int
        - agent_id: str
        - agent_type: str
        - position_x: float
        - position_y: float
        - resource_level: float
        - current_health: float
        - is_defending: bool
    """

    agent_states: List[Tuple[int, str, str, float, float, float, float, bool]]


@dataclass
class SequencePattern:
    """Statistics about action sequences.

    Attributes
    ----------
    count : int
        Number of times this sequence occurred
    probability : float
        Probability of this sequence occurring
    """

    count: int
    probability: float


@dataclass
class TimePattern:
    """Temporal patterns of action occurrences and rewards over time.

    Attributes:
        action_type: The type of action analyzed.
        time_distribution: A list of action counts per time period.
        reward_progression: A list of average rewards per time period.
        rolling_average_rewards: A list of rolling average rewards.
        rolling_average_counts: A list of rolling average action counts.
    """

    action_type: str
    time_distribution: List[int]
    reward_progression: List[float]
    rolling_average_rewards: List[float]
    rolling_average_counts: List[float]


@dataclass
class EventSegment:
    """Metrics segmented by events.

    Attributes:
        start_step: The starting step of the segment.
        end_step: The ending step of the segment (exclusive).
        action_counts: A dictionary of action counts during the segment.
        average_rewards: A dictionary of average rewards per action type during the segment.
    """

    start_step: int
    end_step: Optional[int]
    action_counts: Dict[str, int]
    average_rewards: Dict[str, float]


@dataclass
class InteractionStats:
    """Statistics about agent interactions.

    Attributes
    ----------
    action_type : str
        Type of action analyzed
    interaction_rate : float
        Rate of interactive vs solo actions
    solo_performance : float
        Average reward for solo actions
    interaction_performance : float
        Average reward for interactive actions
    """

    action_type: str
    interaction_rate: float
    solo_performance: float
    interaction_performance: float


@dataclass
class ResourceImpact:
    """Resource impact statistics for an action type.

    Attributes
    ----------
    action_type : str
        The type of action being analyzed
    avg_resources_before : float
        Mean resources available before action execution
    avg_resource_change : float
        Average change in resources from action execution
    resource_efficiency : float
        Resource change per action execution (change/count)
    """

    action_type: str
    avg_resources_before: float
    avg_resource_change: float
    resource_efficiency: float


@dataclass
class AdvancedActionMetrics:
    """Container for all advanced analysis metrics.

    Attributes
    ----------
    exploration_vs_exploitation : Dict[str, Any]
        Exploration vs exploitation patterns
    adversarial_interactions : Dict[str, Any]
        Adversarial interaction patterns
    collaborative_behaviors : Dict[str, Any]
        Collaborative behavior patterns
    learning_curves : Dict[str, Any]
        Learning curve patterns
    environmental_impacts : Dict[str, Any]
        Environmental impact patterns
    conflict_patterns : Dict[str, Any]
        Conflict patterns
    risk_reward_relationships : Dict[str, Any]
        Risk-reward relationships
    counterfactual_scenarios : Dict[str, Any]
        Counterfactual scenario analysis
    resilience_metrics : Dict[str, Any]
        Resilience metrics
    """

    exploration_vs_exploitation: Dict[str, Any]
    adversarial_interactions: Dict[str, Any]
    collaborative_behaviors: Dict[str, Any]
    learning_curves: Dict[str, Any]
    environmental_impacts: Dict[str, Any]
    conflict_patterns: Dict[str, Any]
    risk_reward_relationships: Dict[str, Any]
    counterfactual_scenarios: Dict[str, Any]
    resilience_metrics: Dict[str, Any]


@dataclass
class PopulationMetrics:
    """Basic population count metrics.

    Attributes
    ----------
    total_agents : int
        Total number of agents in the population
    system_agents : int
        Number of system agents
    independent_agents : int
        Number of independent agents
    control_agents : int
        Number of control agents
    """

    total_agents: int
    system_agents: int
    independent_agents: int
    control_agents: int


@dataclass
class PopulationStatistics:
    """Combined population statistics.

    Attributes
    ----------
    population_metrics : PopulationMetrics
        Basic population count metrics
    population_variance : PopulationVariance
        Statistical variance metrics for the population
    """

    population_metrics: PopulationMetrics
    population_variance: PopulationVariance


@dataclass
class LearningStatistics:
    """Statistics related to learning performance.

    Attributes
    ----------
    learning_progress : LearningProgress
        Overall learning and adaptation statistics
    module_performance : Dict[str, ModulePerformance]
        Performance metrics for individual learning modules
    agent_learning_stats : Dict[str, float]
        Learning statistics for individual agents
    learning_efficiency : float
        Overall learning efficiency
    """

    learning_progress: LearningProgress
    module_performance: Dict[str, ModulePerformance]
    agent_learning_stats: Dict[str, float]
    learning_efficiency: float


@dataclass
class AdvancedStatistics:
    """Comprehensive simulation statistics.

    Attributes
    ----------
    population_metrics : PopulationStatistics
        Detailed population statistics
    interaction_metrics : InteractionPattern
        Statistics about agent interactions
    resource_metrics : ResourceMetrics
        Resource distribution and efficiency metrics
    agent_type_distribution : AgentDistribution
        Distribution of different agent types
    """

    population_metrics: PopulationStatistics
    interaction_metrics: InteractionPattern
    resource_metrics: ResourceMetrics
    agent_type_distribution: AgentDistribution


@dataclass
class InteractionMetrics:
    """Detailed metrics about agent interactions.

    Attributes
    ----------
    total_actions : int
        Total number of interaction actions
    conflict_rate : float
        Rate of conflict interactions
    cooperation_rate : float
        Rate of cooperative interactions
    reproduction_rate : float
        Rate of reproduction events
    interaction_density : float
        Density of interactions in the population
    avg_reward_conflict : float
        Average reward from conflict interactions
    avg_reward_coop : float
        Average reward from cooperative interactions
    interaction_success : float
        Overall success rate of interactions
    """

    total_actions: int
    conflict_rate: float
    cooperation_rate: float
    reproduction_rate: float
    interaction_density: float
    avg_reward_conflict: float
    avg_reward_coop: float
    interaction_success: float


@dataclass
class ActionMetrics:
    """Statistics for a specific action type."""

    action_type: str
    count: int
    frequency: float
    avg_reward: float
    min_reward: float
    max_reward: float
    variance_reward: float
    std_dev_reward: float
    median_reward: float
    quartiles_reward: List[float]
    confidence_interval: float
    interaction_rate: float
    solo_performance: float
    interaction_performance: float
    temporal_patterns: List[Any]
    resource_impacts: List[Any]
    decision_patterns: List[Any]


@dataclass
class DecisionPatternStats:
    """Statistics for a single decision/action type.

    Attributes
    ----------
    action_type : str
        The type of action
    count : int
        Number of times this action was taken
    frequency : float
        Frequency of this action relative to total decisions
    reward_stats : Dict[str, float]
        Statistics about rewards for this action type including:
        - average: Mean reward
        - stddev: Standard deviation of rewards
        - min: Minimum reward received
        - max: Maximum reward received
    """

    action_type: str
    count: int
    frequency: float
    reward_stats: Dict[str, float]


@dataclass
class SequencePattern:
    """Statistics about action sequences.

    Attributes
    ----------
    sequence : str
        The sequence of actions
    count : int
        Number of times this sequence occurred
    probability : float
        Probability of this sequence occurring
    """

    sequence: str
    count: int
    probability: float


@dataclass
class TimePattern:
    """Temporal patterns of action occurrences and rewards over time.

    Attributes:
        action_type: The type of action analyzed.
        time_distribution: A list of action counts per time period.
        reward_progression: A list of average rewards per time period.
        rolling_average_rewards: A list of rolling average rewards.
        rolling_average_counts: A list of rolling average action counts.
    """

    action_type: str
    time_distribution: List[int]
    reward_progression: List[float]
    rolling_average_rewards: List[float]  # Add this field
    rolling_average_counts: List[float]  # Add this field


@dataclass
class InteractionStats:
    """Statistics about agent interactions.

    Attributes
    ----------
    action_type : str
        Type of action analyzed
    interaction_rate : float
        Rate of interactive vs solo actions
    solo_performance : float
        Average reward for solo actions
    interaction_performance : float
        Average reward for interactive actions
    """

    action_type: str
    interaction_rate: float
    solo_performance: float
    interaction_performance: float


@dataclass
class DecisionSummary:
    """Summary statistics about decision making.

    Attributes
    ----------
    total_decisions : int
        Total number of decisions made
    unique_actions : int
        Number of unique action types used
    most_frequent : Optional[str]
        Most frequently taken action
    most_rewarding : Optional[str]
        Action with highest average reward
    action_diversity : float
        Shannon entropy of action distribution
    """

    total_decisions: int
    unique_actions: int
    most_frequent: Optional[str]
    most_rewarding: Optional[str]
    action_diversity: float


@dataclass
class DecisionPatterns:
    """Comprehensive analysis of decision-making patterns.

    Attributes
    ----------
    decision_patterns : Dict[str, DecisionPatternStats]
        Statistics for each action type
    sequence_analysis : Dict[str, SequencePattern]
        Analysis of action sequences
    resource_impact : Dict[str, ResourceImpact]
        Resource impact of different actions
    temporal_patterns : Dict[str, TimePattern]
        Temporal patterns of actions
    interaction_analysis : Dict[str, InteractionStats]
        Analysis of interactive behaviors
    decision_summary : DecisionSummary
        Overall decision-making summary
    """

    decision_patterns: Dict[str, DecisionPatternStats]
    sequence_analysis: Dict[str, SequencePattern]
    resource_impact: Dict[str, ResourceImpact]
    temporal_patterns: Dict[str, TimePattern]
    interaction_analysis: Dict[str, InteractionStats]
    decision_summary: DecisionSummary


@dataclass
class ResourceStates:
    """Represents the state of a resource at a specific simulation step.

    Attributes
    ----------
    resource_id : int
        Unique identifier for the resource instance
    amount : float
        Current amount of the resource available (0.0 to max_amount)
    position_x : float
        X coordinate of resource position in simulation grid (0.0 to grid_width)
    position_y : float
        Y coordinate of resource position in simulation grid (0.0 to grid_height)
    """

    resource_id: int
    amount: float
    position_x: float
    position_y: float


@dataclass
class AgentStates:
    """Data representing agent states from database queries.

    Attributes
    ----------
    step_number : int
        Simulation step number when this state was recorded
    agent_id : str
        Unique identifier for the agent
    agent_type : str
        Type of the agent (system, independent, or control)
    position_x : float
        X coordinate of the agent's position in the simulation grid
    position_y : float
        Y coordinate of the agent's position in the simulation grid
    resource_level : float
        Current resource level of the agent (0.0 to max_resources)
    current_health : float
        Current health level of the agent (0.0 to starting_health)
    is_defending : bool
        Whether the agent is currently in a defensive state
    """

    step_number: int
    agent_id: str
    agent_type: str
    position_x: float
    position_y: float
    resource_level: float
    current_health: float
    is_defending: bool


@dataclass
class LifespanStatistics:
    """Statistics about agent lifespans by type and generation.

    Attributes
    ----------
    average_lifespan : float
        Mean lifespan across all agents
    lifespan_by_type : Dict[str, float]
        Mean lifespan per agent type
    lifespan_by_generation : Dict[int, float]
        Mean lifespan per generation
    maximum_lifespan : float
        Maximum lifespan of any agent
    minimum_lifespan : float
        Minimum lifespan of any agent
    """

    average_lifespan: float
    maximum_lifespan: float
    minimum_lifespan: float
    lifespan_by_type: Dict[str, float]
    lifespan_by_generation: Dict[int, float]


@dataclass
class SurvivalRatesByGeneration:
    """Statistics about survival rates by generation.

    Attributes
    ----------
    rates : Dict[int, float]
        Dictionary mapping generation numbers to their survival rates (0-100).
        Survival rate is the percentage of agents still alive in each generation.
    """

    rates: Dict[int, float]


@dataclass
class AgentLifespanResults:
    """Combined agent lifespan statistics and survival rates.

    Attributes
    ----------
    lifespan_statistics : LifespanStatistics
        Comprehensive lifespan statistics including averages and breakdowns
        by type and generation
    survival_rates : SurvivalRatesByGeneration
        Survival rates for each generation of agents
    """

    lifespan_statistics: LifespanStatistics
    survival_rates: SurvivalRatesByGeneration


@dataclass
class Population:
    """Data for population and resource metrics at a specific step.

    Attributes
    ----------
    step_number : int
        Step number in the simulation
    total_agents : int
        Total number of agents alive at this step
    total_resources : float
        Total resources available across all grid cells
    resources_consumed : float
        Total amount of resources consumed by agents in this step
    """

    step_number: int
    total_agents: int
    total_resources: float
    resources_consumed: float


@dataclass
class BasicPopulationStatistics:
    """Basic population statistics for a simulation run.

    Attributes
    ----------
    avg_population : float
        Average population across all steps
    death_step : int
        Final step number of the simulation
    peak_population : int
        Maximum population reached
    lowest_population : int
        Minimum population reached
    resources_consumed : float
        Total resources consumed across all steps
    resources_available : float
        Total resources available across all steps
    sum_squared : float
        Sum of squared population counts (for variance calculations)
    initial_population : int
        Initial population at the start of the simulation
    final_population : int
        Final population at the end of the simulation
    step_count : int
        Total number of simulation steps
    """

    avg_population: float
    death_step: int
    peak_population: int
    lowest_population: int
    resources_consumed: float
    resources_available: float
    sum_squared: float
    initial_population: int
    final_population: int
    step_count: int


@dataclass
class ResourceDistributionData:
    """Resource distribution data over time.

    Attributes
    ----------
    steps : List[int]
        List of step numbers
    total_resources : List[float]
        Total resources at each step
    average_per_cell : List[float]
        Average resources per grid cell
    distribution_entropy : List[float]
        Resource distribution entropy
    """

    steps: List[int]
    total_resources: List[float]
    average_per_cell: List[float]
    distribution_entropy: List[float]


@dataclass
class ConsumptionStats:
    """Statistics about resource consumption in the simulation.

    Attributes
    ----------
    total_consumed : float
        Total amount of resources consumed across all agents
    avg_consumption_rate : float
        Average rate of resource consumption per step
    peak_consumption : float
        Maximum resource consumption in any single step
    consumption_variance : float
        Variance in resource consumption rates
    """

    total_consumed: float
    avg_consumption_rate: float
    peak_consumption: float
    consumption_variance: float


@dataclass
class ResourceHotspot:
    """Represents a location of high resource concentration.

    Attributes
    ----------
    position_x : float
        X coordinate of the hotspot
    position_y : float
        Y coordinate of the hotspot
    concentration : float
        Resource concentration at this location
    """

    position_x: float
    position_y: float
    concentration: float


@dataclass
class ResourceEfficiencyMetrics:
    """Resource efficiency metrics.

    Attributes
    ----------
    utilization_rate : float
        Resource utilization rate (0-1)
    distribution_efficiency : float
        Resource distribution efficiency (0-1)
    consumption_efficiency : float
        Resource consumption efficiency (0-1)
    regeneration_rate : float
        Resource regeneration rate
    """

    utilization_rate: float
    distribution_efficiency: float
    consumption_efficiency: float
    regeneration_rate: float


@dataclass
class ResourceAnalysis:
    """Comprehensive resource statistics and analysis.

    Attributes
    ----------
    distribution : List[ResourceDistributionStep]
        Resource distribution data over time
    consumption : ConsumptionStats
        Consumption pattern statistics
    hotspots : List[ResourceHotspot]
        Resource concentration points, sorted by concentration
    efficiency : ResourceEfficiencyMetrics
        Resource efficiency metrics
    """

    distribution: List[ResourceDistributionStep]
    consumption: ConsumptionStats
    hotspots: List[ResourceHotspot]
    efficiency: ResourceEfficiencyMetrics


@dataclass
class AgentLearningStats:
    """Statistics about an agent's learning performance.

    Attributes
    ----------
    agent_id : str
        Identifier of the agent
    reward_mean : float
        Average reward achieved by the agent
    total_actions : int
        Total number of actions taken by the agent
    actions_used : List[str]
        List of unique actions performed by the agent
    """

    agent_id: str
    reward_mean: float
    total_actions: int
    actions_used: List[str]


@dataclass
class LearningEfficiencyMetrics:
    """Represents efficiency metrics for learning performance.

    Attributes
    ----------
    reward_efficiency : float
        Average reward across all learning experiences
    action_diversity : float
        Ratio of unique actions to total actions (0-1)
    learning_stability : float
        Measure of learning consistency based on reward variance (0-1)
    """

    reward_efficiency: float
    action_diversity: float
    learning_stability: float


@dataclass
class BasicAgentInfo:
    """Basic statistics for an agent.

    Attributes
    ----------
    agent_id : str
        Unique identifier for the agent
    agent_type : str
        Type of agent (system, independent, control)
    birth_time : datetime
        When the agent was created
    death_time : Optional[datetime]
        When the agent died (None if still alive)
    lifespan : Optional[timedelta]
        How long the agent lived (None if still alive)
    initial_resources : float
        Starting resource amount
    starting_health : float
        Maximum possible health value
    starvation_threshold : float
        Resource level below which agent starts starving
    last_known_position : Tuple[float, float]
        Last recorded position (x, y) coordinates
    generation : int
        Which generation this agent belongs to
    genome_id : str
        Unique identifier for agent's genetic code
    total_actions : int
        Total number of actions taken by agent
    total_health_incidents : int
        Number of health-affecting events
    learning_experiences_count : int
        Number of learning experiences recorded
    times_targeted : int
        Number of times this agent was targeted by others
    """

    agent_id: str
    agent_type: str
    birth_time: datetime
    death_time: Optional[datetime]
    lifespan: Optional[timedelta]
    initial_resources: float
    starting_health: float
    starvation_threshold: float
    genome_id: str
    generation: int


@dataclass
class AgentPerformance:
    """Performance metrics for an agent.

    Attributes
    ----------
    total_actions : int
        Total number of actions taken
    average_reward : float
        Mean reward per action
    action_diversity : int
        Number of unique actions used
    """

    total_actions: int
    average_reward: float
    action_diversity: int


@dataclass
class AgentMetrics:
    """Combined metrics for an agent.

    Attributes
    ----------
    basic_info : BasicAgentInfo
        Basic statistical measures
    performance : AgentPerformance
        Performance-related metrics
    """

    basic_info: BasicAgentInfo
    performance: AgentPerformance


@dataclass
class AgentEvolutionMetrics:
    """Evolution metrics for agents.

    Attributes
    ----------
    total_agents : int
        Total number of agents
    unique_genomes : int
        Number of unique genomes
    average_lifespan : float
        Mean lifespan across agents
    generation : Optional[int]
        Generation number if specific to one generation
    """

    total_agents: int
    unique_genomes: int
    average_lifespan: float
    generation: Optional[int]


@dataclass
class AgentInfo:
    """Basic information about an agent.

    Attributes
    ----------
    agent_id : str
        The unique identifier of the agent
    agent_type : str
        The type of agent
    birth_time : datetime
        When the agent was created
    death_time : Optional[datetime]
        When the agent died (None if still alive)
    lifespan : Optional[timedelta]
        How long the agent lived (None if still alive)
    initial_resources : float
        Starting resource amount
    starting_health : float
        Maximum possible health value
    starvation_threshold : float
        Resource level below which agent starts starving
    """

    agent_id: str
    agent_type: str
    birth_time: datetime
    death_time: Optional[datetime]
    lifespan: Optional[timedelta]
    initial_resources: float
    starting_health: float
    starvation_threshold: float


@dataclass
class AgentGenetics:
    """Genetic information about an agent.

    Attributes
    ----------
    genome_id : str
        Unique identifier for this genome
    generation : int
        Which generation this agent belongs to
    """

    genome_id: str
    generation: int


@dataclass
class AgentHistory:
    """Historical metrics for an agent.

    Attributes
    ----------
    average_health : float
        Mean health value across all states
    average_resources : float
        Mean resource level across all states
    total_steps : int
        Total number of simulation steps
    total_reward : float
        Cumulative reward earned
    """

    average_health: float
    average_resources: float
    total_steps: int
    total_reward: float


@dataclass
class HealthIncidentData:
    """Data about a single health incident.

    Attributes
    ----------
    step : int
        Simulation step when incident occurred
    health_before : float
        Health value before incident
    health_after : float
        Health value after incident
    cause : str
        Reason for health change
    details : Dict[str, Any]
        Additional incident-specific information
    """

    step: int
    health_before: float
    health_after: float
    cause: str
    details: Dict[str, Any]


@dataclass
class ActionStats:
    """Statistics about a specific action type.

    Attributes
    ----------
    count : int
        Number of times this action was taken
    average_reward : float
        Mean reward received for this action
    total_actions : int
        Total number of actions taken by the agent
    action_diversity : int
        Number of unique action types used
    """

    count: int
    average_reward: float
    total_actions: int
    action_diversity: int


@dataclass
class AgentActionHistory:
    """Complete action history for an agent.

    Attributes
    ----------
    actions : Dict[str, ActionStats]
        Dictionary mapping action types to their statistics
    """

    actions: Dict[str, ActionStats]


@dataclass
class CausalAnalysis:
    """Measures the causal impact of actions on outcomes.

    Attributes
    ----------
    action_type : str
        Type of action analyzed
    causal_impact : float
        Average causal effect of the action
    state_transition_probs : Dict[str, float]
        Probabilities of transitioning to states
    """

    action_type: str
    causal_impact: float
    state_transition_probs: Dict[str, float]


@dataclass
class BehaviorClustering:
    """Clustering analysis of agent behavioral patterns.

    Attributes
    ----------
    clusters : Dict[str, List[int]]
        Groups of agent IDs with similar behaviors, where:
        - 'aggressive': Agents with high attack rates
        - 'cooperative': Agents focused on sharing and interaction
        - 'efficient': Agents with high success rates and rewards
        - 'balanced': Agents with mixed behavior patterns

    cluster_characteristics : Dict[str, Dict[str, float]]
        Key behavioral metrics for each cluster, containing:
        - 'attack_rate': Proportion of attack actions
        - 'cooperation': Combined rate of sharing and interactions
        - 'risk_taking': Measure of risky behavior
        - 'success_rate': Proportion of successful actions
        - 'resource_efficiency': Efficiency of resource management

    cluster_performance : Dict[str, float]
        Average reward per action for each cluster
    reduced_features : Optional[Dict[str, Dict]] = None

    Examples
    --------
    >>> clusters = retriever.get_behavior_clustering()
    >>> for name, agents in clusters.clusters.items():
    ...     print(f"{name}: {len(agents)} agents")
    ...     print(f"Performance: {clusters.cluster_performance[name]:.2f}")
    ...     print("Characteristics:", clusters.cluster_characteristics[name])
    Aggressive: 12 agents
    Performance: 1.85
    Characteristics: {'attack_rate': 0.75, 'cooperation': 0.15, 'risk_taking': 0.85}
    Cooperative: 8 agents
    Performance: 2.15
    Characteristics: {'attack_rate': 0.25, 'cooperation': 0.85, 'risk_taking': 0.35}
    """

    clusters: Dict[str, List[int]]
    cluster_characteristics: Dict[str, Dict[str, float]]
    cluster_performance: Dict[str, float]
    reduced_features: Optional[Dict[str, Dict]] = None


@dataclass
class ExplorationExploitation:
    """Analysis of exploration versus exploitation in decision-making.

    Attributes
    ----------
    exploration_rate : float
        Proportion of unique or novel actions
    exploitation_rate : float
        Proportion of repeated successful actions
    reward_comparison : Dict[str, float]
        Comparison of rewards between new and known actions
    """

    exploration_rate: float
    exploitation_rate: float
    reward_comparison: Dict[str, float]


@dataclass
class AdversarialInteractionAnalysis:
    """Performance in competitive scenarios.

    Attributes
    ----------
    win_rate : float
        Proportion of successful adversarial actions
    damage_efficiency : float
        Reward or resources gained per adversarial action
    counter_strategies : Dict[str, float]
        Actions used to counter adversarial behavior
    """

    win_rate: float
    damage_efficiency: float
    counter_strategies: Dict[str, float]


@dataclass
class CollaborativeInteractionAnalysis:
    """Performance in cooperative scenarios.

    Attributes
    ----------
    collaboration_rate : float
        Proportion of actions involving cooperation
    group_reward_impact : float
        Rewards gained from collaborative actions
    synergy_metrics : float
        Efficiency of combined actions by agents
    """

    collaboration_rate: float
    group_reward_impact: float
    synergy_metrics: float


@dataclass
class LearningCurveAnalysis:
    """Tracking agent improvement over time.

    Attributes
    ----------
    action_success_over_time : List[float]
        Improvement in success rates over time
    reward_progression : List[float]
        Changes in average rewards over time
    mistake_reduction : float
        Reduction in suboptimal actions over time
    """

    action_success_over_time: List[float]
    reward_progression: List[float]
    mistake_reduction: float


@dataclass
class EnvironmentalImpactAnalysis:
    """Impact of the environment on actions.

    Attributes
    ----------
    environmental_state_impact : Dict[str, float]
        Correlation between environment and actions
    adaptive_behavior : Dict[str, float]
        Changes in actions due to environment shifts
    spatial_analysis : Dict[str, float]
        Impact of spatial factors on actions
    """

    environmental_state_impact: Dict[str, float]
    adaptive_behavior: Dict[str, float]
    spatial_analysis: Dict[str, float]


@dataclass
class ConflictAnalysis:
    """Insights into conflict dynamics.

    Attributes
    ----------
    conflict_trigger_actions : Dict[str, float]
        Actions most likely to lead to conflicts
    conflict_resolution_actions : Dict[str, float]
        Actions resolving conflicts
    conflict_outcome_metrics : Dict[str, float]
        Success or resource metrics post-conflict
    """

    conflict_trigger_actions: Dict[str, float]
    conflict_resolution_actions: Dict[str, float]
    conflict_outcome_metrics: Dict[str, float]


@dataclass
class RiskRewardAnalysis:
    """Understanding risk-taking behavior.

    Attributes
    ----------
    high_risk_actions : Dict[str, float]
        Actions with high variability in outcomes
    low_risk_actions : Dict[str, float]
        Actions with consistent rewards
    risk_appetite : float
        Proportion of risky actions
    """

    high_risk_actions: Dict[str, float]
    low_risk_actions: Dict[str, float]
    risk_appetite: float


@dataclass
class CounterfactualAnalysis:
    """Simulating alternative outcomes for actions.

    Attributes
    ----------
    counterfactual_rewards : Dict[str, float]
        Estimated rewards for alternative actions
    missed_opportunities : Dict[str, float]
        Gains from unused or underused actions
    strategy_comparison : Dict[str, float]
        Performance of alternative strategies
    """

    counterfactual_rewards: Dict[str, float]
    missed_opportunities: Dict[str, float]
    strategy_comparison: Dict[str, float]


@dataclass
class ResilienceAnalysis:
    """Agent adaptability post-failure.

    Attributes
    ----------
    recovery_rate : float
        Time to recover from failure
    adaptation_rate : float
        Speed of behavioral adjustment
    failure_impact : float
        Resources or rewards lost due to failures
    """

    recovery_rate: float
    adaptation_rate: float
    failure_impact: float


@dataclass
class ActionAnalysis:
    """Comprehensive analysis for a specific action type.

    Attributes
    ----------
    stats : ActionStats
        Basic action metrics
    time_pattern : TimePattern
        Temporal patterns
    resource_impact : ResourceImpact
        Resource effects
    interaction_stats : InteractionStats
        Interaction performance
    sequence_patterns : Dict[str, SequencePattern]
        Action sequence analysis
    """

    stats: ActionStats
    time_pattern: TimePattern
    resource_impact: ResourceImpact
    interaction_stats: InteractionStats
    sequence_patterns: Dict[str, SequencePattern]


@dataclass
class AdvancedActionMetrics:
    """Advanced metrics for action analysis.

    Attributes
    ----------
    causal_analysis : CausalAnalysis
        Causal impact analysis
    behavior_clustering : BehaviorClustering
        Behavioral pattern clusters
    exploration_exploitation : ExplorationExploitation
        Exploration vs exploitation metrics
    adversarial_analysis : AdversarialInteractionAnalysis
        Competitive interaction metrics
    collaborative_analysis : CollaborativeInteractionAnalysis
        Cooperative interaction metrics
    learning_curve : LearningCurveAnalysis
        Learning progression metrics
    environmental_impact : EnvironmentalImpactAnalysis
        Environmental influence metrics
    conflict_analysis : ConflictAnalysis
        Conflict pattern metrics
    risk_reward : RiskRewardAnalysis
        Risk-taking behavior metrics
    counterfactual : CounterfactualAnalysis
        Alternative outcome analysis
    resilience : ResilienceAnalysis
        Failure recovery metrics
    """

    causal_analysis: CausalAnalysis
    behavior_clustering: BehaviorClustering
    exploration_exploitation: ExplorationExploitation
    adversarial_analysis: AdversarialInteractionAnalysis
    collaborative_analysis: CollaborativeInteractionAnalysis
    learning_curve: LearningCurveAnalysis
    environmental_impact: EnvironmentalImpactAnalysis
    conflict_analysis: ConflictAnalysis
    risk_reward: RiskRewardAnalysis
    counterfactual: CounterfactualAnalysis
    resilience: ResilienceAnalysis


@dataclass
class ActionSummary:
    """Summary of an action's overall performance.

    Attributes
    ----------
    total_executions : int
        Total number of times action was executed
    success_rate : float
        Proportion of successful executions
    average_impact : Dict[str, float]
        Average impact on various metrics
    context_effectiveness : Dict[str, float]
        Effectiveness in different contexts
    """

    total_executions: int
    success_rate: float
    average_impact: Dict[str, float]
    context_effectiveness: Dict[str, float]


@dataclass
class ActionContext:
    """Context in which an action was taken.

    Attributes
    ----------
    environmental_state : Dict[str, float]
        State of environment when action was taken
    agent_state : Dict[str, float]
        State of agent when action was taken
    social_context : Dict[str, Any]
        Social/interaction context of the action
    resource_context : Dict[str, float]
        Resource availability context
    """

    environmental_state: Dict[str, float]
    agent_state: Dict[str, float]
    social_context: Dict[str, Any]
    resource_context: Dict[str, float]


@dataclass
class ActionOutcome:
    """Outcome of an action execution.

    Attributes
    ----------
    immediate_reward : float
        Immediate reward received
    state_changes : Dict[str, float]
        Changes in state variables
    side_effects : Dict[str, Any]
        Unintended consequences
    long_term_impact : Dict[str, float]
        Long-term effects of the action
    """

    immediate_reward: float
    state_changes: Dict[str, float]
    side_effects: Dict[str, Any]
    long_term_impact: Dict[str, float]


@dataclass
class ActionAnalysis:
    """Comprehensive analysis of a specific action type.

    Attributes
    ----------
    action_type : str
        The type of action being analyzed
    decision_patterns : DecisionPatterns
        Detailed analysis of decision patterns
    advanced_metrics : AdvancedActionMetrics
        Advanced metrics for this action type
    """

    action_type: str
    decision_patterns: DecisionPatterns
    advanced_metrics: AdvancedActionMetrics


@dataclass
class DecisionPatterns:
    """Comprehensive analysis of decision-making patterns.

    Attributes
    ----------
    decision_patterns : Dict[str, DecisionPatternStats]
        Statistics for each action type
    decision_summary : DecisionSummary
        Overall decision-making summary
    """

    decision_patterns: Dict[str, DecisionPatternStats]
    decision_summary: DecisionSummary


@dataclass
class AgentActionData:
    """Data representing a single agent action.

    Attributes
    ----------
    agent_id : str
        ID of the acting agent
    action_type : str
        Type of action performed
    action_target_id : Optional[int]
        Target agent ID (if any)
    step_number : int
        Simulation step when action occurred
    resources_before : float
        Resource level before action
    resources_after : float
        Resource level after action
    reward : Optional[float]
        Reward received from action
    details : Optional[Dict[str, Any]]
        Additional action-specific details
    """

    agent_id: str
    action_type: str
    step_number: int
    action_target_id: Optional[str] = None
    resources_before: Optional[float] = None
    resources_after: Optional[float] = None
    state_before_id: Optional[str] = None
    state_after_id: Optional[str] = None
    reward: Optional[float] = None
    details: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DecisionPatternStats:
    """Statistics for a single decision/action type."""

    action_type: str
    count: int
    frequency: float
    reward_stats: dict = field(default_factory=dict)
    contribution_metrics: dict = field(default_factory=dict)
    temporal_stats: dict = field(default_factory=dict)
    first_occurrence: dict = field(default_factory=dict)


@dataclass
class DecisionSummary:
    """Summary statistics about decision making."""

    total_decisions: int
    unique_actions: int
    most_frequent: Optional[str]
    most_rewarding: Optional[str]
    action_diversity: float
    normalized_diversity: float
    co_occurrence_patterns: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )


class GenomeId(BaseModel):
    """Structured representation of a genome identifier.

    Format: 'AgentType:generation:parents:time'
    where parents is either 'none' or parent IDs joined by '_'
    """

    agent_type: str
    generation: int
    parent_ids: list[str]
    creation_time: int

    @classmethod
    def from_string(cls, genome_id: str) -> "GenomeId":
        """Parse a genome ID string into a structured object."""
        agent_type, generation, parents, time = genome_id.split(":")
        parent_ids = [] if parents == "none" else parents.split("_")
        return cls(
            agent_type=agent_type,
            generation=int(generation),
            parent_ids=parent_ids,
            creation_time=int(time),
        )

    def to_string(self) -> str:
        """Convert the genome ID object back to string format."""
        parent_str = "_".join(self.parent_ids) if self.parent_ids else "none"
        return f"{self.agent_type}:{self.generation}:{parent_str}:{self.creation_time}"
