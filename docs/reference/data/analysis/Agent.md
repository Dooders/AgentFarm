# Agent Analysis

The `AgentAnalysis` class provides comprehensive analysis of individual agent behavior, learning patterns, and performance metrics in the Farm simulation. This analyzer examines multiple aspects of agent behavior including exploration/exploitation balance, interactions, learning efficiency, and environmental adaptation.

## Overview

The analyzer examines:
- Basic agent information and metrics
- Exploration vs exploitation patterns
- Adversarial and collaborative interactions
- Learning curves and adaptation
- Risk-reward profiles
- Environmental impact and resilience

## Key Features

### 1. Basic Agent Analysis
```python
basic_info = analyzer.analyze(agent_id="agent_1")
```
Returns `BasicAgentInfo` containing:
- Agent identification and type
- Temporal data (birth/death times)
- Resource metrics
- Genealogical information

### 2. Exploration/Exploitation Analysis
```python
explore_stats = analyzer.analyze_exploration_exploitation(agent_id="agent_1")
```
Examines:
- Exploration rate (new action attempts)
- Exploitation rate (repeated actions)
- Reward comparisons between strategies
- Action diversity metrics

### 3. Interaction Analysis

#### Adversarial Interactions
```python
adversarial = analyzer.analyze_adversarial_interactions(agent_id="agent_1")
```
Tracks:
- Win rates in competitive scenarios
- Damage efficiency
- Counter-strategy patterns
- Combat effectiveness

#### Collaborative Interactions
```python
collab = analyzer.analyze_collaboration(agent_id="agent_1")
```
Measures:
- Collaboration frequency
- Group reward impact
- Synergy metrics
- Cooperative efficiency

### 4. Learning Analysis
```python
learning = analyzer.analyze_learning_curve(agent_id="agent_1")
```
Tracks:
- Success rates over time
- Reward progression
- Error reduction
- Skill acquisition patterns

### 5. Risk-Reward Analysis
```python
risk = analyzer.analyze_risk_reward(agent_id="agent_1")
```
Evaluates:
- High-risk action patterns
- Low-risk action patterns
- Risk appetite metrics
- Reward optimization

### 6. Environmental Impact
```python
impact = analyzer.analyze_environmental_impact(agent_id="agent_1")
```
Analyzes:
- Resource utilization
- Environmental state impact
- Adaptive behaviors
- Spatial effects

### 7. Resilience Analysis
```python
resilience = analyzer.analyze_resilience(agent_id="agent_1")
```
Measures:
- Recovery from failures
- Adaptation speed
- Impact assessment
- Stability metrics

### 8. Conflict Analysis
```python
conflicts = analyzer.analyze_conflicts(agent_id="agent_1")
```
Analyzes:
- Conflict frequency and patterns
- Resolution strategies
- Impact on performance
- Conflict avoidance behaviors

### 9. Counterfactual Analysis
```python
counterfactuals = analyzer.analyze_counterfactuals(agent_id="agent_1")
```
Examines:
- Alternative action scenarios
- What-if analysis
- Opportunity costs
- Decision quality assessment

## Detailed Metrics

### Basic Agent Metrics

The fundamental metrics that define an agent's characteristics and state:

- **Agent Identification**
  - `agent_id`: Unique identifier for the agent
  - `agent_type`: Classification of agent's primary role (e.g., harvester, defender)
  - `genome_id`: Identifier for agent's genetic template. Format: `parent1:parent2:counter` where parents are agent IDs and counter (>= 1) distinguishes multiple offspring from the same parent(s). First offspring is :1, second is :2, etc.

- **Temporal Metrics**
  - `birth_time`: When agent was created/spawned
  - `death_time`: When agent ceased to exist (if applicable)
  - `lifespan`: Total duration of agent's existence

- **Resource Metrics**
  - `initial_resources`: Starting resource allocation
  - `starting_health`: Initial health value
  - `starvation_counter`: Current count of consecutive steps with zero resources

- **Genealogical Data**
  - `generation`: Agent's position in evolutionary lineage
  - `parent_ids`: Identifiers of agent's predecessors

### Exploration/Exploitation Metrics

Measures how agents balance discovering new behaviors versus utilizing known strategies:

- **Exploration Metrics**
  - `exploration_rate`: Proportion of new action attempts
  - `new_action_rewards`: Success of experimental behaviors
  - `discovery_efficiency`: Value gained from exploration

- **Exploitation Metrics**
  - `exploitation_rate`: Proportion of repeated actions
  - `known_action_rewards`: Success of established behaviors
  - `optimization_level`: Efficiency in using known strategies

### Interaction Metrics

#### Adversarial Interactions
Measures competitive performance and combat effectiveness:

- **Combat Effectiveness**
  - `win_rate`: Success rate in competitive encounters
  - `damage_efficiency`: Ratio of damage dealt vs. received
  - `survival_rate`: Success in avoiding elimination

- **Strategic Metrics**
  - `counter_strategy_diversity`: Range of defensive responses
  - `adaptation_speed`: Rate of tactical adjustment
  - `engagement_selectivity`: Wisdom in choosing battles

#### Collaborative Interactions
Measures cooperative behaviors and team effectiveness:

- **Cooperation Metrics**
  - `collaboration_frequency`: Rate of cooperative actions
  - `synergy_score`: Enhanced outcomes from cooperation
  - `resource_sharing_efficiency`: Effectiveness of resource distribution

- **Social Metrics**
  - `group_contribution`: Impact on collective success
  - `coordination_level`: Ability to align with others
  - `trust_rating`: Reliability in cooperative scenarios

### Learning Metrics

Measures how agents improve and adapt over time:

- **Progress Metrics**
  - `success_rate_progression`: Improvement in task completion
  - `mistake_reduction`: Decrease in error frequency
  - `adaptation_speed`: Rate of behavioral improvement

- **Skill Development**
  - `skill_mastery_levels`: Proficiency in different abilities
  - `learning_efficiency`: Resource cost of improvement
  - `knowledge_retention`: Stability of learned behaviors

### Risk-Reward Metrics

Evaluates agent's decision-making under uncertainty:

- **Risk Assessment**
  - `risk_appetite`: Willingness to take chances
  - `risk_adjusted_returns`: Rewards normalized by risk
  - `loss_avoidance`: Success in preventing negative outcomes

- **Decision Quality**
  - `decision_consistency`: Stability of choice patterns
  - `opportunity_recognition`: Ability to spot advantages
  - `risk_diversification`: Balance in risk exposure

### Environmental Impact Metrics

Measures agent's interaction with and impact on environment:

- **Resource Interaction**
  - `resource_efficiency`: Effective use of available resources
  - `environmental_footprint`: Impact on resource availability
  - `sustainability_score`: Long-term resource management

- **Adaptation Metrics**
  - `environmental_responsiveness`: Adjustment to changes
  - `habitat_utilization`: Effective use of space
  - `niche_specialization`: Development of specialized behaviors

### Resilience Metrics

Measures agent's ability to handle adversity:

- **Recovery Metrics**
  - `recovery_rate`: Speed of bouncing back from setbacks
  - `adaptation_flexibility`: Range of coping strategies
  - `stability_index`: Resistance to disruption

- **Sustainability Metrics**
  - `long_term_viability`: Projected survival probability
  - `stress_tolerance`: Ability to function under pressure
  - `resource_buffer`: Maintenance of safety margins

These metrics provide a comprehensive view of agent performance and behavior across multiple dimensions, enabling detailed analysis of individual and collective agent dynamics within the simulation.

## Metric Interpretation Guide

### Basic Agent Metrics Interpretation

```python
basic_info = analyzer.analyze(agent_id="agent_1")
```

**Example Values:**
```
agent_type: "harvester"
lifespan: 32109 seconds
initial_resources: 100.0
generation: 2
```

**How to Interpret:**
- Lifespans < 10000 seconds suggest early termination or survival issues
- Initial resources > 150 indicate favorable starting conditions
- Higher generations (>5) suggest successful evolutionary lineage
- Starvation threshold should be ~20% of initial resources for balance

### Exploration/Exploitation Balance

```python
explore_stats = analyzer.analyze_exploration_exploitation(agent_id="agent_1")
```

**Example Values:**
```
exploration_rate: 0.35
exploitation_rate: 0.65
new_action_rewards: 2.5
known_action_rewards: 3.8
```

**How to Interpret:**
- Healthy exploration rate: 0.2-0.4 (too high = inefficient, too low = stagnant)
- Optimal exploitation rate: 0.6-0.8
- New/known reward ratio > 0.7 indicates effective exploration
- Declining exploration rate over time is normal and often desirable

### Combat and Interaction Analysis

```python
adversarial = analyzer.analyze_adversarial_interactions(agent_id="agent_1")
```

**Example Values:**
```
win_rate: 0.65
damage_efficiency: 0.8
counter_strategy_diversity: 0.7
```

**What It Means:**
- Win rates > 0.6 indicate strong competitive ability
- Damage efficiency > 1.0 shows favorable trade-offs
- Counter strategy diversity > 0.5 suggests adaptability
- Low win rate + high damage efficiency = efficient fighter but poor finisher

### Learning Progress Indicators

```python
learning = analyzer.analyze_learning_curve(agent_id="agent_1")
```

**Example Values:**
```
success_rate_progression: [0.4, 0.5, 0.6, 0.7]
mistake_reduction: 0.6
learning_efficiency: 0.8
```

**Key Patterns:**
- Steady increase in success rate indicates healthy learning
- Mistake reduction > 0.5 shows good improvement
- Learning efficiency > 0.7 is excellent
- Plateaus in progression may indicate:
  - Reached skill ceiling
  - Environmental limitations
  - Need for new strategies

### Risk-Reward Pattern Analysis

```python
risk = analyzer.analyze_risk_reward(agent_id="agent_1")
```

**Example Values:**
```
risk_appetite: 0.4
risk_adjusted_returns: 1.8
decision_consistency: 0.75
```

**Interpretation Guidelines:**
- Risk appetite 0.3-0.5 is balanced
- Risk-adjusted returns > 1.5 indicate efficient risk-taking
- Decision consistency > 0.7 shows stable strategy
- High risk + low returns suggests need for strategy adjustment

### Environmental Impact Assessment

```python
impact = analyzer.analyze_environmental_impact(agent_id="agent_1")
```

**Example Values:**
```
resource_efficiency: 0.85
environmental_footprint: 0.3
sustainability_score: 0.75
```

**What to Look For:**
- Resource efficiency > 0.8 is excellent
- Environmental footprint < 0.4 is sustainable
- Sustainability score > 0.7 indicates long-term viability
- Trends in these metrics more important than absolute values

### Resilience Metric Patterns

```python
resilience = analyzer.analyze_resilience(agent_id="agent_1")
```

**Example Values:**
```
recovery_rate: 0.8
stability_index: 0.7
stress_tolerance: 0.65
```

**Pattern Analysis:**
- Recovery rate > 0.7 shows strong resilience
- Stability index > 0.6 indicates robust behavior
- Stress tolerance patterns:
  - Increasing = developing resilience
  - Decreasing = potential burnout
  - Stable = established coping mechanisms

## Common Metric Combinations

### Survival Potential
```python
survival_potential = (
    basic_info.starting_health * 
    resilience.stability_index * 
    impact.resource_efficiency
)
```
- High (>0.8): Excellent survival prospects
- Medium (0.4-0.8): Viable but may need support
- Low (<0.4): At risk, needs intervention

### Learning Effectiveness
```python
learning_effectiveness = (
    learning.success_rate_progression[-1] *
    explore_stats.exploitation_rate *
    resilience.adaptation_flexibility
)
```
- High (>0.7): Effective learner
- Medium (0.3-0.7): Steady progress
- Low (<0.3): Learning challenges

### Competitive Fitness
```python
competitive_fitness = (
    adversarial.win_rate *
    risk.risk_adjusted_returns *
    resilience.recovery_rate
)
```
- High (>0.6): Strong competitor
- Medium (0.3-0.6): Competent
- Low (<0.3): Needs strategic improvement

## Usage Examples

### Comprehensive Agent Analysis
```python
from farm.database.analyzers.agent_analyzer import AgentAnalysis
from farm.database.repositories.agent_repository import AgentRepository

# Initialize analyzer
repository = AgentRepository(session)
analyzer = AgentAnalysis(repository)

# Get comprehensive analysis
agent_id = "agent_1"
basic_info = analyzer.analyze(agent_id)
exploration = analyzer.analyze_exploration_exploitation(agent_id)
learning = analyzer.analyze_learning_curve(agent_id)
resilience = analyzer.analyze_resilience(agent_id)

# Print key metrics
print(f"Agent Type: {basic_info.agent_type}")
print(f"Exploration Rate: {exploration.exploration_rate:.2%}")
print(f"Learning Progress: {learning.mistake_reduction:.2%}")
print(f"Recovery Rate: {resilience.recovery_rate:.2f}")
```

### Focused Interaction Analysis
```python
# Analyze both competitive and cooperative behaviors
adversarial = analyzer.analyze_adversarial_interactions(agent_id)
collaborative = analyzer.analyze_collaboration(agent_id)

print(f"Win Rate: {adversarial.win_rate:.2%}")
print(f"Collaboration Rate: {collaborative.collaboration_rate:.2%}")
```

### Environmental Adaptation
```python
# Analyze environmental impact and adaptation
impact = analyzer.analyze_environmental_impact(agent_id)
print(f"Resource Efficiency: {impact.resource_efficiency:.2%}")
```

### Conflict and Counterfactual Analysis
```python
# Analyze conflicts and alternative scenarios
conflicts = analyzer.analyze_conflicts(agent_id)
counterfactuals = analyzer.analyze_counterfactuals(agent_id)

print(f"Conflict Frequency: {conflicts.conflict_frequency:.2%}")
print(f"Opportunity Cost: {counterfactuals.opportunity_cost:.2f}")
```

## Analysis Parameters

Most analysis methods accept common parameters:
- `agent_id`: Target agent identifier (optional for population-level analysis)
- `scope`: Analysis scope (SIMULATION, EPISODE)
- `step`: Specific timestep
- `step_range`: Analysis period

## Integration Points

The AgentAnalysis integrates with:
- Population analysis for context
- Resource analysis for efficiency metrics
- Learning analysis for performance evaluation
- Spatial analysis for movement patterns

## Performance Considerations

- Caches agent data for repeated analysis
- Supports incremental analysis
- Handles missing data gracefully
- Scales with agent history size

## Best Practices

1. **Analysis Scope**
   - Start with basic agent info
   - Add specific analyses as needed
   - Consider temporal context

2. **Performance Optimization**
   - Use step ranges for large datasets
   - Cache frequent analyses
   - Filter unnecessary metrics

3. **Data Interpretation**
   - Consider agent lifespan
   - Account for environmental factors
   - Compare against population averages

## Error Handling

The analyzer handles common issues:
- Missing agent data
- Incomplete histories
- Invalid time ranges
- Null values in metrics

## Future Considerations

The analyzer is designed for extension:
- New analysis metrics
- Custom scoring systems
- Additional behavioral patterns
- Enhanced visualization support
