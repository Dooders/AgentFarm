# Generic Simulation Scenario How-To Guide

## Overview

This guide provides a systematic approach to creating any type of agent-based simulation scenario using the existing AgentFarm codebase. The framework is designed to be flexible and extensible for various simulation types including cooperation, competition, resource management, social dynamics, and more.

## Prerequisites

- Understanding of the existing codebase structure
- Familiarity with the configuration system (`config/default.yaml`)
- Basic knowledge of the agent framework and action systems

## Step-by-Step Implementation Process

### Step 1: Define Your Simulation Scenario

**Identify Key Elements:**
- **Agent Types**: What different types of agents do you need?
- **Environment**: What does your world look like and how is it structured?
- **Objectives**: What are the goals for different agent types?
- **Interactions**: How do agents interact with each other and the environment?
- **Success Metrics**: How do you measure success or completion?

**Example Scenario Types:**
- **Cooperation**: Agents working together to achieve common goals
- **Competition**: Agents competing for limited resources
- **Social Dynamics**: Agents forming relationships, hierarchies, or societies
- **Resource Management**: Agents managing and distributing resources
- **Exploration**: Agents discovering and mapping unknown environments
- **Learning**: Agents adapting and improving over time
- **Ecosystem**: Agents representing different species in an ecological system

### Step 2: Plan Your Environment Structure

**Environment Design Strategy:**
```python
# Example: Custom environment for any scenario
from farm.core.environment import Environment

class CustomScenarioEnvironment(Environment):
    def __init__(self, width=100, height=100, **kwargs):
        super().__init__(width, height, **kwargs)
        
        # Define environment zones or features
        self.resource_zones = []
        self.safe_zones = []
        self.danger_zones = []
        self.interaction_points = []
        
        # Scenario-specific attributes
        self.scenario_phase = "initialization"
        self.success_conditions = {}
        self.failure_conditions = {}
```

**Environment Considerations:**
- **Spatial Structure**: Grid layout, zones, boundaries
- **Resource Distribution**: Where and how resources are placed
- **Environmental Features**: Obstacles, hazards, beneficial areas
- **Dynamic Elements**: Changing conditions, events, or weather
- **Interaction Spaces**: Areas where agents can meet or cooperate

### Step 3: Design Your Agent Types

**Agent Specialization Strategy:**
```python
# Example: Create specialized agent types using AgentFactory
from farm.core.agent import AgentFactory, AgentCore

class SpecialistAgent(AgentCore):
    def __init__(self, specialization="general", **kwargs):
        super().__init__(**kwargs)
        self.specialization = specialization
        self.special_abilities = self._get_special_abilities()
        self.behavior_patterns = self._get_behavior_patterns()

class CooperativeAgent(AgentCore):
    def __init__(self, cooperation_level=0.5, **kwargs):
        super().__init__(**kwargs)
        self.cooperation_level = cooperation_level
        self.trust_network = {}
        self.shared_goals = []

class CompetitiveAgent(AgentCore):
    def __init__(self, aggression_level=0.5, **kwargs):
        super().__init__(**kwargs)
        self.aggression_level = aggression_level
        self.territory_claims = []
        self.rival_agents = []
```

**Key Attributes to Customize:**
- **Behavioral Traits**: Cooperation, aggression, curiosity, etc.
- **Capabilities**: Movement, communication, resource handling
- **Special Abilities**: Unique skills or advantages
- **Learning Parameters**: How quickly they adapt
- **Social Preferences**: Who they prefer to interact with

### Step 4: Configure Interaction Mechanics

**Leverage Existing Action Systems:**

The action system is located in `farm/core/action.py` and includes:
- Action registry for managing available actions
- ActionType enum with predefined action types
- Action execution functions for each behavior type
```yaml
# config/default.yaml additions for any scenario
# Add to existing agent_type_ratios section
agent_type_ratios:
  SystemAgent: 0.4
  IndependentAgent: 0.3
  ControlAgent: 0.3

# Add custom agent parameters
agent_parameters:
  SystemAgent:
    gather_efficiency_multiplier: 0.4
    gather_cost_multiplier: 0.4
    min_resource_threshold: 0.2
    share_weight: 0.3
    attack_weight: 0.05
    # Custom scenario parameters
    cooperation_bonus: 1.3
    specialization_bonus: 1.2
  
  IndependentAgent:
    gather_efficiency_multiplier: 0.7
    gather_cost_multiplier: 0.2
    min_resource_threshold: 0.05
    share_weight: 0.05
    attack_weight: 0.25
    # Custom scenario parameters
    adaptability_bonus: 1.1
    learning_rate: 1.2

# Interaction mechanics (add to existing config)
cooperation_rewards:
  shared_success: 1.5
  team_formation: 1.2
  resource_sharing: 1.1

competition_penalties:
  conflict_cost: -0.3
  territory_dispute: -0.2
```

**Interaction Types to Consider:**
- **Cooperation**: Sharing resources (share action), working together, forming teams
- **Competition**: Fighting (attack action), territory control, resource hoarding
- **Communication**: Information sharing, coordination, negotiation
- **Learning**: Teaching, imitation, knowledge transfer via decision modules
- **Social**: Relationship building, hierarchy formation, group dynamics

**Available Actions in the System:**
- `DEFEND`: Enter defensive stance, reducing incoming damage
- `ATTACK`: Attack nearby agents within range
- `GATHER`: Collect resources from nearby nodes
- `SHARE`: Share resources with nearby allies
- `MOVE`: Move to a new position
- `REPRODUCE`: Create offspring if conditions are met
- `PASS`: Take no action this turn

### Step 5: Implement Success Conditions

**Success Logic Examples:**
```python
# Add to your custom environment class
def check_success_conditions(self):
    """Check if scenario objectives have been met."""
    
    if self.scenario_type == "cooperation":
        return self._check_cooperation_success()
    elif self.scenario_type == "exploration":
        return self._check_exploration_success()
    elif self.scenario_type == "resource_management":
        return self._check_resource_success()
    
    return None

def _check_cooperation_success(self):
    """Check if cooperation goals are met."""
    total_cooperation_events = self.scenario_stats.get("cooperation_events", 0)
    required_cooperation = getattr(self.config, 'required_cooperation_threshold', 100)
    
    if total_cooperation_events >= required_cooperation:
        return "Cooperation Success"
    return None
```

**Common Success Types:**
- **Achievement**: Complete specific tasks or reach goals
- **Survival**: Maintain population or health above threshold
- **Efficiency**: Achieve objectives with minimal resource use
- **Learning**: Demonstrate improved performance over time
- **Social**: Form stable relationships or hierarchies
- **Exploration**: Discover or map significant portions of environment

### Step 6: Create Configuration System

**Extend Configuration Structure:**
```python
# Extend existing SimulationConfig in farm/core/config.py
# Add these fields to the existing SimulationConfig class:

@dataclass
class SimulationConfig:
    # ... existing fields ...
    
    # Scenario-specific additions
    scenario_type: str = "general"
    max_turns: int = 1000
    success_threshold: float = 0.8
    
    # Environment settings
    resource_distribution: str = "random"
    interaction_zones: int = 5
    dynamic_events: bool = False
    
    # Success conditions
    required_cooperation_threshold: int = 100
    exploration_coverage_threshold: float = 0.8
    resource_efficiency_threshold: float = 0.7
    
    # Custom scenario parameters
    cooperation_rewards: Dict[str, float] = field(
        default_factory=lambda: {
            "shared_success": 1.5,
            "team_formation": 1.2,
            "resource_sharing": 1.1
        }
    )
    
    competition_penalties: Dict[str, float] = field(
        default_factory=lambda: {
            "conflict_cost": -0.3,
            "territory_dispute": -0.2
        }
    )
```

### Step 7: Implement Special Mechanics

**Advanced Scenario Features:**
```python
# Add to your custom environment class
def calculate_cooperation_bonus(self, agent, partner):
    """Calculate bonus for cooperative actions."""
    cooperation_rewards = getattr(self.config, 'cooperation_rewards', {})
    base_bonus = cooperation_rewards.get('shared_success', 1.0)
    trust_bonus = getattr(agent, 'trust_network', {}).get(partner.agent_id, 0.1)
    return base_bonus * (1 + trust_bonus)

def apply_learning_improvement(self, agent, action_success):
    """Apply learning improvements based on action outcomes."""
    if action_success:
        if not hasattr(agent, 'learning_progress'):
            agent.learning_progress = 0
        agent.learning_progress += 0.1
        # Learning is handled by the decision modules in farm/core/decision/
        # The agent's decision module will automatically update based on rewards
```

**Common Special Mechanics:**
- **Cooperation Networks**: Trust building and team formation
- **Learning Curves**: Skill improvement over time
- **Social Dynamics**: Relationship formation and maintenance
- **Environmental Adaptation**: Response to changing conditions
- **Resource Economics**: Supply, demand, and value systems

### Step 8: Set Up Analysis and Tracking

**Extend Existing Tracking:**
```python
# Add to custom environment class extending Environment
from farm.core.environment import Environment
from farm.core.metrics_tracker import MetricsTracker

class CustomScenarioEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add scenario-specific tracking
        self.scenario_stats = {
            "cooperation_events": 0,
            "learning_progress": {},
            "social_networks": {},
            "resource_efficiency": 0.0,
            "exploration_coverage": 0.0
        }
    
    def track_scenario_event(self, event_type, agents, outcome):
        """Track scenario-specific events."""
        if event_type == "cooperation":
            self.scenario_stats["cooperation_events"] += 1
        elif event_type == "learning":
            for agent in agents:
                self.scenario_stats["learning_progress"][agent.agent_id] = getattr(agent, 'learning_progress', 0)
```

### Step 9: Testing and Iteration

**Testing Strategy:**
1. **Functionality Testing**: Ensure all mechanics work as intended
2. **Balance Testing**: Verify fair and engaging gameplay
3. **Performance Testing**: Check efficiency with larger populations
4. **Edge Case Testing**: Handle unusual or extreme situations

**Iteration Process:**
- Start with core mechanics
- Add complexity gradually
- Test each addition thoroughly
- Balance based on simulation results
- Refine based on observed behaviors

## Template Files Structure

**Required New Files:**
```
farm/core/your_scenario_environment.py  # Custom environment class
farm/core/your_agent_types.py          # Custom agent classes
farm/core/your_scenario_config.py      # Extended configuration (optional)
farm/core/action.py                     # Add new action types if needed
```

**Files to Modify:**
```
config/default.yaml - Add scenario parameters to existing sections
farm/core/config.py - Extend SimulationConfig class
farm/core/action.py - Add new action types if needed
farm/core/environment.py - Extend Environment class if needed
```

## Integration with Existing Systems

### Decision System Integration

The AgentFarm system uses a sophisticated decision-making system located in `farm/core/decision/`:

- **DecisionModule**: Main decision-making interface using DQN (Deep Q-Network)
- **Action Algorithms**: Specialized algorithms for different behaviors
- **Feature Engineering**: State representation and feature extraction
- **Training System**: Experience replay and model training

To integrate with the decision system:

```python
from farm.core.decision.decision import DecisionModule
from farm.core.decision.config import DecisionConfig

# Your custom agent can use the existing decision system
class CustomAgent(AgentCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The decision module is automatically initialized by AgentCore
        # You can customize the decision config if needed
        
    def decide_action(self):
        # This method is already implemented in AgentCore
        # It uses the DecisionModule to select actions
        return super().decide_action()
```

### Memory System Integration

The system includes Redis-based memory for persistent learning:

```python
from farm.memory.redis_memory import AgentMemoryManager

# Memory is automatically available if configured
# Access through agent.memory_manager if use_memory=True
```

## Common Patterns and Best Practices

### Agent Design Patterns
- **Inheritance**: Extend `AgentCore` from `farm.core.agent` for specialized behavior
- **Composition**: Use existing action modules from `farm.core.action` when possible
- **Configuration**: Make behavior configurable via YAML in `config/default.yaml`
- **Learning**: Implement adaptive behavior mechanisms using existing decision modules

### Environment Design Patterns
- **Zone-based Design**: Define clear areas for different activities
- **Resource Management**: Strategic placement and regeneration using `ResourceManager`
- **Dynamic Elements**: Changing conditions to maintain interest
- **Interaction Spaces**: Areas that encourage desired behaviors
- **Spatial Indexing**: Use existing `SpatialIndex` for efficient proximity queries

### Scenario Enhancement Patterns
- **Progressive Complexity**: Start simple, add features gradually
- **Multiple Objectives**: Give agents competing or complementary goals
- **Social Networks**: Track and influence agent relationships
- **Environmental Feedback**: Let the environment respond to agent actions

## Troubleshooting Common Issues

**Balance Problems:**
- One strategy dominates → Adjust rewards and penalties
- No interesting dynamics → Add more interaction types
- Stagnant behavior → Introduce dynamic elements or learning

**Performance Issues:**
- Slow simulation → Optimize agent queries and calculations
- Memory leaks → Ensure proper cleanup of unused data
- Database bloat → Implement data archiving for long simulations

**Learning Problems:**
- Agents not adapting → Check learning parameters and reward structure
- Stuck in local optima → Increase exploration or add noise
- Poor coordination → Implement better communication mechanisms

## Scenario-Specific Considerations

### Cooperation Scenarios
- **Trust Building**: Implement mechanisms for agents to build trust
- **Team Formation**: Allow agents to form and maintain teams
- **Shared Goals**: Create objectives that require multiple agents
- **Communication**: Enable information sharing and coordination

### Competition Scenarios
- **Resource Scarcity**: Create limited resources to drive competition
- **Territory Control**: Implement spatial competition mechanisms
- **Conflict Resolution**: Provide ways to resolve disputes
- **Balance**: Ensure no single strategy dominates

### Exploration Scenarios
- **Unknown Territory**: Create areas that need to be discovered
- **Information Value**: Make exploration rewarding
- **Risk vs Reward**: Balance safety with discovery incentives
- **Mapping**: Track and visualize discovered areas

### Social Dynamics Scenarios
- **Relationship Building**: Implement social network formation
- **Hierarchy Development**: Allow for leadership and followership
- **Group Formation**: Enable agents to form stable groups
- **Social Learning**: Allow agents to learn from each other

## Conclusion

This framework provides a solid foundation for creating diverse simulation scenarios using the existing AgentFarm codebase. The key is to start simple and gradually add complexity while maintaining balance and performance. The existing agent and environment systems provide powerful tools that can be extended for almost any scenario.

### Key Integration Points:
- **Environment**: Extend `Environment` class from `farm.core.environment`
- **Agents**: Extend `AgentCore` class from `farm.core.agent`
- **Actions**: Use existing action system in `farm.core.action` or add new actions
- **Decisions**: Leverage existing DQN-based decision system in `farm.core.decision`
- **Configuration**: Extend `SimulationConfig` in `farm.core.config`
- **Memory**: Use Redis-based memory system in `farm.memory`

### Remember to:
- Leverage existing systems rather than rebuilding
- Use the configuration system for easy parameter tuning
- Test thoroughly at each development stage
- Document your specific scenario requirements clearly
- Focus on the core mechanics that make your scenario interesting
- Follow the existing code patterns and architecture
- Use the existing service interfaces for dependency injection
