# Generic Grid-Based Combat Simulation How-To Guide

## Overview

This guide provides a systematic approach to creating any type of grid-based combat simulation using the existing AgentFarm codebase. The framework is designed to be flexible and extensible for various combat scenarios.

## Prerequisites

- Understanding of the existing codebase structure
- Familiarity with the configuration system (`config/default.yaml`)
- Basic knowledge of the agent framework and combat systems

## Step-by-Step Implementation Process

### Step 1: Define Your Combat Scenario

**Identify Key Elements:**
- **Factions**: How many different types of agents/teams?
- **Territory**: How should the grid be divided initially?
- **Objectives**: What are the victory conditions?
- **Combat Mechanics**: What advantages/disadvantages should each faction have?

**Example Scenarios:**
- Team A vs Team B (symmetric)
- Defenders vs Attackers (asymmetric)
- Multiple factions (free-for-all)
- Territory control with resource competition

### Step 2: Plan Your Environment Structure

**Grid Layout Strategy:**
```python
# Example: 100x100 grid with different spawn zones
class CustomEnvironment(Environment):
    def __init__(self, width=100, height=100, **kwargs):
        super().__init__(width, height, **kwargs)
        
        # Define spawn zones
        self.team_a_zone = (0, 0, 50, 100)  # Left half
        self.team_b_zone = (50, 0, 100, 100)  # Right half
        # Or center vs surrounding areas
        self.center_zone = (30, 30, 70, 70)  # Center 40x40
        self.outer_zone = "surrounding"  # Everything else
```

**Spawn Logic Considerations:**
- Random placement within zones
- Strategic positioning (corners, edges, center)
- Balanced distribution
- Initial resource placement

### Step 3: Design Your Agent Types

**Agent Specialization Strategy:**
```python
# Example: Use BaseAgent with different agent_type and configuration
# Team A agents (aggressive)
def create_team_a_agent(agent_id, position, spatial_service, environment):
    agent = BaseAgent(
        agent_id=agent_id,
        position=position,
        resource_level=5,
        spatial_service=spatial_service,
        environment=environment,
        agent_type="TeamAAgent"  # Custom type identifier
    )
    # Configure combat parameters
    agent.attack_strength = 12.0  # Higher attack
    agent.defense_strength = 1.6  # Lower defense
    agent.max_movement = 8  # Higher movement range
    return agent

# Team B agents (defensive)
def create_team_b_agent(agent_id, position, spatial_service, environment):
    agent = BaseAgent(
        agent_id=agent_id,
        position=position,
        resource_level=5,
        spatial_service=spatial_service,
        environment=environment,
        agent_type="TeamBAgent"  # Custom type identifier
    )
    # Configure combat parameters
    agent.attack_strength = 8.0   # Lower attack
    agent.defense_strength = 2.4  # Higher defense
    agent.max_movement = 6  # Lower movement range
    return agent
```

**Key Attributes to Customize:**
- `attack_strength`: Base damage dealt in combat
- `defense_strength`: Damage reduction when defending
- `max_movement`: Maximum movement distance per turn
- `gathering_range`: Range for resource gathering
- `agent_type`: String identifier for agent classification
- `resource_level`: Starting and current resource amount

### Step 4: Configure Combat Mechanics

**Leverage Existing Combat System:**
```yaml
# config/default.yaml additions
# Combat Parameters (already exist in current config)
starting_health: 100.0
attack_range: 20.0
attack_base_damage: 10.0
attack_kill_reward: 5.0

# Agent-specific parameters for different teams
agent_parameters:
  TeamAAgent:
    attack_strength: 12.0
    defense_strength: 1.6
    max_movement: 8
    gather_efficiency_multiplier: 0.8
    attack_weight: 0.3
  TeamBAgent:
    attack_strength: 8.0
    defense_strength: 2.4
    max_movement: 6
    gather_efficiency_multiplier: 1.2
    attack_weight: 0.1

# Combat enhancements (custom additions)
territory_bonus: 1.3  # Bonus for fighting in own territory
formation_bonus: 1.2  # Bonus for grouped agents
```

**Combat Enhancements to Consider:**
- Territory-based bonuses
- Formation fighting mechanics
- Special abilities or weapons
- Environmental effects
- Resource-based combat advantages

### Step 5: Implement Victory Conditions

**Victory Logic Examples:**
```python
def check_victory_conditions(self):
    """Check if any team has won."""
    team_a_alive = sum(1 for agent in self._agent_objects.values() 
                      if agent.agent_type == "TeamAAgent" and agent.alive)
    team_b_alive = sum(1 for agent in self._agent_objects.values() 
                      if agent.agent_type == "TeamBAgent" and agent.alive)
    
    if team_a_alive == 0:
        return "Team B Victory"
    elif team_b_alive == 0:
        return "Team A Victory"
    elif self.time >= self.max_steps:
        return "Time Limit - Draw"
    
    return None
```

**Common Victory Types:**
- Elimination (destroy all enemies)
- Territory control (control X% of grid)
- Resource accumulation (gather X resources)
- Survival (survive for X turns)
- Objective completion (reach specific locations)

### Step 6: Create Configuration System

**Extend Configuration Structure:**
```python
# Add to existing SimulationConfig in farm/core/config.py
@dataclass
class SimulationConfig:
    # ... existing fields ...
    
    # Combat scenario additions
    scenario_type: str = "team_vs_team"
    team_a_count: int = 10
    team_b_count: int = 10
    victory_condition: str = "elimination"
    max_combat_turns: int = 1000
    
    # Territory settings
    territory_control_threshold: float = 0.8
    resource_control_threshold: int = 100
    
    # Combat modifiers
    territory_bonus: float = 1.3
    formation_bonus: float = 1.2
    special_ability_cooldown: int = 5
```

### Step 7: Implement Special Mechanics

**Advanced Combat Features:**
```python
def calculate_formation_bonus(self, agent, nearby_allies):
    """Calculate bonus for fighting in formation."""
    if len(nearby_allies) >= 3:
        return getattr(self.config, 'formation_bonus', 1.2)
    return 1.0

def apply_territory_bonus(self, agent, position):
    """Apply bonus for fighting in own territory."""
    if self.is_in_territory(agent.agent_type, position):
        return getattr(self.config, 'territory_bonus', 1.3)
    return 1.0

def is_in_territory(self, agent_type, position):
    """Check if position is in agent's territory."""
    x, y = position
    if agent_type == "TeamAAgent":
        return x < self.width // 2  # Left half
    elif agent_type == "TeamBAgent":
        return x >= self.width // 2  # Right half
    return False
```

**Common Special Mechanics:**
- Formation bonuses
- Territory advantages
- Resource-based power scaling
- Environmental hazards
- Special abilities with cooldowns

### Step 8: Set Up Analysis and Tracking

**Extend Existing Tracking:**
```python
# Add to environment
self.team_stats = {
    "TeamAAgent": {"kills": 0, "deaths": 0, "territory_controlled": 0},
    "TeamBAgent": {"kills": 0, "deaths": 0, "territory_controlled": 0}
}

def track_combat_event(self, attacker, defender, damage, killed):
    """Track combat statistics by team."""
    attacker_team = attacker.agent_type
    defender_team = defender.agent_type
    
    if killed:
        self.team_stats[attacker_team]["kills"] += 1
        self.team_stats[defender_team]["deaths"] += 1
```

### Step 9: Testing and Iteration

**Testing Strategy:**
1. **Balance Testing**: Run multiple simulations to ensure fair gameplay
2. **Parameter Tuning**: Adjust multipliers and bonuses for desired difficulty
3. **Edge Case Testing**: Test extreme scenarios and boundary conditions
4. **Performance Testing**: Ensure simulation runs efficiently

**Iteration Process:**
- Start with simple mechanics
- Add complexity gradually
- Test each addition thoroughly
- Balance based on simulation results

## Template Files Structure

**Required New Files:**
```
farm/core/your_scenario_environment.py  # Custom environment class
farm/core/your_scenario_runner.py       # Custom simulation runner
```

**Files to Modify:**
```
config/default.yaml - Add scenario parameters
farm/core/config.py - Add new configuration fields to SimulationConfig
farm/core/action.py - Attack action already uses spatial index for efficient combat
```

## Common Patterns and Best Practices

### Agent Design Patterns
- **Type-based**: Use `BaseAgent` with different `agent_type` strings
- **Configuration**: Customize behavior via `agent_parameters` in config
- **Composition**: Use existing action modules (move, attack, gather)
- **Service Injection**: Leverage spatial service and other services

### Environment Design Patterns
- **Zone-based Spawning**: Define clear spawn areas for each team
- **Resource Management**: Strategic resource placement for conflict
- **Territory Tracking**: Monitor control of different grid areas
- **Agent Lifecycle**: Use existing agent creation and management systems

### Combat Enhancement Patterns
- **Multiplier System**: Use configurable multipliers in `agent_parameters`
- **Bonus Stacking**: Allow multiple bonuses to combine
- **Cooldown Management**: Implement ability cooldowns for balance
- **Spatial Queries**: Leverage existing spatial index for efficient combat targeting

## Troubleshooting Common Issues

**Balance Problems:**
- One faction consistently wins → Adjust attack/defense multipliers
- Combat too fast/slow → Modify health/damage values
- No strategic depth → Add territory or formation bonuses

**Performance Issues:**
- Slow simulation → Optimize agent queries and combat calculations
- Memory leaks → Ensure proper cleanup of dead agents
- Database bloat → Implement data archiving for long simulations

**Learning Problems:**
- Agents not adapting → Check DQN parameters and reward structure
- Stuck in local optima → Increase exploration (epsilon) or add noise
- Poor coordination → Implement team-based rewards

## Conclusion

This framework provides a solid foundation for creating diverse combat simulations. The key is to start simple and gradually add complexity while maintaining balance and performance. The existing combat and learning systems provide powerful tools that can be extended for almost any scenario.

Remember to:
- Leverage existing systems rather than rebuilding
- Use the configuration system for easy parameter tuning
- Test thoroughly at each development stage
- Document your specific scenario requirements clearly
