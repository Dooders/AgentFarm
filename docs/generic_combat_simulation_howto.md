# Generic Grid-Based Combat Simulation How-To Guide

## Overview

This guide provides a systematic approach to creating any type of grid-based combat simulation using the existing AgentFarm codebase. The framework is designed to be flexible and extensible for various combat scenarios.

## Prerequisites

- Understanding of the existing codebase structure
- Familiarity with the configuration system (`config.yaml`)
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
class CustomEnvironment(BaseEnvironment):
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
# Example: Extend BaseAgent for different factions
class TeamAAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attack_multiplier = 1.2
        self.defense_multiplier = 0.8
        self.movement_range = 8

class TeamBAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attack_multiplier = 0.8
        self.defense_multiplier = 1.2
        self.movement_range = 6
```

**Key Attributes to Customize:**
- Attack strength and range
- Defense capabilities
- Movement speed and range
- Resource gathering efficiency
- Special abilities or behaviors

### Step 4: Configure Combat Mechanics

**Leverage Existing Combat System:**
```yaml
# config.yaml additions
combat_scenario:
  team_a:
    attack_multiplier: 1.2
    defense_multiplier: 0.8
    movement_range: 8
    special_ability: "charge_attack"
  
  team_b:
    attack_multiplier: 0.8
    defense_multiplier: 1.2
    movement_range: 6
    special_ability: "defensive_formation"

# Combat enhancements
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
    """Check if any faction has won."""
    team_a_alive = sum(1 for agent in self.agents if agent.faction == "A" and agent.alive)
    team_b_alive = sum(1 for agent in self.agents if agent.faction == "B" and agent.alive)
    
    if team_a_alive == 0:
        return "Team B Victory"
    elif team_b_alive == 0:
        return "Team A Victory"
    elif self.time >= self.max_turns:
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
# farm/core/config.py additions
@dataclass
class CombatScenarioConfig:
    scenario_type: str = "team_vs_team"
    team_a_count: int = 10
    team_b_count: int = 10
    victory_condition: str = "elimination"
    max_turns: int = 1000
    
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
        return self.config.formation_bonus
    return 1.0

def apply_territory_bonus(self, agent, position):
    """Apply bonus for fighting in own territory."""
    if self.is_in_territory(agent.faction, position):
        return self.config.territory_bonus
    return 1.0
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
self.faction_stats = {
    "team_a": {"kills": 0, "deaths": 0, "territory_controlled": 0},
    "team_b": {"kills": 0, "deaths": 0, "territory_controlled": 0}
}

def track_combat_event(self, attacker, defender, damage, killed):
    """Track combat statistics by faction."""
    attacker_faction = attacker.faction
    defender_faction = defender.faction
    
    if killed:
        self.faction_stats[attacker_faction]["kills"] += 1
        self.faction_stats[defender_faction]["deaths"] += 1
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
farm/environments/your_scenario_environment.py
farm/agents/your_faction_agent.py
farm/core/your_scenario_config.py
```

**Files to Modify:**
```
config.yaml - Add scenario parameters
farm/core/config.py - Add new configuration classes
farm/core/action.py - Attack action already uses spatial index for efficient combat
```

## Common Patterns and Best Practices

### Agent Design Patterns
- **Inheritance**: Extend `BaseAgent` for faction-specific behavior
- **Composition**: Use existing action modules (move, attack, gather)
- **Configuration**: Make behavior configurable via YAML

### Environment Design Patterns
- **Zone-based Spawning**: Define clear spawn areas for each faction
- **Resource Management**: Strategic resource placement for conflict
- **Territory Tracking**: Monitor control of different grid areas

### Combat Enhancement Patterns
- **Multiplier System**: Use configurable multipliers for balance
- **Bonus Stacking**: Allow multiple bonuses to combine
- **Cooldown Management**: Implement ability cooldowns for balance

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
