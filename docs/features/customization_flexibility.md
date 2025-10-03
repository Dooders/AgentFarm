# Customization & Flexibility

![Feature](https://img.shields.io/badge/feature-customization-green)

## Overview

AgentFarm's Customization & Flexibility framework empowers researchers and developers to tailor every aspect of their simulations to match specific research goals and experimental designs. From simple parameter adjustments to complete behavioral overhauls, AgentFarm provides the tools and extensibility needed for any agent-based modeling scenario.

### Why Customization Matters

Agent-based modeling requires flexibility because:
- **Research Diversity**: Different research questions demand different simulation designs
- **Domain Specificity**: Each field (ecology, economics, social science) has unique requirements
- **Experimental Control**: Scientific rigor requires precise control over variables
- **Iterative Refinement**: Research evolves, and your simulation framework should too

---

## Core Capabilities

### 1. Define Custom Parameters, Rules, and Environments

AgentFarm provides a comprehensive configuration system that allows you to customize every aspect of your simulation without writing code.

#### Hierarchical Configuration System

AgentFarm uses a layered configuration approach:

```
Base Configuration (default.yaml)
    ↓
Environment Override (development/production/testing)
    ↓
Profile Override (benchmark/research/simulation)
    ↓
Runtime Parameters
```

**Benefits:**
- **Separation of Concerns**: Keep different aspects of configuration organized
- **Reusability**: Share base configurations across experiments
- **Flexibility**: Override only what you need
- **Version Control**: Track configuration changes alongside code

#### Basic Configuration Usage

```python
from farm.config import SimulationConfig

# Load with environment-specific settings
config = SimulationConfig.from_centralized_config(
    environment="development"  # or "production", "testing"
)

# Load with specialized profile
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="research"  # or "benchmark", "simulation"
)

# Override specific parameters at runtime
config.num_steps = 2000
config.system_agents = 50
config.initial_resources = 500
```

#### Custom Environment Rules

Create environments with specialized rules and constraints:

```python
from farm.core.environment import Environment

class CustomEnvironment(Environment):
    """Environment with custom rules for resource dynamics."""
    
    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, **kwargs)
        
        # Custom environment properties
        self.resource_decay_rate = kwargs.get('resource_decay_rate', 0.05)
        self.resource_clustering_factor = kwargs.get('clustering_factor', 2.0)
        self.seasonal_variation = kwargs.get('seasonal_variation', True)
        self.current_season = 'spring'
        
    def step_resources(self):
        """Custom resource regeneration logic."""
        # Seasonal resource dynamics
        if self.seasonal_variation:
            regen_multiplier = self._get_seasonal_multiplier()
        else:
            regen_multiplier = 1.0
            
        # Apply custom regeneration
        for resource_id, resource in self.resources.items():
            # Decay existing resources
            resource.amount *= (1 - self.resource_decay_rate)
            
            # Regenerate based on season and location
            regen_amount = self.base_regen_rate * regen_multiplier
            resource.amount = min(
                resource.amount + regen_amount,
                self.max_resource_amount
            )
            
    def _get_seasonal_multiplier(self) -> float:
        """Get resource multiplier based on season."""
        season_multipliers = {
            'spring': 1.5,  # Abundant growth
            'summer': 1.2,  # Steady resources
            'fall': 0.8,    # Declining resources
            'winter': 0.3   # Scarce resources
        }
        return season_multipliers.get(self.current_season, 1.0)
        
    def advance_season(self):
        """Progress to next season."""
        seasons = ['spring', 'summer', 'fall', 'winter']
        current_idx = seasons.index(self.current_season)
        self.current_season = seasons[(current_idx + 1) % 4]
```

#### Custom Resource Distribution

Define exactly how resources are distributed in your environment:

```python
# Clustered distribution (resources in groups)
environment = Environment(
    width=100,
    height=100,
    resource_distribution={
        "type": "clustered",
        "amount": 500,
        "num_clusters": 10,
        "cluster_radius": 15
    }
)

# Gradient distribution (resources vary spatially)
environment = Environment(
    width=100,
    height=100,
    resource_distribution={
        "type": "gradient",
        "amount": 500,
        "gradient_direction": "north",  # More resources in north
        "gradient_strength": 0.8
    }
)

# Custom distribution function
def custom_distribution(width: int, height: int, amount: int):
    """Place resources in a ring pattern."""
    resources = []
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    for i in range(amount):
        angle = (2 * np.pi * i) / amount
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        resources.append((x, y, 5))  # position and amount
        
    return resources

environment = Environment(
    width=100,
    height=100,
    resource_distribution={
        "type": "custom",
        "function": custom_distribution,
        "amount": 500
    }
)
```

#### Environmental Zones and Features

Create specialized zones with different properties:

```python
class ZonedEnvironment(Environment):
    """Environment with distinct zones having different properties."""
    
    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, **kwargs)
        
        # Define zones
        self.zones = {
            'safe_zone': {
                'bounds': (0, 0, 30, 30),
                'properties': {
                    'no_combat': True,
                    'resource_bonus': 1.5,
                    'movement_cost': 0.5
                }
            },
            'danger_zone': {
                'bounds': (70, 70, 100, 100),
                'properties': {
                    'combat_damage_multiplier': 2.0,
                    'resource_penalty': 0.5,
                    'movement_cost': 1.5
                }
            },
            'neutral_zone': {
                'bounds': (30, 30, 70, 70),
                'properties': {
                    'no_modifiers': True
                }
            }
        }
        
    def get_zone_at_position(self, position: Tuple[float, float]) -> str:
        """Determine which zone contains a position."""
        x, y = position
        for zone_name, zone_data in self.zones.items():
            x1, y1, x2, y2 = zone_data['bounds']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return 'neutral_zone'
        
    def get_zone_properties(self, position: Tuple[float, float]) -> Dict:
        """Get properties for zone at position."""
        zone_name = self.get_zone_at_position(position)
        return self.zones[zone_name]['properties']
```

---

### 2. Create Specialized Agent Behaviors and Properties

Customize agent behaviors to match your research requirements, from simple trait adjustments to complete behavioral overhauls.

#### Custom Agent Types

Extend `BaseAgent` to create specialized agent types:

```python
from farm.core.agent import BaseAgent
from typing import Tuple, Optional, List

class ResearcherAgent(BaseAgent):
    """Agent specialized in exploration and knowledge gathering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Research-specific attributes
        self.discovered_locations = set()
        self.knowledge_base = {}
        self.exploration_radius = kwargs.get('exploration_radius', 10)
        self.curiosity_level = kwargs.get('curiosity_level', 0.8)
        
    def decide_action(self):
        """Decision-making prioritizing exploration."""
        # Prioritize exploring unknown areas
        if self._should_explore():
            return self._plan_exploration_move()
            
        # Otherwise, use standard behavior
        return super().decide_action()
        
    def _should_explore(self) -> bool:
        """Decide whether to explore."""
        # Higher curiosity = more exploration
        return random.random() < self.curiosity_level
        
    def _plan_exploration_move(self):
        """Plan movement to unexplored area."""
        # Find least-visited nearby location
        candidates = self._get_unexplored_positions()
        if candidates:
            target = random.choice(candidates)
            return {
                'action_type': 'move',
                'target_position': target
            }
        return {'action_type': 'pass'}
        
    def _get_unexplored_positions(self) -> List[Tuple[int, int]]:
        """Get nearby positions not yet visited."""
        unexplored = []
        x, y = self.position
        
        for dx in range(-self.exploration_radius, self.exploration_radius + 1):
            for dy in range(-self.exploration_radius, self.exploration_radius + 1):
                pos = (x + dx, y + dy)
                if (self.environment.is_valid_position(pos) and 
                    pos not in self.discovered_locations):
                    unexplored.append(pos)
                    
        return unexplored
        
    def update_knowledge(self, observation: Dict):
        """Record observations in knowledge base."""
        position = tuple(self.position)
        self.discovered_locations.add(position)
        self.knowledge_base[position] = {
            'timestamp': self.environment.current_step,
            'resources': observation.get('nearby_resources', 0),
            'agents': observation.get('nearby_agents', [])
        }


class TraderAgent(BaseAgent):
    """Agent specialized in resource trading and exchange."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Trading-specific attributes
        self.inventory = {}
        self.trade_history = []
        self.trading_partners = set()
        self.price_memory = {}  # Remember what resources are worth
        self.negotiation_skill = kwargs.get('negotiation_skill', 0.5)
        
    def decide_action(self):
        """Decision-making prioritizing trade."""
        # Look for trading opportunities
        if self._should_trade():
            partner, offer = self._find_trade_opportunity()
            if partner:
                return {
                    'action_type': 'trade',
                    'partner': partner,
                    'offer': offer
                }
                
        return super().decide_action()
        
    def _should_trade(self) -> bool:
        """Decide whether to attempt trade."""
        # Trade when we have surplus or need specific resources
        has_surplus = self.resource_level > 150
        has_demand = any(
            self.inventory.get(res_type, 0) < 10 
            for res_type in ['food', 'materials', 'energy']
        )
        return has_surplus or has_demand
        
    def _find_trade_opportunity(self) -> Tuple[Optional[str], Dict]:
        """Find beneficial trade with nearby agent."""
        nearby = self.environment.get_nearby_agents(
            self.position, 
            radius=5
        )
        
        best_partner = None
        best_offer = {}
        best_value = 0
        
        for agent_id in nearby:
            if agent_id == self.agent_id:
                continue
                
            agent = self.environment.agents[agent_id]
            offer = self._evaluate_trade_with(agent)
            value = offer.get('expected_value', 0)
            
            if value > best_value:
                best_partner = agent_id
                best_offer = offer
                best_value = value
                
        return best_partner, best_offer
        
    def _evaluate_trade_with(self, agent: BaseAgent) -> Dict:
        """Evaluate potential trade value with agent."""
        # Complex trading logic based on:
        # - Our needs vs their surplus
        # - Historical prices
        # - Negotiation skills
        # - Relationship history
        
        offer = {
            'give': min(self.resource_level // 10, 20),
            'request': 15,
            'expected_value': 0
        }
        
        # Adjust based on negotiation skill
        offer['expected_value'] = (
            offer['request'] - offer['give']
        ) * (1 + self.negotiation_skill)
        
        return offer
        
    def record_trade(self, partner_id: str, details: Dict):
        """Record completed trade for future reference."""
        self.trade_history.append({
            'step': self.environment.current_step,
            'partner': partner_id,
            'details': details
        })
        self.trading_partners.add(partner_id)


class LeaderAgent(BaseAgent):
    """Agent that can coordinate and direct other agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Leadership attributes
        self.followers = set()
        self.leadership_radius = kwargs.get('leadership_radius', 15)
        self.charisma = kwargs.get('charisma', 0.7)
        self.strategy = kwargs.get('strategy', 'balanced')
        self.commands_issued = []
        
    def decide_action(self):
        """Leaders decide for themselves and coordinate followers."""
        # First, assess situation and decide group strategy
        group_action = self._coordinate_followers()
        
        # Then make own decision
        own_action = super().decide_action()
        
        return {
            'own_action': own_action,
            'group_coordination': group_action
        }
        
    def _coordinate_followers(self) -> Dict:
        """Issue coordinated instructions to followers."""
        nearby_agents = self.environment.get_nearby_agents(
            self.position,
            self.leadership_radius
        )
        
        # Assess group needs
        group_status = self._assess_group_status(nearby_agents)
        
        # Decide strategy based on situation
        if group_status['avg_resources'] < 50:
            # Focus on gathering
            command = {
                'type': 'gather',
                'priority': 'high',
                'target_area': self._find_resource_rich_area()
            }
        elif group_status['threats_detected']:
            # Defensive formation
            command = {
                'type': 'defend',
                'priority': 'urgent',
                'rally_point': self.position
            }
        else:
            # Explore and expand
            command = {
                'type': 'explore',
                'priority': 'medium',
                'direction': self._choose_exploration_direction()
            }
            
        self.commands_issued.append(command)
        return command
        
    def _assess_group_status(self, agent_ids: List[str]) -> Dict:
        """Evaluate status of follower group."""
        status = {
            'total_followers': len(agent_ids),
            'avg_resources': 0,
            'threats_detected': False,
            'cohesion': 0
        }
        
        if not agent_ids:
            return status
            
        resources = []
        positions = []
        
        for agent_id in agent_ids:
            agent = self.environment.agents.get(agent_id)
            if agent and agent.alive:
                resources.append(agent.resource_level)
                positions.append(agent.position)
                
        status['avg_resources'] = np.mean(resources) if resources else 0
        status['cohesion'] = self._calculate_cohesion(positions)
        
        return status
        
    def _calculate_cohesion(self, positions: List[Tuple]) -> float:
        """Calculate how tightly grouped followers are."""
        if len(positions) < 2:
            return 1.0
            
        # Calculate average distance between all pairs
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(
                    np.array(positions[i]) - np.array(positions[j])
                )
                distances.append(dist)
                
        avg_distance = np.mean(distances)
        # Convert to cohesion score (0-1, higher = more cohesive)
        cohesion = 1.0 / (1.0 + avg_distance / 10)
        return cohesion
```

#### Custom Behavioral Traits

Add personality traits that influence behavior:

```python
from dataclasses import dataclass

@dataclass
class AgentPersonality:
    """Define agent personality traits."""
    
    # Social traits
    cooperativeness: float = 0.5  # 0=selfish, 1=altruistic
    aggressiveness: float = 0.5   # 0=passive, 1=hostile
    trustfulness: float = 0.5     # 0=suspicious, 1=trusting
    
    # Behavioral traits
    risk_tolerance: float = 0.5   # 0=cautious, 1=reckless
    patience: float = 0.5          # 0=impulsive, 1=patient
    curiosity: float = 0.5         # 0=conservative, 1=exploratory
    
    # Cognitive traits
    memory_strength: float = 0.5   # How well they remember past events
    learning_rate: float = 0.5     # How quickly they adapt
    planning_depth: int = 3        # How many steps ahead they think
    
    def apply_to_action_weights(self, base_weights: Dict) -> Dict:
        """Modify action probabilities based on personality."""
        weights = base_weights.copy()
        
        # Cooperative agents share more
        weights['share'] *= (1 + self.cooperativeness)
        
        # Aggressive agents attack more
        weights['attack'] *= (1 + self.aggressiveness)
        
        # Curious agents explore more
        weights['move'] *= (1 + self.curiosity * 0.5)
        
        # Risk-tolerant agents try riskier actions
        if self.risk_tolerance > 0.7:
            weights['attack'] *= 1.3
            weights['explore'] *= 1.2
            
        return weights


class PersonalizedAgent(BaseAgent):
    """Agent with personality-driven behavior."""
    
    def __init__(self, *args, personality: Optional[AgentPersonality] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Assign or generate personality
        self.personality = personality or AgentPersonality(
            cooperativeness=random.random(),
            aggressiveness=random.random(),
            trustfulness=random.random(),
            risk_tolerance=random.random(),
            patience=random.random(),
            curiosity=random.random()
        )
        
    def decide_action(self):
        """Make decisions influenced by personality."""
        # Get base action weights
        base_weights = self._get_base_action_weights()
        
        # Apply personality modifications
        modified_weights = self.personality.apply_to_action_weights(base_weights)
        
        # Choose action based on modified weights
        return self._select_action(modified_weights)
```

---

### 3. Configure Simulation Parameters and Conditions

Fine-tune simulation behavior through comprehensive parameter configuration.

#### Configuration Files

AgentFarm uses YAML configuration files for easy parameter management:

**`farm/config/default.yaml`** - Base configuration:
```yaml
# Environment settings
width: 100
height: 100

# Agent population
system_agents: 10
independent_agents: 10
control_agents: 10

# Resource settings
initial_resources: 20
resource_regen_rate: 0.1
max_resource_amount: 30

# Learning parameters
learning_rate: 0.001
gamma: 0.95
epsilon_start: 1.0
epsilon_min: 0.01
epsilon_decay: 0.995

# Simulation control
max_steps: 1000
seed: 1234567890
```

**`farm/config/environments/research.yaml`** - Research-specific overrides:
```yaml
# Override for intensive research simulations
width: 200
height: 200
system_agents: 50
independent_agents: 50
max_steps: 5000

# More detailed learning
learning_rate: 0.0001
memory_size: 10000
batch_size: 64
```

#### Parameter Presets

Use built-in presets for common scenarios:

```python
from farm.config import SimulationConfig

# Quick testing preset
config = SimulationConfig.from_centralized_config(
    environment="testing",
    profile="benchmark"
)
# Results in: small world, few agents, fast execution

# Research preset
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="research"
)
# Results in: large world, many agents, detailed logging

# Performance benchmarking preset
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)
# Results in: optimized for speed, minimal logging
```

#### Runtime Parameter Modification

Adjust parameters programmatically:

```python
# Load base configuration
config = SimulationConfig.from_centralized_config("development")

# Modify for specific experiment
config.width = 150
config.height = 150
config.system_agents = 30
config.independent_agents = 20
config.initial_resources = 400
config.resource_regen_rate = 0.02

# Adjust learning parameters
config.learning_rate = 0.0005
config.epsilon_decay = 0.998
config.memory_size = 5000

# Set termination conditions
config.max_steps = 2000
config.min_population = 5  # Stop if population drops below this
config.resource_depletion_threshold = 0.1  # Stop if <10% resources remain
```

#### Configuration Validation

Ensure configurations are valid:

```python
from farm.config.validation import validate_config

# Validate configuration
is_valid, errors = validate_config(config)

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration valid!")
    
# Auto-fix common issues
config = auto_correct_config(config)
```

---

### 4. Design Custom Experiments and Scenarios

Create specialized experimental setups for targeted research.

#### Configuration Templates

Use templates for systematic parameter sweeps:

```python
from farm.config.template import ConfigTemplate, ConfigTemplateManager

# Create template with placeholders
template_dict = {
    "width": "{{env_size}}",
    "height": "{{env_size}}",
    "system_agents": "{{system_count}}",
    "independent_agents": "{{independent_count}}",
    "initial_resources": "{{resource_amount}}",
    "resource_regen_rate": "{{regen_rate}}"
}

template = ConfigTemplate(template_dict)

# Generate multiple configurations
manager = ConfigTemplateManager()
variable_sets = [
    {
        "env_size": 100,
        "system_count": 20,
        "independent_count": 10,
        "resource_amount": 200,
        "regen_rate": 0.01
    },
    {
        "env_size": 150,
        "system_count": 30,
        "independent_count": 15,
        "resource_amount": 300,
        "regen_rate": 0.02
    },
    {
        "env_size": 200,
        "system_count": 40,
        "independent_count": 20,
        "resource_amount": 400,
        "regen_rate": 0.03
    }
]

# Generate all experiment configurations
configs = [template.instantiate(vars) for vars in variable_sets]

# Run batch experiment
from farm.core.simulation import run_simulation_batch
results = run_simulation_batch(configs)
```

#### Custom Scenario Framework

Create complete custom scenarios:

```python
from farm.core.scenario import BaseScenario

class ResourceCompetitionScenario(BaseScenario):
    """Scenario studying competition for declining resources."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        
        self.resource_decline_rate = 0.02  # 2% per step
        self.competition_threshold = 0.3    # Resource level triggering competition
        
    def setup(self, environment: Environment):
        """Initialize scenario-specific elements."""
        # Create initial resource-rich state
        environment.resources = self._create_clustered_resources(
            amount=500,
            num_clusters=5
        )
        
        # Place agents near resource clusters
        self._position_agents_near_resources(environment)
        
        # Set up monitoring
        self.resource_history = []
        self.competition_events = []
        
    def step(self, environment: Environment, step_number: int):
        """Execute scenario-specific per-step logic."""
        # Apply resource decline
        self._apply_resource_decline(environment)
        
        # Monitor competition
        if self._check_competition_threshold(environment):
            self._record_competition_event(environment, step_number)
            
        # Record metrics
        self.resource_history.append({
            'step': step_number,
            'total_resources': sum(r.amount for r in environment.resources.values()),
            'num_agents': len([a for a in environment.agents.values() if a.alive])
        })
        
    def _apply_resource_decline(self, environment: Environment):
        """Gradually reduce available resources."""
        for resource in environment.resources.values():
            resource.amount *= (1 - self.resource_decline_rate)
            
    def _check_competition_threshold(self, environment: Environment) -> bool:
        """Check if resource scarcity triggers competition."""
        total_resources = sum(r.amount for r in environment.resources.values())
        agent_count = len([a for a in environment.agents.values() if a.alive])
        
        if agent_count == 0:
            return False
            
        resources_per_agent = total_resources / agent_count
        return resources_per_agent < self.competition_threshold
        
    def _record_competition_event(self, environment: Environment, step: int):
        """Record when competition intensifies."""
        event = {
            'step': step,
            'combat_events': len(environment.combat_log[-10:]),  # Recent combat
            'avg_agent_resources': np.mean([
                a.resource_level 
                for a in environment.agents.values() 
                if a.alive
            ])
        }
        self.competition_events.append(event)
        
    def analyze_results(self) -> Dict:
        """Analyze scenario-specific outcomes."""
        return {
            'resource_decline_pattern': self.resource_history,
            'competition_events': self.competition_events,
            'competition_onset_step': (
                self.competition_events[0]['step'] 
                if self.competition_events 
                else None
            ),
            'total_competition_events': len(self.competition_events)
        }


# Use the custom scenario
scenario = ResourceCompetitionScenario(config)
environment = Environment(width=100, height=100, config=config)

scenario.setup(environment)

for step in range(config.max_steps):
    # Standard simulation step
    actions = {aid: agent.decide_action() 
               for aid, agent in environment.agents.items() 
               if agent.alive}
    environment.step(actions)
    
    # Scenario-specific step
    scenario.step(environment, step)
    
# Analyze scenario results
results = scenario.analyze_results()
```

#### Experiment Management

Organize and track multiple experiments:

```python
from farm.core.experiment_tracker import ExperimentTracker

# Create experiment tracker
tracker = ExperimentTracker("cooperation_vs_competition")

# Define experiment variations
variations = [
    {"name": "cooperative", "system_agents": 40, "independent_agents": 10},
    {"name": "competitive", "system_agents": 10, "independent_agents": 40},
    {"name": "balanced", "system_agents": 25, "independent_agents": 25}
]

# Run multiple iterations of each variation
for variation in variations:
    for iteration in range(10):
        # Create configuration
        config = SimulationConfig.from_centralized_config("production")
        config.system_agents = variation["system_agents"]
        config.independent_agents = variation["independent_agents"]
        
        # Run simulation
        from farm.core.simulation import run_simulation
        results = run_simulation(config)
        
        # Log results
        tracker.log_run(
            variation_name=variation["name"],
            iteration=iteration,
            parameters=variation,
            metrics={
                "final_population": results["surviving_agents"],
                "total_resources_consumed": results["resources_consumed"],
                "cooperation_events": results["sharing_count"],
                "competition_events": results["combat_count"]
            }
        )

# Generate comparative report
tracker.generate_comparative_report()
```

---

## Advanced Customization

### Custom Action Types

Define entirely new actions agents can perform:

```python
from farm.core.action import Action, action_registry

class ResearchAction(Action):
    """Custom action for researching/discovering."""
    
    def __init__(self, agent_id: str, target_location: Tuple[int, int]):
        super().__init__(action_type="research", agent_id=agent_id)
        self.target_location = target_location
        self.discovery_chance = 0.3
        
    def execute(self, environment: Environment) -> Dict:
        """Execute research action."""
        agent = environment.agents[self.agent_id]
        
        # Check if agent is at target location
        if agent.position == self.target_location:
            # Attempt discovery
            if random.random() < self.discovery_chance:
                discovery = self._make_discovery(environment)
                return {
                    "success": True,
                    "discovery": discovery,
                    "reward": 10.0
                }
                    
        return {"success": False, "reward": -0.1}
        
    def _make_discovery(self, environment: Environment) -> Dict:
        """Generate a discovery."""
        return {
            "type": "resource_location",
            "location": self.target_location,
            "value": random.randint(10, 50)
        }

# Register custom action
action_registry.register("research", ResearchAction)

# Use in agent
class ResearcherAgent(BaseAgent):
    def decide_action(self):
        if self._should_research():
            target = self._choose_research_location()
            return {
                'action_type': 'research',
                'target_location': target
            }
        return super().decide_action()
```

### Custom Observation Channels

Add new types of environmental information:

```python
from farm.core.channels import ChannelHandler

class TemperatureChannel(ChannelHandler):
    """Channel for temperature information."""
    
    channel_name = "TEMPERATURE"
    
    def populate(self, obs_tensor, agent, environment):
        """Populate temperature information in observation."""
        # Get temperature at each observable location
        for dx in range(-agent.observation_radius, agent.observation_radius + 1):
            for dy in range(-agent.observation_radius, agent.observation_radius + 1):
                x = agent.position[0] + dx
                y = agent.position[1] + dy
                
                if environment.is_valid_position((x, y)):
                    # Calculate temperature (example: based on distance from "heat source")
                    temp = self._calculate_temperature((x, y), environment)
                    
                    # Normalize to 0-1 range
                    normalized_temp = temp / 100.0
                    
                    # Write to observation tensor
                    obs_x = dx + agent.observation_radius
                    obs_y = dy + agent.observation_radius
                    obs_tensor[obs_x, obs_y, self.channel_index] = normalized_temp
                    
    def _calculate_temperature(self, position, environment):
        """Calculate temperature at position."""
        # Example: temperature varies with latitude
        y = position[1]
        base_temp = 20  # Celsius
        variation = (y / environment.height - 0.5) * 40
        return base_temp + variation

# Register channel
from farm.core.observations import ObservationConfig

obs_config = ObservationConfig(
    R=6,
    custom_channels=["TEMPERATURE"]
)
```

### Custom Learning Algorithms

Implement specialized learning approaches:

```python
from farm.core.decision.algorithms.base import BaseDecisionAlgorithm

class EvolutionaryLearning(BaseDecisionAlgorithm):
    """Custom evolutionary learning algorithm."""
    
    def __init__(self, config):
        super().__init__(config)
        self.population_size = 50
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        self.population = self._initialize_population()
        
    def select_action(self, state):
        """Select action using best individual in population."""
        best_individual = max(self.population, key=lambda ind: ind.fitness)
        return best_individual.get_action(state)
        
    def update(self, state, action, reward, next_state, done):
        """Evolve population based on rewards."""
        # Assign fitness based on reward
        self._assign_fitness(reward)
        
        # Selection
        parents = self._tournament_selection()
        
        # Crossover
        offspring = self._crossover(parents)
        
        # Mutation
        mutated = self._mutate(offspring)
        
        # Replace population
        self.population = self._create_next_generation(mutated)
        
    def _initialize_population(self):
        """Create initial population of strategies."""
        return [Individual() for _ in range(self.population_size)]
        
    def _tournament_selection(self):
        """Select parents via tournament."""
        # Implementation details...
        pass
        
    def _crossover(self, parents):
        """Combine parent strategies."""
        # Implementation details...
        pass
        
    def _mutate(self, individuals):
        """Apply mutations to individuals."""
        # Implementation details...
        pass
```

---

## Configuration Best Practices

### 1. Version Control Your Configurations

```bash
# Keep configurations in version control
git add farm/config/*.yaml
git commit -m "Update research configuration for experiment 3"

# Tag important configurations
git tag -a exp3-config -m "Configuration for experiment 3"
```

### 2. Document Configuration Changes

```yaml
# config/custom_experiment.yaml
# Experiment: Resource Competition Study
# Date: 2024-10-03
# Changes from baseline:
#   - Reduced initial resources by 50%
#   - Increased agent population by 100%
#   - Extended simulation duration to 5000 steps
# Rationale: Testing scarcity effects on competition

width: 200
height: 200
system_agents: 50
independent_agents: 50
initial_resources: 100  # Reduced from 200
max_steps: 5000  # Increased from 1000
```

### 3. Use Configuration Validation

```python
from farm.config.validation import validate_config, ConfigValidationError

try:
    config = SimulationConfig.from_centralized_config("production")
    validate_config(config)
except ConfigValidationError as e:
    print(f"Invalid configuration: {e}")
    print("Errors:")
    for error in e.errors:
        print(f"  - {error.field}: {error.message}")
```

### 4. Create Reusable Configuration Profiles

```yaml
# config/profiles/my_research.yaml
# Reusable profile for my research area

# Population settings optimized for cooperation studies
system_agents: 40
independent_agents: 20
control_agents: 10

# Extended observation radius for social dynamics
perception_radius: 5

# Tuned learning parameters
learning_rate: 0.0005
epsilon_decay: 0.998

# Long-term dynamics
max_steps: 3000
```

---

## Example: Complete Custom Simulation

Here's a complete example bringing together all customization features:

```python
#!/usr/bin/env python3
"""
Complete custom simulation example: Ecological Succession Study

This simulation studies how an ecosystem evolves when different
species (agent types) with different strategies compete and cooperate.
"""

from farm.config import SimulationConfig
from farm.core.environment import Environment
from farm.core.agent import BaseAgent
from farm.core.simulation import run_simulation
import numpy as np
import random
from typing import Dict, List, Tuple

# ============================================================================
# STEP 1: Define Custom Environment
# ============================================================================

class EcosystemEnvironment(Environment):
    """Environment modeling ecological succession."""
    
    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, **kwargs)
        
        # Ecosystem-specific properties
        self.successional_stage = "early"  # early, mid, late
        self.biodiversity_index = 0.0
        self.ecosystem_stability = 1.0
        
        # Environmental gradients
        self.fertility_map = self._create_fertility_map()
        self.moisture_map = self._create_moisture_map()
        
    def _create_fertility_map(self) -> np.ndarray:
        """Create spatial fertility gradient."""
        fertility = np.random.rand(self.height, self.width)
        # Smooth with convolution
        from scipy.ndimage import gaussian_filter
        fertility = gaussian_filter(fertility, sigma=5)
        return fertility
        
    def _create_moisture_map(self) -> np.ndarray:
        """Create spatial moisture gradient."""
        # Higher moisture at lower elevations (bottom of map)
        y_gradient = np.linspace(1.0, 0.3, self.height)
        moisture = np.tile(y_gradient.reshape(-1, 1), (1, self.width))
        # Add some noise
        moisture += np.random.rand(self.height, self.width) * 0.2
        return np.clip(moisture, 0, 1)
        
    def update_successional_stage(self):
        """Progress ecosystem succession."""
        # Calculate biodiversity
        species_counts = self._count_species()
        self.biodiversity_index = self._calculate_biodiversity(species_counts)
        
        # Update successional stage based on biodiversity and time
        if self.current_step > 1000 and self.biodiversity_index > 0.7:
            self.successional_stage = "late"
        elif self.current_step > 500 or self.biodiversity_index > 0.4:
            self.successional_stage = "mid"
        else:
            self.successional_stage = "early"
            
    def _count_species(self) -> Dict[str, int]:
        """Count individuals of each species."""
        counts = {}
        for agent in self.agents.values():
            if agent.alive:
                species = agent.agent_type
                counts[species] = counts.get(species, 0) + 1
        return counts
        
    def _calculate_biodiversity(self, species_counts: Dict) -> float:
        """Calculate Shannon diversity index."""
        if not species_counts:
            return 0.0
            
        total = sum(species_counts.values())
        shannon_index = 0.0
        
        for count in species_counts.values():
            if count > 0:
                proportion = count / total
                shannon_index -= proportion * np.log(proportion)
                
        # Normalize to 0-1
        max_diversity = np.log(len(species_counts))
        return shannon_index / max_diversity if max_diversity > 0 else 0.0

# ============================================================================
# STEP 2: Define Custom Agent Types
# ============================================================================

class PioneerSpecies(BaseAgent):
    """Fast-growing, resource-inefficient species for early succession."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="PioneerSpecies", **kwargs)
        
        self.growth_rate = 1.5          # Fast reproduction
        self.resource_efficiency = 0.6  # Poor efficiency
        self.competition_ability = 0.3  # Weak competitor
        self.stress_tolerance = 0.8     # High stress tolerance
        
    def decide_action(self):
        """Pioneers prioritize rapid expansion."""
        # Reproduce quickly when possible
        if self.resource_level > 60:  # Lower threshold
            if random.random() < 0.3:  # High reproduction chance
                return {'action_type': 'reproduce'}
                
        return super().decide_action()


class IntermediateSpecies(BaseAgent):
    """Balanced species for mid-succession."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="IntermediateSpecies", **kwargs)
        
        self.growth_rate = 1.0
        self.resource_efficiency = 0.8
        self.competition_ability = 0.6
        self.stress_tolerance = 0.5


class ClimaxSpecies(BaseAgent):
    """Slow-growing, resource-efficient species for late succession."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="ClimaxSpecies", **kwargs)
        
        self.growth_rate = 0.6          # Slow reproduction
        self.resource_efficiency = 1.2  # Highly efficient
        self.competition_ability = 0.9  # Strong competitor
        self.stress_tolerance = 0.3     # Low stress tolerance
        
    def decide_action(self):
        """Climax species prioritize competition and efficiency."""
        # Only reproduce when resource-rich
        if self.resource_level > 150:  # High threshold
            if random.random() < 0.1:   # Low reproduction chance
                return {'action_type': 'reproduce'}
                
        return super().decide_action()

# ============================================================================
# STEP 3: Create Custom Configuration
# ============================================================================

def create_ecosystem_config() -> SimulationConfig:
    """Create configuration for ecosystem simulation."""
    
    config = SimulationConfig(
        # Large environment for ecosystem
        width=200,
        height=200,
        
        # Initial population composition
        system_agents=0,      # Not using standard types
        independent_agents=0,
        control_agents=0,
        
        # Resource settings for ecosystem
        initial_resources=500,
        resource_regen_rate=0.015,
        max_resource_amount=50,
        
        # Extended simulation for succession
        max_steps=3000,
        
        # Seed for reproducibility
        seed=42
    )
    
    return config

# ============================================================================
# STEP 4: Run Custom Simulation
# ============================================================================

def run_ecosystem_simulation():
    """Execute the ecosystem succession simulation."""
    
    print("=== Ecological Succession Simulation ===\n")
    
    # Create configuration
    config = create_ecosystem_config()
    
    # Create custom environment
    environment = EcosystemEnvironment(
        width=config.width,
        height=config.height,
        config=config
    )
    
    # Add initial pioneer population
    print("Initializing pioneer population...")
    for i in range(30):
        agent = PioneerSpecies(
            agent_id=f"pioneer_{i}",
            position=(
                random.randint(0, config.width - 1),
                random.randint(0, config.height - 1)
            ),
            resource_level=100,
            spatial_service=environment.spatial_service,
            environment=environment,
            generation=0
        )
        environment.add_agent(agent)
        
    # Add small intermediate population
    print("Adding intermediate species...")
    for i in range(10):
        agent = IntermediateSpecies(
            agent_id=f"intermediate_{i}",
            position=(
                random.randint(0, config.width - 1),
                random.randint(0, config.height - 1)
            ),
            resource_level=100,
            spatial_service=environment.spatial_service,
            environment=environment,
            generation=0
        )
        environment.add_agent(agent)
        
    # Add few climax individuals
    print("Adding climax species...")
    for i in range(5):
        agent = ClimaxSpecies(
            agent_id=f"climax_{i}",
            position=(
                random.randint(0, config.width - 1),
                random.randint(0, config.height - 1)
            ),
            resource_level=150,
            spatial_service=environment.spatial_service,
            environment=environment,
            generation=0
        )
        environment.add_agent(agent)
        
    print(f"\nStarting simulation with {len(environment.agents)} agents")
    print(f"Environment: {config.width}x{config.height}")
    print(f"Steps: {config.max_steps}\n")
    
    # Track ecosystem metrics
    succession_data = []
    
    # Run simulation
    from tqdm import tqdm
    for step in tqdm(range(config.max_steps), desc="Simulation Progress"):
        # Get actions from all agents
        actions = {}
        for agent_id, agent in environment.agents.items():
            if agent.alive:
                actions[agent_id] = agent.decide_action()
                
        # Execute step
        environment.step(actions)
        
        # Update ecosystem
        environment.update_successional_stage()
        
        # Record metrics every 10 steps
        if step % 10 == 0:
            species_counts = environment._count_species()
            succession_data.append({
                'step': step,
                'stage': environment.successional_stage,
                'biodiversity': environment.biodiversity_index,
                'pioneer_count': species_counts.get('PioneerSpecies', 0),
                'intermediate_count': species_counts.get('IntermediateSpecies', 0),
                'climax_count': species_counts.get('ClimaxSpecies', 0),
                'total_population': sum(species_counts.values())
            })
            
    # Analysis
    print("\n=== Simulation Complete ===\n")
    print("Final Ecosystem State:")
    final_data = succession_data[-1]
    print(f"  Successional Stage: {final_data['stage']}")
    print(f"  Biodiversity Index: {final_data['biodiversity']:.3f}")
    print(f"  Total Population: {final_data['total_population']}")
    print(f"\nSpecies Composition:")
    print(f"  Pioneer Species: {final_data['pioneer_count']}")
    print(f"  Intermediate Species: {final_data['intermediate_count']}")
    print(f"  Climax Species: {final_data['climax_count']}")
    
    return succession_data


if __name__ == "__main__":
    results = run_ecosystem_simulation()
```

---

## Additional Resources

### Documentation
- [Configuration System](config/README.md) - Detailed configuration guide
- [Agent System](agents.md) - Agent customization details
- [Action System](action_system.md) - Custom action development
- [Scenario Design](generic_simulation_scenario_howto.md) - Complete scenario guide

### Examples
- [Usage Examples](usage_examples.md) - Practical tutorials
- [Experiment QuickStart](ExperimentQuickStart.md) - Running experiments
- [Custom Scenarios](experiments/) - Example scenarios

### API Reference
- [SimulationConfig](api_reference.md#simulationconfig) - Configuration API
- [Environment](api_reference.md#environment) - Environment API
- [BaseAgent](api_reference.md#baseagent) - Agent API

---

## Support

For help with customization:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Examples**: Check `examples/` directory for more samples

---

**Ready to customize?** Start with the [Configuration System](config/README.md) or explore our [Usage Examples](usage_examples.md) to see customization in action!
