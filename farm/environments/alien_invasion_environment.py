import random
import math
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from farm.core.environment import Environment
from farm.core.resources import Resource
from farm.agents.alien_agent import AlienAgent
from farm.agents.human_agent import HumanAgent

if TYPE_CHECKING:
    from farm.core.config import SimulationConfig


class AlienInvasionEnvironment(Environment):
    """Environment specialized for alien invasion combat scenarios.
    
    This environment manages the spatial setup where:
    - Humans start in the center of the grid
    - Aliens spawn around the edges 
    - Resources are strategically placed to create conflict zones
    - Territory control mechanics are tracked
    """

    def __init__(
        self,
        width: int,
        height: int,
        resource_distribution: int,
        config: "SimulationConfig",
        db_path: str = "alien_invasion.db",
        simulation_id: str = None,
        seed: int = None,
    ):
        """Initialize the alien invasion environment.
        
        Parameters
        ----------
        width : int
            Width of the simulation grid
        height : int
            Height of the simulation grid
        resource_distribution : int
            Number of resources to distribute
        config : SimulationConfig
            Configuration object with simulation parameters
        db_path : str
            Path for the simulation database
        simulation_id : str
            Unique identifier for this simulation
        seed : int
            Random seed for reproducibility
        """
        # Initialize base environment
        super().__init__(
            width=width,
            height=height,
            resource_distribution=resource_distribution,
            db_path=db_path,
            max_resource=config.max_resource_amount,
            config=config,
            simulation_id=simulation_id,
            seed=seed,
        )
        
        # Alien invasion specific attributes
        self.center_radius = min(width, height) * 0.2  # Central safe zone radius
        self.edge_buffer = min(width, height) * 0.1   # Buffer from absolute edge
        self.conflict_zones = []  # Strategic resource placement zones
        
        # Territory control tracking
        self.human_territory = set()
        self.alien_territory = set()
        self.contested_territory = set()
        
        # Combat statistics
        self.humans_eliminated = 0
        self.aliens_eliminated = 0
        self.territorial_control_ratio = 0.5  # 0 = all alien, 1 = all human
        
        # Setup strategic resource placement
        self._setup_conflict_zones()
        self._redistribute_resources_strategically()

    def _setup_conflict_zones(self):
        """Define strategic zones where resources will be concentrated."""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Create conflict zones between center and edges
        conflict_positions = [
            # Cardinal directions from center
            (center_x, center_y + self.center_radius),  # North
            (center_x, center_y - self.center_radius),  # South
            (center_x + self.center_radius, center_y),  # East
            (center_x - self.center_radius, center_y),  # West
            
            # Diagonal positions
            (center_x + self.center_radius * 0.7, center_y + self.center_radius * 0.7),  # NE
            (center_x - self.center_radius * 0.7, center_y + self.center_radius * 0.7),  # NW
            (center_x + self.center_radius * 0.7, center_y - self.center_radius * 0.7),  # SE
            (center_x - self.center_radius * 0.7, center_y - self.center_radius * 0.7),  # SW
        ]
        
        self.conflict_zones = [
            (max(self.edge_buffer, min(self.width - self.edge_buffer, x)),
             max(self.edge_buffer, min(self.height - self.edge_buffer, y)))
            for x, y in conflict_positions
        ]

    def _redistribute_resources_strategically(self):
        """Redistribute resources to create strategic conflict zones."""
        # Clear existing resources
        self.resources.clear()
        
        total_resources = self.config.initial_resources
        
        # Place 60% of resources in conflict zones
        conflict_resources = int(total_resources * 0.6)
        remaining_resources = total_resources - conflict_resources
        
        # Distribute resources in conflict zones
        for _ in range(conflict_resources):
            if self.conflict_zones:
                zone = random.choice(self.conflict_zones)
                # Add some randomness around the conflict zone
                x = zone[0] + random.uniform(-15, 15)
                y = zone[1] + random.uniform(-15, 15)
                
                # Ensure within bounds
                x = max(0, min(self.width, x))
                y = max(0, min(self.height, y))
                
                resource = Resource(
                    resource_id=self.get_next_resource_id(),
                    position=(x, y),
                    amount=random.uniform(5, self.config.max_resource_amount),
                    max_amount=self.config.max_resource_amount,
                    regeneration_rate=self.config.resource_regen_rate,
                )
                self.resources.append(resource)
        
        # Distribute remaining resources randomly
        for _ in range(remaining_resources):
            position = self._get_random_position()
            resource = Resource(
                resource_id=self.get_next_resource_id(),
                position=position,
                amount=random.uniform(3, self.config.max_resource_amount),
                max_amount=self.config.max_resource_amount,
                regeneration_rate=self.config.resource_regen_rate,
            )
            self.resources.append(resource)

    def get_human_spawn_position(self) -> Tuple[float, float]:
        """Get a spawn position in the central human area."""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Spawn humans in a circular area around the center
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, self.center_radius * 0.8)  # 80% of center radius
        
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        # Ensure within bounds
        x = max(5, min(self.width - 5, x))
        y = max(5, min(self.height - 5, y))
        
        return (x, y)

    def get_alien_spawn_position(self) -> Tuple[float, float]:
        """Get a spawn position around the edges for aliens."""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Choose a random edge
        edge = random.choice(['north', 'south', 'east', 'west'])
        
        if edge == 'north':
            x = random.uniform(self.edge_buffer, self.width - self.edge_buffer)
            y = random.uniform(self.height - self.edge_buffer - 20, self.height - self.edge_buffer)
        elif edge == 'south':
            x = random.uniform(self.edge_buffer, self.width - self.edge_buffer)
            y = random.uniform(self.edge_buffer, self.edge_buffer + 20)
        elif edge == 'east':
            x = random.uniform(self.width - self.edge_buffer - 20, self.width - self.edge_buffer)
            y = random.uniform(self.edge_buffer, self.height - self.edge_buffer)
        else:  # west
            x = random.uniform(self.edge_buffer, self.edge_buffer + 20)
            y = random.uniform(self.edge_buffer, self.height - self.edge_buffer)
        
        return (x, y)

    def create_human_agent(self, agent_id: str, resource_level: int, generation: int = 0) -> HumanAgent:
        """Create a human agent at an appropriate spawn position."""
        position = self.get_human_spawn_position()
        return HumanAgent(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=self,
            generation=generation,
        )

    def create_alien_agent(self, agent_id: str, resource_level: int, generation: int = 0) -> AlienAgent:
        """Create an alien agent at an appropriate spawn position."""
        position = self.get_alien_spawn_position()
        return AlienAgent(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=self,
            generation=generation,
        )

    def update_territorial_control(self):
        """Update territory control metrics based on agent positions."""
        # Reset territory sets
        self.human_territory.clear()
        self.alien_territory.clear()
        self.contested_territory.clear()
        
        # Define territory influence radius
        influence_radius = 25.0
        
        # Get all living agents
        humans = [agent for agent in self.agents if isinstance(agent, HumanAgent) and agent.alive]
        aliens = [agent for agent in self.agents if isinstance(agent, AlienAgent) and agent.alive]
        
        # Create grid points for territory calculation
        grid_size = 10
        for x in range(0, self.width, grid_size):
            for y in range(0, self.height, grid_size):
                pos = (x, y)
                
                # Check influence from each faction
                human_influence = sum(
                    1 for human in humans 
                    if self._calculate_distance(pos, human.position) <= influence_radius
                )
                alien_influence = sum(
                    1 for alien in aliens 
                    if self._calculate_distance(pos, alien.position) <= influence_radius
                )
                
                # Determine territorial control
                if human_influence > alien_influence:
                    self.human_territory.add(pos)
                elif alien_influence > human_influence:
                    self.alien_territory.add(pos)
                elif human_influence > 0 or alien_influence > 0:
                    self.contested_territory.add(pos)
        
        # Calculate territorial control ratio
        total_controlled = len(self.human_territory) + len(self.alien_territory)
        if total_controlled > 0:
            self.territorial_control_ratio = len(self.human_territory) / total_controlled
        else:
            self.territorial_control_ratio = 0.5

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def check_victory_conditions(self) -> Tuple[bool, str]:
        """Check if either faction has achieved victory.
        
        Returns
        -------
        Tuple[bool, str]
            (victory_achieved, victor_description)
        """
        humans = [agent for agent in self.agents if isinstance(agent, HumanAgent) and agent.alive]
        aliens = [agent for agent in self.agents if isinstance(agent, AlienAgent) and agent.alive]
        
        # Check elimination victory conditions
        if len(humans) == 0:
            return True, "Aliens achieve total victory - All humans eliminated"
        
        if len(aliens) == 0:
            return True, "Humans achieve total victory - All aliens eliminated"
        
        # Check territorial victory conditions
        if self.territorial_control_ratio >= 0.9:
            return True, "Humans achieve territorial victory - 90% control"
        
        if self.territorial_control_ratio <= 0.1:
            return True, "Aliens achieve territorial victory - 90% control"
        
        # Check survival victory (humans survive for extended period)
        if self.time > 1500 and len(humans) >= 3:  # Humans survive 1500+ steps
            return True, "Humans achieve survival victory - Lasted 1500+ steps"
        
        return False, ""

    def update(self):
        """Update environment with alien invasion specific mechanics."""
        # Standard environment update
        super().update()
        
        # Update territorial control
        self.update_territorial_control()
        
        # Check victory conditions
        victory, message = self.check_victory_conditions()
        if victory:
            print(f"VICTORY ACHIEVED: {message}")
            if hasattr(self, 'db') and self.db:
                # Log victory condition
                self.db.update_notes({
                    'victory_achieved': True,
                    'victory_message': message,
                    'victory_step': self.time,
                    'final_humans': len([a for a in self.agents if isinstance(a, HumanAgent) and a.alive]),
                    'final_aliens': len([a for a in self.agents if isinstance(a, AlienAgent) and a.alive]),
                    'territorial_control': self.territorial_control_ratio
                })

    def _calculate_metrics(self):
        """Calculate environment metrics with invasion-specific additions."""
        # Get base metrics
        metrics = super()._calculate_metrics() if hasattr(super(), '_calculate_metrics') else {}
        
        # Count faction populations
        humans = [agent for agent in self.agents if isinstance(agent, HumanAgent) and agent.alive]
        aliens = [agent for agent in self.agents if isinstance(agent, AlienAgent) and agent.alive]
        
        # Add invasion-specific metrics
        invasion_metrics = {
            'human_agents': len(humans),
            'alien_agents': len(aliens),
            'humans_eliminated': self.humans_eliminated,
            'aliens_eliminated': self.aliens_eliminated,
            'territorial_control_ratio': self.territorial_control_ratio,
            'human_territory_size': len(self.human_territory),
            'alien_territory_size': len(self.alien_territory),
            'contested_territory_size': len(self.contested_territory),
        }
        
        # Merge with base metrics
        if isinstance(metrics, dict):
            metrics.update(invasion_metrics)
        else:
            metrics = invasion_metrics
            
        return metrics

    def remove_agent(self, agent):
        """Remove agent and update elimination counters."""
        if isinstance(agent, HumanAgent):
            self.humans_eliminated += 1
        elif isinstance(agent, AlienAgent):
            self.aliens_eliminated += 1
            
        super().remove_agent(agent)

    def get_invasion_status(self) -> dict:
        """Get current status of the invasion simulation."""
        humans = [agent for agent in self.agents if isinstance(agent, HumanAgent) and agent.alive]
        aliens = [agent for agent in self.agents if isinstance(agent, AlienAgent) and agent.alive]
        
        return {
            'step': self.time,
            'humans_alive': len(humans),
            'aliens_alive': len(aliens),
            'humans_eliminated': self.humans_eliminated,
            'aliens_eliminated': self.aliens_eliminated,
            'territorial_control': {
                'human_ratio': self.territorial_control_ratio,
                'human_territory': len(self.human_territory),
                'alien_territory': len(self.alien_territory),
                'contested_territory': len(self.contested_territory),
            },
            'victory_status': self.check_victory_conditions(),
        }


def create_alien_invasion_simulation(config: "SimulationConfig") -> AlienInvasionEnvironment:
    """Create a fully configured alien invasion simulation environment.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration object with invasion-specific parameters
        
    Returns
    -------
    AlienInvasionEnvironment
        Configured environment ready for simulation
    """
    # Create environment
    env = AlienInvasionEnvironment(
        width=config.width,
        height=config.height,
        resource_distribution=config.initial_resources,
        config=config,
        db_path=f"simulations/alien_invasion_{env.simulation_id}.db" if hasattr(env, 'simulation_id') else "alien_invasion.db",
        seed=config.seed,
    )
    
    # Create human agents in center
    human_count = getattr(config, 'human_agents', 10)
    for i in range(human_count):
        agent = env.create_human_agent(
            agent_id=f"human_{i}",
            resource_level=config.initial_resource_level,
            generation=0
        )
        env.add_agent(agent)
        
        # Log human agent
        if env.db:
            env.db.data_logger.log_agent(
                agent_id=agent.agent_id,
                birth_time=0,
                agent_type="HumanAgent",
                position=agent.position,
                initial_resources=agent.resource_level,
                starting_health=agent.starting_health,
                starvation_threshold=agent.starvation_threshold,
                genome_id=agent.genome_id,
                generation=agent.generation,
                action_weights=agent.get_action_weights(),
            )
    
    # Create alien agents around edges
    alien_count = getattr(config, 'alien_agents', 15)
    for i in range(alien_count):
        agent = env.create_alien_agent(
            agent_id=f"alien_{i}",
            resource_level=config.initial_resource_level,
            generation=0
        )
        env.add_agent(agent)
        
        # Log alien agent
        if env.db:
            env.db.data_logger.log_agent(
                agent_id=agent.agent_id,
                birth_time=0,
                agent_type="AlienAgent",
                position=agent.position,
                initial_resources=agent.resource_level,
                starting_health=agent.starting_health,
                starvation_threshold=agent.starvation_threshold,
                genome_id=agent.genome_id,
                generation=agent.generation,
                action_weights=agent.get_action_weights(),
            )
    
    # Log initial resources
    if env.db:
        for resource in env.resources:
            env.db.data_logger.log_resource(
                resource_id=resource.resource_id,
                initial_amount=resource.amount,
                position=resource.position,
            )
    
    return env