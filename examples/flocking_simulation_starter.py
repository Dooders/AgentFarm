"""
Thermodynamic Flocking Simulation - Starter Template

A minimal implementation of flocking behavior with energy constraints
for the AgentFarm library.

Usage:
    python examples/flocking_simulation_starter.py

This creates a simple flocking simulation with 50 agents demonstrating:
- Local alignment, cohesion, and separation rules
- Energy-based movement costs (thermodynamic realism)
- Emergent swarm behavior
"""

import numpy as np
from typing import List
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class FlockingAgent(BaseAgent):
    """Agent implementing flocking behavior with energy constraints.
    
    This agent follows Reynolds' three flocking rules:
    1. Alignment: Steer toward average heading of neighbors
    2. Cohesion: Steer toward average position of neighbors  
    3. Separation: Avoid crowding neighbors
    
    Energy is consumed based on movement speed (thermodynamic costs).
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize flocking agent with velocity and flocking parameters."""
        super().__init__(*args, **kwargs)
        
        # Initialize velocity (random initial direction)
        self.velocity = np.random.uniform(-2.0, 2.0, 2).astype(float)
        
        # Flocking parameters
        self.max_speed = 2.0
        self.max_force = 0.5
        self.perception_radius = 10.0
        self.separation_radius = 5.0
        
        # Flocking weights
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5
        
        # Energy parameters
        self.velocity_cost_coefficient = 0.25  # E ∝ v²
        self.base_metabolic_cost = 0.03
    
    def get_neighbors(self, radius: float) -> List['FlockingAgent']:
        """Get all alive neighbors within given radius.
        
        Args:
            radius: Search radius for neighbors
            
        Returns:
            List of neighboring agents
        """
        nearby = self.spatial_service.get_nearby(
            self.position,
            radius,
            ["agents"]
        )
        
        neighbors = []
        for agent in nearby.get("agents", []):
            if agent.agent_id != self.agent_id and agent.alive:
                neighbors.append(agent)
        
        return neighbors
    
    def compute_alignment(self) -> np.ndarray:
        """Steer towards average velocity of neighbors.
        
        Returns:
            Steering force vector
        """
        neighbors = self.get_neighbors(self.perception_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        # Calculate average velocity
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        
        # Desired change in velocity
        desired = avg_velocity - self.velocity
        
        return desired
    
    def compute_cohesion(self) -> np.ndarray:
        """Steer towards center of mass of neighbors.
        
        Returns:
            Steering force vector
        """
        neighbors = self.get_neighbors(self.perception_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        # Calculate center of mass
        center = np.mean([n.position for n in neighbors], axis=0)
        
        # Desired direction (gentle steering)
        desired = (center - np.array(self.position)) * 0.01
        
        return desired
    
    def compute_separation(self) -> np.ndarray:
        """Steer away from neighbors that are too close.
        
        Returns:
            Steering force vector
        """
        neighbors = self.get_neighbors(self.separation_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        steering = np.zeros(2)
        
        for neighbor in neighbors:
            diff = np.array(self.position) - np.array(neighbor.position)
            distance = np.linalg.norm(diff)
            
            if distance > 0:
                # Weight by inverse distance squared (closer = stronger repulsion)
                steering += diff / (distance ** 2)
        
        return steering
    
    def apply_toroidal_boundary(self, position: np.ndarray) -> tuple:
        """Apply wrap-around boundary conditions.
        
        Args:
            position: Position to wrap
            
        Returns:
            Wrapped position as tuple
        """
        wrapped = (
            position[0] % self.environment.width,
            position[1] % self.environment.height
        )
        return wrapped
    
    def act(self):
        """Execute one step of flocking behavior with energy constraints.
        
        This method:
        1. Computes flocking forces (alignment, cohesion, separation)
        2. Updates velocity based on forces
        3. Moves agent based on velocity
        4. Applies energy costs based on movement
        5. Checks for death by energy depletion
        """
        if not self.alive:
            return
        
        # Compute flocking steering forces
        align = self.compute_alignment() * self.alignment_weight
        cohere = self.compute_cohesion() * self.cohesion_weight
        separate = self.compute_separation() * self.separation_weight
        
        # Combine all forces
        total_force = align + cohere + separate
        
        # Limit force magnitude to max_force
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > self.max_force:
            total_force = total_force / force_magnitude * self.max_force
        
        # Update velocity
        self.velocity += total_force
        
        # Limit speed to max_speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Update position based on velocity
        new_position = np.array(self.position) + self.velocity
        new_position = self.apply_toroidal_boundary(new_position)
        
        self.update_position(tuple(new_position))
        
        # Energy consumption (thermodynamic costs)
        # Quadratic cost: energy ∝ velocity²
        energy_cost = self.velocity_cost_coefficient * (speed ** 2)
        
        # Base metabolic cost (staying alive costs energy)
        base_cost = self.base_metabolic_cost
        
        # Deduct total energy cost
        total_cost = energy_cost + base_cost
        self.resource_level -= total_cost
        
        # Check for death by energy depletion
        if self.resource_level <= 0:
            logger.info(
                f"Agent {self.agent_id} died from energy depletion "
                f"at position {self.position}"
            )
            self.terminate()
            return


class FlockingMetrics:
    """Track emergence metrics for flocking behavior analysis."""
    
    def __init__(self):
        """Initialize metric history storage."""
        self.time = []
        self.alive_count = []
        self.avg_energy = []
        self.avg_speed = []
        self.alignment = []
        self.cohesion = []
    
    def compute_alignment(self, agents: List[FlockingAgent]) -> float:
        """Compute velocity coherence (how aligned are velocities).
        
        Returns value between 0 (random directions) and 1 (perfect alignment).
        """
        if not agents:
            return 0.0
        
        velocities = np.array([a.velocity for a in agents])
        avg_velocity = np.mean(velocities, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        if avg_speed < 1e-6:
            return 0.0
        
        alignment = np.linalg.norm(avg_velocity) / avg_speed
        return float(alignment)
    
    def compute_cohesion(self, agents: List[FlockingAgent]) -> float:
        """Compute spatial clustering (how close are agents to center).
        
        Returns value between 0 (dispersed) and 1 (clustered).
        """
        if not agents:
            return 0.0
        
        positions = np.array([a.position for a in agents])
        center = np.mean(positions, axis=0)
        avg_distance = np.mean([
            np.linalg.norm(p - center) for p in positions
        ])
        
        # Normalized cohesion (arbitrary scaling)
        cohesion = 1.0 / (1.0 + avg_distance / 10.0)
        return float(cohesion)
    
    def update(self, agents: List[FlockingAgent], step: int):
        """Compute and record all metrics for current step.
        
        Args:
            agents: List of all agents in simulation
            step: Current simulation step number
        """
        alive_agents = [a for a in agents if a.alive]
        
        self.time.append(step)
        self.alive_count.append(len(alive_agents))
        
        if alive_agents:
            self.avg_energy.append(
                np.mean([a.resource_level for a in alive_agents])
            )
            self.avg_speed.append(
                np.mean([np.linalg.norm(a.velocity) for a in alive_agents])
            )
        else:
            self.avg_energy.append(0.0)
            self.avg_speed.append(0.0)
        
        self.alignment.append(self.compute_alignment(alive_agents))
        self.cohesion.append(self.compute_cohesion(alive_agents))


def run_flocking_simulation(
    n_agents: int = 50,
    n_steps: int = 1000,
    width: int = 100,
    height: int = 100,
    initial_energy_min: float = 30.0,
    initial_energy_max: float = 100.0,
    db_path: str = "flocking_simulation.db",
    seed: int = 42
):
    """Run a thermodynamic flocking simulation.
    
    Args:
        n_agents: Number of flocking agents to create
        n_steps: Number of simulation steps to run
        width: Environment width
        height: Environment height
        initial_energy_min: Minimum initial agent energy
        initial_energy_max: Maximum initial agent energy
        db_path: Path to save simulation database
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (environment, metrics)
    """
    logger.info(
        f"Starting flocking simulation with {n_agents} agents for {n_steps} steps"
    )
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment
    env = Environment(
        width=width,
        height=height,
        resource_distribution={
            "type": "random",
            "amount": 8,  # Energy sources
        },
        db_path=db_path,
        seed=seed
    )
    
    # Create flocking agents
    for i in range(n_agents):
        # Random position
        position = (
            np.random.uniform(0, width),
            np.random.uniform(0, height)
        )
        
        # Varied initial energy (creates diversity)
        initial_energy = np.random.uniform(
            initial_energy_min,
            initial_energy_max
        )
        
        # Create agent
        agent = FlockingAgent(
            agent_id=env.get_next_agent_id(),
            position=position,
            resource_level=initial_energy,
            spatial_service=env.spatial_service,
            environment=env,
            agent_type="FlockingAgent"
        )
        
        # Add to environment
        env.add_agent(agent)
    
    logger.info(f"Created {n_agents} flocking agents")
    
    # Initialize metrics tracker
    metrics = FlockingMetrics()
    
    # Run simulation
    logger.info("Running simulation...")
    
    for step in range(n_steps):
        # Step all agents
        env.step()
        
        # Update metrics every step
        all_agents = list(env._agent_objects.values())
        metrics.update(all_agents, step)
        
        # Log progress every 100 steps
        if step % 100 == 0:
            alive = sum(1 for a in all_agents if a.alive)
            avg_energy = (
                np.mean([a.resource_level for a in all_agents if a.alive])
                if alive > 0 else 0
            )
            logger.info(
                f"Step {step}/{n_steps}: {alive} agents alive, "
                f"avg energy: {avg_energy:.1f}"
            )
    
    # Finalize simulation
    env.finalize()
    
    logger.info("Simulation complete!")
    logger.info(f"Final alive agents: {metrics.alive_count[-1]}")
    logger.info(f"Final avg energy: {metrics.avg_energy[-1]:.1f}")
    logger.info(f"Final alignment: {metrics.alignment[-1]:.3f}")
    logger.info(f"Final cohesion: {metrics.cohesion[-1]:.3f}")
    
    return env, metrics


def plot_metrics(metrics: FlockingMetrics, save_path: str = "flocking_metrics.png"):
    """Create visualization of flocking metrics.
    
    Args:
        metrics: FlockingMetrics instance with recorded history
        save_path: Path to save plot image
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Population
    axes[0, 0].plot(metrics.time, metrics.alive_count, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Alive Agents')
    axes[0, 0].set_title('Population Dynamics')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy
    axes[0, 1].plot(metrics.time, metrics.avg_energy, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Average Energy')
    axes[0, 1].set_title('Energy Dynamics')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Speed
    axes[0, 2].plot(metrics.time, metrics.avg_speed, 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Average Speed')
    axes[0, 2].set_title('Movement Dynamics')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Alignment
    axes[1, 0].plot(metrics.time, metrics.alignment, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Alignment (0-1)')
    axes[1, 0].set_title('Velocity Coherence')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cohesion
    axes[1, 1].plot(metrics.time, metrics.cohesion, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Cohesion (0-1)')
    axes[1, 1].set_title('Spatial Clustering')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary stats
    axes[1, 2].axis('off')
    summary_text = f"""
    Simulation Summary
    
    Total Steps: {len(metrics.time)}
    Final Alive: {metrics.alive_count[-1]}
    
    Avg Alignment: {np.mean(metrics.alignment):.3f}
    Avg Cohesion: {np.mean(metrics.cohesion):.3f}
    
    Final Energy: {metrics.avg_energy[-1]:.1f}
    Final Speed: {metrics.avg_speed[-1]:.2f}
    """
    axes[1, 2].text(
        0.1, 0.5, summary_text,
        fontsize=12, verticalalignment='center',
        family='monospace'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved metrics plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Run simulation
    env, metrics = run_flocking_simulation(
        n_agents=50,
        n_steps=1000,
        seed=42
    )
    
    # Plot results
    plot_metrics(metrics)
    
    print("\nSimulation complete! Check flocking_metrics.png for results.")
    print(f"Database saved to: flocking_simulation.db")
