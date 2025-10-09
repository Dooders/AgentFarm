"""Static Catapult RL Environment Example.

This example demonstrates how to use the new physics engine abstraction
to create a simple catapult aiming problem.

Problem:
    An agent controls a catapult with two parameters:
    - angle: Launch angle (0-90 degrees)
    - power: Launch power (0-100%)
    
    Goal: Hit a target at a specific distance
    
    This is a "static" environment because there's no spatial movement
    after the launch - just finding the right angle/power combination.

Note:
    This example shows the proposed API. The physics implementations
    will be added in future commits.
"""

import numpy as np
from typing import Tuple, List, Any, Dict
from gymnasium import spaces

# When implemented, these will be actual imports:
# from farm.core.physics import StaticPhysics
# from farm.core.environment import Environment
# from farm.config import SimulationConfig


class CatapultPhysics:
    """Static physics for catapult aiming problem.
    
    This demonstrates a minimal IPhysicsEngine implementation for
    a non-spatial, discrete state space problem.
    """
    
    def __init__(
        self,
        target_distance: float = 50.0,
        gravity: float = 9.8
    ):
        """Initialize catapult physics.
        
        Args:
            target_distance: Distance to target in meters
            gravity: Gravitational constant
        """
        self.target_distance = target_distance
        self.gravity = gravity
        
        # Define discrete state space
        self.angles = np.linspace(0, 90, 91)  # 0-90 degrees, 1 degree steps
        self.powers = np.linspace(0, 100, 101)  # 0-100% power, 1% steps
        
        # Create position mapping (angle, power) -> index
        self.positions = [
            (angle, power) 
            for angle in self.angles 
            for power in self.powers
        ]
        self.position_to_idx = {pos: idx for idx, pos in enumerate(self.positions)}
        
        # Entity storage (not really used for catapult, but required by protocol)
        self.entities: Dict[str, List[Any]] = {
            "agents": [],
            "resources": [],
            "target": []
        }
    
    def validate_position(self, position: Tuple[float, float]) -> bool:
        """Check if (angle, power) combination is valid."""
        if not isinstance(position, tuple) or len(position) != 2:
            return False
        
        angle, power = position
        return (0 <= angle <= 90) and (0 <= power <= 100)
    
    def get_nearby_entities(
        self,
        position: Tuple[float, float],
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """For static physics, 'nearby' means similar parameters.
        
        Returns entities with angle/power within radius of given position.
        """
        if not self.validate_position(position):
            return []
        
        angle, power = position
        nearby = []
        
        for entity in self.entities.get(entity_type, []):
            if hasattr(entity, 'position'):
                e_angle, e_power = entity.position
                # "Distance" in parameter space
                param_dist = np.sqrt((angle - e_angle)**2 + (power - e_power)**2)
                if param_dist <= radius:
                    nearby.append(entity)
        
        return nearby
    
    def compute_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Compute distance in parameter space."""
        angle1, power1 = pos1
        angle2, power2 = pos2
        return np.sqrt((angle1 - angle2)**2 + (power1 - power2)**2)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """State is 2D: (angle, power)."""
        return (2,)
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Observation: [angle, power, distance_to_target].
        
        Returns:
            Box space with shape (3,)
            - obs[0]: current angle (0-90)
            - obs[1]: current power (0-100)  
            - obs[2]: distance error from target (-100 to 100)
        """
        return spaces.Box(
            low=np.array([0, 0, -100], dtype=np.float32),
            high=np.array([90, 100, 100], dtype=np.float32),
            dtype=np.float32
        )
    
    def sample_position(self) -> Tuple[float, float]:
        """Sample random angle and power."""
        angle = np.random.choice(self.angles)
        power = np.random.choice(self.powers)
        return (angle, power)
    
    def update(self, dt: float = 1.0) -> None:
        """No physics update needed for static problem."""
        pass
    
    def reset(self) -> None:
        """Clear entity lists."""
        for entity_type in self.entities:
            self.entities[entity_type].clear()
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "type": "catapult",
            "target_distance": self.target_distance,
            "gravity": self.gravity,
            "angle_steps": len(self.angles),
            "power_steps": len(self.powers)
        }
    
    def calculate_projectile_distance(self, angle: float, power: float) -> float:
        """Calculate how far the projectile travels.
        
        Uses simplified projectile motion physics:
        distance = (v^2 * sin(2*theta)) / g
        
        Args:
            angle: Launch angle in degrees
            power: Launch power (0-100% of max velocity)
            
        Returns:
            Distance traveled in meters
        """
        # Convert to radians
        angle_rad = np.radians(angle)
        
        # Power maps to velocity (assume max velocity = 50 m/s)
        max_velocity = 50.0
        velocity = (power / 100.0) * max_velocity
        
        # Projectile motion formula
        distance = (velocity ** 2 * np.sin(2 * angle_rad)) / self.gravity
        
        return distance
    
    def calculate_reward(self, angle: float, power: float) -> float:
        """Calculate reward based on how close to target.
        
        Args:
            angle: Launch angle
            power: Launch power
            
        Returns:
            Reward value:
            - High reward for hitting target
            - Decreasing reward as error increases
            - Bonus for exact hit
        """
        distance = self.calculate_projectile_distance(angle, power)
        error = abs(distance - self.target_distance)
        
        # Base reward: inverse of error
        reward = 10.0 / (1.0 + error)
        
        # Bonus for hitting target within 1 meter
        if error < 1.0:
            reward += 100.0
        
        # Small penalty for very high/low angles (encourage realistic solutions)
        if angle < 10 or angle > 80:
            reward -= 1.0
        
        return reward


def example_usage():
    """Demonstrate how to use the catapult physics engine."""
    
    # Create physics engine
    physics = CatapultPhysics(target_distance=50.0)
    
    print("Catapult RL Environment Example")
    print("=" * 50)
    print(f"Target distance: {physics.target_distance} meters")
    print(f"State space: {len(physics.positions)} discrete positions")
    print()
    
    # Test different angle/power combinations
    test_cases = [
        (45, 50),  # Optimal angle, medium power
        (30, 70),  # Lower angle, more power
        (60, 40),  # Higher angle, less power
        (90, 100), # Straight up, max power (bad!)
        (0, 100),  # Flat, max power (also bad!)
    ]
    
    print("Testing angle/power combinations:")
    print("-" * 50)
    
    for angle, power in test_cases:
        # Calculate result
        distance = physics.calculate_projectile_distance(angle, power)
        reward = physics.calculate_reward(angle, power)
        error = abs(distance - physics.target_distance)
        
        print(f"Angle: {angle:2.0f}°, Power: {power:3.0f}% -> "
              f"Distance: {distance:5.1f}m, Error: {error:5.1f}m, "
              f"Reward: {reward:6.2f}")
    
    print()
    print("-" * 50)
    
    # Find optimal solution
    print("\nSearching for optimal solution...")
    best_reward = -float('inf')
    best_params = None
    
    for angle in physics.angles[::5]:  # Check every 5 degrees
        for power in physics.powers[::5]:  # Check every 5%
            reward = physics.calculate_reward(angle, power)
            if reward > best_reward:
                best_reward = reward
                best_params = (angle, power)
    
    angle, power = best_params
    distance = physics.calculate_projectile_distance(angle, power)
    error = abs(distance - physics.target_distance)
    
    print(f"Best found: Angle={angle:.0f}°, Power={power:.0f}%")
    print(f"  Distance: {distance:.2f}m (error: {error:.2f}m)")
    print(f"  Reward: {best_reward:.2f}")
    
    print()
    print("=" * 50)
    print("\nThis example demonstrates:")
    print("1. Non-spatial physics engine (static positions)")
    print("2. Custom reward function for domain-specific problem")
    print("3. Discrete state space (angle/power combinations)")
    print("4. How to implement IPhysicsEngine protocol")
    print("\nTo use with Environment class:")
    print("  env = Environment(physics_engine=physics, config=config)")


if __name__ == "__main__":
    example_usage()
