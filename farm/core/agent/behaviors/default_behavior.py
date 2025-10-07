"""
Default agent behavior with random action selection.

Simple behavior for testing and baseline comparisons.
"""

import random
from typing import TYPE_CHECKING, List, Optional
from farm.core.agent.behaviors.base_behavior import IAgentBehavior

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class DefaultAgentBehavior(IAgentBehavior):
    """
    Simple behavior that selects random actions.

    This behavior is useful for:
    - Testing agent systems
    - Baseline comparisons
    - Simple simulations without learning

    The behavior randomly selects from available actions based on
    the agent's current capabilities (components).
    """

    def __init__(self):
        """Initialize default behavior."""
        self._turn_count = 0

    def execute_turn(self, agent: "AgentCore") -> None:
        """
        Execute one turn with random action selection.

        Actions are selected based on available components:
        - If has movement: might move randomly
        - If has resources and can reproduce: might reproduce
        - If has combat and enemies nearby: might attack
        - Otherwise: pass (do nothing)

        Args:
            agent: The agent to control
        """
        self._turn_count += 1

        # Get available components
        has_movement = agent.has_component("movement")
        has_resource = agent.has_component("resource")
        has_combat = agent.has_component("combat")
        has_perception = agent.has_component("perception")
        has_reproduction = agent.has_component("reproduction")

        # Build list of possible actions
        possible_actions = []

        if has_movement:
            possible_actions.append("move")

        if has_resource:
            resource = agent.get_component("resource")
            if resource and resource.level > 50:
                possible_actions.append("gather")  # Try to gather more

        if has_reproduction:
            reproduction = agent.get_component("reproduction")
            if reproduction and reproduction.can_reproduce():
                possible_actions.append("reproduce")

        if has_combat and has_perception:
            perception = agent.get_component("perception")
            if perception:
                nearby = perception.get_nearby_entities(["agents"])
                if nearby.get("agents", []):
                    possible_actions.append("attack")
                    possible_actions.append("defend")

        # Always can pass
        possible_actions.append("pass")

        # Randomly select action
        action = random.choice(possible_actions)

        # Execute selected action
        if action == "move":
            self._do_random_move(agent)
        elif action == "gather":
            self._do_gather(agent)
        elif action == "reproduce":
            self._do_reproduce(agent)
        elif action == "attack":
            self._do_attack(agent)
        elif action == "defend":
            self._do_defend(agent)
        else:
            # pass - do nothing
            pass

    def _do_random_move(self, agent: "AgentCore") -> None:
        """Execute random movement."""
        movement = agent.get_component("movement")
        if movement:
            movement.random_move()

    def _do_gather(self, agent: "AgentCore") -> None:
        """Try to gather resources."""
        perception = agent.get_component("perception")
        movement = agent.get_component("movement")
        resource = agent.get_component("resource")

        if not (perception and movement and resource):
            return

        # Find nearest resource
        nearest = perception.get_nearest_entity(["resources"])
        nearest_resource = nearest.get("resources")

        if nearest_resource:
            # Move toward it
            movement.move_toward_entity(nearest_resource.position, stop_distance=0.5)

            # If close enough, gather it
            if movement.distance_to(nearest_resource.position) < 1.0:
                # Assume gathering adds resources (would be implemented in action)
                resource.add(10)

    def _do_reproduce(self, agent: "AgentCore") -> None:
        """Attempt reproduction."""
        reproduction = agent.get_component("reproduction")
        if reproduction:
            reproduction.reproduce()

    def _do_attack(self, agent: "AgentCore") -> None:
        """Attack a nearby agent."""
        perception = agent.get_component("perception")
        combat = agent.get_component("combat")

        if not (perception and combat):
            return

        # Get nearby agents
        nearby = perception.get_nearby_entities(["agents"])
        nearby_agents = nearby.get("agents", [])

        if nearby_agents:
            # Attack random nearby agent
            target = random.choice(nearby_agents)
            combat.attack(target)

    def _do_defend(self, agent: "AgentCore") -> None:
        """Enter defensive stance."""
        combat = agent.get_component("combat")
        if combat:
            combat.start_defense()

    def reset(self) -> None:
        """Reset behavior state."""
        self._turn_count = 0

    def get_state(self) -> dict:
        """Get behavior state for serialization."""
        return {
            "turn_count": self._turn_count,
        }

    def load_state(self, state: dict) -> None:
        """Load behavior state from dictionary."""
        self._turn_count = state.get("turn_count", 0)