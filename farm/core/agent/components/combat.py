"""
Combat component for agent combat mechanics.

Handles health, attacks, defense, and combat-related death.
"""

from typing import TYPE_CHECKING, Optional
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import CombatConfig

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class CombatComponent(IAgentComponent):
    """
    Component handling agent combat mechanics.

    Responsibilities:
    - Track health
    - Execute attacks on other agents
    - Handle incoming damage
    - Manage defensive stance
    - Trigger death when health depleted

    Single Responsibility: Only combat logic.
    """

    def __init__(self, config: CombatConfig):
        """
        Initialize combat component.

        Args:
            config: Combat configuration
        """
        self._config = config
        self._agent: Optional["AgentCore"] = None
        self._health = config.starting_health
        self._is_defending = False
        self._defense_timer = 0

    @property
    def name(self) -> str:
        """Component identifier."""
        return "combat"

    @property
    def health(self) -> float:
        """Current health points."""
        return self._health

    @property
    def max_health(self) -> float:
        """Maximum health points."""
        return self._config.starting_health

    @property
    def health_ratio(self) -> float:
        """
        Health as ratio of maximum (0.0 to 1.0).

        Returns:
            float: Current health / max health

        Example:
            >>> ratio = combat.health_ratio
            >>> print(f"Agent is at {ratio * 100}% health")
        """
        return self._health / self._config.starting_health

    @property
    def is_defending(self) -> bool:
        """Whether agent is currently in defensive stance."""
        return self._is_defending

    @property
    def defense_turns_remaining(self) -> int:
        """Number of turns remaining in defensive stance."""
        return self._defense_timer

    @property
    def is_alive(self) -> bool:
        """Whether agent is alive (health > 0)."""
        return self._health > 0

    def attack(self, target: "AgentCore") -> dict:
        """
        Attack another agent.

        Calculates damage based on current health, then applies it to target.

        Args:
            target: Agent to attack

        Returns:
            dict: Attack result with:
                - success: bool, whether attack succeeded
                - damage_dealt: float, actual damage applied
                - target_killed: bool, whether target died from attack
                - error: str (optional), error message if failed

        Example:
            >>> result = combat.attack(enemy)
            >>> if result['success']:
            ...     print(f"Dealt {result['damage_dealt']} damage!")
        """
        if self._agent is None:
            return {
                "success": False,
                "error": "Component not attached to agent"
            }

        # Get target's combat component
        target_combat = target.get_component("combat")
        if not target_combat:
            return {
                "success": False,
                "error": "Target has no combat component"
            }

        # Calculate damage based on our health
        damage = self._calculate_attack_damage()

        # Apply damage to target
        actual_damage = target_combat.take_damage(damage)

        return {
            "success": True,
            "damage_dealt": actual_damage,
            "target_killed": not target_combat.is_alive,
            "target_health": target_combat.health,
        }

    def take_damage(self, damage: float) -> float:
        """
        Take damage from an attack.

        Applies damage reduction if defending, then reduces health.
        Triggers agent death if health reaches 0.

        Args:
            damage: Base damage amount

        Returns:
            float: Actual damage taken after defense

        Example:
            >>> damage_taken = combat.take_damage(25.0)
            >>> print(f"Took {damage_taken} damage")
        """
        # Apply defense reduction if defending
        if self._is_defending:
            damage *= self._config.defense_reduction

        # Apply damage
        self._health -= damage

        # Check for death
        if self._health <= 0:
            self._health = 0
            if self._agent:
                self._agent.terminate()

        return damage

    def heal(self, amount: float) -> float:
        """
        Restore health.

        Args:
            amount: Health points to restore

        Returns:
            float: Actual amount healed (capped at max health)

        Example:
            >>> healed = combat.heal(50.0)
            >>> print(f"Restored {healed} health")
        """
        old_health = self._health
        self._health = min(self._health + amount, self._config.starting_health)
        return self._health - old_health

    def set_health(self, amount: float) -> None:
        """
        Set health directly.

        Args:
            amount: New health value (clamped to 0-max)

        Note:
            Use sparingly - prefer heal() or take_damage() for tracking changes.
        """
        self._health = max(0, min(amount, self._config.starting_health))

        # Check for death
        if self._health <= 0 and self._agent:
            self._agent.terminate()

    def start_defense(self, duration: Optional[int] = None) -> None:
        """
        Enter defensive stance.

        While defending, incoming damage is reduced by defense_reduction factor.

        Args:
            duration: Turns to defend (uses config default if None)

        Example:
            >>> combat.start_defense(3)  # Defend for 3 turns
        """
        if duration is None:
            duration = self._config.defense_duration

        self._is_defending = True
        self._defense_timer = duration

    def end_defense(self) -> None:
        """
        End defensive stance immediately.

        Example:
            >>> combat.end_defense()  # Stop defending
        """
        self._is_defending = False
        self._defense_timer = 0

    def _calculate_attack_damage(self) -> float:
        """
        Calculate attack damage based on current health.

        Damage is scaled by health ratio - wounded agents deal less damage.

        Returns:
            float: Damage to deal
        """
        return self._config.base_attack_strength * self.health_ratio

    def get_defense_strength(self) -> float:
        """
        Get current defense strength.

        Returns:
            float: Defense strength (0 if not defending)

        Example:
            >>> defense = combat.get_defense_strength()
        """
        if not self._is_defending:
            return 0.0
        return self._config.base_defense_strength

    def on_step_end(self) -> None:
        """
        Called at end of each step.

        Updates defense timer and ends defensive stance when timer expires.
        """
        if self._defense_timer > 0:
            self._defense_timer -= 1
            if self._defense_timer <= 0:
                self._is_defending = False

    def get_state(self) -> dict:
        """
        Get serializable state.

        Returns:
            dict: Component state including health and defense status
        """
        return {
            "health": self._health,
            "is_defending": self._is_defending,
            "defense_timer": self._defense_timer,
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self._health = state.get("health", self._config.starting_health)
        self._is_defending = state.get("is_defending", False)
        self._defense_timer = state.get("defense_timer", 0)