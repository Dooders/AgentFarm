"""
Combat component.

Handles health, attacks, defense status, and combat mechanics.
"""

from farm.core.agent.config.component_configs import CombatConfig
from farm.core.agent.services import AgentServices

from .base import AgentComponent


class CombatComponent(AgentComponent):
    """
    Manages agent combat and health.

    Responsibilities:
    - Track health points
    - Handle attacks and damage
    - Manage defensive stance
    - Track defense timer
    """

    def __init__(self, services: AgentServices, config: CombatConfig):
        """
        Initialize combat component.

        Args:
            services: Agent services
            config: Combat configuration
        """
        super().__init__(services, "CombatComponent")
        self.config = config
        self.health = config.starting_health
        self.is_defending = False
        self.defense_timer = 0

    def attach(self, core) -> None:
        """Attach to core."""
        super().attach(core)

    def on_step_start(self) -> None:
        """Handle defense timer countdown at start of step."""
        if self.defense_timer > 0:
            self.defense_timer -= 1
            self.is_defending = self.defense_timer > 0
        else:
            self.is_defending = False

    def on_step_end(self) -> None:
        """Called at end of step."""
        pass

    def on_terminate(self) -> None:
        """Called when agent dies."""
        pass

    def take_damage(self, damage: float) -> float:
        """
        Apply damage to this agent.

        Damage is reduced if agent is defending. Agent dies if health reaches 0.

        Args:
            damage: Base damage amount

        Returns:
            float: Actual damage dealt after defense calculations
        """
        # Store health before damage for logging
        health_before = self.health

        # Reduce damage if defending
        actual_damage = damage
        if self.is_defending:
            actual_damage = damage * self.config.defense_damage_reduction

        # Apply damage
        self.health = max(0, self.health - actual_damage)

        # Log health incident if damage was dealt and logging service available
        if actual_damage > 0 and self.logging_service:
            try:
                self.logging_service.log_health_incident(
                    step_number=self.current_time,
                    agent_id=self.core.agent_id if self.core else "unknown",
                    health_before=health_before,
                    health_after=self.health,
                    cause="combat_damage",
                    details={
                        "damage_dealt": actual_damage,
                        "was_defending": self.is_defending,
                        "defense_reduction": self.config.defense_damage_reduction if self.is_defending else 1.0,
                    },
                )
            except Exception:
                pass

        # Check for death
        if self.health <= 0 and self.core:
            self.core.terminate()

        return actual_damage

    def heal(self, amount: float) -> None:
        """Heal the agent (capped at starting health)."""
        health_before = self.health
        new_health = min(self.config.starting_health, self.health + amount)

        self.health = new_health

        # Log health incident if healing occurred and logging service available
        if amount > 0 and self.logging_service:
            try:
                self.logging_service.log_health_incident(
                    step_number=self.current_time,
                    agent_id=self.core.agent_id if self.core else "unknown",
                    health_before=health_before,
                    health_after=self.health,
                    cause="healing",
                    details={
                        "healing_amount": amount,
                        "actual_healing": self.health - health_before,
                        "health_capped": self.health == self.config.starting_health,
                    },
                )
            except Exception:
                pass

    def start_defense(self) -> None:
        """Start defensive stance for configured duration."""
        self.is_defending = True
        self.defense_timer = self.config.defense_timer_duration

    def end_defense(self) -> None:
        """End defensive stance immediately."""
        self.is_defending = False
        self.defense_timer = 0

    @property
    def attack_strength(self) -> float:
        """Get current attack strength scaled by health."""
        health_ratio = self.health / self.config.starting_health
        return self.config.base_attack_strength * health_ratio

    @property
    def defense_strength(self) -> float:
        """Get current defense strength (0 if not defending)."""
        if not self.is_defending:
            return 0.0
        return self.config.base_defense_strength

    @property
    def health_ratio(self) -> float:
        """Get health as ratio of starting health (0.0 to 1.0)."""
        return self.health / self.config.starting_health

    @property
    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.health > 0
