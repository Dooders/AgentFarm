"""Attack action logging module.

This module handles logging of attack-related actions and outcomes in the simulation,
including attacks, defenses, and their results.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent
    from farm.database.database import SimulationDatabase

logger = logging.getLogger(__name__)


class AttackLogger:
    """Logger for attack-related actions and outcomes."""

    def __init__(self, db: Optional["SimulationDatabase"] = None):
        self.db = db

    def log_defense(
        self,
        step_number: int,
        agent: "BaseAgent",
        resources_before: float,
        resources_after: float,
    ) -> None:
        """Log defensive stance action."""
        if self.db is None:
            return

        self.db.logger.log_agent_action(
            step_number=step_number,
            agent_id=agent.agent_id,
            action_type="defend",
            resources_before=resources_before,
            resources_after=resources_after,
            reward=0,
            details={
                "is_defending": True,
                "health_ratio": agent.current_health / agent.starting_health,
            },
        )

        # Also log as interaction for comprehensive tracking
        self.db.logger.log_interaction_edge(
            step_number=step_number,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",  # Self-targeted defense
            target_id=agent.agent_id,
            interaction_type="defend",
            action_type="defend",
            details={},  # Minimal details - full data already in ActionModel
        )
        logger.debug(f"Agent {agent.agent_id} took defensive stance")

    def log_attack_attempt(
        self,
        step_number: int,
        agent: "BaseAgent",
        action_target_id: Optional[str],
        target_position: Tuple[float, float],
        resources_before: float,
        resources_after: float,
        success: bool,
        targets_found: int = 0,
        damage_dealt: float = 0.0,
        reason: Optional[str] = None,
    ) -> None:
        """Log attack attempt and its outcome."""
        if self.db is None:
            return

        details = {
            "success": success,
            "target_position": target_position,
            "targets_found": targets_found,
            "damage_dealt": damage_dealt,
            "health_ratio": agent.current_health / agent.starting_health,
        }

        if reason:
            details["reason"] = reason

        self.db.logger.log_agent_action(
            step_number=step_number,
            agent_id=agent.agent_id,
            action_type="attack",
            action_target_id=action_target_id,
            resources_before=resources_before,
            resources_after=resources_after,
            reward=0,  # Reward will be handled separately by the learning system
            details=details,
        )

        # Also log as interaction for comprehensive tracking
        interaction_type = "attack" if success else "attack_failed"
        target_type = "agent" if action_target_id else "position"
        target_id = action_target_id if action_target_id else f"{target_position[0]},{target_position[1]}"

        self.db.logger.log_interaction_edge(
            step_number=step_number,
            source_type="agent",
            source_id=agent.agent_id,
            target_type=target_type,
            target_id=target_id,
            interaction_type=interaction_type,
            action_type="attack",
            details={},  # Minimal details - full data already in ActionModel
        )

        if success:
            logger.debug(
                f"Agent {agent.agent_id} successful attack at {target_position}"
                f" dealing {damage_dealt} damage to {targets_found} targets"
            )
        else:
            logger.debug(
                f"Agent {agent.agent_id} failed attack at {target_position}."
                f" Reason: {reason}"
            )
