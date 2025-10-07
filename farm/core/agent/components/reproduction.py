"""
Reproduction component for agent reproduction mechanics.

Handles offspring creation, resource costs, and generation tracking.
"""

from typing import TYPE_CHECKING, Optional, Callable
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import ReproductionConfig

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore
    from farm.core.services.interfaces import IAgentLifecycleService


class ReproductionComponent(IAgentComponent):
    """
    Component handling agent reproduction.

    Responsibilities:
    - Check reproduction requirements
    - Create offspring agents
    - Manage resource costs
    - Track generation lineage

    Single Responsibility: Only reproduction logic.

    Note:
        Requires lifecycle service for creating new agents.
        Requires resource component for cost management.
    """

    def __init__(
        self,
        config: ReproductionConfig,
        lifecycle_service: Optional["IAgentLifecycleService"] = None,
        offspring_factory: Optional[Callable] = None,
    ):
        """
        Initialize reproduction component.

        Args:
            config: Reproduction configuration
            lifecycle_service: Service for adding new agents (optional)
            offspring_factory: Factory function to create offspring (optional)
        """
        self._config = config
        self._lifecycle_service = lifecycle_service
        self._offspring_factory = offspring_factory
        self._agent: Optional["AgentCore"] = None
        self._reproduction_count = 0

    @property
    def name(self) -> str:
        """Component identifier."""
        return "reproduction"

    @property
    def offspring_cost(self) -> int:
        """Resource cost to reproduce."""
        return self._config.offspring_cost

    @property
    def reproduction_count(self) -> int:
        """Number of times this agent has reproduced."""
        return self._reproduction_count

    def can_reproduce(self) -> bool:
        """
        Check if agent can reproduce.

        Requirements:
        - Agent must be alive
        - Must have resource component
        - Must have enough resources

        Returns:
            bool: True if agent can reproduce

        Example:
            >>> if reproduction.can_reproduce():
            ...     reproduction.reproduce()
        """
        if self._agent is None or not self._agent.alive:
            return False

        # Check if agent has resource component
        resource = self._agent.get_component("resource")
        if not resource:
            return False

        # Check resource threshold
        return resource.level >= self._config.reproduction_threshold

    def reproduce(self) -> dict:
        """
        Create offspring agent.

        Consumes resources from parent and creates new agent at same position.

        Returns:
            dict: Reproduction result with:
                - success: bool, whether reproduction succeeded
                - offspring_id: str (optional), ID of created offspring
                - cost: int, resources consumed
                - error: str (optional), error message if failed

        Example:
            >>> result = reproduction.reproduce()
            >>> if result['success']:
            ...     print(f"Created offspring {result['offspring_id']}")
        """
        if self._agent is None:
            return {
                "success": False,
                "error": "Component not attached to agent"
            }

        if not self._agent.alive:
            return {
                "success": False,
                "error": "Dead agents cannot reproduce"
            }

        # Get resource component
        resource = self._agent.get_component("resource")
        if not resource:
            return {
                "success": False,
                "error": "Agent has no resource component"
            }

        # Check resource requirements
        if resource.level < self._config.reproduction_threshold:
            return {
                "success": False,
                "error": f"Insufficient resources (need {self._config.reproduction_threshold}, have {resource.level})"
            }

        # Get lifecycle service
        lifecycle_service = self._lifecycle_service
        if not lifecycle_service and hasattr(self._agent, "_lifecycle_service"):
            lifecycle_service = self._agent._lifecycle_service

        if not lifecycle_service:
            return {
                "success": False,
                "error": "No lifecycle service available"
            }

        try:
            # Generate offspring ID
            offspring_id = lifecycle_service.get_next_agent_id()

            # Create offspring (if factory provided)
            if self._offspring_factory:
                # Get parent position
                position = self._agent.state_manager.position
                generation = self._agent.state_manager.generation + 1

                # Create offspring using factory
                offspring = self._offspring_factory(
                    agent_id=offspring_id,
                    position=position,
                    initial_resources=self._config.offspring_initial_resources,
                    parent_ids=[self._agent.agent_id],
                    generation=generation,
                )

                # Add offspring to environment
                lifecycle_service.add_agent(offspring, flush_immediately=True)
            else:
                # Factory not provided - just consume resources
                # Actual offspring creation handled externally
                pass

            # Consume parent's resources
            resource.consume(self._config.offspring_cost)

            # Increment reproduction counter
            self._reproduction_count += 1

            return {
                "success": True,
                "offspring_id": offspring_id,
                "cost": self._config.offspring_cost,
                "offspring_resources": self._config.offspring_initial_resources,
                "parent_resources_after": resource.level,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Reproduction failed: {str(e)}"
            }

    def get_reproduction_info(self) -> dict:
        """
        Get information about reproduction requirements and status.

        Returns:
            dict: Reproduction info including costs and requirements

        Example:
            >>> info = reproduction.get_reproduction_info()
            >>> print(f"Need {info['required_resources']} resources to reproduce")
        """
        resource = None
        if self._agent:
            resource = self._agent.get_component("resource")

        return {
            "can_reproduce": self.can_reproduce(),
            "required_resources": self._config.reproduction_threshold,
            "current_resources": resource.level if resource else 0,
            "offspring_cost": self._config.offspring_cost,
            "offspring_initial_resources": self._config.offspring_initial_resources,
            "reproduction_count": self._reproduction_count,
        }

    def get_state(self) -> dict:
        """
        Get serializable state.

        Returns:
            dict: Component state including reproduction count
        """
        return {
            "reproduction_count": self._reproduction_count,
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self._reproduction_count = state.get("reproduction_count", 0)