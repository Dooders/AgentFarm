"""
Agent core - the coordinator for component-based agents.

AgentCore is the minimal agent that coordinates components and behaviors.
It has no agent-specific logic - all capabilities are provided by components.
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch

from farm.core.action import Action, action_registry
from farm.core.agent.behaviors.base import IAgentBehavior
from farm.core.agent.components.base import AgentComponent
from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.services import AgentServices
from farm.core.agent.state.state_manager import StateManager, StateSnapshot
from farm.core.device_utils import create_device_from_config

if TYPE_CHECKING:
    from farm.core.environment import Environment


class AgentCore:
    """
    Minimal coordinator for component-based agents.
    
    Responsibilities:
    - Manage component lifecycle
    - Coordinate components and behavior
    - Provide interface for actions to interact with agent
    - Delegate all capability logic to components
    
    Design principles:
    - Single Responsibility: Only coordinates, doesn't implement capabilities
    - Composition over Inheritance: Built from pluggable components
    - Open/Closed: Easy to extend with new components
    """
    
    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        services: AgentServices,
        behavior: IAgentBehavior,
        components: list[AgentComponent],
        config: AgentComponentConfig,
        environment: Optional["Environment"] = None,
        device: Optional[torch.device] = None,
        initial_resources: float = 100.0,
        generation: int = 0,
        parent_ids: Optional[list[str]] = None,
        agent_type: str = "AgentCore",
    ):
        """
        Initialize agent core.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial (x, y) position
            services: AgentServices container with all services
            behavior: IAgentBehavior implementation for decision-making
            components: List of components to attach to agent
            config: AgentComponentConfig with all settings
            environment: Optional environment reference
            device: Optional torch device for computations
            initial_resources: Starting resource level
            generation: Generation number
            parent_ids: IDs of parent agents
            agent_type: Type of agent (e.g., 'system', 'independent', 'control')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.services = services
        self.behavior = behavior
        self.config = config
        self.environment = environment
        self.alive = True
        self.total_reward = 0.0
        self.generation = generation
        
        # Device setup
        if device is not None:
            self.device = device
        elif config:
            self.device = create_device_from_config(config)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actions
        self.actions = action_registry.get_all(normalized=True)
        
        # State management
        current_time = services.get_current_time()
        self.state = StateManager(
            agent_id=agent_id,
            position=position,
            generation=generation,
            parent_ids=parent_ids,
            genome_id="",  # Will be set by factory
            birth_time=current_time,
        )
        
        # Component storage
        self._components: Dict[str, AgentComponent] = {}
        
        # Attach all components
        for component in components:
            self.attach_component(component)
        
        # Initialize resource level
        resource_comp = self.get_component("resource")
        if resource_comp:
            resource_comp.level = initial_resources
            self.state.resource_level = initial_resources
        
        # Initialize combat component
        combat_comp = self.get_component("combat")
        if combat_comp:
            self.state.health = combat_comp.health
        
        # Initialize movement component
        movement_comp = self.get_component("movement")
        if movement_comp:
            movement_comp.position = position
            self.state.position = position
    
    def attach_component(self, component: AgentComponent) -> None:
        """
        Attach a component to this agent.
        
        Args:
            component: Component to attach
        """
        component.attach(self)
        name = component.name.lower().replace("component", "")
        self._components[name] = component
    
    def get_component(self, name: str) -> Optional[AgentComponent]:
        """
        Get a component by name.
        
        Args:
            name: Component name (e.g., "movement", "resource", "combat")
            
        Returns:
            Component or None if not found
        """
        return self._components.get(name.lower())
    
    def step(self) -> None:
        """
        Execute one simulation step.
        
        Orchestrates:
        1. Component on_step_start (pre-decision logic)
        2. Decision and action execution
        3. Component on_step_end (post-action logic)
        """
        if not self.alive:
            return
        
        # Call on_step_start on all components
        for component in self._components.values():
            try:
                component.on_step_start()
            except Exception:
                pass
        
        # Check if agent should die from starvation
        resource_comp = self.get_component("resource")
        if resource_comp and resource_comp.starvation_counter >= resource_comp.config.starvation_threshold:
            self.terminate()
            return
        
        # Get current state and decision
        state_tensor = self._create_observation()
        
        # Get enabled actions (for curriculum learning)
        enabled_actions = self._get_enabled_actions()
        
        # Decide and execute action
        try:
            action = self.behavior.decide_action(self, state_tensor, enabled_actions)
            self._execute_action(action, state_tensor)
        except Exception:
            pass
        
        # Call on_step_end on all components
        for component in self._components.values():
            try:
                component.on_step_end()
            except Exception:
                pass
        
        # Update state snapshot
        self._update_state_snapshot()
    
    def act(self) -> None:
        """
        Alias for step() for consistency with agent lifecycle.
        
        Executes one simulation step orchestrating component lifecycle,
        decision-making, and action execution.
        """
        self.step()
    
    def _create_observation(self) -> torch.Tensor:
        """
        Create observation tensor for decision-making.
        
        Delegates to perception component if available, otherwise returns environment observation.
        """
        perception_comp = self.get_component("perception")
        if perception_comp:
            try:
                return perception_comp.get_observation_tensor(self.device)
            except Exception:
                pass
        
        # Fallback to environment if available
        if self.environment:
            try:
                observation_np = self.environment.observe(self.agent_id)
                return torch.from_numpy(observation_np).to(device=self.device, dtype=torch.float32)
            except Exception:
                pass
        
        # Final fallback
        return torch.zeros((1, 11, 11), dtype=torch.float32, device=self.device)
    
    def _get_enabled_actions(self) -> Optional[list[Action]]:
        """Get list of enabled actions based on curriculum learning if configured."""
        if not self.config.decision.curriculum_phases:
            return None
        
        current_step = self.services.get_current_time()
        curriculum_phases = self.config.decision.curriculum_phases
        
        for phase in curriculum_phases:
            if current_step < phase.get("steps", -1) or phase.get("steps") == -1:
                enabled_action_names = phase.get("enabled_actions", [])
                return [a for a in self.actions if a.name in enabled_action_names]
        
        return None
    
    def _execute_action(self, action: Action, state_tensor: torch.Tensor) -> None:
        """
        Execute an action and handle learning update.
        
        Args:
            action: Action to execute
            state_tensor: State tensor before action
        """
        # Store pre-action state
        pre_action_state = self.state.snapshot(self.services.get_current_time())
        
        # Capture resource level before action execution for accurate logging
        resources_before = pre_action_state.resource_level
        
        # Execute action
        try:
            action_result = action.execute(self)
        except Exception:
            action_result = {"success": False, "error": "Action execution failed"}
        
        # Get post-action state
        post_action_state = self.state.snapshot(self.services.get_current_time())
        
        # Capture resource level after action execution
        resources_after = post_action_state.resource_level
        
        # Log action to database if logger is available
        if (
            self.environment
            and hasattr(self.environment, "db")
            and self.environment.db
            and hasattr(self.environment.db, "logger")
        ):
            try:
                # Get current time step
                current_time = self.services.get_current_time()

                # Log the agent action
                self.environment.db.logger.log_agent_action(
                    step_number=current_time,
                    agent_id=self.agent_id,
                    action_type=action.name,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    reward=0,  # Reward will be calculated later
                    details=action_result.get("details", {}) if isinstance(action_result, dict) else {},
                )
            except Exception as e:
                # Log warning but don't crash on database logging failure
                from farm.utils.logging import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Failed to log agent action {action.name}: {e}")
        
        # Calculate reward (simple: based on state changes)
        reward = self._calculate_reward(pre_action_state, post_action_state, action)
        
        # Get next state
        next_state_tensor = self._create_observation()
        
        # Update behavior with experience
        try:
            self.behavior.update(
                state=state_tensor,
                action=action,
                reward=reward,
                next_state=next_state_tensor,
                done=not self.alive,
            )
        except Exception:
            pass
    
    def _calculate_reward(self, pre_state: StateSnapshot, post_state: StateSnapshot, action: Action) -> float:
        """
        Calculate reward for state transition.
        
        Args:
            pre_state: State before action
            post_state: State after action
            action: Action executed
            
        Returns:
            Calculated reward
        """
        resource_delta = (post_state.resource_level - pre_state.resource_level) * 0.1
        health_delta = (post_state.health - pre_state.health) * 0.5
        survival = 0.1 if self.alive else -10.0
        action_bonus = 0.05 if action.name != "pass" else 0.0
        
        return resource_delta + health_delta + survival + action_bonus
    
    def _update_state_snapshot(self) -> None:
        """Update state snapshot from components."""
        resource_comp = self.get_component("resource")
        if resource_comp:
            self.state.resource_level = resource_comp.level
        
        combat_comp = self.get_component("combat")
        if combat_comp:
            self.state.health = combat_comp.health
            self.state.is_defending = combat_comp.is_defending
            self.state.defense_timer = combat_comp.defense_timer
        
        movement_comp = self.get_component("movement")
        if movement_comp:
            self.state.position = movement_comp.position
    
    def terminate(self) -> None:
        """
        Terminate the agent.
        
        Calls on_terminate on all components and marks agent as dead.
        """
        if not self.alive:
            return
        
        self.alive = False
        self.state.set_dead(self.services.get_current_time())
        
        # Notify all components
        for component in self._components.values():
            try:
                component.on_terminate()
            except Exception:
                pass
        
        # Remove from lifecycle service
        if self.services.lifecycle_service:
            try:
                self.services.lifecycle_service.remove_agent(self)
            except Exception:
                pass
    
    # Properties delegating to components
    
    @property
    def position(self) -> tuple[float, float]:
        """Get agent position."""
        return self.state.position
    
    @position.setter
    def position(self, value: tuple[float, float]) -> None:
        """Set agent position."""
        movement_comp = self.get_component("movement")
        if movement_comp:
            movement_comp.set_position(value)
            self.state.position = value
    
    @property
    def resource_level(self) -> float:
        """Get resource level."""
        resource_comp = self.get_component("resource")
        return resource_comp.level if resource_comp else 0.0
    
    @resource_level.setter
    def resource_level(self, value: float) -> None:
        """Set resource level."""
        resource_comp = self.get_component("resource")
        if resource_comp:
            resource_comp.level = value
    
    @property
    def current_health(self) -> float:
        """Get current health."""
        combat_comp = self.get_component("combat")
        return combat_comp.health if combat_comp else 0.0
    
    @property
    def is_defending(self) -> bool:
        """Check if agent is defending."""
        combat_comp = self.get_component("combat")
        return combat_comp.is_defending if combat_comp else False
    
    @is_defending.setter
    def is_defending(self, value: bool) -> None:
        """Set agent defending status."""
        combat_comp = self.get_component("combat")
        if combat_comp:
            combat_comp.is_defending = value
    
    @property
    def defense_timer(self) -> int:
        """Get defense timer."""
        combat_comp = self.get_component("combat")
        return combat_comp.defense_timer if combat_comp else 0
    
    @defense_timer.setter
    def defense_timer(self, value: int) -> None:
        """Set defense timer."""
        combat_comp = self.get_component("combat")
        if combat_comp:
            combat_comp.defense_timer = value
    
    @property
    def birth_time(self) -> int:
        """Get birth time."""
        return self.state.birth_time
    
    @property
    def genome_id(self) -> str:
        """Get genome ID."""
        return self.state.genome_id
    
    @property
    def starting_health(self) -> float:
        """Get starting health."""
        combat_comp = self.get_component("combat")
        return combat_comp.config.starting_health if combat_comp else 100.0
    
    def get_state(self) -> StateSnapshot:
        """Get complete state snapshot."""
        return self.state.snapshot(self.services.get_current_time())
    
    @property
    def step_reward(self) -> float:
        """Get current step reward from reward component."""
        reward_comp = self.get_component("reward")
        if reward_comp:
            return reward_comp.step_reward
        return 0.0