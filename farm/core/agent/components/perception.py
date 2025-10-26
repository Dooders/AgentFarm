"""
Perception component.

Handles agent perception, spatial awareness, and observation generation for decision-making.
Consolidates all perception logic from the environment into a single component.
"""

import math
import time as _time
from typing import Dict, List, Tuple

import numpy as np
import torch

from farm.core.agent.config.component_configs import PerceptionConfig
from farm.core.agent.services import AgentServices
from farm.core.geometry import discretize_position_continuous
from farm.core.observations import AgentObservation, ObservationConfig
from farm.core.perception import PerceptionData
from farm.utils.logging import get_logger

from .base import AgentComponent

# Import bilinear_distribute_value from environment
try:
    from farm.core.environment import bilinear_distribute_value
except ImportError:
    # Fallback implementation if not available
    def bilinear_distribute_value(
        position: Tuple[float, float],
        value: float,
        grid: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> None:
        """Distribute a value across grid cells using bilinear interpolation."""
        x, y = position
        width, height = grid_size

        # Get the four nearest grid cells
        x_floor = int(math.floor(x))
        y_floor = int(math.floor(y))
        x_ceil = min(x_floor + 1, width - 1)
        y_ceil = min(y_floor + 1, height - 1)

        # Calculate interpolation weights
        wx = x - x_floor
        wy = y - y_floor

        # Distribute value across four cells
        grid[y_floor, x_floor] += value * (1 - wx) * (1 - wy)
        grid[y_floor, x_ceil] += value * wx * (1 - wy)
        grid[y_ceil, x_floor] += value * (1 - wx) * wy
        grid[y_ceil, x_ceil] += value * wx * wy


logger = get_logger(__name__)


class PerceptionComponent(AgentComponent):
    """
    Manages agent perception and observation using the full multi-channel observation system.

    Responsibilities:
    - Query spatial service for nearby entities
    - Generate multi-channel observation tensors
    - Build world layers for observation system
    - Handle egocentric perception with agent orientation
    - Integrate with AgentObservation for full perception capabilities
    """

    def __init__(self, services: AgentServices, config: PerceptionConfig):
        """
        Initialize perception component.

        Args:
            services: Agent services
            config: Perception configuration
        """
        super().__init__(services, "PerceptionComponent")
        self.config = config
        self.last_perception = None
        self.agent_observation = None
        self._perception_profile = {
            "spatial_query_time_s": 0.0,
            "bilinear_time_s": 0.0,
            "nearest_time_s": 0.0,
            "bilinear_points": 0,
            "nearest_points": 0,
        }
        # Cache for spatial queries to avoid redundant calls
        self._spatial_cache = {
            "resources": None,
            "agents": None,
            "last_position": None,
            "last_radius": None,
        }

    def attach(self, core) -> None:
        """Attach to core and initialize observation system."""
        super().attach(core)

        # Initialize AgentObservation using PerceptionConfig to maintain radius consistency
        # This ensures the observation system uses the same radius as the perception component
        # Use CPU as default device during initialization - will be updated on first observation
        obs_config = self._create_observation_config_from_perception_config("cpu")
        self.agent_observation = AgentObservation(obs_config)
        self._last_obs_config = obs_config

    def on_step_start(self) -> None:
        """Called at start of step."""
        pass

    def on_step_end(self) -> None:
        """Called at end of step."""
        pass

    def on_terminate(self) -> None:
        """Called when agent dies."""
        pass

    def _create_observation_config_from_perception_config(self, device: str = "cpu") -> ObservationConfig:
        """
        Create an ObservationConfig from the PerceptionConfig to maintain radius consistency.
        
        This method ensures that the AgentObservation system uses the same radius
        as the PerceptionComponent, eliminating inconsistencies between simple
        perception grids and multi-channel observations.
        
        Args:
            device: Device to use for observation tensors
        
        Returns:
            ObservationConfig with radius matching PerceptionConfig.perception_radius
        """
        return ObservationConfig(
            R=self.config.perception_radius,
            fov_radius=self.config.perception_radius,
            # Use other defaults from ObservationConfig
            device=device,
            dtype="float32",
            initialization="zeros",
            storage_mode="hybrid",
            enable_metrics=True,
            gamma_trail=0.90,
            gamma_dmg=0.85,
            gamma_sig=0.92,
            gamma_known=0.98,
            sparse_backend="scatter",
            default_point_reduction="max",
            channel_reduction_overrides={},
            high_frequency_channels=[],
            random_min=0.0,
            random_max=1.0,
        )

    def _get_current_observation_config(self) -> ObservationConfig:
        """
        Get the current observation config from the environment.

        This method ensures we always use the most up-to-date observation config,
        even if the environment changes after component initialization.

        Returns:
            Current ObservationConfig from environment or default if not available
        """
        if (
            self.core
            and hasattr(self.core, "environment")
            and self.core.environment
            and hasattr(self.core.environment, "observation_config")
            and self.core.environment.observation_config
        ):
            return self.core.environment.observation_config
        else:
            return ObservationConfig()

    def _ensure_observation_config_current(self, device: str = "cpu") -> None:
        """
        Ensure the AgentObservation instance uses the current observation config.

        This method updates the AgentObservation with the PerceptionConfig-based
        observation config to maintain radius consistency.

        Args:
            device: Device to use for observation tensors
        """
        if not self.agent_observation:
            return

        current_config = self._create_observation_config_from_perception_config(device)

        # Check if we need to update the observation config
        # We'll compare key attributes to detect changes
        if not hasattr(self, "_last_obs_config") or self._last_obs_config != current_config:
            # Log config change for debugging
            logger.debug(f"Updating observation config for agent {getattr(self.core, 'agent_id', 'unknown')}")

            # Update the AgentObservation with current config
            self.agent_observation = AgentObservation(current_config)
            self._last_obs_config = current_config

    def validate_observation_config_consistency(self) -> bool:
        """
        Validate that the current observation config is consistent with the environment.

        This method can be called to check if there are any configuration mismatches
        that could cause issues with observation generation.

        Returns:
            True if config is consistent, False otherwise
        """
        if not self.core or not hasattr(self.core, "environment") or not self.core.environment:
            return True  # No environment to validate against

        env = self.core.environment
        current_config = self._get_current_observation_config()

        # Check if environment has observation config
        if not hasattr(env, "observation_config") or not env.observation_config:
            logger.warning("Environment has no observation_config - using default")
            return False

        # Check if configs match
        if self._last_obs_config != current_config:
            logger.warning("Observation config mismatch detected - will be updated on next observation")
            return False

        return True

    def _get_cached_spatial_query(self, radius: float, index_names: List[str]) -> Dict[str, List]:
        """
        Get spatial query results with caching to avoid redundant calls.

        Args:
            radius: Query radius
            index_names: List of index names to query

        Returns:
            Dictionary mapping index names to lists of entities
        """
        if not self.core or not self.spatial_service:
            return {name: [] for name in index_names}

        current_position = self.core.position
        current_radius = radius

        # Check if we can use cached results
        cache_valid = (
            self._spatial_cache["last_position"] == current_position
            and self._spatial_cache["last_radius"] == current_radius
        )

        if cache_valid:
            # Return cached results for requested index names
            result = {}
            for name in index_names:
                if name in self._spatial_cache:
                    result[name] = self._spatial_cache[name] or []
                else:
                    result[name] = []
            return result

        # Perform new spatial query
        try:
            nearby = self.spatial_service.get_nearby(current_position, current_radius, index_names)

            # Update cache
            self._spatial_cache["last_position"] = current_position
            self._spatial_cache["last_radius"] = current_radius
            for name in index_names:
                self._spatial_cache[name] = nearby.get(name, [])

            return nearby
        except Exception as e:
            logger.warning(f"Failed to query nearby entities: {e}")
            return {name: [] for name in index_names}

    def get_perception(self) -> PerceptionData:
        """
        Get agent's current perception of surrounding environment.

        Creates a grid representation of nearby entities:
        - 0: Empty space
        - 1: Resource
        - 2: Other agent
        - 3: Boundary/obstacle

        Returns:
            PerceptionData: Structured perception grid
        """
        if not self.core:
            return PerceptionData(
                np.zeros((2 * self.config.perception_radius + 1, 2 * self.config.perception_radius + 1), dtype=np.int8)
            )

        radius = self.config.perception_radius
        size = 2 * radius + 1
        perception = np.zeros((size, size), dtype=np.int8)

        # Get nearby entities using cached spatial query
        nearby = self._get_cached_spatial_query(radius, ["resources", "agents"])
        nearby_resources = nearby.get("resources", [])
        nearby_agents = nearby.get("agents", [])

        # Helper to convert world coords to grid
        def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
            if self.config.position_discretization_method == "round":
                gx = int(round(wx - self.core.position[0] + radius))
                gy = int(round(wy - self.core.position[1] + radius))
            elif self.config.position_discretization_method == "ceil":
                gx = int(math.ceil(wx - self.core.position[0] + radius))
                gy = int(math.ceil(wy - self.core.position[1] + radius))
            else:  # "floor" (default)
                gx = int(math.floor(wx - self.core.position[0] + radius))
                gy = int(math.floor(wy - self.core.position[1] + radius))
            return gx, gy

        # Add resources
        for resource in nearby_resources:
            try:
                gx, gy = world_to_grid(resource.position[0], resource.position[1])
                if 0 <= gx < size and 0 <= gy < size:
                    perception[gy, gx] = 1
            except Exception as e:
                logger.warning(f"Failed to add resource to perception: {e}")

        # Add other agents
        for agent in nearby_agents:
            try:
                if agent.agent_id != self.core.agent_id:
                    gx, gy = world_to_grid(agent.position[0], agent.position[1])
                    if 0 <= gx < size and 0 <= gy < size:
                        perception[gy, gx] = 2
            except Exception as e:
                logger.warning(f"Failed to add agent to perception: {e}")

        # Mark boundaries
        x_min = self.core.position[0] - radius
        y_min = self.core.position[1] - radius

        for i in range(size):
            for j in range(size):
                world_x = x_min + j
                world_y = y_min + i
                try:
                    if self.validation_service and not self.validation_service.is_valid_position((world_x, world_y)):
                        perception[i, j] = 3
                except Exception:
                    perception[i, j] = 3

        self.last_perception = PerceptionData(perception)
        return self.last_perception

    def _create_world_layers(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Create world layers for the observation system.

        This method consolidates the resource layer creation logic from the environment
        and creates the world layers needed for AgentObservation.perceive_world().

        Args:
            device: Device to create tensors on

        Returns:
            Dictionary mapping layer names to tensors
        """
        if not self.core or not hasattr(self.core, "environment") or not self.core.environment:
            # Return empty layers if no environment
            R = self.config.perception_radius
            S = 2 * R + 1
            empty_layer = torch.zeros((S, S), dtype=torch.float32, device=device)
            return {
                "RESOURCES": empty_layer,
                "OBSTACLES": empty_layer,
                "TERRAIN_COST": empty_layer,
            }

        env = self.core.environment

        # Use PerceptionConfig radius for consistency with perception component
        R = self.config.perception_radius
        S = 2 * R + 1
        
        # Get discretization method from config
        if env.config and getattr(env.config, "environment", None) is not None:
            discretization_method = getattr(env.config.environment, "position_discretization_method", "floor")
            use_bilinear = getattr(env.config.environment, "use_bilinear_interpolation", True)
        else:
            discretization_method = (
                getattr(env.config, "position_discretization_method", "floor") if env.config else "floor"
            )
            use_bilinear = getattr(env.config, "use_bilinear_interpolation", True) if env.config else True

        # Agent position as (y, x) using configured discretization method
        height, width = int(env.height), int(env.width)
        grid_size = (width, height)
        ax, ay = discretize_position_continuous(self.core.position, grid_size, discretization_method)

        # Ensure spatial index is up to date before observation generation
        if hasattr(env, "spatial_index") and env.spatial_index:
            env.spatial_index.update()

        # Build local resource layer
        resource_local = torch.zeros(
            (S, S),
            dtype=torch.float32,
            device=device,
        )

        # Get max resource amount
        if hasattr(env, "max_resource") and env.max_resource is not None:
            max_amount = env.max_resource
        else:
            from farm.utils.config_utils import get_nested_then_flat

            max_amount = get_nested_then_flat(
                config=env.config,
                nested_parent_attr="resources",
                nested_attr_name="max_resource_amount",
                flat_attr_name="max_resource_amount",
                default_value=10,
                expected_types=(int, float),
            )

        # Query nearby resources using cached spatial query
        nearby_resources = []
        used_memmap = False

        try:
            if (
                hasattr(env, "resource_manager")
                and getattr(env.resource_manager, "has_memmap", False)
                and env.resource_manager.has_memmap
            ):
                used_memmap = True
                # Compute world-space window bounds (y,x) centered at (ay, ax)
                y0 = ay - R
                y1 = ay + R + 1
                x0 = ax - R
                x1 = ax + R + 1
                window_np = env.resource_manager.get_resource_window(y0, y1, x0, x1, normalize=True)
                # Convert to torch tensor of correct dtype/device with minimal copies
                if (
                    window_np.dtype == np.float32
                ):
                    resource_local = torch.from_numpy(window_np)
                else:
                    resource_local = torch.tensor(
                        window_np,
                        dtype=torch.float32,
                        device="cpu",
                        copy=False,
                    )
            else:
                # Use cached spatial query to avoid redundant calls
                nearby = self._get_cached_spatial_query(R + 1, ["resources"])
                nearby_resources = nearby.get("resources", [])
        except Exception as e:
            logger.warning(f"Failed to query nearby resources: {e}")
            nearby_resources = []

        # Distribute resources to local grid
        if not used_memmap and use_bilinear:
            _tb0 = _time.perf_counter()
            for res in nearby_resources:
                # Convert world to local continuous coords where (R, R) is agent center
                lx = float(res.position[0]) - (ax - R)
                ly = float(res.position[1]) - (ay - R)
                bilinear_distribute_value(
                    (lx, ly),
                    float(res.amount) / float(max_amount),
                    resource_local,
                    (S, S),
                )
                # 4 target points per bilinear distribution
                self._perception_profile["bilinear_points"] += 4
            _tb1 = _time.perf_counter()
            self._perception_profile["bilinear_time_s"] += max(0.0, _tb1 - _tb0)
        elif not used_memmap:
            _tn0 = _time.perf_counter()
            for res in nearby_resources:
                rx, ry = discretize_position_continuous(res.position, (width, height), discretization_method)
                lx = rx - (ax - R)
                ly = ry - (ay - R)
                if 0 <= lx < S and 0 <= ly < S:
                    resource_local[int(ly), int(lx)] += float(res.amount) / float(max_amount)
                    self._perception_profile["nearest_points"] += 1
            _tn1 = _time.perf_counter()
            self._perception_profile["nearest_time_s"] += max(0.0, _tn1 - _tn0)

        # Create empty layers for obstacles and terrain cost
        obstacles_local = torch.zeros_like(resource_local)
        terrain_cost_local = torch.zeros_like(resource_local)

        return {
            "RESOURCES": resource_local,
            "OBSTACLES": obstacles_local,
            "TERRAIN_COST": terrain_cost_local,
        }

    def get_observation_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Get full multi-channel observation tensor for decision-making.

        This method uses the complete AgentObservation system to generate
        a multi-channel observation tensor with all available perception data.

        Args:
            device: Torch device to place tensor on

        Returns:
            Multi-channel observation tensor
        """
        if not self.core:
            return torch.zeros((1, 11, 11), dtype=torch.float32)

        # Determine target device early to ensure consistency throughout the pipeline
        if device is None:
            device = getattr(self.core, "device", torch.device("cpu"))
        
        # Convert device to string for consistency with ObservationConfig
        device_str = str(device)

        # Use the full observation system if available
        if self.agent_observation and hasattr(self.core, "environment") and self.core.environment:
            try:
                # Ensure we're using the current observation config with correct device
                self._ensure_observation_config_current(device_str)

                # Create world layers with correct device
                world_layers = self._create_world_layers(device_str)

                # Get agent health
                self_hp01 = self.core.current_health / self.core.starting_health

                # Get agent position in world coordinates
                env = self.core.environment
                height, width = int(env.height), int(env.width)
                grid_size = (width, height)

                # Get discretization method from PerceptionConfig
                discretization_method = self.config.position_discretization_method

                ax, ay = discretize_position_continuous(self.core.position, grid_size, discretization_method)

                # Update observation with full perception data
                self.agent_observation.perceive_world(
                    world_layers=world_layers,
                    agent_world_pos=(ay, ax),
                    self_hp01=self_hp01,
                    allies=None,  # Let observation system use spatial index for efficiency
                    enemies=None,  # Let observation system use spatial index for efficiency
                    goal_world_pos=None,  # TODO: Set if needed
                    recent_damage_world=[],  # TODO: Implement if needed
                    ally_signals_world=[],  # TODO: Implement if needed
                    trails_world_points=[],  # TODO: Implement if needed
                    spatial_index=env.spatial_index if hasattr(env, "spatial_index") else None,
                    agent_object=self.core,
                    agent_orientation=getattr(self.core, "orientation", 0.0),
                )

                # Get the observation tensor - it should already be on the correct device
                tensor = self.agent_observation.tensor()
                return tensor.to(device=device)

            except Exception as e:
                logger.warning(f"Failed to generate full observation: {e}")
                # Fall back to simple perception grid
                pass

        # Fallback to simple perception grid
        perception = self.get_perception()
        grid = perception.grid.astype(np.float32)
        tensor = torch.from_numpy(grid).unsqueeze(0).to(device=device)
        return tensor
