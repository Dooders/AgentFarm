import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from farm.core.resources import Resource

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manage resource initialization, regeneration, and lifecycle.

    Responsibilities:
    - Initialize resource nodes using a provided distribution or config defaults
    - Regenerate resources deterministically (seeded) or stochastically per step
    - Track consumption, regeneration, and depletion events
    - Provide nearby/nearest queries (via spatial index when available)

    Notes:
    - Initialization and update logic is aligned with the environment's original
      behavior to preserve deterministic seeds and metrics.
    - Methods return lightweight stats for logging and analysis where relevant.
    """

    def __init__(
        self,
        width: float,
        height: float,
        config=None,
        seed: Optional[int] = None,
        database_logger=None,
        spatial_index=None,
    ):
        """
        Initialize the resource manager.

        Parameters
        ----------
        width : float
            Environment width
        height : float
            Environment height
        config : object, optional
            Configuration object containing resource parameters
        seed : int, optional
            Random seed for deterministic behavior
        database_logger : object, optional
            Database logger for resource logging
        spatial_index : SpatialIndex, optional
            Spatial index for efficient spatial queries
        """
        self.width = width
        self.height = height
        self.config = config
        self.seed_value = seed
        self.database_logger = database_logger
        self.spatial_index = spatial_index

        # Resource tracking
        self.resources: List[Resource] = []
        self.next_resource_id = 0

        # Regeneration tracking
        self.regeneration_step = 0
        self.total_resources_consumed = 0.0
        self.total_resources_regenerated = 0.0

        # Performance tracking
        self.regeneration_events = 0
        self.depletion_events = 0

        # Set random seed if provided
        if self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)

    def set_spatial_index(self, spatial_index):
        """Set the spatial index reference for efficient spatial queries.

        Parameters
        ----------
        spatial_index : SpatialIndex
            Spatial index instance to use for queries
        """
        self.spatial_index = spatial_index

    def mark_positions_dirty(self):
        """Mark that resource positions have changed and spatial index needs updating."""
        if self.spatial_index is not None:
            self.spatial_index.mark_positions_dirty()

    def initialize_resources(
        self, distribution: Union[Dict[str, Any], Callable]
    ) -> List[Resource]:
        """
        Initialize resources in the environment using the same logic as the original Environment.

        Parameters
        ----------
        distribution : Dict
            Distribution configuration (for compatibility, but uses config.initial_resources)

        Returns
        -------
        List[Resource]
            List of created resources
        """
        # Check if distribution has amount, otherwise use config.initial_resources
        if distribution and isinstance(distribution, dict) and "amount" in distribution:
            amount = distribution["amount"]
        else:
            amount = (
                getattr(self.config, "initial_resources", 20) if self.config else 20
            )

        logger.info(f"Initializing {amount} resources using original Environment logic")

        # Clear existing resources
        self.resources.clear()
        self.next_resource_id = 0

        # Use seeded random generator for deterministic behavior
        if self.seed_value is not None:
            rng = random.Random(self.seed_value)
        else:
            rng = random

        for i in range(amount):
            # Random position (same as original)
            x = rng.uniform(0, self.width)
            y = rng.uniform(0, self.height)

            # Create resource with regeneration parameters (same as original)
            resource = Resource(
                resource_id=i,
                position=(x, y),
                amount=(
                    self.config.resources.max_resource_amount if self.config else 10
                ),
                max_amount=(
                    self.config.resources.max_resource_amount if self.config else 10
                ),
                regeneration_rate=(
                    getattr(self.config, "resource_regen_rate", 0.1)
                    if self.config
                    else 0.1
                ),
            )
            self.resources.append(resource)

        # Update next_resource_id to match the number of resources created (same as original)
        self.next_resource_id = amount

        # Log resources to database if logger is available
        if self.database_logger:
            for resource in self.resources:
                self.database_logger.log_resource(
                    resource_id=resource.resource_id,
                    initial_amount=resource.amount,
                    position=resource.position,
                )

        logger.info(f"Successfully initialized {len(self.resources)} resources")
        return self.resources

    def _create_random_distribution(
        self, amount: int, distribution: Dict
    ) -> List[Resource]:
        """Create resources with random distribution."""
        resources = []

        # Use seeded random generator for deterministic behavior
        if self.seed_value is not None:
            rng = random.Random(self.seed_value)
        else:
            rng = random

        for _ in range(amount):
            position = (rng.uniform(0, self.width), rng.uniform(0, self.height))
            resource = self._create_resource(position, distribution)
            resources.append(resource)

        return resources

    def _create_grid_distribution(
        self, amount: int, distribution: Dict
    ) -> List[Resource]:
        """Create resources in a grid pattern."""
        resources = []

        # Calculate grid dimensions
        grid_size = int(np.sqrt(amount))
        actual_amount = grid_size * grid_size

        # Calculate spacing
        x_spacing = self.width / (grid_size + 1)
        y_spacing = self.height / (grid_size + 1)

        for i in range(grid_size):
            for j in range(grid_size):
                x = x_spacing * (i + 1)
                y = y_spacing * (j + 1)
                position = (x, y)
                resource = self._create_resource(position, distribution)
                resources.append(resource)

        return resources

    def _create_clustered_distribution(
        self, amount: int, distribution: Dict
    ) -> List[Resource]:
        """Create resources in clustered patterns."""
        resources = []

        # Use seeded random generator
        if self.seed_value is not None:
            rng = random.Random(self.seed_value)
        else:
            rng = random

        # Create cluster centers
        num_clusters = max(1, amount // 10)  # One cluster per 10 resources
        cluster_centers = [
            (rng.uniform(0, self.width), rng.uniform(0, self.height))
            for _ in range(num_clusters)
        ]

        # Distribute resources around cluster centers
        resources_per_cluster = amount // num_clusters
        remaining_resources = amount % num_clusters

        for i, center in enumerate(cluster_centers):
            cluster_amount = resources_per_cluster + (
                1 if i < remaining_resources else 0
            )

            for _ in range(cluster_amount):
                # Add some randomness around cluster center
                offset_x = rng.gauss(0, self.width * 0.1)
                offset_y = rng.gauss(0, self.height * 0.1)

                x = max(0, min(self.width, center[0] + offset_x))
                y = max(0, min(self.height, center[1] + offset_y))

                position = (x, y)
                resource = self._create_resource(position, distribution)
                resources.append(resource)

        return resources

    def _create_resource(
        self, position: Tuple[float, float], distribution: Dict
    ) -> Resource:
        """Create a single resource with the given parameters."""
        # Determine initial amount
        if self.seed_value is not None:
            # Deterministic amount based on position
            pos_sum = position[0] + position[1]
            amount = 3 + int(pos_sum % 6)  # Range from 3 to 8
        else:
            # Random amount
            min_amount = distribution.get("min_amount", 3)
            max_amount = distribution.get("max_amount", 8)
            amount = random.randint(min_amount, max_amount)

        # Get configuration parameters
        max_amount = self.config.resources.max_resource_amount if self.config else 10
        regeneration_rate = (
            getattr(self.config, "resource_regen_rate", 0.1) if self.config else 0.1
        )

        resource = Resource(
            resource_id=self.next_resource_id,
            position=position,
            amount=amount,
            max_amount=max_amount,
            regeneration_rate=regeneration_rate,
        )

        self.next_resource_id += 1
        return resource

    def update_resources(self, time_step: int) -> Dict:
        """
        Update all resources for the current time step using the same logic as the original Environment.

        This matches the original Environment.update() resource regeneration logic exactly.

        Parameters
        ----------
        time_step : int
            Current simulation time step

        Returns
        -------
        Dict
            Update statistics including regeneration events, consumption, etc.
        """
        stats: Dict[str, float] = {
            "regeneration_events": 0,
            "resources_regenerated": 0.0,
            "depletion_events": 0,
            "total_resources": len(self.resources),
        }

        if self.seed_value is not None:
            # Create deterministic RNG based on seed and current time (same as original)
            rng = random.Random(self.seed_value + time_step)

            # Deterministically decide which resources regenerate (same as original)
            for resource in self.resources:
                # Use resource ID and position as additional entropy sources (same as original)
                decision_seed = hash(
                    (
                        resource.resource_id,
                        resource.position[0],
                        resource.position[1],
                        time_step,
                    )
                )
                # Mix with simulation seed (same as original)
                combined_seed = (self.seed_value * 100000) + decision_seed
                # Create a deterministic random generator for this resource (same as original)
                resource_rng = random.Random(combined_seed)

                # Check if this resource should regenerate (same as original)
                regen_rate = (
                    getattr(self.config, "resource_regen_rate", 0.1)
                    if self.config
                    else 0.1
                )
                max_resource = (
                    self.config.resources.max_resource_amount if self.config else None
                )

                if resource_rng.random() < regen_rate and (
                    max_resource is None or resource.amount < max_resource
                ):
                    regen_amount = (
                        getattr(self.config, "resource_regen_amount", 2)
                        if self.config
                        else 2
                    )
                    old_amount = resource.amount
                    resource.amount = min(
                        resource.amount + regen_amount,
                        max_resource or float("inf"),
                    )
                    regenerated = resource.amount - old_amount

                    stats["regeneration_events"] += 1
                    stats["resources_regenerated"] += regenerated
        else:
            # Use standard random method if no seed is set (same as original)
            regen_rate = (
                getattr(self.config, "resource_regen_rate", 0.1) if self.config else 0.1
            )
            max_resource = (
                self.config.resources.max_resource_amount if self.config else None
            )

            regen_mask = np.random.random(len(self.resources)) < regen_rate
            for resource, should_regen in zip(self.resources, regen_mask):
                if should_regen and (
                    max_resource is None or resource.amount < max_resource
                ):
                    regen_amount = (
                        getattr(self.config, "resource_regen_amount", 2)
                        if self.config
                        else 2
                    )
                    old_amount = resource.amount
                    resource.amount = min(
                        resource.amount + regen_amount,
                        max_resource or float("inf"),
                    )
                    regenerated = resource.amount - old_amount

                    stats["regeneration_events"] += 1
                    stats["resources_regenerated"] += regenerated

        # Update tracking
        self.regeneration_step = time_step
        self.regeneration_events += stats["regeneration_events"]
        self.total_resources_regenerated += stats["resources_regenerated"]

        return stats

    def _update_resources_deterministic(self, time_step: int, stats: Dict):
        """Update resources using deterministic regeneration logic."""
        for resource in self.resources:
            # Create deterministic decision seed
            decision_seed = hash(
                (
                    resource.resource_id,
                    resource.position[0],
                    resource.position[1],
                    time_step,
                )
            )

            # Mix with simulation seed
            combined_seed = ((self.seed_value or 0) * 100000) + decision_seed
            resource_rng = random.Random(combined_seed)

            # Check if resource should regenerate
            regen_rate = self.config.resource_regen_rate if self.config else 0.1

            if (
                resource_rng.random() < regen_rate
                and resource.amount < resource.max_amount
            ):

                regen_amount = self.config.resource_regen_amount if self.config else 2

                old_amount = resource.amount
                resource.regenerate(regen_amount)
                regenerated = resource.amount - old_amount

                stats["regeneration_events"] += 1
                stats["resources_regenerated"] += regenerated

    def _update_resources_random(self, stats: Dict):
        """Update resources using random regeneration logic."""
        regen_rate = self.config.resource_regen_rate if self.config else 0.1

        # Create regeneration mask
        regen_mask = np.random.random(len(self.resources)) < regen_rate

        for resource, should_regen in zip(self.resources, regen_mask):
            if should_regen and resource.amount < resource.max_amount:
                regen_amount = self.config.resource_regen_amount if self.config else 2

                old_amount = resource.amount
                resource.regenerate(regen_amount)
                regenerated = resource.amount - old_amount

                stats["regeneration_events"] += 1
                stats["resources_regenerated"] += regenerated

    def consume_resource(self, resource: Resource, amount: float) -> float:
        """
        Consume resources from a specific resource node.

        Parameters
        ----------
        resource : Resource
            Resource to consume from
        amount : float
            Amount to consume

        Returns
        -------
        float
            Actual amount consumed
        """
        if resource.is_depleted():
            return 0.0

        actual_consumption = min(amount, resource.amount)
        resource.consume(actual_consumption)

        self.total_resources_consumed += actual_consumption

        if resource.is_depleted():
            self.depletion_events += 1

        return actual_consumption

    def get_nearby_resources(
        self, position: Tuple[float, float], radius: float
    ) -> List[Resource]:
        """
        Get all resources within a specified radius of a position.

        Parameters
        ----------
        position : Tuple[float, float]
            Center position (x, y)
        radius : float
            Search radius

        Returns
        -------
        List[Resource]
            List of resources within radius
        """
        # Use spatial index if available for O(log n) performance
        if self.spatial_index is not None:
            return self.spatial_index.get_nearby_resources(position, radius)

        # Fallback to linear search if no spatial index
        nearby = []
        for resource in self.resources:
            distance = np.sqrt(
                (resource.position[0] - position[0]) ** 2
                + (resource.position[1] - position[1]) ** 2
            )
            if distance <= radius:
                nearby.append(resource)
        return nearby

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Resource]:
        """
        Find the nearest resource to a given position.

        Parameters
        ----------
        position : Tuple[float, float]
            Position to search from

        Returns
        -------
        Resource or None
            Nearest resource if any exist
        """
        # Use spatial index if available for O(log n) performance
        if self.spatial_index is not None:
            return self.spatial_index.get_nearest_resource(position)

        # Fallback to linear search if no spatial index
        if not self.resources:
            return None

        nearest = None
        min_distance = float("inf")

        for resource in self.resources:
            distance = np.sqrt(
                (resource.position[0] - position[0]) ** 2
                + (resource.position[1] - position[1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                nearest = resource

        return nearest

    def add_resource(
        self, position: Tuple[float, float], amount: Optional[float] = None
    ) -> Resource:
        """
        Add a new resource to the environment.

        Parameters
        ----------
        position : Tuple[float, float]
            Position for the new resource
        amount : float, optional
            Initial amount (uses default if not specified)

        Returns
        -------
        Resource
            The newly created resource
        """
        if amount is None:
            amount = 5  # Default amount

        resource = Resource(
            resource_id=self.next_resource_id,
            position=position,
            amount=amount,
            max_amount=self.config.resources.max_resource_amount if self.config else 10,
            regeneration_rate=self.config.resources.resource_regen_rate if self.config else 0.1,
        )

        self.next_resource_id += 1
        self.resources.append(resource)

        # Mark spatial index as dirty when resources change
        if self.spatial_index is not None:
            self.spatial_index.mark_positions_dirty()

        # Log to database if available
        if self.database_logger:
            self.database_logger.log_resource(
                resource_id=resource.resource_id,
                initial_amount=resource.amount,
                position=resource.position,
            )

        return resource

    def remove_resource(self, resource: Resource) -> bool:
        """
        Remove a resource from the environment.

        Parameters
        ----------
        resource : Resource
            Resource to remove

        Returns
        -------
        bool
            True if resource was removed, False if not found
        """
        if resource in self.resources:
            self.resources.remove(resource)

            # Mark spatial index as dirty when resources change
            if self.spatial_index is not None:
                self.spatial_index.mark_positions_dirty()

            return True
        return False

    def get_resource_statistics(self) -> Dict:
        """
        Get comprehensive statistics about current resources.

        Returns
        -------
        Dict
            Resource statistics
        """
        if not self.resources:
            return {
                "total_resources": 0,
                "average_amount": 0,
                "depleted_resources": 0,
                "full_resources": 0,
                "total_capacity": 0,
                "utilization_rate": 0,
            }

        amounts = [r.amount for r in self.resources]
        max_amounts = [r.max_amount for r in self.resources]

        return {
            "total_resources": len(self.resources),
            "average_amount": np.mean(amounts),
            "depleted_resources": sum(1 for r in self.resources if r.is_depleted()),
            "full_resources": sum(
                1 for r in self.resources if r.amount >= r.max_amount
            ),
            "total_capacity": sum(max_amounts),
            "utilization_rate": (
                sum(amounts) / sum(max_amounts) if sum(max_amounts) > 0 else 0
            ),
            "min_amount": min(amounts),
            "max_amount": max(amounts),
            "std_amount": np.std(amounts),
        }

    def reset(self):
        """Reset the resource manager state."""
        self.resources.clear()
        self.next_resource_id = 0
        self.regeneration_step = 0
        self.total_resources_consumed = 0
        self.total_resources_regenerated = 0
        self.regeneration_events = 0
        self.depletion_events = 0
