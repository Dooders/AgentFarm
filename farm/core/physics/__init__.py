"""Physics engines for AgentFarm environments.

This module provides physics engine implementations that define how agents
interact with space, compute distances, and represent state in different
environment types.

Available Physics Engines:
    - Grid2DPhysics: 2D grid-based physics (default, backward compatible)
    - StaticPhysics: Fixed position physics for static environments
    - ContinuousPhysics: Continuous space physics for robotics/navigation
    - CustomPhysics: User-defined physics via protocol implementation

Quick Start:
    >>> from farm.core.physics import Grid2DPhysics
    >>> from farm.core.environment import Environment
    >>> 
    >>> physics = Grid2DPhysics(width=100, height=100)
    >>> env = Environment(physics_engine=physics, config=config)

Custom Physics:
    >>> class MyPhysics:
    ...     def validate_position(self, position): ...
    ...     def get_nearby_entities(self, position, radius, entity_type): ...
    ...     # ... implement IPhysicsEngine protocol
    >>> 
    >>> physics = MyPhysics()
    >>> env = Environment(physics_engine=physics, config=config)

See Also:
    - farm.core.physics.interface: IPhysicsEngine protocol definition
    - docs/design/environment_module_design_report.md: Full design documentation
"""

from farm.core.physics.interface import (
    IObservationBuilder,
    IPhysicsEngine,
    ObservationBuilder,
    PhysicsEngine,
)

# Note: Actual implementations (Grid2DPhysics, StaticPhysics, etc.) will be
# added in future commits as part of the refactoring process

__all__ = [
    "IPhysicsEngine",
    "IObservationBuilder",
    "PhysicsEngine",
    "ObservationBuilder",
    # Implementations will be added here:
    # "Grid2DPhysics",
    # "StaticPhysics", 
    # "ContinuousPhysics",
]
