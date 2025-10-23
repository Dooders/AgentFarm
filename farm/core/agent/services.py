"""
Services container for agent components.

Bundles all required services (spatial queries, time, metrics, logging, etc.)
into a single container that can be injected into components and the agent core.
This follows the Dependency Inversion Principle by decoupling components from
concrete service implementations.
"""

from dataclasses import dataclass
from typing import Optional

from farm.core.services.interfaces import (
    IAgentLifecycleService,
    ILoggingService,
    IMetricsService,
    ISpatialQueryService,
    ITimeService,
    IValidationService,
)


@dataclass
class AgentServices:
    """
    Container for all services required by agents and their components.
    
    This design follows the Service Locator pattern combined with Dependency Injection.
    Components receive an AgentServices instance and can access the services they need.
    All services are optional (can be None) to support different runtime configurations.
    
    Attributes:
        spatial_service: Service for spatial queries (nearby entities, range queries)
        time_service: Service for accessing current simulation time
        metrics_service: Service for recording metrics and events
        logging_service: Service for logging events to database
        validation_service: Service for validating positions and actions
        lifecycle_service: Service for agent creation/removal lifecycle management
    """
    
    spatial_service: ISpatialQueryService
    time_service: Optional[ITimeService] = None
    metrics_service: Optional[IMetricsService] = None
    logging_service: Optional[ILoggingService] = None
    validation_service: Optional[IValidationService] = None
    lifecycle_service: Optional[IAgentLifecycleService] = None
    
    def get_current_time(self) -> int:
        """Get current simulation time, with fallback to 0 if no time service."""
        if self.time_service:
            return self.time_service.current_time()
        return 0
