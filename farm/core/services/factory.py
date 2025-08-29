from typing import TYPE_CHECKING, Optional

from farm.core.services.implementations import (
    EnvironmentAgentLifecycleService,
    EnvironmentLoggingService,
    EnvironmentMetricsService,
    EnvironmentTimeService,
    EnvironmentValidationService,
)
from farm.core.services.interfaces import (
    IAgentLifecycleService,
    ILoggingService,
    IMetricsService,
    ITimeService,
    IValidationService,
)

if TYPE_CHECKING:
    from farm.core.environment import Environment


class AgentServiceFactory:
    """Factory for creating and configuring agent services.

    This factory encapsulates the logic for deriving services from an environment
    when they are not explicitly provided, separating this concern from the agent
    initialization logic.
    """

    @staticmethod
    def create_services(
        environment: Optional["Environment"] = None,
        *,
        metrics_service: Optional[IMetricsService] = None,
        logging_service: Optional[ILoggingService] = None,
        validation_service: Optional[IValidationService] = None,
        time_service: Optional[ITimeService] = None,
        lifecycle_service: Optional[IAgentLifecycleService] = None,
        config: Optional[object] = None,
    ) -> tuple[
        Optional[IMetricsService],
        Optional[ILoggingService],
        Optional[IValidationService],
        Optional[ITimeService],
        Optional[IAgentLifecycleService],
        Optional[object],
    ]:
        """Create and configure agent services.

        Args:
            environment: The environment to derive services from if not explicitly provided
            metrics_service: Explicitly provided metrics service
            logging_service: Explicitly provided logging service
            validation_service: Explicitly provided validation service
            time_service: Explicitly provided time service
            lifecycle_service: Explicitly provided lifecycle service
            config: Explicitly provided config

        Returns:
            Tuple of (metrics_service, logging_service, validation_service,
                     time_service, lifecycle_service, config)
        """
        # Derive services from environment if provided and not explicitly passed
        if environment is not None:
            metrics_service = metrics_service or EnvironmentMetricsService(environment)
            logging_service = logging_service or EnvironmentLoggingService(environment)
            validation_service = validation_service or EnvironmentValidationService(
                environment
            )
            time_service = time_service or EnvironmentTimeService(environment)
            lifecycle_service = lifecycle_service or EnvironmentAgentLifecycleService(
                environment
            )
            config = config or getattr(environment, "config", None)

        return (
            metrics_service,
            logging_service,
            validation_service,
            time_service,
            lifecycle_service,
            config,
        )
