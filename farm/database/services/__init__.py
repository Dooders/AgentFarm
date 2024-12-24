"""
Services Module

This module contains high-level service classes that orchestrate complex operations by coordinating
between multiple components of the system. Services act as an abstraction layer between the
application logic and the underlying implementation details.

Key Services:
-------------
ActionsService
    Orchestrates analysis of agent actions using various analyzers. Provides a unified interface
    for analyzing action patterns, behaviors, resource impacts, and other metrics.

Service Design Principles:
-------------------------
1. Separation of Concerns
    - Services coordinate between different components but don't implement core logic
    - Each service focuses on a specific domain area (e.g., actions, resources)

2. Dependency Injection
    - Services receive their dependencies through constructor injection
    - Makes services more testable and loosely coupled

3. High-Level Interface
    - Services provide simple, intuitive interfaces for complex operations
    - Hide implementation details and coordinate between multiple components

4. Stateless Operation
    - Services generally don't maintain state between operations
    - Each method call is independent and self-contained

Usage Example:
-------------
    # Initialize repository and service
    action_repo = AgentActionRepository(session_manager)
    actions_service = ActionsService(action_repo)

    # Perform comprehensive analysis
    results = actions_service.analyze_actions(
        scope="EPISODE",
        agent_id=123,
        analysis_types=['stats', 'behavior']
    )

    # Get high-level summary
    summary = actions_service.get_action_summary(
        scope="SIMULATION",
        agent_id=123
    )
"""

from farm.database.services.actions_service import ActionsService

__all__ = ['ActionsService']
