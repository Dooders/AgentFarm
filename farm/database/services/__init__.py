"""
Services Module

This module contains high-level service classes that orchestrate complex operations by coordinating
between multiple components of the system. Services act as an abstraction layer between the
application logic and the underlying implementation details.

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

Note: Legacy services have been removed. Use the farm.analysis module for analysis operations.
"""

__all__ = []
