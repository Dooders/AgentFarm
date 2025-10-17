"""Unified AgentFarm API for agentic control.

This package provides a unified API that allows agentsic systems to easily control
simulations and experiments through a clean, intuitive interface.

Main Components:
    - AgentFarmController: Main unified controller class
    - SessionManager: Manages research sessions
    - UnifiedAdapter: Adapts existing controllers to the new API
    - ConfigTemplateManager: Manages configuration templates
    - Data models: Structured data classes for API responses

Usage:
    ```python
    from farm.api import AgentFarmController

    # Initialize controller
    controller = AgentFarmController()

    # Create a session
    session_id = controller.create_session("My Research", "Testing agent behaviors")

    # Create and run a simulation
    simulation_id = controller.create_simulation(session_id, {
        "name": "Basic Test",
        "steps": 1000,
        "agents": {"system_agents": 10, "independent_agents": 10}
    })

    # Start simulation
    controller.start_simulation(session_id, simulation_id)

    # Monitor progress
    status = controller.get_simulation_status(session_id, simulation_id)
    print(f"Progress: {status.progress_percentage:.1f}%")

    # Get results
    results = controller.get_simulation_results(session_id, simulation_id)
    print(f"Final agents: {results.final_agent_count}")
    ```
"""

from .unified_controller import AgentFarmController
from .models import (
    SessionInfo, SimulationStatus, SimulationResults, ExperimentStatus,
    ExperimentResults, ConfigTemplate, ValidationResult, AnalysisResults,
    ComparisonResults, Event, EventSubscription
)
from .session_manager import SessionManager
from .unified_adapter import UnifiedAdapter
from .config_templates import ConfigTemplateManager

__version__ = "1.0.0"
__author__ = "AgentFarm Team"

__all__ = [
    # Main controller
    "AgentFarmController",

    # Data models
    "SessionInfo",
    "SimulationStatus",
    "SimulationResults",
    "ExperimentStatus",
    "ExperimentResults",
    "ConfigTemplate",
    "ValidationResult",
    "AnalysisResults",
    "ComparisonResults",
    "Event",
    "EventSubscription",

    # Supporting classes
    "SessionManager",
    "UnifiedAdapter",
    "ConfigTemplateManager",
]
