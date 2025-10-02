"""Data models and schemas for the unified AgentFarm API.

This module defines the core data structures used by the unified API,
providing clear interfaces for agentic systems to interact with simulations
and experiments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(str, Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SimulationStatus(str, Enum):
    """Simulation status enumeration."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class ConfigCategory(str, Enum):
    """Configuration category enumeration."""

    SIMULATION = "simulation"
    EXPERIMENT = "experiment"
    RESEARCH = "research"


@dataclass
class SessionInfo:
    """Information about a research session."""

    session_id: str
    name: str
    description: str
    created_at: datetime
    status: SessionStatus
    simulations: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "simulations": self.simulations,
            "experiments": self.experiments,
            "metadata": self.metadata,
        }


@dataclass
class SimulationStatusInfo:
    """Status information for a simulation."""

    simulation_id: str
    status: SimulationStatus
    current_step: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "simulation_id": self.simulation_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percentage": self.progress_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class SimulationResults:
    """Results from a completed simulation."""

    simulation_id: str
    status: SimulationStatus
    total_steps: int
    final_agent_count: int
    final_resource_count: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    data_files: List[str] = field(default_factory=list)
    analysis_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "simulation_id": self.simulation_id,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "final_agent_count": self.final_agent_count,
            "final_resource_count": self.final_resource_count,
            "metrics": self.metrics,
            "data_files": self.data_files,
            "analysis_available": self.analysis_available,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentStatusInfo:
    """Status information for an experiment."""

    experiment_id: str
    status: ExperimentStatus
    current_iteration: int = 0
    total_iterations: int = 0
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "progress_percentage": self.progress_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentResults:
    """Results from a completed experiment."""

    experiment_id: str
    status: ExperimentStatus
    total_iterations: int
    completed_iterations: int
    results_summary: Dict[str, Any] = field(default_factory=dict)
    data_files: List[str] = field(default_factory=list)
    analysis_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "total_iterations": self.total_iterations,
            "completed_iterations": self.completed_iterations,
            "results_summary": self.results_summary,
            "data_files": self.data_files,
            "analysis_available": self.analysis_available,
            "metadata": self.metadata,
        }


@dataclass
class ConfigTemplate:
    """Configuration template for simulations and experiments."""

    name: str
    description: str
    category: ConfigCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "examples": self.examples,
        }


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validated_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "validated_config": self.validated_config,
        }


@dataclass
class AnalysisResults:
    """Results from simulation or experiment analysis."""

    analysis_id: str
    analysis_type: str
    summary: Dict[str, Any] = field(default_factory=dict)
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    charts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "summary": self.summary,
            "detailed_results": self.detailed_results,
            "output_files": self.output_files,
            "charts": self.charts,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResults:
    """Results from comparing multiple simulations."""

    comparison_id: str
    simulation_ids: List[str]
    comparison_summary: Dict[str, Any] = field(default_factory=dict)
    detailed_comparison: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    charts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "comparison_id": self.comparison_id,
            "simulation_ids": self.simulation_ids,
            "comparison_summary": self.comparison_summary,
            "detailed_comparison": self.detailed_comparison,
            "output_files": self.output_files,
            "charts": self.charts,
            "metadata": self.metadata,
        }


@dataclass
class Event:
    """Event in the simulation or experiment lifecycle."""

    event_id: str
    event_type: str
    timestamp: datetime
    session_id: str
    simulation_id: Optional[str] = None
    experiment_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "simulation_id": self.simulation_id,
            "experiment_id": self.experiment_id,
            "data": self.data,
            "message": self.message,
        }


@dataclass
class EventSubscription:
    """Subscription to simulation or experiment events."""

    subscription_id: str
    session_id: str
    event_types: List[str]
    simulation_id: Optional[str] = None
    experiment_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subscription_id": self.subscription_id,
            "session_id": self.session_id,
            "event_types": self.event_types,
            "simulation_id": self.simulation_id,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "active": self.active,
        }
