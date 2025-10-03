from __future__ import annotations

"""
Core Experiment abstractions for the benchmarking framework.

This module defines:
- ExperimentContext: shared run-time context passed to experiments
- Experiment: abstract base class with a clean lifecycle

Notes:
- Experiments should be small, composable units that execute a single measured run.
- Iteration orchestration is handled by the Runner.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentContext:
    """Context object provided to experiments during execution.

    Attributes
    ----------
    run_id : str
        Unique identifier for the current run
    output_dir : str
        Base directory where run artifacts should be written
    run_dir : str
        Directory specific to this run (inside output_dir)
    iteration_index : Optional[int]
        Current measured iteration index (0-based). None during setup/teardown.
    seed : Optional[int]
        Seed used for reproducibility, if provided
    instruments : List[Any]
        Active instrumentation instances configured by the Runner
    extras : Dict[str, Any]
        Arbitrary key-value store for cross-component data
    """

    run_id: str
    output_dir: str
    run_dir: str
    iteration_index: Optional[int] = None
    seed: Optional[int] = None
    instruments: List[Any] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


class Experiment(abc.ABC):
    """Abstract base class for all experiments.

    Concrete experiments must implement the lifecycle methods below.
    Experiments should avoid performing iteration loops; the Runner will
    orchestrate warmups and measured iterations, calling `execute_once` per
    measured iteration.
    """

    # Optional: subclasses may override to provide a JSON-schema-like dict
    # for parameters validation and defaulting.
    param_schema: Dict[str, Any] = {}

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = params or {}

    def setup(self, context: ExperimentContext) -> None:  # pragma: no cover - optional override
        """Allocate resources needed for the experiment.

        Called once before warmups and measured iterations.
        """
        return None

    @abc.abstractmethod
    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        """Execute the workload exactly once and return metrics/artifacts.

        Returns
        -------
        Dict[str, Any]
            A dictionary with scalar metrics and optional nested structures.
            Keys should be JSON-serializable. Artifacts should be recorded by
            the Runner using the results API rather than written here, unless
            the file emission is intrinsic to the workload.
        """
        raise NotImplementedError

    def teardown(self, context: ExperimentContext) -> None:  # pragma: no cover - optional override
        """Release resources after all iterations complete."""
        return None

