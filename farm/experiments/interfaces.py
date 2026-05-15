"""Common interface for dashboard-backed experiment adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from farm.experiments.manifest import ExperimentManifest, ManifestValidationResult


@dataclass(frozen=True)
class ViewDescriptor:
    """Metadata describing a dashboard view exposed by an experiment adapter."""

    view_id: str
    view_type: str
    title: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "view_id": self.view_id,
            "view_type": self.view_type,
            "title": self.title,
            "description": self.description,
        }


class ExperimentAdapterProtocol(Protocol):
    """Contract implemented by each experiment type adapter."""

    experiment_type: str

    def validate_manifest(self, manifest: ExperimentManifest) -> ManifestValidationResult:
        """Validate adapter-specific sections of a normalized manifest."""

    def run_experiment(
        self,
        run_id: str,
        manifest: ExperimentManifest,
        runtime_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the experiment and return a run summary."""

    def list_views(self, run_context: Dict[str, Any]) -> List[ViewDescriptor]:
        """List available dashboard views for a completed/in-progress run."""

    def get_view_data(
        self,
        run_context: Dict[str, Any],
        view_id: str,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return normalized payload for a single dashboard view."""


class ExperimentRegistry:
    """In-memory experiment type -> adapter registry."""

    def __init__(self) -> None:
        self._adapters: Dict[str, ExperimentAdapterProtocol] = {}

    def register(self, adapter: ExperimentAdapterProtocol) -> None:
        self._adapters[adapter.experiment_type] = adapter

    def get(self, experiment_type: str) -> ExperimentAdapterProtocol:
        if experiment_type not in self._adapters:
            supported = ", ".join(sorted(self._adapters.keys()))
            raise KeyError(
                f"Unsupported experiment_type={experiment_type!r}. Supported: {supported}"
            )
        return self._adapters[experiment_type]

    def list_experiment_types(self) -> List[str]:
        return sorted(self._adapters.keys())
