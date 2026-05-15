"""Versioned experiment manifest schema used by the dashboard APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


SUPPORTED_SCHEMA_VERSION = 1
INTRINSIC_EVOLUTION_EXPERIMENT_TYPE = "intrinsic_evolution"


class DashboardPreset(BaseModel):
    """View-level defaults selected by the no-code editor."""

    default_views: List[str] = Field(default_factory=list)
    default_gene_ids: List[str] = Field(default_factory=list)
    step_window: Optional[Dict[str, int]] = None


class ExperimentManifest(BaseModel):
    """Top-level, versioned experiment descriptor."""

    schema_version: int = Field(default=SUPPORTED_SCHEMA_VERSION)
    experiment_type: str
    experiment_name: str = Field(min_length=1)
    base_simulation_config: Dict[str, Any] = Field(default_factory=dict)
    experiment_config: Dict[str, Any] = Field(default_factory=dict)
    dashboard_preset: DashboardPreset = Field(default_factory=DashboardPreset)

    @model_validator(mode="after")
    def _validate_supported_values(self) -> "ExperimentManifest":
        schema_version = self.schema_version
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version}. "
                f"Expected {SUPPORTED_SCHEMA_VERSION}."
            )

        experiment_type = self.experiment_type
        if experiment_type != INTRINSIC_EVOLUTION_EXPERIMENT_TYPE:
            raise ValueError(
                f"Unsupported experiment_type={experiment_type!r}. "
                f"Supported values: [{INTRINSIC_EVOLUTION_EXPERIMENT_TYPE!r}]"
            )
        return self


class ManifestValidationResult(BaseModel):
    """Validation response shape returned by backend endpoints."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    normalized_manifest: Optional[Dict[str, Any]] = None


def validate_experiment_manifest(payload: Dict[str, Any]) -> ManifestValidationResult:
    """Validate and normalize a manifest payload."""
    try:
        manifest = ExperimentManifest.model_validate(payload)
    except ValidationError as exc:
        errors: List[str] = []
        for issue in exc.errors():
            location = ".".join(str(part) for part in issue.get("loc", []))
            errors.append(f"{location}: {issue.get('msg')}")
        return ManifestValidationResult(is_valid=False, errors=errors)
    except ValueError as exc:
        return ManifestValidationResult(is_valid=False, errors=[str(exc)])

    return ManifestValidationResult(
        is_valid=True,
        normalized_manifest=manifest.model_dump(),
    )
