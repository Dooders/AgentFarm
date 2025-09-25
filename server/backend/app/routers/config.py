from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, Optional, List

from farm.core.config_schema import generate_combined_config_schema
from farm.core.config import SimulationConfig


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/schema")
def get_config_schema():
    return generate_combined_config_schema()


class ConfigLoadRequest(BaseModel):
    path: Optional[str] = Field(default=None, description="Absolute or relative path to a YAML config file")


class ConfigSaveRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration object to save")
    path: Optional[str] = Field(default=None, description="Absolute or relative path to write the YAML file")


class ConfigValidateRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration object to validate")


class ConfigResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    path: Optional[str] = None


def _get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def _resolve_default_config_path() -> Path:
    root = _get_repo_root()
    config_path = root / "config.yaml"
    return config_path.resolve()


def _resolve_safe_path(user_path: Optional[str]) -> Path:
    root = _get_repo_root()
    if user_path is None or user_path.strip() == "":
        return _resolve_default_config_path()

    candidate = Path(user_path)
    # If relative, treat as relative to repo root
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        # Prefer pathlib's relative_to when available (Python 3.9+)
        candidate.relative_to(root)  # type: ignore[attr-defined]
    except Exception:
        # Robust fallback using commonpath
        import os
        root_str = str(root)
        cand_str = str(candidate)
        common = os.path.commonpath([root_str, cand_str])
        if common != root_str:
            raise ValueError("Path outside allowed base directory")

    return candidate


@router.post("/load", response_model=ConfigResponse)
def load_config(req: ConfigLoadRequest) -> ConfigResponse:
    try:
        path = _resolve_safe_path(req.path)
        if not path.exists():
            return ConfigResponse(success=False, message="Config file not found", errors=[f"Missing: {str(path)}"], path=str(path))
        if path.suffix.lower() not in (".yml", ".yaml"):
            return ConfigResponse(success=False, message="Unsupported file extension", errors=["Use .yml or .yaml"], path=str(path))

        cfg = SimulationConfig.from_yaml(str(path))
        return ConfigResponse(success=True, config=cfg.to_dict(), path=str(path))
    except Exception:
        # Do not leak internal error details
        return ConfigResponse(success=False, message="Failed to load configuration", errors=["An error occurred while loading the configuration."])


@router.post("/save", response_model=ConfigResponse)
def save_config(req: ConfigSaveRequest) -> ConfigResponse:
    try:
        # Validate and normalize first
        cfg = SimulationConfig.from_dict(dict(req.config))

        # Resolve path
        path = _resolve_safe_path(req.path)
        if path.suffix.lower() not in (".yml", ".yaml"):
            # If user gave a path without extension, add .yaml
            if path.suffix == "":
                path = path.with_suffix(".yaml")
            else:
                return ConfigResponse(success=False, message="Unsupported file extension", errors=["Use .yml or .yaml"], path=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(str(path))
        return ConfigResponse(success=True, message="Configuration saved", config=cfg.to_dict(), path=str(path))
    except Exception:
        return ConfigResponse(success=False, message="Failed to save configuration", errors=["An error occurred while saving the configuration."])


@router.post("/validate", response_model=ConfigResponse)
def validate_config(req: ConfigValidateRequest) -> ConfigResponse:
    try:
        cfg = SimulationConfig.from_dict(dict(req.config))
        # If construction succeeds, consider it valid; return normalized config
        return ConfigResponse(success=True, message="Configuration is valid", config=cfg.to_dict())
    except Exception:
        return ConfigResponse(success=False, message="Invalid configuration", errors=["Configuration failed validation."])

