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


def _resolve_default_config_path() -> Path:
    # Prefer repo root config.yaml if present
    candidates = [
        Path("config.yaml"),
        Path("./config.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    # Fallback to repo root location even if missing
    return candidates[0].resolve()


@router.post("/load", response_model=ConfigResponse)
def load_config(req: ConfigLoadRequest) -> ConfigResponse:
    try:
        path = Path(req.path) if req.path else _resolve_default_config_path()
        if not path.exists():
            return ConfigResponse(success=False, message="Config file not found", errors=[f"Missing: {str(path)}"], path=str(path))
        if path.suffix.lower() not in (".yml", ".yaml"):
            return ConfigResponse(success=False, message="Unsupported file extension", errors=["Use .yml or .yaml"], path=str(path))

        cfg = SimulationConfig.from_yaml(str(path))
        return ConfigResponse(success=True, config=cfg.to_dict(), path=str(path))
    except Exception as e:
        # Do not leak stack traces, return unified error payload
        return ConfigResponse(success=False, message="Failed to load configuration", errors=[str(e)])


@router.post("/save", response_model=ConfigResponse)
def save_config(req: ConfigSaveRequest) -> ConfigResponse:
    try:
        # Validate and normalize first
        cfg = SimulationConfig.from_dict(dict(req.config))

        # Resolve path
        path = Path(req.path) if req.path else _resolve_default_config_path()
        if path.suffix.lower() not in (".yml", ".yaml"):
            # If user gave a path without extension, add .yaml
            if path.suffix == "":
                path = path.with_suffix(".yaml")
            else:
                return ConfigResponse(success=False, message="Unsupported file extension", errors=["Use .yml or .yaml"], path=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(str(path))
        return ConfigResponse(success=True, message="Configuration saved", config=cfg.to_dict(), path=str(path))
    except Exception as e:
        return ConfigResponse(success=False, message="Failed to save configuration", errors=[str(e)])


@router.post("/validate", response_model=ConfigResponse)
def validate_config(req: ConfigValidateRequest) -> ConfigResponse:
    try:
        cfg = SimulationConfig.from_dict(dict(req.config))
        # If construction succeeds, consider it valid; return normalized config
        return ConfigResponse(success=True, message="Configuration is valid", config=cfg.to_dict())
    except Exception as e:
        return ConfigResponse(success=False, message="Invalid configuration", errors=[str(e)])

