from fastapi import APIRouter

from farm.core.config_schema import generate_combined_config_schema


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/schema")
def get_config_schema():
    return generate_combined_config_schema()

