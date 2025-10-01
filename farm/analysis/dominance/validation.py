from farm.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from farm.analysis.dominance.models import DominanceDataModel
except Exception as exc:
    DominanceDataModel = None
    logger.debug(f"DominanceDataModel unavailable: {exc}")


def validate_sim_data(sim_data: dict):
    if DominanceDataModel is None:
        return sim_data
    try:
        return DominanceDataModel(**sim_data).dict()
    except Exception as exc:
        logger.warning(f"Data validation failed: {exc}")
        return sim_data

