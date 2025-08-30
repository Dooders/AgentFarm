import logging

try:
    from farm.analysis.dominance.models import DominanceDataModel
except Exception as exc:
    DominanceDataModel = None
    logging.debug(f"DominanceDataModel unavailable: {exc}")


def validate_sim_data(sim_data: dict):
    if DominanceDataModel is None:
        return sim_data
    try:
        return DominanceDataModel(**sim_data).dict()
    except Exception as exc:
        logging.warning(f"Data validation failed: {exc}")
        return sim_data

