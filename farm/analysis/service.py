from dataclasses import dataclass
from typing import Any, Dict, Optional

from farm.analysis.registry import get_module, register_modules
from farm.core.services import IConfigService


@dataclass
class AnalysisRequest:
    module_name: str
    experiment_path: str
    output_path: str
    group: str = "all"
    processor_kwargs: Optional[Dict[str, Any]] = None
    analysis_kwargs: Optional[Dict[str, Dict[str, Any]]] = None


class AnalysisService:
    def __init__(self, config_service: IConfigService):
        register_modules(config_service=config_service)

    def run(self, request: AnalysisRequest):
        module = get_module(request.module_name)
        if module is None:
            raise ValueError(f"Unknown analysis module: {request.module_name}")
        return module.run_analysis(
            experiment_path=request.experiment_path,
            output_path=request.output_path,
            group=request.group,
            processor_kwargs=request.processor_kwargs,
            analysis_kwargs=request.analysis_kwargs,
        )