from typing import Callable, Optional

from farm.analysis.base_module import AnalysisModule


class NullModule(AnalysisModule):
    def __init__(self, name: str = "null", description: str = "No-op analysis module"):
        super().__init__(name=name, description=description)

    def register_analysis(self) -> None:
        self._analysis_functions = {}
        self._analysis_groups = {"all": []}

    def get_data_processor(self) -> Callable:
        def _noop(*args, **kwargs):
            return None

        return _noop

    def get_db_loader(self) -> Optional[Callable]:
        return None

    def get_db_filename(self) -> Optional[str]:
        return None


# Provide a lightweight singleton for registry-based discovery
null_module = NullModule()