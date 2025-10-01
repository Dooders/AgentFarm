"""
Null analysis module for testing and as a minimal example.
"""

from typing import Callable, Optional
import pandas as pd

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor


class NullModule(BaseAnalysisModule):
    """No-op analysis module that does nothing.
    
    Useful for testing and as a minimal example of the module structure.
    """
    
    def __init__(self, name: str = "null", description: str = "No-op analysis module"):
        super().__init__(name=name, description=description)

    def register_functions(self) -> None:
        """Register no functions (null module)."""
        self._functions = {}
        self._groups = {"all": []}

    def get_data_processor(self) -> SimpleDataProcessor:
        """Return a processor that returns empty DataFrame."""
        def _noop_processor(*args, **kwargs) -> pd.DataFrame:
            return pd.DataFrame()
        
        return SimpleDataProcessor(_noop_processor)


# Provide a lightweight singleton for registry-based discovery
null_module = NullModule()