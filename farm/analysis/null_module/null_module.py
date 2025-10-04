"""
Null analysis module for testing purposes.

This is a minimal implementation that does nothing but satisfies the AnalysisModule protocol.
"""

import pandas as pd
from typing import List, Dict, Any, Optional

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function


def null_data_processor(df: pd.DataFrame) -> pd.DataFrame:
    """Null data processor that returns data unchanged."""
    return df


def null_analysis_function(df: pd.DataFrame, ctx, **kwargs) -> Optional[Any]:
    """Null analysis function that does nothing."""
    return None


class NullModule(BaseAnalysisModule):
    """Null module for testing the analysis registry."""

    def __init__(self):
        super().__init__(
            name="null",
            description="Null module for testing purposes"
        )

    def register_functions(self) -> None:
        """Register null analysis functions."""

        # Single null function
        self._functions = {
            "null_function": make_analysis_function(null_analysis_function),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "basic": list(self._functions.values()),
        }

    def get_data_processor(self):
        """Get null data processor."""
        return SimpleDataProcessor(null_data_processor)


# Create singleton instance
null_module = NullModule()
