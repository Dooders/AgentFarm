"""
Comparative analysis module - backward compatibility layer.

This module provides backward compatibility for the old comparative_analysis
import path while delegating to the new comparative module.
"""

from farm.analysis.comparative.compare import compare_simulations

__all__ = ["compare_simulations"]
