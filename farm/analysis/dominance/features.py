"""
Dominance Features Module

This module provides feature analysis functions for dominance analysis.
Uses the orchestrator pattern for clean dependency management.
"""

import pandas as pd

from farm.analysis.dominance.orchestrator import create_dominance_orchestrator
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create module-level orchestrator instance
_orchestrator = create_dominance_orchestrator()


# All analysis functions now use the orchestrator
# These are thin wrappers that delegate to the orchestrator for consistency

def analyze_dominance_switch_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze dominance switch factors using orchestrator."""
    return _orchestrator.analyze_dominance_switch_factors(df)


def analyze_reproduction_dominance_switching(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze reproduction dominance switching using orchestrator."""
    return _orchestrator.analyze_reproduction_dominance_switching(df)


def analyze_high_vs_low_switching(df: pd.DataFrame, numeric_repro_cols):
    """Analyze high vs low switching using orchestrator."""
    return _orchestrator.analyze_high_vs_low_switching(df, numeric_repro_cols)


def analyze_reproduction_timing(df: pd.DataFrame, numeric_repro_cols):
    """Analyze reproduction timing using orchestrator."""
    return _orchestrator.analyze_reproduction_timing(df, numeric_repro_cols)


def analyze_reproduction_efficiency(df: pd.DataFrame, numeric_repro_cols):
    """Analyze reproduction efficiency using orchestrator."""
    return _orchestrator.analyze_reproduction_efficiency(df, numeric_repro_cols)


def analyze_reproduction_advantage(df: pd.DataFrame, numeric_repro_cols):
    """Analyze reproduction advantage using orchestrator."""
    return _orchestrator.analyze_reproduction_advantage(df, numeric_repro_cols)


def analyze_by_agent_type(df: pd.DataFrame, numeric_repro_cols):
    """Analyze by agent type using orchestrator."""
    return _orchestrator.analyze_by_agent_type(df, numeric_repro_cols)
