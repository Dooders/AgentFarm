"""
Dominance Features Module

This module provides feature analysis functions for dominance analysis.
Uses the orchestrator pattern for clean dependency management.

All functions accept an optional orchestrator parameter for dependency injection.
If no orchestrator is provided, a new instance is created using the factory function.
This avoids global state while maintaining backward compatibility.
"""

from typing import Optional

import pandas as pd

from farm.analysis.dominance.orchestrator import DominanceAnalysisOrchestrator, create_dominance_orchestrator
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


# All analysis functions now use the orchestrator
# These are thin wrappers that delegate to the orchestrator for consistency


def analyze_dominance_switch_factors(
    df: pd.DataFrame, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze dominance switch factors using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added dominance switch factor columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_dominance_switch_factors(df)


def analyze_reproduction_dominance_switching(
    df: pd.DataFrame, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze reproduction dominance switching using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added reproduction analysis columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_reproduction_dominance_switching(df)


def analyze_high_vs_low_switching(
    df: pd.DataFrame, numeric_repro_cols, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze high vs low switching using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added high vs low switching comparison columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_high_vs_low_switching(df, numeric_repro_cols)


def analyze_reproduction_timing(
    df: pd.DataFrame, numeric_repro_cols, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze reproduction timing using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added reproduction timing analysis columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_reproduction_timing(df, numeric_repro_cols)


def analyze_reproduction_efficiency(
    df: pd.DataFrame, numeric_repro_cols, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze reproduction efficiency using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added reproduction efficiency analysis columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_reproduction_efficiency(df, numeric_repro_cols)


def analyze_reproduction_advantage(
    df: pd.DataFrame, numeric_repro_cols, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze reproduction advantage using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added reproduction advantage analysis columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_reproduction_advantage(df, numeric_repro_cols)


def analyze_by_agent_type(
    df: pd.DataFrame, numeric_repro_cols, orchestrator: Optional[DominanceAnalysisOrchestrator] = None
) -> pd.DataFrame:
    """
    Analyze by agent type using orchestrator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names
    orchestrator : Optional[DominanceAnalysisOrchestrator]
        Optional orchestrator instance. If None, creates a new instance.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added agent type analysis columns
    """
    if orchestrator is None:
        orchestrator = create_dominance_orchestrator()
    return orchestrator.analyze_by_agent_type(df, numeric_repro_cols)
