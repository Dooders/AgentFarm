"""
Protocol interfaces for the dominance analysis module.

This module defines protocols to break the circular dependency between
analyze.py and compute.py through dependency injection and protocol-based design.

The protocols follow the Dependency Inversion Principle (DIP) by defining
abstractions that both analyzer and computer components can depend on,
rather than depending directly on concrete implementations.
"""

from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


class DominanceComputerProtocol(Protocol):
    """
    Protocol for dominance computation operations.

    This protocol defines the interface for computing dominance-related metrics
    from simulation data. Implementations should provide pure computation logic
    without depending on analysis modules.
    """

    def compute_population_dominance(self, sim_session) -> Optional[str]:
        """
        Compute the dominant agent type by final population.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[str]
            The agent type with highest final population, or None if no data
        """
        ...

    def compute_survival_dominance(self, sim_session) -> Optional[str]:
        """
        Compute the dominant agent type by average survival time.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[str]
            The agent type with highest average survival time, or None if no data
        """
        ...

    def compute_comprehensive_dominance(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Compute comprehensive dominance scores considering entire simulation history.

        Uses multiple metrics including:
        - Area Under the Curve (AUC)
        - Recency-weighted AUC
        - Dominance duration
        - Growth trends
        - Final population ratios

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - dominant_type: Overall dominant agent type
            - scores: Composite scores for each agent type
            - metrics: Individual metric values
            - normalized_metrics: Normalized metric values
            Returns None if no data available
        """
        ...

    def compute_dominance_switches(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Analyze dominance switching patterns during simulation.

        Identifies:
        - Total number of dominance switches
        - Average duration of dominance periods
        - Phase-specific switch frequency
        - Transition probabilities between agent types

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - total_switches: Total number of switches
            - switches_per_step: Switch frequency
            - avg_dominance_periods: Average period durations by type
            - phase_switches: Switches by simulation phase
            - transition_matrix: Raw transition counts
            - transition_probabilities: Normalized transition probabilities
            Returns None if no data available
        """
        ...

    def compute_dominance_switch_factors(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate factors that correlate with dominance switching patterns.

        Analyzes:
        - Initial condition correlations
        - Dominant type relationships
        - Reproduction metric correlations

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results including total_switches column

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - top_positive_correlations: Factors associated with more switches
            - top_negative_correlations: Factors associated with fewer switches
            - switches_by_dominant_type: Average switches by final dominant type
            - reproduction_correlations: Reproduction factors correlated with switching
            Returns None if insufficient data
        """
        ...

    def aggregate_reproduction_analysis_results(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple reproduction analysis functions.

        This method requires an analyzer dependency to be injected into the
        DominanceComputer instance for proper functionality.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names to analyze

        Returns
        -------
        Dict[str, Any]
            Dictionary with aggregated reproduction analysis results including:
            - High vs low switching comparisons
            - Reproduction timing analysis
            - Reproduction efficiency analysis
            - Reproduction advantage analysis
            - Agent type-specific analysis

        Raises
        ------
        RuntimeError
            If no analyzer dependency is injected into the DominanceComputer instance
        """
        ...


class DominanceAnalyzerProtocol(Protocol):
    """
    Protocol for dominance analysis operations.

    This protocol defines the interface for analyzing dominance patterns
    and their relationships with other simulation factors. Implementations
    should coordinate data analysis and interpretation.
    """

    def analyze_by_agent_type(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze relationship between reproduction metrics and dominance switching by agent type.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added agent type analysis columns
        """
        ...

    def analyze_high_vs_low_switching(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Compare reproduction metrics between high and low switching groups.

        Splits simulations by median total_switches and compares reproduction
        metrics between the two groups.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added high vs low switching comparison columns
        """
        ...

    def analyze_reproduction_advantage(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze reproduction advantage and its relationship to dominance switching.

        Focuses on advantage metrics (rate, efficiency) and their correlation
        with dominance stability.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added reproduction advantage analysis columns
        """
        ...

    def analyze_reproduction_efficiency(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze if reproduction efficiency correlates with dominance stability.

        Examines the relationship between reproduction efficiency metrics
        and the inverse of switching frequency (stability).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added reproduction efficiency analysis columns
        """
        ...

    def analyze_reproduction_timing(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze how first reproduction timing relates to dominance switching.

        Examines correlations between first reproduction time and
        dominance switching frequency by agent type.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added reproduction timing analysis columns
        """
        ...

    def analyze_dominance_switch_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze what factors correlate with dominance switching patterns.

        Wraps the computation of switch factors and adds results to DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added dominance switch factor columns
        """
        ...

    def analyze_reproduction_dominance_switching(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze relationship between reproduction strategies and dominance switching.

        Orchestrates multiple reproduction analysis functions and aggregates
        their results into the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added reproduction analysis columns
        """
        ...


class DominanceDataProviderProtocol(Protocol):
    """
    Protocol for dominance data retrieval operations.

    This protocol defines the interface for retrieving simulation data
    needed for dominance analysis.
    """

    def get_final_population_counts(self, sim_session) -> Optional[Dict[str, int]]:
        """
        Get final population counts by agent type.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, int]]
            Dictionary with population counts by agent type, or None if no data
        """
        ...

    def get_agent_survival_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Get agent survival statistics by type.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with survival statistics by agent type, or None if no data
        """
        ...

    def get_reproduction_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Get reproduction statistics by agent type.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with reproduction statistics by agent type, or None if no data
        """
        ...

    def get_initial_positions_and_resources(self, sim_session, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get initial positioning and resource data.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data
        config : dict
            Simulation configuration dictionary

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with initial position and resource data, or None if no data
        """
        ...
