"""
Orchestrator for dominance analysis components.

This module provides the DominanceAnalysisOrchestrator class that manages
dependency wiring between analyzer, computer, and data provider components,
breaking the circular dependency through runtime dependency injection.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from farm.analysis.dominance.compute import DominanceComputer
from farm.analysis.dominance.implementations import (
    DominanceAnalyzer,
    DominanceDataProvider,
)
from farm.analysis.dominance.interfaces import (
    DominanceAnalyzerProtocol,
    DominanceComputerProtocol,
    DominanceDataProviderProtocol,
)
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class DominanceAnalysisOrchestrator:
    """
    Orchestrator for dominance analysis components.

    This class manages the dependency wiring between analyzer, computer,
    and data provider components, providing a unified API for dominance
    analysis operations.

    The orchestrator breaks the circular dependency by:
    1. Creating all components
    2. Wiring their dependencies at runtime (not import time)
    3. Providing a clean API for external consumers

    Example
    -------
    >>> orchestrator = DominanceAnalysisOrchestrator()
    >>>
    >>> # Compute dominance metrics
    >>> result = orchestrator.compute_population_dominance(session)
    >>>
    >>> # Analyze switching patterns
    >>> df = orchestrator.analyze_dominance_switch_factors(df)
    """

    def __init__(
        self,
        computer: Optional[DominanceComputerProtocol] = None,
        analyzer: Optional[DominanceAnalyzerProtocol] = None,
        data_provider: Optional[DominanceDataProviderProtocol] = None,
    ):
        """
        Initialize the dominance analysis orchestrator.

        Parameters
        ----------
        computer : Optional[DominanceComputerProtocol]
            Optional computer implementation. If None, creates default DominanceComputer.
        analyzer : Optional[DominanceAnalyzerProtocol]
            Optional analyzer implementation. If None, creates default DominanceAnalyzer.
        data_provider : Optional[DominanceDataProviderProtocol]
            Optional data provider implementation. If None, creates default DominanceDataProvider.

        Notes
        -----
        When None is provided for any component, a default implementation is created.
        All components are stateless, so the same orchestrator instance can be safely
        reused across different contexts without unexpected behavior. For explicit control
        over component creation, use create_dominance_orchestrator() factory function.
        """
        # Create components if not provided
        self.data_provider = data_provider or DominanceDataProvider()

        # Create computer and analyzer with None dependencies initially
        self.computer = computer or DominanceComputer()
        self.analyzer = analyzer or DominanceAnalyzer()

        # Wire bidirectional dependencies
        self.computer.analyzer = self.analyzer
        self.analyzer.computer = self.computer

        logger.debug(
            "dominance_orchestrator_initialized",
            computer=type(self.computer).__name__,
            analyzer=type(self.analyzer).__name__,
            data_provider=type(self.data_provider).__name__,
        )

    # ========================================================================
    # Computation Methods (delegate to computer)
    # ========================================================================

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
        return self.computer.compute_population_dominance(sim_session)

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
        return self.computer.compute_survival_dominance(sim_session)

    def compute_comprehensive_dominance(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Compute comprehensive dominance scores considering entire simulation history.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing dominance scores and metrics, or None if no data
        """
        return self.computer.compute_comprehensive_dominance(sim_session)

    def compute_dominance_switches(self, sim_session) -> Optional[Dict[str, Any]]:
        """
        Analyze dominance switching patterns during simulation.

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing switching statistics, or None if no data
        """
        return self.computer.compute_dominance_switches(sim_session)

    def compute_dominance_switch_factors(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate factors that correlate with dominance switching patterns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with correlation analysis results, or None if insufficient data
        """
        return self.computer.compute_dominance_switch_factors(df)

    def aggregate_reproduction_analysis_results(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple reproduction analysis functions.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : list of str
            List of numeric reproduction column names

        Returns
        -------
        Dict[str, Any]
            Dictionary with aggregated reproduction analysis results
        """
        return self.computer.aggregate_reproduction_analysis_results(df, numeric_repro_cols)

    # ========================================================================
    # Analysis Methods (delegate to analyzer)
    # ========================================================================

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
        return self.analyzer.analyze_by_agent_type(df, numeric_repro_cols)

    def analyze_high_vs_low_switching(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Compare reproduction metrics between high and low switching groups.

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
        return self.analyzer.analyze_high_vs_low_switching(df, numeric_repro_cols)

    def analyze_reproduction_advantage(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze reproduction advantage and its relationship to dominance switching.

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
        return self.analyzer.analyze_reproduction_advantage(df, numeric_repro_cols)

    def analyze_reproduction_efficiency(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze if reproduction efficiency correlates with dominance stability.

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
        return self.analyzer.analyze_reproduction_efficiency(df, numeric_repro_cols)

    def analyze_reproduction_timing(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """
        Analyze how first reproduction timing relates to dominance switching.

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
        return self.analyzer.analyze_reproduction_timing(df, numeric_repro_cols)

    def analyze_dominance_switch_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze what factors correlate with dominance switching patterns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added dominance switch factor columns
        """
        return self.analyzer.analyze_dominance_switch_factors(df)

    def analyze_reproduction_dominance_switching(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze relationship between reproduction strategies and dominance switching.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added reproduction analysis columns
        """
        return self.analyzer.analyze_reproduction_dominance_switching(df)

    # ========================================================================
    # Data Provider Methods (delegate to data_provider)
    # ========================================================================

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
        return self.data_provider.get_final_population_counts(sim_session)

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
        return self.data_provider.get_agent_survival_stats(sim_session)

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
        return self.data_provider.get_reproduction_stats(sim_session)

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
        return self.data_provider.get_initial_positions_and_resources(sim_session, config)

    # ========================================================================
    # High-Level Orchestration Methods
    # ========================================================================

    def run_full_analysis(self, sim_session, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete dominance analysis for a simulation.

        This orchestrates all components to perform a full analysis:
        1. Compute dominance metrics
        2. Collect data from simulation
        3. Return comprehensive results

        Parameters
        ----------
        sim_session : SQLAlchemy session
            Database session connected to simulation data
        config : dict
            Simulation configuration dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all dominance analysis results
        """
        logger.info("starting_full_dominance_analysis")

        results = {}

        try:
            # Compute dominance metrics
            results["population_dominance"] = self.compute_population_dominance(sim_session)
            results["survival_dominance"] = self.compute_survival_dominance(sim_session)
            results["comprehensive_dominance"] = self.compute_comprehensive_dominance(sim_session)
            results["dominance_switches"] = self.compute_dominance_switches(sim_session)

            # Get data from simulation
            results["initial_data"] = self.get_initial_positions_and_resources(sim_session, config)
            results["final_counts"] = self.get_final_population_counts(sim_session)
            results["survival_stats"] = self.get_agent_survival_stats(sim_session)
            results["reproduction_stats"] = self.get_reproduction_stats(sim_session)

            logger.info("full_dominance_analysis_complete")

        except Exception as e:
            logger.error("error_in_full_dominance_analysis", error=str(e), exc_info=True)
            raise

        return results

    def analyze_dataframe_comprehensively(
        self, df: pd.DataFrame, numeric_repro_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive analysis on a DataFrame of simulation results.

        This orchestrates all DataFrame-based analysis methods:
        1. Switch factor analysis
        2. Reproduction-dominance switching analysis
        3. High vs low switching analysis
        4. Agent type analysis

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with simulation analysis results
        numeric_repro_cols : Optional[list of str]
            List of numeric reproduction column names. If None, auto-detected.

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with all analysis columns added
        """
        logger.info("starting_comprehensive_dataframe_analysis", rows=len(df))

        try:
            # Auto-detect numeric reproduction columns if not provided
            if numeric_repro_cols is None:
                from farm.analysis.common.metrics import get_valid_numeric_columns

                reproduction_cols = [col for col in df.columns if "reproduction" in col]
                numeric_repro_cols = get_valid_numeric_columns(df, reproduction_cols)
                logger.debug("auto_detected_repro_cols", count=len(numeric_repro_cols))

            # Run all analysis methods
            df = self.analyze_dominance_switch_factors(df)
            df = self.analyze_reproduction_dominance_switching(df)

            if numeric_repro_cols:
                df = self.analyze_high_vs_low_switching(df, numeric_repro_cols)
                df = self.analyze_by_agent_type(df, numeric_repro_cols)

            logger.info("comprehensive_dataframe_analysis_complete", final_columns=len(df.columns))

        except Exception as e:
            logger.error("error_in_comprehensive_dataframe_analysis", error=str(e), exc_info=True)
            raise

        return df


# ============================================================================
# Convenience Factory Function
# ============================================================================


def create_dominance_orchestrator(
    custom_computer: Optional[DominanceComputerProtocol] = None,
    custom_analyzer: Optional[DominanceAnalyzerProtocol] = None,
    custom_data_provider: Optional[DominanceDataProviderProtocol] = None,
) -> DominanceAnalysisOrchestrator:
    """
    Factory function to create a properly configured DominanceAnalysisOrchestrator.

    This is the recommended way to create an orchestrator instance.

    Parameters
    ----------
    custom_computer : Optional[DominanceComputerProtocol]
        Optional custom computer implementation
    custom_analyzer : Optional[DominanceAnalyzerProtocol]
        Optional custom analyzer implementation
    custom_data_provider : Optional[DominanceDataProviderProtocol]
        Optional custom data provider implementation

    Returns
    -------
    DominanceAnalysisOrchestrator
        Fully configured orchestrator with all dependencies wired

    Example
    -------
    >>> # Create with default implementations
    >>> orchestrator = create_dominance_orchestrator()
    >>>
    >>> # Create with custom computer
    >>> custom_comp = MyCustomComputer()
    >>> orchestrator = create_dominance_orchestrator(custom_computer=custom_comp)
    """
    return DominanceAnalysisOrchestrator(
        computer=custom_computer,
        analyzer=custom_analyzer,
        data_provider=custom_data_provider,
    )
