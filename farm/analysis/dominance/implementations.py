"""
Concrete implementations of dominance analysis protocols.

This module provides the concrete classes that implement the protocol interfaces,
using dependency injection to break circular dependencies.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from farm.analysis.common.metrics import (
    analyze_correlations,
    get_valid_numeric_columns,
    group_and_analyze,
    split_and_compare_groups,
)
from farm.analysis.dominance.interfaces import (
    DominanceAnalyzerProtocol,
    DominanceComputerProtocol,
    DominanceDataProviderProtocol,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class DominanceAnalyzer:
    """
    Concrete implementation of DominanceAnalyzerProtocol.

    Provides analysis and interpretation of dominance patterns using
    dependency injection to avoid circular dependencies.
    """

    def __init__(self, computer: Optional[DominanceComputerProtocol] = None):
        """
        Initialize the dominance analyzer.

        Parameters
        ----------
        computer : Optional[DominanceComputerProtocol]
            Optional computer dependency for computation operations.
            Should be injected by orchestrator if needed.
        """
        self.computer = computer

    def analyze_by_agent_type(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """Analyze relationship between reproduction metrics and dominance switching by agent type."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_by_agent_type")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        results = {}

        if "comprehensive_dominance" in result_df.columns:
            # Define the analysis function to apply to each group
            def analyze_group_correlations(group_df):
                # Calculate correlations between reproduction metrics and switching
                group_correlations = analyze_correlations(
                    group_df,
                    target_column="total_switches",
                    metric_columns=numeric_repro_cols,
                    min_data_points=5,
                )
                return group_correlations

            # Use the utility function to group and analyze
            agent_types = ["system", "independent", "control"]
            type_results = group_and_analyze(
                result_df,
                group_column="comprehensive_dominance",
                group_values=agent_types,
                analysis_func=analyze_group_correlations,
                min_group_size=5,
            )

            # Log top correlations for each agent type
            for agent_type, type_correlations in type_results.items():
                if type_correlations:
                    logger.info(
                        "top_reproduction_factors_affecting_switching",
                        agent_type=agent_type,
                    )
                    sorted_corrs = sorted(type_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    for col, corr in sorted_corrs[:3]:  # Top 3
                        if abs(corr) > 0.2:  # Only report stronger correlations
                            direction = "more" if corr > 0 else "fewer"
                            logger.info(
                                "reproduction_factor_correlation",
                                column=col,
                                correlation=corr,
                                direction=direction,
                            )

        return result_df

    def analyze_high_vs_low_switching(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """Compare reproduction metrics between high and low switching groups."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_high_vs_low_switching")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        if result_df.empty or "total_switches" not in result_df.columns:
            logger.warning("no_dominance_switch_data_for_high_vs_low_switching")
            return result_df

        # Use the utility function to split and compare groups
        comparison_results = split_and_compare_groups(
            result_df,
            split_column="total_switches",
            metrics=numeric_repro_cols,
            split_method="median",
        )

        # Extract comparison results
        if "comparison_results" in comparison_results:
            repro_comparison = comparison_results["comparison_results"]

            # Add these values to the DataFrame with the specific naming convention used in this module
            for col, stats in repro_comparison.items():
                result_df[f"{col}_high_switching_mean"] = stats["high_group_mean"]
                result_df[f"{col}_low_switching_mean"] = stats["low_group_mean"]
                result_df[f"{col}_difference"] = stats["difference"]
                result_df[f"{col}_percent_difference"] = stats["percent_difference"]

            # Log the most significant differences
            logger.info("reproduction_differences_high_vs_low_switching")
            sorted_diffs = sorted(
                repro_comparison.items(),
                key=lambda x: abs(x[1]["percent_difference"]),
                reverse=True,
            )

            for col, stats in sorted_diffs[:5]:  # Top 5 differences
                if abs(stats["percent_difference"]) > 10:  # Only report meaningful differences
                    direction = "higher" if stats["difference"] > 0 else "lower"
                    logger.info(
                        "reproduction_difference_detail",
                        column=col,
                        high_mean=stats["high_group_mean"],
                        low_mean=stats["low_group_mean"],
                        percent_difference=abs(stats["percent_difference"]),
                        direction=direction,
                    )

        return result_df

    def analyze_reproduction_advantage(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """Analyze reproduction advantage and dominance switching."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_reproduction_advantage")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        advantage_cols = [
            col
            for col in numeric_repro_cols
            if "reproduction_rate_advantage" in col or "reproduction_efficiency_advantage" in col
        ]

        if advantage_cols and "switches_per_step" in result_df.columns:
            # Calculate stability metric (inverse of switches per step)
            if "dominance_stability" not in result_df.columns:
                result_df["dominance_stability"] = 1 / (result_df["switches_per_step"] + 0.01)

            # Use the utility function to analyze correlations
            def filter_valid(data_df, advantage_cols=advantage_cols):
                return data_df[data_df[advantage_cols].notna().all(axis=1)]

            advantage_stability_corr = analyze_correlations(
                result_df,
                target_column="dominance_stability",
                metric_columns=advantage_cols,
                min_data_points=5,
                filter_condition=filter_valid,
            )

            # Log the results
            logger.info("correlation_reproduction_advantage_dominance_stability")
            for col, corr in advantage_stability_corr.items():
                if abs(corr) > 0.1:  # Only report meaningful correlations
                    if "_vs_" in col:
                        types = (
                            col.split("_vs_")[0],
                            col.split("_vs_")[1].split("_reproduction")[0],
                        )
                        direction = "more" if corr > 0 else "less"
                        logger.info(
                            "reproduction_advantage_correlation",
                            advantage_type=types[0],
                            over_type=types[1],
                            direction=direction,
                            correlation=corr,
                        )

        return result_df

    def analyze_reproduction_efficiency(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """Analyze if reproduction efficiency correlates with dominance stability."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_reproduction_efficiency")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        efficiency_cols = [col for col in numeric_repro_cols if "reproduction_efficiency" in col]

        if efficiency_cols and "switches_per_step" in result_df.columns:
            # Calculate stability metric (inverse of switches per step)
            result_df["dominance_stability"] = 1 / (result_df["switches_per_step"] + 0.01)

            # Use the utility function to analyze correlations
            def filter_valid(data_df, efficiency_cols=efficiency_cols):
                return data_df[
                    (data_df[efficiency_cols].notna()).all(axis=1) & (data_df[efficiency_cols] != 0).all(axis=1)
                ]

            efficiency_stability_corr = analyze_correlations(
                result_df,
                target_column="dominance_stability",
                metric_columns=efficiency_cols,
                min_data_points=5,
                filter_condition=filter_valid,
            )

            # Log the results
            logger.info("correlation_reproduction_efficiency_dominance_stability")
            for col, corr in efficiency_stability_corr.items():
                if abs(corr) > 0.1:  # Only report meaningful correlations
                    agent_type = col.split("_reproduction")[0]
                    direction = "more" if corr > 0 else "less"
                    logger.info(
                        "reproduction_efficiency_correlation",
                        agent_type=agent_type,
                        direction=direction,
                        correlation=corr,
                    )

        return result_df

    def analyze_reproduction_timing(self, df: pd.DataFrame, numeric_repro_cols: List[str]) -> pd.DataFrame:
        """Analyze how first reproduction timing relates to dominance switching."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_reproduction_timing")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        # Filter to get only first reproduction columns
        first_repro_cols = [col for col in numeric_repro_cols if "first_reproduction_time" in col]

        if first_repro_cols:
            # Use the utility function to analyze correlations with filtering
            first_repro_corr = {}
            for col in first_repro_cols:
                # Create a single-column filter for this specific column
                def col_filter(df, col=col):
                    return df[df[col] > 0]

                correlations = analyze_correlations(
                    result_df,
                    target_column="total_switches",
                    metric_columns=[col],
                    min_data_points=5,
                    filter_condition=col_filter,
                )

                # Add the correlation if found
                if correlations and col in correlations:
                    first_repro_corr[col] = correlations[col]

            # Log the results
            logger.info("correlation_reproduction_timing_dominance_switches")
            for col, corr in first_repro_corr.items():
                agent_type = col.split("_first_reproduction")[0]
                if abs(corr) > 0.1:  # Only report meaningful correlations
                    direction = "more" if corr > 0 else "fewer"
                    logger.info(
                        "reproduction_timing_correlation",
                        agent_type=agent_type,
                        direction=direction,
                        correlation=corr,
                    )
        else:
            logger.info("no_first_reproduction_timing_data_available")

        return result_df

    def analyze_dominance_switch_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze what factors correlate with dominance switching patterns."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_dominance_switch_factors")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        if result_df.empty or "total_switches" not in result_df.columns:
            logger.warning("no_dominance_switch_data_available")
            return result_df

        # Calculate dominance switch factors using computer if available
        if self.computer is not None:
            results = self.computer.compute_dominance_switch_factors(result_df)
        else:
            logger.warning("no_computer_injected_for_switch_factors")
            return result_df

        if results is None:
            return result_df

        # Add dominance switch factors to the DataFrame
        if results:
            # Add top positive correlations
            if "top_positive_correlations" in results:
                for factor, corr in results["top_positive_correlations"].items():
                    result_df[f"positive_corr_{factor}"] = corr

            # Add top negative correlations
            if "top_negative_correlations" in results:
                for factor, corr in results["top_negative_correlations"].items():
                    result_df[f"negative_corr_{factor}"] = corr

            # Add switches by dominant type
            if "switches_by_dominant_type" in results:
                for agent_type, avg_switches in results["switches_by_dominant_type"].items():
                    result_df[f"{agent_type}_avg_switches"] = avg_switches

            # Add reproduction correlations
            if "reproduction_correlations" in results:
                for factor, corr in results["reproduction_correlations"].items():
                    result_df[f"repro_corr_{factor}"] = corr

            logger.info("added_dominance_switch_factor_analysis")

        return result_df

    def analyze_reproduction_dominance_switching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze relationship between reproduction strategies and dominance switching."""
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("input_not_dataframe", function="analyze_reproduction_dominance_switching")
            return df

        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()

        if result_df.empty or "total_switches" not in result_df.columns:
            logger.warning("no_dominance_switch_data_for_reproduction_switching")
            return result_df

        reproduction_cols = [col for col in result_df.columns if "reproduction" in col]

        # Filter to only include numeric reproduction columns
        numeric_repro_cols = get_valid_numeric_columns(result_df, reproduction_cols)

        if not numeric_repro_cols:
            logger.warning("no_numeric_reproduction_data_columns")
            return result_df

        # Use the aggregation function from computer to collect all results
        if self.computer is not None:
            results = self.computer.aggregate_reproduction_analysis_results(result_df, numeric_repro_cols)
        else:
            logger.warning("no_computer_injected_for_reproduction_analysis")
            return result_df

        if not results:
            return result_df

        # Add results to the DataFrame
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for key, value in category_results.items():
                    if isinstance(value, dict):
                        # For nested dictionaries (like high vs low switching comparison)
                        for subkey, subvalue in value.items():
                            col_name = f"{category}_{key}_{subkey}"
                            result_df[col_name] = subvalue
                    else:
                        # For simple key-value pairs
                        col_name = f"{category}_{key}"
                        result_df[col_name] = value

        logger.info("added_reproduction_analysis_categories", count=len(results))

        return result_df


class DominanceDataProvider:
    """
    Concrete implementation of DominanceDataProviderProtocol.

    Provides data retrieval operations for dominance analysis.
    """

    def get_final_population_counts(self, sim_session) -> Optional[Dict[str, int]]:
        """Get the final population counts for each agent type."""
        from farm.analysis.dominance.data import get_final_population_counts as _get_final_population_counts

        return _get_final_population_counts(sim_session)

    def get_agent_survival_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """Get detailed survival statistics for each agent type."""
        from farm.analysis.dominance.data import get_agent_survival_stats as _get_agent_survival_stats

        return _get_agent_survival_stats(sim_session)

    def get_reproduction_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """Analyze reproduction patterns for each agent type."""
        from farm.analysis.dominance.data import get_reproduction_stats as _get_reproduction_stats

        return _get_reproduction_stats(sim_session)

    def get_initial_positions_and_resources(self, sim_session, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the initial positions of agents and resources."""
        from farm.analysis.dominance.data import (
            get_initial_positions_and_resources as _get_initial_positions_and_resources,
        )

        return _get_initial_positions_and_resources(sim_session, config)
