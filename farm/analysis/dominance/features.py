import logging
import pandas as pd

from farm.analysis.base_module import get_valid_numeric_columns, group_and_analyze
from farm.analysis.common.metrics import analyze_correlations, split_and_compare_groups
from farm.analysis.dominance.compute import (
    aggregate_reproduction_analysis_results,
    compute_dominance_switch_factors,
)


def analyze_dominance_switch_factors(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_dominance_switch_factors is not a DataFrame")
        return df
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for analysis")
        return df
    results = compute_dominance_switch_factors(df)
    if results is None:
        return df
    if results:
        if "top_positive_correlations" in results:
            for factor, corr in results["top_positive_correlations"].items():
                df[f"positive_corr_{factor}"] = corr
        if "top_negative_correlations" in results:
            for factor, corr in results["top_negative_correlations"].items():
                df[f"negative_corr_{factor}"] = corr
        if "switches_by_dominant_type" in results:
            for agent_type, avg_switches in results["switches_by_dominant_type"].items():
                df[f"{agent_type}_avg_switches"] = avg_switches
        if "reproduction_correlations" in results:
            for factor, corr in results["reproduction_correlations"].items():
                df[f"repro_corr_{factor}"] = corr
        logging.info("Added dominance switch factor analysis to the DataFrame")
    return df


def analyze_reproduction_dominance_switching(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logging.warning(
            "Input to analyze_reproduction_dominance_switching is not a DataFrame"
        )
        return df
    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for reproduction-switching analysis"
        )
        return df
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    numeric_repro_cols = get_valid_numeric_columns(df, reproduction_cols)
    if not numeric_repro_cols:
        logging.warning("No numeric reproduction data columns found")
        return df
    results = aggregate_reproduction_analysis_results(df, numeric_repro_cols)
    if not results:
        return df
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            for key, value in category_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        col_name = f"{category}_{key}_{subkey}"
                        df[col_name] = subvalue
                else:
                    col_name = f"{category}_{key}"
                    df[col_name] = value
    logging.info(f"Added {len(results)} reproduction analysis categories to the DataFrame")
    return df


def analyze_high_vs_low_switching(df: pd.DataFrame, numeric_repro_cols):
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_high_vs_low_switching is not a DataFrame")
        return df
    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for high vs low switching analysis"
        )
        return df
    comparison_results = split_and_compare_groups(
        df,
        split_column="total_switches",
        metrics=numeric_repro_cols,
        split_method="median",
    )
    if "comparison_results" in comparison_results:
        repro_comparison = comparison_results["comparison_results"]
        for col, stats in repro_comparison.items():
            df[f"{col}_high_switching_mean"] = stats["high_group_mean"]
            df[f"{col}_low_switching_mean"] = stats["low_group_mean"]
            df[f"{col}_difference"] = stats["difference"]
            df[f"{col}_percent_difference"] = stats["percent_difference"]
        logging.info("Computed high vs low switching comparison for reproduction metrics")
    return df


def analyze_reproduction_timing(df: pd.DataFrame, numeric_repro_cols):
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_reproduction_timing is not a DataFrame")
        return df
    first_repro_cols = [col for col in numeric_repro_cols if "first_reproduction_time" in col]
    if first_repro_cols:
        first_repro_corr = {}
        for col in first_repro_cols:
            col_filter = lambda df_: df_[df_[col] > 0]
            correlations = analyze_correlations(
                df,
                target_column="total_switches",
                metric_columns=[col],
                min_data_points=5,
                filter_condition=col_filter,
            )
            if correlations and col in correlations:
                first_repro_corr[col] = correlations[col]
        logging.info("Computed correlation: first reproduction timing vs dominance switches")
    else:
        logging.info("No first reproduction timing data available for analysis")
    return df


def analyze_reproduction_efficiency(df: pd.DataFrame, numeric_repro_cols):
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_reproduction_efficiency is not a DataFrame")
        return df
    efficiency_cols = [col for col in numeric_repro_cols if "reproduction_efficiency" in col]
    if efficiency_cols and "switches_per_step" in df.columns:
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)
        filter_valid = lambda data_df: data_df[
            (data_df[efficiency_cols].notna()).all(axis=1)
            & (data_df[efficiency_cols] != 0).all(axis=1)
        ]
        analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=efficiency_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )
        logging.info("Computed correlation: reproduction efficiency vs dominance stability")
    return df


def analyze_reproduction_advantage(df: pd.DataFrame, numeric_repro_cols):
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_reproduction_advantage is not a DataFrame")
        return df
    advantage_cols = [
        col
        for col in numeric_repro_cols
        if "reproduction_rate_advantage" in col or "reproduction_efficiency_advantage" in col
    ]
    if advantage_cols and "switches_per_step" in df.columns:
        if "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)
        filter_valid = lambda data_df: data_df[data_df[advantage_cols].notna().all(axis=1)]
        analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=advantage_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )
        logging.info("Computed correlation: reproduction advantage vs dominance stability")
    return df


def analyze_by_agent_type(df: pd.DataFrame, numeric_repro_cols):
    if not isinstance(df, pd.DataFrame):
        logging.warning("Input to analyze_by_agent_type is not a DataFrame")
        return df
    results = {}
    if "comprehensive_dominance" in df.columns:
        def analyze_group_correlations(group_df):
            group_correlations = analyze_correlations(
                group_df,
                target_column="total_switches",
                metric_columns=numeric_repro_cols,
                min_data_points=5,
            )
            return group_correlations

        agent_types = ["system", "independent", "control"]
        type_results = group_and_analyze(
            df,
            group_column="comprehensive_dominance",
            group_values=agent_types,
            analysis_func=analyze_group_correlations,
            min_group_size=5,
        )
        for agent_type, type_correlations in type_results.items():
            if type_correlations:
                top = sorted(type_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                logging.info(
                    f"Top reproduction factors affecting switching in {agent_type}-dominant simulations: {top}"
                )
    return df

