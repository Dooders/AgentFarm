"""
Advantage Analysis Module

This module provides functions to analyze advantages across multiple simulations,
identifying patterns and correlations between advantages and dominance outcomes.
"""

import logging
import time
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
from scipy import stats

from farm.analysis.advantage.compute import (
    compute_advantage_dominance_correlation,
    compute_advantages,
)
# Note: BaseAnalysisModule is from the old system - kept for backwards compatibility
# For new modules, use: from farm.analysis.core import BaseAnalysisModule
from farm.analysis.base_module import BaseAnalysisModule
from farm.analysis.dominance.compute import compute_comprehensive_dominance
from scripts.analysis_config import setup_and_process_simulations


def process_single_simulation(session, iteration, config, **kwargs):
    """
    Process data from a single simulation for advantage analysis.

    Parameters
    ----------
    session : SQLAlchemy session
        Session connected to the simulation database
    iteration : int
        Iteration number of the simulation
    config : dict
        Configuration dictionary for the simulation
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    dict or None
        Dictionary containing processed data for this simulation,
        or None if processing failed
    """
    try:
        logging.debug(f"Computing dominance for iteration {iteration}")
        dominance_result = compute_comprehensive_dominance(session)

        if not dominance_result:
            logging.warning(
                f"Skipping iteration {iteration}: Failed to compute dominance"
            )
            return None

        dominant_type = dominance_result["dominant_type"]
        logging.debug(f"Iteration {iteration} dominant type: {dominant_type}")

        # Compute advantages
        logging.debug(f"Computing advantages for iteration {iteration}")
        advantages = compute_advantages(session)

        # Create a row of data for this simulation
        sim_data = {"iteration": iteration, "dominant_type": dominant_type}

        # Add dominance scores
        for agent_type in ["system", "independent", "control"]:
            sim_data[f"{agent_type}_dominance_score"] = dominance_result["scores"][
                agent_type
            ]

        # Add advantage metrics
        for category in advantages:
            if category == "composite_advantage":
                # Add composite advantage scores
                for pair, details in advantages[category].items():
                    sim_data[f"{pair}_composite_advantage"] = details["score"]

                    # Add component contributions
                    for component, value in details["components"].items():
                        sim_data[f"{pair}_{component}_contribution"] = value
            else:
                # For each agent pair in this category
                for pair, metrics in advantages[category].items():
                    # Add relevant advantage metrics
                    for metric, value in metrics.items():
                        if "advantage" in metric or "trajectory" in metric:
                            sim_data[f"{pair}_{category}_{metric}"] = value

        # Add advantage to dominance correlation
        logging.debug(
            f"Computing advantage-dominance correlation for iteration {iteration}"
        )
        advantage_correlation = compute_advantage_dominance_correlation(session)
        if advantage_correlation:
            sim_data["advantage_ratio"] = advantage_correlation["summary"][
                "advantage_ratio"
            ]

            # Add summary statistics
            sim_data["advantages_favoring_dominant"] = advantage_correlation["summary"][
                "advantages_favoring_dominant"
            ]
            sim_data["total_advantages"] = advantage_correlation["summary"][
                "total_advantages"
            ]

        return sim_data

    except Exception as e:
        logging.error(f"Error processing iteration {iteration}: {e}")
        return None


def analyze_advantages(
    experiment_path: str,
    save_to_db: bool = False,
    db_path: str = "sqlite:///advantage.db",
) -> pd.DataFrame:
    """
    Analyze advantages across all simulations in an experiment.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing simulation data
    save_to_db : bool, optional
        If True, save the data directly to the database instead of
        returning a DataFrame
    db_path : str, optional
        Path to the database to save the data to, defaults to
        'sqlite:///advantage.db'

    Returns
    -------
    pandas.DataFrame
        DataFrame containing advantage metrics for all
        simulations if save_to_db is False, otherwise None
    """
    # Use the helper function to process all simulations
    data = setup_and_process_simulations(
        experiment_path=experiment_path,
        process_simulation_func=process_single_simulation,
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Handle the save_to_db option if implemented
    if save_to_db and not df.empty:
        logging.info(f"Saving analysis data to database: {db_path}")
        # Return empty DataFrame when saving to database
        return pd.DataFrame()

    return df


def analyze_advantage_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in advantages across simulations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage metrics for all simulations

    Returns
    -------
    Dict
        Dictionary containing analysis results
    """
    results = {
        "advantage_significance": {},
        "dominance_correlations": {},
        "advantage_category_importance": {},
        "advantage_timing_analysis": {},
        "agent_type_specific_analysis": {},
    }

    # Skip if no data
    if df.empty:
        logging.warning("No data to analyze")
        return results

    # 1. Analyze statistical significance of advantages
    # Find all advantage columns
    advantage_cols = [
        col
        for col in df.columns
        if (
            ("advantage" in col or "trajectory" in col)
            and "composite" not in col
            and "ratio" not in col
        )
    ]

    for col in advantage_cols:
        # Test if the advantage is significantly different from zero
        try:
            t_stat, p_value = stats.ttest_1samp(df[col].dropna(), 0)
            # Handle tuple return from scipy.stats and ensure float type
            if isinstance(p_value, tuple):
                p_value = p_value[0] if len(p_value) > 0 else 1.0
            elif p_value is None:
                p_value = 1.0
            else:
                try:
                    p_value = (
                        float(cast(float, p_value)) if p_value is not None else 1.0
                    )
                except (ValueError, TypeError):
                    p_value = 1.0
            significance = p_value < 0.05

            results["advantage_significance"][col] = {
                "mean": df[col].mean(),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": significance,
            }
        except Exception as e:
            logging.warning(f"Error analyzing significance for {col}: {e}")

    # 2. Analyze correlations between advantages and dominance outcomes
    # For each agent type, find correlations with its dominance score
    for agent_type in ["system", "independent", "control"]:
        score_col = f"{agent_type}_dominance_score"
        if score_col in df.columns:
            correlations = {}

            for col in advantage_cols:
                # Only include advantage columns relevant to this agent type
                if f"{agent_type}_vs_" in col or f"_vs_{agent_type}" in col:
                    try:
                        # For advantages where agent_type is second (e.g., other_vs_system),
                        # we need to invert the correlation because a positive value
                        # means an advantage for the first agent, not our agent_type
                        invert = f"_vs_{agent_type}" in col

                        corr_result = df[[col, score_col]].corr().iloc[0, 1]
                        # Handle NaN values and ensure float type
                        if pd.isna(corr_result):
                            corr = 0.0
                        else:
                            try:
                                corr = (
                                    float(cast(float, corr_result))
                                    if corr_result is not None
                                    else 0.0
                                )
                            except (ValueError, TypeError):
                                corr = 0.0
                        if invert:
                            corr = -corr

                        correlations[col] = corr
                    except Exception as e:
                        logging.warning(f"Error calculating correlation for {col}: {e}")

            # Sort by absolute correlation strength
            sorted_corrs = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )

            results["dominance_correlations"][agent_type] = {
                k: v for k, v in sorted_corrs
            }

    # 3. Analyze importance of different advantage categories
    categories = [
        "resource_acquisition",
        "reproduction",
        "survival",
        "population_growth",
        "combat",
        "initial_positioning",
    ]

    category_importance = {}

    for category in categories:
        # Find all advantage columns for this category
        category_cols = [col for col in advantage_cols if category in col]

        if not category_cols:
            continue

        # For each agent type, calculate the average correlation strength
        for agent_type in ["system", "independent", "control"]:
            if agent_type not in results["dominance_correlations"]:
                continue

            relevance_scores = []

            for col in category_cols:
                if col in results["dominance_correlations"][agent_type]:
                    relevance_scores.append(
                        abs(results["dominance_correlations"][agent_type][col])
                    )

            if relevance_scores:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                max_relevance = max(relevance_scores)

                if category not in category_importance:
                    category_importance[category] = {}

                category_importance[category][agent_type] = {
                    "average_relevance": avg_relevance,
                    "max_relevance": max_relevance,
                }

    # Sort categories by overall importance
    overall_importance = {}
    for category, agent_data in category_importance.items():
        overall_scores = [
            data["average_relevance"] for agent_type, data in agent_data.items()
        ]
        if overall_scores:
            overall_importance[category] = sum(overall_scores) / len(overall_scores)

    sorted_importance = sorted(
        overall_importance.items(), key=lambda x: x[1], reverse=True
    )

    results["advantage_category_importance"] = {
        "by_category": category_importance,
        "overall_ranking": {k: v for k, v in sorted_importance},
    }

    # 4. Analyze timing of advantages (early vs late phase)
    timing_analysis = {}

    for agent_type in ["system", "independent", "control"]:
        # Get the simulations where this agent type was dominant
        agent_dominant = df[df["dominant_type"] == agent_type]

        if len(agent_dominant) < 5:  # Need enough samples
            continue

        phase_advantages = {"early": {}, "mid": {}, "late": {}}

        # Analyze each phase
        for phase in ["early", "mid", "late"]:
            # Find advantage columns for this phase
            phase_cols = [
                col
                for col in advantage_cols
                if f"{phase}_phase_advantage" in col
                or (
                    phase == "early" and "first_" in col
                )  # Special case for first reproduction
            ]

            for col in phase_cols:
                # Calculate average advantage value in simulations where this agent type dominated
                try:
                    avg_value = agent_dominant[col].mean()
                    all_avg = df[col].mean()  # Average across all simulations

                    # Is this advantage typically positive for this agent type when it dominates?
                    if (f"{agent_type}_vs_" in col and avg_value > 0) or (
                        f"_vs_{agent_type}" in col and avg_value < 0
                    ):
                        favors_agent = True
                    else:
                        favors_agent = False

                    # How much stronger is this advantage when this agent type dominates?
                    strength = avg_value / all_avg if all_avg != 0 else float("inf")

                    phase_advantages[phase][col] = {
                        "average_value": avg_value,
                        "favors_agent": favors_agent,
                        "strength": strength,
                    }
                except Exception as e:
                    logging.warning(f"Error analyzing phase advantage for {col}: {e}")

        timing_analysis[agent_type] = phase_advantages

    results["advantage_timing_analysis"] = timing_analysis

    # 5. Agent type-specific advantage analysis
    type_specific = {}

    for agent_type in ["system", "independent", "control"]:
        # Get simulations where this agent type was dominant
        agent_dominant = df[df["dominant_type"] == agent_type]

        if len(agent_dominant) < 5:  # Need enough samples
            continue

        # Find the strongest predictors of this agent type's dominance
        advantage_predictors = {}

        for col in advantage_cols:
            if f"{agent_type}_vs_" in col or f"_vs_{agent_type}" in col:
                try:
                    # Calculate the average value when this agent type dominates vs. when it doesn't
                    dominant_avg = agent_dominant[col].mean()
                    non_dominant_avg = df[df["dominant_type"] != agent_type][col].mean()

                    # Calculate significance of the difference
                    if (
                        len(agent_dominant[col].dropna()) > 5
                        and len(df[df["dominant_type"] != agent_type][col].dropna()) > 5
                    ):
                        t_stat, p_value = stats.ttest_ind(
                            agent_dominant[col].dropna(),
                            df[df["dominant_type"] != agent_type][col].dropna(),
                        )
                    else:
                        t_stat, p_value = 0, 1.0

                    # Determine if this advantage significantly predicts dominance
                    # Handle tuple return from scipy.stats
                    if isinstance(p_value, tuple):
                        p_value = p_value[0] if len(p_value) > 0 else 1.0
                    elif p_value is None:
                        p_value = 1.0
                    else:
                        try:
                            p_value = (
                                float(cast(float, p_value))
                                if p_value is not None
                                else 1.0
                            )
                        except (ValueError, TypeError):
                            p_value = 1.0
                    is_significant = p_value < 0.05

                    # Calculate effect size (Cohen's d)
                    try:
                        pooled_std = np.sqrt(
                            (
                                (len(agent_dominant[col].dropna()) - 1)
                                * agent_dominant[col].std() ** 2
                                + (
                                    len(
                                        df[df["dominant_type"] != agent_type][
                                            col
                                        ].dropna()
                                    )
                                    - 1
                                )
                                * df[df["dominant_type"] != agent_type][col].std() ** 2
                            )
                            / (
                                len(agent_dominant[col].dropna())
                                + len(
                                    df[df["dominant_type"] != agent_type][col].dropna()
                                )
                                - 2
                            )
                        )
                        effect_size = (dominant_avg - non_dominant_avg) / pooled_std
                    except Exception:
                        effect_size = 0

                    # Store results
                    advantage_predictors[col] = {
                        "dominant_avg": dominant_avg,
                        "non_dominant_avg": non_dominant_avg,
                        "difference": dominant_avg - non_dominant_avg,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": is_significant,
                        "effect_size": effect_size,
                    }
                except Exception as e:
                    logging.warning(f"Error analyzing advantage predictor {col}: {e}")

        # Sort predictors by effect size
        sorted_predictors = sorted(
            advantage_predictors.items(),
            key=lambda x: abs(x[1]["effect_size"]),
            reverse=True,
        )

        type_specific[agent_type] = {
            "top_predictors": {k: v for k, v in sorted_predictors[:10]},
            "significant_predictors": {
                k: v for k, v in advantage_predictors.items() if v["significant"]
            },
        }

    results["agent_type_specific_analysis"] = type_specific

    # 6. Advantage Threshold Analysis
    # Check if there are threshold values where advantages become decisive
    threshold_analysis = {}

    for agent_type in ["system", "independent", "control"]:
        score_col = f"{agent_type}_dominance_score"
        if score_col not in df.columns:
            continue

        # Find the strongest correlating advantages for this agent type
        if agent_type in results["dominance_correlations"]:
            top_advantages = sorted(
                results["dominance_correlations"][agent_type].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[
                :5
            ]  # Take top 5

            advantage_thresholds = {}

            for adv_col, corr in top_advantages:
                try:
                    # Create 10 equal-sized bins for this advantage
                    df_valid = df[df[adv_col].notna()]
                    if len(df_valid) < 20:  # Need enough samples
                        continue

                    # Get the correct sign - positive/negative correlation means different things
                    # depending on whether the agent is first or second in the advantage column
                    invert = f"_vs_{agent_type}" in adv_col
                    sign_multiplier = -1 if invert else 1

                    # Sort values
                    sorted_values = sorted(df_valid[adv_col].values)
                    step_size = len(sorted_values) // 10

                    thresholds = []
                    for i in range(1, 10):
                        threshold = sorted_values[i * step_size]

                        # Calculate dominance when advantage is above/below threshold
                        if sign_multiplier * corr > 0:  # Positive correlation
                            above_threshold = df_valid[df_valid[adv_col] >= threshold][
                                score_col
                            ].mean()
                            below_threshold = df_valid[df_valid[adv_col] < threshold][
                                score_col
                            ].mean()
                        else:  # Negative correlation
                            above_threshold = df_valid[df_valid[adv_col] <= threshold][
                                score_col
                            ].mean()
                            below_threshold = df_valid[df_valid[adv_col] > threshold][
                                score_col
                            ].mean()

                        dominance_ratio = (
                            above_threshold / below_threshold
                            if below_threshold > 0
                            else float("inf")
                        )

                        thresholds.append(
                            {
                                "threshold": threshold,
                                "dominance_above": above_threshold,
                                "dominance_below": below_threshold,
                                "dominance_ratio": dominance_ratio,
                            }
                        )

                    # Find the threshold with the highest dominance ratio
                    optimal_threshold = max(
                        thresholds, key=lambda x: x["dominance_ratio"]
                    )

                    advantage_thresholds[adv_col] = {
                        "optimal_threshold": optimal_threshold["threshold"],
                        "dominance_ratio": optimal_threshold["dominance_ratio"],
                        "all_thresholds": thresholds,
                    }
                except Exception as e:
                    logging.warning(f"Error analyzing thresholds for {adv_col}: {e}")

            threshold_analysis[agent_type] = advantage_thresholds

    results["advantage_threshold_analysis"] = threshold_analysis

    return results


def get_advantage_recommendations(
    analysis_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate recommendations based on advantage analysis.

    Parameters
    ----------
    analysis_results : Dict
        Results from analyze_advantage_patterns

    Returns
    -------
    Dict
        Dictionary containing recommendations for each agent type
    """
    # Initialize empty recommendations
    recommendations = {
        "system": {
            "key_advantages": [],
            "critical_thresholds": [],
            "phase_importance": {},
            "advantage_categories": [],
        },
        "independent": {
            "key_advantages": [],
            "critical_thresholds": [],
            "phase_importance": {},
            "advantage_categories": [],
        },
        "control": {
            "key_advantages": [],
            "critical_thresholds": [],
            "phase_importance": {},
            "advantage_categories": [],
        },
    }

    # Check if analysis_results is empty or None
    if not analysis_results:
        logging.warning("No analysis results available for generating recommendations")
        return recommendations

    # Check if required data is available
    if "agent_type_specific_analysis" not in analysis_results:
        logging.warning(
            "No agent-specific analysis available for generating recommendations"
        )
        return recommendations

    # Generate agent-specific recommendations
    for agent_type in ["system", "independent", "control"]:
        if agent_type not in analysis_results["agent_type_specific_analysis"]:
            continue

        agent_recommendations = {
            "key_advantages": [],
            "critical_thresholds": [],
            "phase_importance": {},
            "advantage_categories": [],
        }

        # 1. Find key advantages
        agent_analysis = analysis_results["agent_type_specific_analysis"][agent_type]
        if "top_predictors" in agent_analysis:
            for adv_col, data in list(agent_analysis["top_predictors"].items())[:3]:
                # Extract category and specific advantage type
                parts = adv_col.split("_")
                category = None
                specific_type = None

                for cat in [
                    "resource_acquisition",
                    "reproduction",
                    "survival",
                    "population_growth",
                    "combat",
                    "initial_positioning",
                ]:
                    if cat in adv_col:
                        category = cat
                        break

                if "early_phase" in adv_col:
                    specific_type = "early phase"
                elif "mid_phase" in adv_col:
                    specific_type = "middle phase"
                elif "late_phase" in adv_col:
                    specific_type = "late phase"
                elif "first_reproduction" in adv_col:
                    specific_type = "first reproduction timing"
                elif "success_rate" in adv_col:
                    specific_type = "success rate"

                # Determine which agent types are involved
                types_involved = []
                if "_vs_" in adv_col:
                    pair = (
                        adv_col.split("_vs_")[0]
                        + "_vs_"
                        + adv_col.split("_vs_")[1].split("_")[0]
                    )
                    types_involved = pair.split("_vs_")

                # Format a human-readable recommendation
                favors_direction = (
                    "advantage over"
                    if data["dominant_avg"] > 0
                    else "disadvantage against"
                )
                other_type = (
                    types_involved[1]
                    if types_involved[0] == agent_type
                    else types_involved[0]
                )

                recommendation = {
                    "advantage": adv_col,
                    "category": category,
                    "specific_type": specific_type,
                    "description": f"{category} {specific_type} {favors_direction} {other_type}",
                    "effect_size": data["effect_size"],
                    "significance": data["p_value"] < 0.05,
                }

                agent_recommendations["key_advantages"].append(recommendation)

        # 2. Find critical thresholds
        if (
            "advantage_threshold_analysis" in analysis_results
            and agent_type in analysis_results["advantage_threshold_analysis"]
        ):
            for adv_col, data in analysis_results["advantage_threshold_analysis"][
                agent_type
            ].items():
                if data["dominance_ratio"] > 1.5:  # Only include meaningful thresholds
                    # Extract the same information as above
                    parts = adv_col.split("_")
                    category = None
                    specific_type = None

                    for cat in [
                        "resource_acquisition",
                        "reproduction",
                        "survival",
                        "population_growth",
                        "combat",
                        "initial_positioning",
                    ]:
                        if cat in adv_col:
                            category = cat
                            break

                    if "early_phase" in adv_col:
                        specific_type = "early phase"
                    elif "mid_phase" in adv_col:
                        specific_type = "middle phase"
                    elif "late_phase" in adv_col:
                        specific_type = "late phase"

                    # Determine which agent types are involved
                    types_involved = []
                    if "_vs_" in adv_col:
                        pair = (
                            adv_col.split("_vs_")[0]
                            + "_vs_"
                            + adv_col.split("_vs_")[1].split("_")[0]
                        )
                        types_involved = pair.split("_vs_")

                    other_type = (
                        types_involved[1]
                        if types_involved[0] == agent_type
                        else types_involved[0]
                    )

                    # Format threshold description
                    threshold = data["optimal_threshold"]
                    threshold_description = (
                        f"When {category} {specific_type} advantage over {other_type} "
                        f"exceeds {threshold:.3f}, dominance likelihood increases by "
                        f"{data['dominance_ratio']:.1f}x"
                    )

                    agent_recommendations["critical_thresholds"].append(
                        {
                            "advantage": adv_col,
                            "threshold": threshold,
                            "dominance_ratio": data["dominance_ratio"],
                            "description": threshold_description,
                        }
                    )

        # 3. Analyze phase importance
        if (
            "advantage_timing_analysis" in analysis_results
            and agent_type in analysis_results["advantage_timing_analysis"]
        ):
            phase_data = analysis_results["advantage_timing_analysis"][agent_type]

            # Calculate the average strength of advantages in each phase
            phase_strengths = {}
            for phase, advantages in phase_data.items():
                if advantages:
                    strengths = [
                        abs(data["average_value"]) for data in advantages.values()
                    ]
                    if strengths:
                        phase_strengths[phase] = sum(strengths) / len(strengths)

            # Sort phases by importance
            sorted_phases = sorted(
                phase_strengths.items(), key=lambda x: x[1], reverse=True
            )

            # Create phase importance ranking
            for i, (phase, strength) in enumerate(sorted_phases):
                phase_name = (
                    "early"
                    if phase == "early"
                    else ("middle" if phase == "mid" else "late")
                )
                agent_recommendations["phase_importance"][phase_name] = {
                    "rank": i + 1,
                    "strength": strength,
                }

        # 4. Rank advantage categories
        if (
            "advantage_category_importance" in analysis_results
            and "by_category" in analysis_results["advantage_category_importance"]
        ):
            category_data = analysis_results["advantage_category_importance"][
                "by_category"
            ]

            agent_category_strengths = {}
            for category, agent_data in category_data.items():
                if agent_type in agent_data:
                    agent_category_strengths[category] = agent_data[agent_type][
                        "average_relevance"
                    ]

            # Sort categories by importance for this agent type
            sorted_categories = sorted(
                agent_category_strengths.items(), key=lambda x: x[1], reverse=True
            )

            # Add top 3 categories
            for category, strength in sorted_categories[:3]:
                agent_recommendations["advantage_categories"].append(
                    {
                        "category": category,
                        "relevance": strength,
                        "description": f"{category.replace('_', ' ')} advantages are critical for {agent_type} dominance",
                    }
                )

        recommendations[agent_type] = agent_recommendations

    return recommendations
