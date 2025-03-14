import glob
import json
import logging
import os

import numpy as np
import pandas as pd
import sqlalchemy
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker

from farm.analysis.dominance.compute import (
    compute_comprehensive_dominance,
    compute_dominance_switches,
    compute_population_dominance,
    compute_survival_dominance,
)
from farm.analysis.dominance.data import (
    get_agent_survival_stats,
    get_final_population_counts,
    get_initial_positions_and_resources,
    get_reproduction_stats,
)


def analyze_simulations(experiment_path):
    """
    Analyze all simulation databases in the experiment folder.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases

    Returns
    -------
    pandas.DataFrame
        DataFrame with analysis results for each simulation
    """
    data = []

    # Find all simulation folders
    sim_folders: list[str] = glob.glob(os.path.join(experiment_path, "iteration_*"))

    for folder in sim_folders:
        # Check if this is a simulation folder with a database
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not (os.path.exists(db_path) and os.path.exists(config_path)):
            logging.warning(f"Skipping {folder}: Missing database or config file")
            continue

        try:
            # Extract the iteration number from the folder name
            folder_name = os.path.basename(folder)
            if folder_name.startswith("iteration_"):
                iteration = int(folder_name.split("_")[1])
            else:
                logging.warning(f"Skipping {folder}: Invalid folder name format")
                continue

            # Load the configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            # Connect to the database
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Compute dominance metrics
            population_dominance = compute_population_dominance(session)
            survival_dominance = compute_survival_dominance(session)
            comprehensive_dominance = compute_comprehensive_dominance(session)

            # Compute dominance switching metrics
            dominance_switches = compute_dominance_switches(session)

            # Get initial positions and resource data
            initial_data = get_initial_positions_and_resources(session, config)

            # Get final population counts
            final_counts = get_final_population_counts(session)

            # Get agent survival statistics
            survival_stats = get_agent_survival_stats(session)

            # Get reproduction statistics
            reproduction_stats = get_reproduction_stats(session)

            # Combine all data
            sim_data = {
                "iteration": iteration,
                "population_dominance": population_dominance,
                "survival_dominance": survival_dominance,
                "comprehensive_dominance": comprehensive_dominance["dominant_type"],
            }

            # Add dominance scores
            for agent_type in ["system", "independent", "control"]:
                sim_data[f"{agent_type}_dominance_score"] = comprehensive_dominance[
                    "scores"
                ][agent_type]
                sim_data[f"{agent_type}_auc"] = comprehensive_dominance["metrics"][
                    "auc"
                ][agent_type]
                sim_data[f"{agent_type}_recency_weighted_auc"] = (
                    comprehensive_dominance["metrics"]["recency_weighted_auc"][
                        agent_type
                    ]
                )
                sim_data[f"{agent_type}_dominance_duration"] = comprehensive_dominance[
                    "metrics"
                ]["dominance_duration"][agent_type]
                sim_data[f"{agent_type}_growth_trend"] = comprehensive_dominance[
                    "metrics"
                ]["growth_trends"][agent_type]
                sim_data[f"{agent_type}_final_ratio"] = comprehensive_dominance[
                    "metrics"
                ]["final_ratios"][agent_type]

            # Add dominance switching data
            if dominance_switches:
                sim_data["total_switches"] = dominance_switches["total_switches"]
                sim_data["switches_per_step"] = dominance_switches["switches_per_step"]

                # Add average dominance periods
                for agent_type in ["system", "independent", "control"]:
                    sim_data[f"{agent_type}_avg_dominance_period"] = dominance_switches[
                        "avg_dominance_periods"
                    ][agent_type]

                # Add phase-specific switch counts
                for phase in ["early", "middle", "late"]:
                    sim_data[f"{phase}_phase_switches"] = dominance_switches[
                        "phase_switches"
                    ][phase]

                # Add transition matrix data
                for from_type in ["system", "independent", "control"]:
                    for to_type in ["system", "independent", "control"]:
                        sim_data[f"{from_type}_to_{to_type}"] = dominance_switches[
                            "transition_probabilities"
                        ][from_type][to_type]

            # Add all other data
            sim_data.update(initial_data)
            sim_data.update(final_counts)
            sim_data.update(survival_stats)
            sim_data.update(reproduction_stats)

            data.append(sim_data)

            # Close the session
            session.close()

        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")

    # Convert to DataFrame
    return pd.DataFrame(data)


def analyze_dominance_switch_factors(df, output_path):
    """
    Analyze what factors correlate with dominance switching patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Directory path where output files will be saved

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for analysis")
        return None

    results = {}

    # 1. Correlation between initial conditions and switching frequency
    initial_condition_cols = [
        col
        for col in df.columns
        if any(x in col for x in ["initial_", "resource_", "proximity"])
    ]

    if initial_condition_cols and len(df) > 5:
        # Calculate correlations with total switches
        corr_with_switches = (
            df[initial_condition_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top positive and negative correlations
        top_positive = corr_with_switches.sort_values(ascending=False).head(5)
        top_negative = corr_with_switches.sort_values().head(5)

        results["top_positive_correlations"] = top_positive.to_dict()
        results["top_negative_correlations"] = top_negative.to_dict()

        logging.info("\nFactors associated with MORE dominance switching:")
        for factor, corr in top_positive.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

        logging.info("\nFactors associated with LESS dominance switching:")
        for factor, corr in top_negative.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

    # 2. Relationship between switching and final dominance
    if "comprehensive_dominance" in df.columns:
        # Average switches by dominant type
        switches_by_dominant = df.groupby("comprehensive_dominance")[
            "total_switches"
        ].mean()
        results["switches_by_dominant_type"] = switches_by_dominant.to_dict()

        logging.info("\nAverage dominance switches by final dominant type:")
        for agent_type, avg_switches in switches_by_dominant.items():
            logging.info(f"  {agent_type}: {avg_switches:.2f}")

    # 3. Relationship between switching and reproduction metrics
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    if reproduction_cols and len(df) > 5:
        # Calculate correlations with total switches
        repro_corr = (
            df[reproduction_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top correlations (absolute value)
        top_repro_corr = repro_corr.abs().sort_values(ascending=False).head(5)
        top_repro_factors = repro_corr[top_repro_corr.index]

        results["reproduction_correlations"] = top_repro_factors.to_dict()

        logging.info("\nReproduction factors most associated with dominance switching:")
        for factor, corr in top_repro_factors.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(f"  {factor}: {corr:.3f} ({direction} switches)")

    # 4. Create a plot showing the relationship between switching and dominance stability
    plt.figure(figsize=(10, 6))

    # Calculate stability metric (inverse of switches per step)
    df["dominance_stability"] = 1 / (
        df["switches_per_step"] + 0.01
    )  # Add small constant to avoid division by zero

    # Plot relationship between stability and dominance score for each agent type
    for agent_type in ["system", "independent", "control"]:
        score_col = f"{agent_type}_dominance_score"
        if score_col in df.columns:
            plt.scatter(
                df["dominance_stability"], df[score_col], label=agent_type, alpha=0.7
            )

    plt.xlabel("Dominance Stability (inverse of switches per step)")
    plt.ylabel("Dominance Score")
    plt.title("Relationship Between Dominance Stability and Final Dominance Score")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(output_path, "dominance_stability_analysis.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved dominance stability analysis to {output_file}")

    return results


def analyze_reproduction_dominance_switching(df, output_path):
    """
    Analyze the relationship between reproduction strategies and dominance switching patterns.

    This function examines how different reproduction metrics correlate with dominance
    switching patterns, including:
    1. How reproduction success rates affect dominance stability
    2. How reproduction timing relates to dominance switches
    3. Differences in reproduction strategies between simulations with high vs. low switching
    4. How reproduction efficiency impacts dominance patterns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Path to the directory where output files will be saved

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for reproduction-switching analysis"
        )
        return None

    # Check if we have reproduction data
    reproduction_cols = [col for col in df.columns if "reproduction" in col]

    # Log all available columns for debugging
    logging.info(f"Available columns in dataframe: {', '.join(df.columns)}")

    if not reproduction_cols:
        logging.warning("No reproduction data columns found in the analysis dataframe")
        # Check if we have any agent type columns that might indicate reproduction data
        agent_type_cols = [
            col
            for col in df.columns
            if any(agent in col for agent in ["system", "independent", "control"])
        ]
        logging.info(f"Agent-related columns: {', '.join(agent_type_cols)}")
        return None

    logging.info(f"Found reproduction columns: {', '.join(reproduction_cols)}")

    # Filter to only include numeric reproduction columns
    numeric_repro_cols = []
    for col in reproduction_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if column has enough non-zero values
                non_zero_count = (df[col] != 0).sum()
                if non_zero_count > 5:  # Need at least 5 non-zero values for analysis
                    numeric_repro_cols.append(col)
                else:
                    logging.info(
                        f"Skipping column {col} with only {non_zero_count} non-zero values"
                    )
            else:
                logging.info(f"Skipping non-numeric reproduction column: {col}")

    if not numeric_repro_cols:
        logging.warning("No numeric reproduction data columns found")
        return None

    results = {}

    # 1. Divide simulations into high and low switching groups
    median_switches = df["total_switches"].median()
    high_switching = df[df["total_switches"] > median_switches]
    low_switching = df[df["total_switches"] <= median_switches]

    logging.info(
        f"Analyzing {len(high_switching)} high-switching and {len(low_switching)} low-switching simulations"
    )

    # Compare reproduction metrics between high and low switching groups
    repro_comparison = {}
    for col in numeric_repro_cols:
        try:
            high_mean = high_switching[col].mean()
            low_mean = low_switching[col].mean()
            difference = high_mean - low_mean
            percent_diff = (
                (difference / low_mean * 100) if low_mean != 0 else float("inf")
            )

            repro_comparison[col] = {
                "high_switching_mean": high_mean,
                "low_switching_mean": low_mean,
                "difference": difference,
                "percent_difference": percent_diff,
            }
        except Exception as e:
            logging.warning(f"Error processing column {col}: {e}")

    results["reproduction_high_vs_low_switching"] = repro_comparison

    # Log the most significant differences
    logging.info(
        "\nReproduction differences between high and low switching simulations:"
    )
    sorted_diffs = sorted(
        repro_comparison.items(),
        key=lambda x: abs(x[1]["percent_difference"]),
        reverse=True,
    )

    for col, stats in sorted_diffs[:5]:  # Top 5 differences
        if abs(stats["percent_difference"]) > 10:  # Only report meaningful differences
            direction = "higher" if stats["difference"] > 0 else "lower"
            logging.info(
                f"  {col}: {stats['high_switching_mean']:.3f} vs {stats['low_switching_mean']:.3f} "
                f"({abs(stats['percent_difference']):.1f}% {direction} in high-switching simulations)"
            )

    # 2. Analyze how first reproduction timing relates to dominance switching
    first_repro_cols = [
        col for col in numeric_repro_cols if "first_reproduction_time" in col
    ]
    if first_repro_cols:
        first_repro_corr = {}
        for col in first_repro_cols:
            try:
                # Filter out -1 values (no reproduction)
                valid_data = df[df[col] > 0]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "total_switches"]].corr().iloc[0, 1]
                    first_repro_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["first_reproduction_timing_correlation"] = first_repro_corr

        logging.info(
            "\nCorrelation between first reproduction timing and dominance switches:"
        )
        for col, corr in first_repro_corr.items():
            agent_type = col.split("_first_reproduction")[0]
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(
                    f"  Earlier {agent_type} reproduction → {direction} dominance switches (r={corr:.3f})"
                )
    else:
        logging.info("No first reproduction timing data available for analysis")

    # 3. Create visualization showing reproduction success rate vs. dominance switching
    success_rate_cols = [
        col for col in numeric_repro_cols if "reproduction_success_rate" in col
    ]
    if success_rate_cols:
        plt.figure(figsize=(12, 8))

        for i, col in enumerate(success_rate_cols):
            try:
                agent_type = col.split("_reproduction")[0]

                # Filter out NaN values before plotting
                valid_data = df.dropna(subset=[col, "total_switches"])

                if len(valid_data) < 5:  # Skip if not enough valid data
                    logging.warning(
                        f"Not enough valid data points for {col} visualization"
                    )
                    continue

                plt.subplot(1, len(success_rate_cols), i + 1)

                # Create scatter plot with only valid data
                plt.scatter(
                    valid_data[col],
                    valid_data["total_switches"],
                    alpha=0.7,
                    label=agent_type,
                    c=f"C{i}",
                )

                # Add trend line - with robust error handling
                if len(valid_data) > 5:
                    try:
                        # Check if we have enough variation in the data
                        if valid_data[col].std() > 0.001:  # Need some variation
                            # Try polynomial fit with regularization
                            from sklearn.linear_model import Ridge
                            from sklearn.pipeline import make_pipeline
                            from sklearn.preprocessing import PolynomialFeatures

                            # Create a simple linear model with regularization
                            X = valid_data[col].values.reshape(-1, 1)
                            y = valid_data["total_switches"].values

                            # Make sure there are no NaN values
                            if np.isnan(X).any() or np.isnan(y).any():
                                logging.warning(
                                    f"Data for {col} still contains NaN values after filtering"
                                )
                                # Fallback to simple mean line
                                plt.axhline(
                                    y=valid_data["total_switches"].mean(),
                                    color=f"C{i}",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                            else:
                                # Use Ridge regression which is more stable
                                model = make_pipeline(
                                    PolynomialFeatures(degree=1), Ridge(alpha=1.0)
                                )
                                model.fit(X, y)

                                # Generate prediction points
                                x_plot = np.linspace(
                                    valid_data[col].min(), valid_data[col].max(), 100
                                ).reshape(-1, 1)
                                y_plot = model.predict(x_plot)

                                # Plot the trend line
                                plt.plot(x_plot, y_plot, f"C{i}--", alpha=0.8)
                        else:
                            logging.info(
                                f"Not enough variation in {col} for trend line"
                            )
                            # Fallback to simple mean line
                            plt.axhline(
                                y=valid_data["total_switches"].mean(),
                                color=f"C{i}",
                                linestyle="--",
                                alpha=0.5,
                            )
                    except Exception as e:
                        logging.warning(f"Error creating trend line for {col}: {e}")
                        # Fallback to simple mean line if trend calculation fails
                        plt.axhline(
                            y=valid_data["total_switches"].mean(),
                            color=f"C{i}",
                            linestyle="--",
                            alpha=0.5,
                        )

                plt.xlabel(f"{agent_type.capitalize()} Reproduction Success Rate")
                plt.ylabel("Total Dominance Switches")
                plt.title(f"{agent_type.capitalize()} Reproduction vs. Switching")
            except Exception as e:
                logging.warning(f"Error creating plot for {col}: {e}")

        # Add caption
        caption = (
            "This multi-panel figure shows the relationship between reproduction success rates and dominance switching "
            "for different agent types. Each panel displays a scatter plot of reproduction success rate (x-axis) versus "
            "the total number of dominance switches (y-axis) for a specific agent type. The dashed trend lines indicate "
            "the general relationship between reproductive success and dominance stability. This visualization helps identify "
            "whether higher reproduction success correlates with more or fewer changes in dominance throughout the simulation."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        output_file = os.path.join(output_path, "reproduction_vs_switching.png")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Saved reproduction vs. switching analysis to {output_file}")
    else:
        logging.info("No reproduction success rate data available for visualization")

    # 4. Analyze if reproduction efficiency correlates with dominance stability
    efficiency_cols = [
        col for col in numeric_repro_cols if "reproduction_efficiency" in col
    ]
    if efficiency_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        efficiency_stability_corr = {}
        for col in efficiency_cols:
            try:
                # Filter out rows with NaN or zero values
                valid_data = df[(df[col].notna()) & (df[col] != 0)]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "dominance_stability"]].corr().iloc[0, 1]
                    efficiency_stability_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["reproduction_efficiency_stability_correlation"] = (
            efficiency_stability_corr
        )

        logging.info(
            "\nCorrelation between reproduction efficiency and dominance stability:"
        )
        for col, corr in efficiency_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                agent_type = col.split("_reproduction")[0]
                direction = "more" if corr > 0 else "less"
                logging.info(
                    f"  Higher {agent_type} reproduction efficiency → {direction} stable dominance (r={corr:.3f})"
                )

    # 5. Analyze reproduction advantage and dominance switching
    advantage_cols = [
        col
        for col in numeric_repro_cols
        if "reproduction_rate_advantage" in col
        or "reproduction_efficiency_advantage" in col
    ]
    if advantage_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        if "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        advantage_stability_corr = {}
        for col in advantage_cols:
            try:
                # Filter out rows with NaN values
                valid_data = df[df[col].notna()]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "dominance_stability"]].corr().iloc[0, 1]
                    advantage_stability_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["reproduction_advantage_stability_correlation"] = (
            advantage_stability_corr
        )

        logging.info(
            "\nCorrelation between reproduction advantage and dominance stability:"
        )
        for col, corr in advantage_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                if "_vs_" in col:
                    types = (
                        col.split("_vs_")[0],
                        col.split("_vs_")[1].split("_reproduction")[0],
                    )
                    direction = "more" if corr > 0 else "less"
                    logging.info(
                        f"  {types[0]} advantage over {types[1]} → {direction} stable dominance (r={corr:.3f})"
                    )

        # Create visualization of reproduction advantage vs. stability
        plt.figure(figsize=(10, 6))

        # Count how many valid columns we have for plotting
        valid_advantage_cols = []
        for col in advantage_cols:
            # Check if column has enough non-NaN values
            valid_count = df[col].notna().sum()
            if valid_count > 5 and "_vs_" in col:
                valid_advantage_cols.append(col)

        if valid_advantage_cols:
            for i, col in enumerate(valid_advantage_cols):
                try:
                    if "_vs_" in col:
                        types = (
                            col.split("_vs_")[0],
                            col.split("_vs_")[1].split("_reproduction")[0],
                        )
                        label = f"{types[0]} vs {types[1]}"

                        # Filter out NaN values
                        valid_data = df[
                            df[col].notna() & df["dominance_stability"].notna()
                        ]

                        if len(valid_data) < 5:  # Skip if not enough valid data
                            logging.warning(
                                f"Not enough valid data points for {col} visualization"
                            )
                            continue

                        plt.scatter(
                            valid_data[col],
                            valid_data["dominance_stability"],
                            alpha=0.7,
                            label=label,
                        )

                        # Add trend line - with robust error handling
                        if len(valid_data) > 5:
                            try:
                                # Check if we have enough variation in the data
                                if valid_data[col].std() > 0.001:  # Need some variation
                                    # Use robust regression
                                    from sklearn.linear_model import RANSACRegressor

                                    # Create a robust regression model
                                    X = valid_data[col].values.reshape(-1, 1)
                                    y = valid_data["dominance_stability"].values

                                    # Double-check for NaN values
                                    if np.isnan(X).any() or np.isnan(y).any():
                                        logging.warning(
                                            f"Data for {col} still contains NaN values after filtering"
                                        )
                                        # Fallback to horizontal line at mean
                                        plt.axhline(
                                            y=valid_data["dominance_stability"].mean(),
                                            linestyle="--",
                                            alpha=0.3,
                                        )
                                    else:
                                        # RANSAC is robust to outliers
                                        model = RANSACRegressor(random_state=42)
                                        model.fit(X, y)

                                        # Generate prediction points
                                        x_sorted = np.sort(X, axis=0)
                                        y_pred = model.predict(x_sorted)

                                        # Plot the trend line
                                        plt.plot(x_sorted, y_pred, "--", alpha=0.6)
                                else:
                                    logging.info(
                                        f"Not enough variation in {col} for trend line"
                                    )
                                    # Fallback to horizontal line at mean
                                    plt.axhline(
                                        y=valid_data["dominance_stability"].mean(),
                                        linestyle="--",
                                        alpha=0.3,
                                    )
                            except Exception as e:
                                logging.warning(
                                    f"Error creating trend line for {col}: {e}"
                                )
                                # Fallback to horizontal line at mean
                                plt.axhline(
                                    y=valid_data["dominance_stability"].mean(),
                                    linestyle="--",
                                    alpha=0.3,
                                )
                except Exception as e:
                    logging.warning(f"Error creating plot for {col}: {e}")

            plt.xlabel("Reproduction Advantage")
            plt.ylabel("Dominance Stability")
            plt.title("Reproduction Advantage vs. Dominance Stability")
            plt.legend()
            plt.tight_layout()

            output_file = os.path.join(
                output_path, "reproduction_advantage_stability.png"
            )
            plt.savefig(output_file)
            plt.close()
            logging.info(
                f"Saved reproduction advantage vs. stability analysis to {output_file}"
            )
        else:
            logging.warning("No valid advantage columns for plotting")
            plt.close()

    # 6. Analyze relationship between reproduction metrics and dominance switching by agent type
    if "comprehensive_dominance" in df.columns:
        # Group by dominant agent type
        for agent_type in ["system", "independent", "control"]:
            type_data = df[df["comprehensive_dominance"] == agent_type]
            if len(type_data) > 5:  # Need enough data points
                type_results = {}

                # Calculate correlations between reproduction metrics and switching for this agent type
                for col in numeric_repro_cols:
                    try:
                        # Filter out NaN values
                        valid_data = type_data[
                            type_data[col].notna() & type_data["total_switches"].notna()
                        ]
                        if len(valid_data) > 5:  # Need enough data points
                            corr = valid_data[[col, "total_switches"]].corr().iloc[0, 1]
                            if not np.isnan(corr):
                                type_results[col] = corr
                        else:
                            logging.info(
                                f"Not enough valid data points for {col} in {agent_type}-dominant simulations"
                            )
                    except Exception as e:
                        logging.warning(
                            f"Error calculating correlation for {col} in {agent_type}-dominant simulations: {e}"
                        )

                # Add to results
                results[f"{agent_type}_dominance_reproduction_correlations"] = (
                    type_results
                )

                # Log top correlations
                if type_results:
                    logging.info(
                        f"\nTop reproduction factors affecting switching in {agent_type}-dominant simulations:"
                    )
                    sorted_corrs = sorted(
                        type_results.items(), key=lambda x: abs(x[1]), reverse=True
                    )
                    for col, corr in sorted_corrs[:3]:  # Top 3
                        if abs(corr) > 0.2:  # Only report stronger correlations
                            direction = "more" if corr > 0 else "fewer"
                            logging.info(f"  {col}: {corr:.3f} ({direction} switches)")

    return results
