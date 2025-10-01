#!/usr/bin/env python
"""
Advantage Analysis Script

This script analyzes advantages between agent types across simulations and
generates visualizations and reports showing how different advantages contribute to dominance.

Usage:
    python advantage_analysis.py

The script will:
1. Analyze advantages in the most recent experiment
2. Generate visualizations for different advantage categories
3. Create a comprehensive analysis report
"""

import glob
import os
import time
from datetime import datetime

import numpy as np

# Import analysis configuration
from analysis_config import DATA_PATH, OUTPUT_PATH, safe_remove_directory, setup_logging

from farm.analysis.advantage.analyze import (
    analyze_advantage_patterns,
    analyze_advantages,
    get_advantage_recommendations,
)
from farm.analysis.advantage.plot import plot_advantage_results
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    start_time = time.time()
    logger.info("Starting advantage analysis script")

    try:
        # Create advantage output directory
        adv_output_path = os.path.join(OUTPUT_PATH, "advantage")

        # Clear the advantage directory if it exists
        if os.path.exists(adv_output_path):
            logger.info(f"Clearing existing advantage directory: {adv_output_path}")
            if not safe_remove_directory(adv_output_path):
                # If we couldn't remove the directory after retries, create a new one with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                adv_output_path = os.path.join(OUTPUT_PATH, f"advantage_{timestamp}")
                logger.info(f"Using alternative directory: {adv_output_path}")

        # Create the directory
        os.makedirs(adv_output_path, exist_ok=True)

        # Set up logging to the advantage directory
        log_file = setup_logging(adv_output_path)

        logger.info(f"Saving results to {adv_output_path}")

        # Find the most recent experiment folder in DATA_PATH
        logger.info(f"Searching for experiment folders in {DATA_PATH}")
        experiment_folders = [
            d for d in glob.glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(d)
        ]
        if not experiment_folders:
            logger.error(f"No experiment folders found in {DATA_PATH}")
            return

        # Sort by modification time (most recent first)
        experiment_folders.sort(key=os.path.getmtime, reverse=True)
        experiment_path = experiment_folders[0]
        logger.info(f"Found most recent experiment folder: {experiment_path}")

        # Check if experiment_path contains iteration folders directly
        logger.info(f"Checking for iteration folders in {experiment_path}")
        iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
        if not iteration_folders:
            # If no iteration folders found directly, look for subdirectories that might contain them
            logger.info("No iteration folders found directly, checking subdirectories")
            subdirs = [
                d
                for d in glob.glob(os.path.join(experiment_path, "*"))
                if os.path.isdir(d)
            ]
            if subdirs:
                # Sort by modification time (most recent first)
                subdirs.sort(key=os.path.getmtime, reverse=True)
                experiment_path = subdirs[0]
                logger.info(f"Using subdirectory as experiment path: {experiment_path}")

                # Verify that this subdirectory contains iteration folders
                iteration_folders = glob.glob(
                    os.path.join(experiment_path, "iteration_*")
                )
                if not iteration_folders:
                    logger.error(f"No iteration folders found in {experiment_path}")
                    return
            else:
                logger.error(f"No subdirectories found in {experiment_path}")
                return

        logger.info(f"Found {len(iteration_folders)} iteration folders")

        # Step 1: Collect advantage data from all simulations
        try:
            logger.info("Starting advantage data collection")
            analysis_start_time = time.time()

            # First collect the advantage data from simulations
            df_original = analyze_advantages(experiment_path)

            analysis_duration = time.time() - analysis_start_time
            logger.info(
                f"Completed advantage data collection in {analysis_duration:.2f} seconds"
            )

            if df_original.empty:
                logger.warning("No simulation data found.")
                return

            logger.info(f"Collected data from {len(df_original)} simulations")

            # Save the raw data
            logger.info("Saving raw analysis data to CSV")
            output_csv = os.path.join(adv_output_path, "advantage_analysis.csv")
            df_original.to_csv(output_csv, index=False)
            logger.info(f"Saved analysis data to {output_csv}")

        except Exception as e:
            logger.error(f"Error in data collection step: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return

        # Step 2: Analyze patterns in advantages
        try:
            logger.info("Starting to analyze patterns in advantages...")
            patterns_start_time = time.time()

            # Now analyze patterns in the collected data
            analysis_results = analyze_advantage_patterns(df_original)

            patterns_duration = time.time() - patterns_start_time
            logger.info(
                f"Completed advantage pattern analysis in {patterns_duration:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error in pattern analysis step: {e}")
            import traceback

            logger.error(traceback.format_exc())
            analysis_results = (
                {}
            )  # Use empty dict to continue with limited functionality

        # Save analysis results
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            return obj

        # Process the analysis results to make them JSON-serializable
        def process_dict(d):
            """
            Recursively process a dictionary to make all values JSON-serializable.
            Handles NumPy types, NaN, Infinity, and other special values.
            """
            if isinstance(d, dict):
                return {k: process_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [process_dict(item) for item in d]
            elif isinstance(d, np.integer):
                return int(d)
            elif isinstance(d, np.floating):
                # Handle NaN and Infinity
                if np.isnan(d):
                    return 0  # Replace NaN with 0 instead of "NaN" string
                elif np.isinf(d):
                    return 1.0 if d > 0 else -1.0  # Replace Infinity with 1 or -1
                return float(d)
            elif isinstance(d, np.ndarray):
                return d.tolist()
            elif isinstance(d, np.bool_):
                return bool(d)
            elif d is None:
                return None
            # Handle other NumPy types
            elif isinstance(d, np.generic):
                return d.item()
            else:
                return convert_for_json(d)

        logger.info("Processing analysis results for JSON serialization")
        serializable_results = process_dict(analysis_results)

        logger.info("Saving advantage patterns to JSON")
        output_json = os.path.join(
            adv_output_path, "advantage_patterns.json", encoding="utf-8"
        )
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved advantage patterns analysis to {output_json}")
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Try to identify the problematic value
            logger.error("Attempting to identify problematic values...")

            def find_non_serializable(obj, path="root"):
                """Recursively find non-serializable values in a nested structure"""
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        find_non_serializable(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        find_non_serializable(v, f"{path}[{i}]")
                else:
                    try:
                        json.dumps(obj)
                    except TypeError:
                        logger.error(
                            f"Non-serializable value at {path}: {type(obj)} - {obj}"
                        )

            find_non_serializable(serializable_results)

            # Save a simplified version without the problematic values
            logger.info("Attempting to save a simplified version...")
            try:
                # Convert to JSON string with error handling for each value
                class SafeEncoder(json.JSONEncoder):
                    def default(self, obj):
                        try:
                            return super().default(obj)
                        except TypeError:
                            return str(obj)

                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=2, cls=SafeEncoder)
                logger.info(
                    f"Saved simplified advantage patterns analysis to {output_json}"
                )
            except Exception as e2:
                logger.error(f"Failed to save simplified version: {e2}")
        except Exception as e:
            logger.error(f"Error saving advantage patterns: {e}")

        # Step 3: Generate visualizations
        try:
            logger.info("Starting to generate visualizations...")
            viz_start_time = time.time()

            # Modify the plot function call to include a flag indicating data has been cleaned
            plot_advantage_results(
                df_original, analysis_results, adv_output_path, data_cleaned=True
            )

            viz_duration = time.time() - viz_start_time
            logger.info(
                f"Completed visualization generation in {viz_duration:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error in visualization generation: {e}")
            import traceback

            logger.error(traceback.format_exc())

        # Step 4: Generate recommendations
        try:
            logger.info(
                "Starting to generate recommendations based on advantage analysis..."
            )
            rec_start_time = time.time()
            recommendations = get_advantage_recommendations(analysis_results)

            rec_duration = time.time() - rec_start_time
            logger.info(
                f"Completed recommendation generation in {rec_duration:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            import traceback

            logger.error(traceback.format_exc())
            recommendations = (
                {}
            )  # Use empty dict to continue with limited functionality

        # Save recommendations
        logger.info("Saving recommendations to JSON")
        recommendations_json = os.path.join(
            adv_output_path, "advantage_recommendations.json", encoding="utf-8"
        )
        try:
            with open(recommendations_json, "w", encoding="utf-8") as f:
                json.dump(process_dict(recommendations), f, indent=2)
            logger.info(f"Saved advantage recommendations to {recommendations_json}")
        except TypeError as e:
            logger.error(f"JSON serialization error for recommendations: {e}")
            # Try to save with the SafeEncoder
            try:
                with open(recommendations_json, "w", encoding="utf-8") as f:
                    json.dump(
                        process_dict(recommendations), f, indent=2, cls=SafeEncoder
                    )
                logger.info(
                    f"Saved simplified recommendations to {recommendations_json}"
                )
            except Exception as e2:
                logger.error(f"Failed to save simplified recommendations: {e2}")
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")

        # Generate a summary report
        try:
            logger.info("Generating summary report")
            report_start_time = time.time()
            generate_summary_report(
                df_original, analysis_results, recommendations, adv_output_path
            )
            report_duration = time.time() - report_start_time
            logger.info(
                f"Completed summary report generation in {report_duration:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error in summary report generation: {e}")
            import traceback

            logger.error(traceback.format_exc())

        total_duration = time.time() - start_time
        logger.info(
            f"\nAnalysis complete. Total execution time: {total_duration:.2f} seconds"
        )
        logger.info("Results saved to CSV, JSON, and PNG files.")
        logger.info(f"Log file saved to: {log_file}")
        logger.info(f"All analysis files saved to: {adv_output_path}")

    except Exception as e:
        logger.error(f"Unhandled exception in main function: {e}")
        import traceback

        logger.error(traceback.format_exc())


def generate_summary_report(df, analysis_results, recommendations, output_path):
    """Generate a summary report of the advantage analysis."""
    logger.info("Starting summary report generation")
    report_path = os.path.join(output_path, "advantage_report.md", encoding="utf-8")

    # Ensure recommendations is a dictionary
    if recommendations is None:
        logger.warning("Recommendations is None, using empty dictionary instead")
        recommendations = {}

    # Open file with UTF-8 encoding
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Advantage Analysis Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Note about data cleaning
        f.write("## ⚠️ Data Processing Note\n\n")
        f.write(
            "This report is based on data that has been processed to remove NaN (Not a Number) "
        )
        f.write(
            "and Infinity values to improve visualization quality. The raw data contained mathematical "
        )
        f.write(
            "anomalies that would otherwise make charts unreadable. This processing may affect "
        )
        f.write("the precision of some insights.\n\n")

        # 1. Overview
        logger.info("Writing overview section")
        f.write("## 1. Overview\n\n")
        f.write(
            f"This report analyzes data from {len(df)} simulations to understand how advantages between agent types influence dominance patterns.\n\n"
        )

        # Show dominance distribution
        if "dominant_type" in df.columns:
            f.write("### Dominance Distribution\n\n")
            dominance_counts = df["dominant_type"].value_counts()
            total = dominance_counts.sum()
            f.write("| Agent Type | Count | Percentage |\n")
            f.write("|------------|-------|------------|\n")
            for agent_type, count in dominance_counts.items():
                percentage = (count / total) * 100
                f.write(
                    f"| {agent_type.capitalize()} | {count} | {percentage:.1f}% |\n"
                )
            f.write("\n")

        # 2. Key Findings
        logger.info("Writing key findings section")
        f.write("## 2. Key Findings\n\n")

        # Category importance
        if (
            analysis_results
            and "advantage_category_importance" in analysis_results
            and "overall_ranking" in analysis_results["advantage_category_importance"]
        ):
            f.write("### Most Important Advantage Categories\n\n")
            overall_ranking = analysis_results["advantage_category_importance"][
                "overall_ranking"
            ]
            f.write("Categories ranked by overall importance:\n\n")
            for i, (category, importance) in enumerate(overall_ranking.items(), 1):
                # Ensure importance is not NaN or Infinity
                if isinstance(importance, float) and (
                    np.isnan(importance) or np.isinf(importance)
                ):
                    importance = 0.0
                f.write(
                    f"{i}. **{category.replace('_', ' ').title()}** (correlation strength: {importance:.3f})\n"
                )
            f.write("\n")

        # 3. Agent-Specific Insights
        logger.info("Writing agent-specific insights section")
        f.write("## 3. Agent-Specific Insights\n\n")

        for agent_type in ["system", "independent", "control"]:
            if agent_type in recommendations:
                f.write(f"### {agent_type.capitalize()} Agent\n\n")

                # Key advantages
                if (
                    "key_advantages" in recommendations[agent_type]
                    and recommendations[agent_type]["key_advantages"]
                ):
                    f.write("#### Key Advantages\n\n")
                    for adv in recommendations[agent_type]["key_advantages"]:
                        # Use "Yes"/"No" instead of Unicode symbols
                        sig_marker = "Yes" if adv.get("significance", False) else "No"

                        # Handle problematic values
                        effect_size = adv.get("effect_size", 0)
                        if isinstance(effect_size, str) or (
                            isinstance(effect_size, float)
                            and (np.isnan(effect_size) or np.isinf(effect_size))
                        ):
                            effect_size = 0.0

                        f.write(
                            f"- {adv.get('description', 'Unknown advantage')} (Effect size: {effect_size:.3f}, Significant: {sig_marker})\n"
                        )
                    f.write("\n")

                # Critical thresholds
                if (
                    "critical_thresholds" in recommendations[agent_type]
                    and recommendations[agent_type]["critical_thresholds"]
                ):
                    f.write("#### Critical Thresholds\n\n")
                    for threshold in recommendations[agent_type]["critical_thresholds"]:
                        # Clean up infinity values in description
                        desc = threshold.get("description", "Unknown threshold")
                        desc = desc.replace("infx", "significantly")
                        f.write(f"- {desc}\n")
                    f.write("\n")

                # Phase importance
                if (
                    "phase_importance" in recommendations[agent_type]
                    and recommendations[agent_type]["phase_importance"]
                ):
                    f.write("#### Phase Importance\n\n")
                    phases = sorted(
                        recommendations[agent_type]["phase_importance"].items(),
                        key=lambda x: x[1].get("rank", 0),
                    )
                    f.write("Phases ranked by importance:\n\n")
                    for phase, data in phases:
                        rel_strength = data.get("relative_strength", 0)
                        if isinstance(rel_strength, float) and (
                            np.isnan(rel_strength) or np.isinf(rel_strength)
                        ):
                            rel_strength = 0.0

                        f.write(
                            f"{data.get('rank', 0)}. **{phase.capitalize()} Phase** (relative strength: {rel_strength:.3f})\n"
                        )
                    f.write("\n")

                # Category importance
                if (
                    "advantage_categories" in recommendations[agent_type]
                    and recommendations[agent_type]["advantage_categories"]
                ):
                    f.write("#### Important Advantage Categories\n\n")
                    for category in recommendations[agent_type]["advantage_categories"]:
                        relevance = category.get("relevance", 0)
                        if isinstance(relevance, str) or (
                            isinstance(relevance, float)
                            and (np.isnan(relevance) or np.isinf(relevance))
                        ):
                            relevance = 0.0

                        f.write(
                            f"- {category.get('description', 'Unknown category')} (relevance: {relevance:.3f})\n"
                        )
                    f.write("\n")

        # 4. Comparative Analysis
        logger.info("Writing comparative analysis section")
        f.write("## 4. Comparative Analysis\n\n")

        # Show overall composite advantages
        composite_cols = [
            col
            for col in df.columns
            if "composite_advantage" in col and "contribution" not in col
        ]
        if composite_cols:
            f.write("### Composite Advantage Scores\n\n")
            f.write("| Agent Pair | Average Composite Advantage |\n")
            f.write("|------------|-----------------------------|\n")

            for col in composite_cols:
                pair = col.replace("_composite_advantage", "").replace("_", " vs ")
                avg_value = df[col].mean()
                if np.isnan(avg_value) or np.isinf(avg_value):
                    avg_value = 0.0
                f.write(f"| {pair.title()} | {avg_value:.4f} |\n")
            f.write("\n")

        # 5. Conclusions and Recommendations
        logger.info("Writing conclusions and recommendations section")
        f.write("## 5. Conclusions and Recommendations\n\n")

        # Generate overall conclusions
        f.write("### Overall Conclusions\n\n")

        # Most important category
        if (
            analysis_results
            and "advantage_category_importance" in analysis_results
            and "overall_ranking" in analysis_results["advantage_category_importance"]
            and analysis_results["advantage_category_importance"]["overall_ranking"]
        ):
            top_category = next(
                iter(
                    analysis_results["advantage_category_importance"]["overall_ranking"]
                )
            )
            f.write(
                f"- **{top_category.replace('_', ' ').title()}** is the most important category of advantage for determining dominance outcomes.\n"
            )

        # Phase importance
        timing_data = {}
        for agent_type in ["system", "independent", "control"]:
            if (
                agent_type in recommendations
                and "phase_importance" in recommendations[agent_type]
                and recommendations[agent_type]["phase_importance"]
            ):
                phases = sorted(
                    recommendations[agent_type]["phase_importance"].items(),
                    key=lambda x: x[1].get("rank", 0),
                )
                if phases:
                    timing_data[agent_type] = phases[0][0]  # Most important phase

        if timing_data:
            f.write(
                "- The most critical simulation phase for establishing dominance varies by agent type:\n"
            )
            for agent_type, phase in timing_data.items():
                f.write(
                    f"  - **{agent_type.capitalize()}**: {phase.capitalize()} phase\n"
                )

        f.write("\n### Strategic Recommendations\n\n")

        # Generate strategic recommendations for each agent type
        for agent_type in ["system", "independent", "control"]:
            if (
                agent_type in recommendations
                and "key_advantages" in recommendations[agent_type]
                and recommendations[agent_type]["key_advantages"]
            ):
                f.write(f"#### For {agent_type.capitalize()} Agents\n\n")

                # Get top advantage
                top_advantage = recommendations[agent_type]["key_advantages"][0]
                f.write(
                    f"- Focus on {top_advantage.get('description', 'key advantages')}.\n"
                )

                # Get most important phase
                if (
                    "phase_importance" in recommendations[agent_type]
                    and recommendations[agent_type]["phase_importance"]
                ):
                    phases = sorted(
                        recommendations[agent_type]["phase_importance"].items(),
                        key=lambda x: x[1].get("rank", 0),
                    )
                    if phases:
                        top_phase = phases[0][0]
                        f.write(
                            f"- Prioritize establishing advantages in the {top_phase} phase of the simulation.\n"
                        )

                # Get critical threshold if available
                if (
                    "critical_thresholds" in recommendations[agent_type]
                    and recommendations[agent_type]["critical_thresholds"]
                ):
                    threshold = recommendations[agent_type]["critical_thresholds"][0]
                    # Clean up infinity values in description
                    desc = threshold.get(
                        "description", "Reach critical advantage thresholds"
                    )
                    desc = desc.replace("infx", "significantly")
                    f.write(f"- {desc}.\n")

                f.write("\n")

        # If no recommendations were available, add a note
        if not recommendations:
            f.write("No specific recommendations available from the analysis.\n\n")
            f.write("Please refer to the visualizations and raw data for insights.\n")

    logger.info(f"Generated summary report at {report_path}")


if __name__ == "__main__":
    main()
