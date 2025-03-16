import glob
import logging
import os
from datetime import datetime

# Import analysis configuration
from analysis_config import (
    DATA_PATH,
    OUTPUT_PATH,
    check_reproduction_events,
    find_latest_experiment_path,
    safe_remove_directory,
    setup_analysis_directory,
    setup_logging,
)

from farm.analysis.dominance.analyze import (
    analyze_dominance_switch_factors,
    analyze_reproduction_dominance_switching,
    analyze_simulations,
)
from farm.analysis.dominance.plot import (
    plot_comprehensive_score_breakdown,
    plot_correlation_matrix,
    plot_dominance_comparison,
    plot_dominance_distribution,
    plot_dominance_switches,
    plot_feature_importance,
    plot_reproduction_vs_dominance,
    plot_resource_proximity_vs_dominance,
)
from farm.analysis.dominance.train import train_classifier


def main():
    # Set up the dominance analysis directory
    dominance_output_path, log_file = setup_analysis_directory("dominance")

    logging.info(f"Saving results to {dominance_output_path}")

    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return

    # Check if reproduction events exist in the databases
    #! Is this still needed
    has_reproduction_events = check_reproduction_events(experiment_path)
    if not has_reproduction_events:
        logging.warning(
            "No reproduction events found in databases. Reproduction analysis may be limited."
        )

    logging.info(f"Analyzing simulations in {experiment_path}...")
    #! This should be the data gathering and processing
    df = analyze_simulations(experiment_path)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Save the raw data
    #! Is this needed???
    output_csv = os.path.join(dominance_output_path, "simulation_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

    # Check if we have reproduction data in the dataframe
    #! Is this needed???
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    if reproduction_cols:
        logging.info(
            f"Found {len(reproduction_cols)} reproduction-related columns in analysis data"
        )
        for col in reproduction_cols[:10]:  # Show first 10
            non_null = df[col].count()
            logging.info(f"  {col}: {non_null} non-null values")
    else:
        logging.warning("No reproduction columns found in analysis data")

    # Continue with the rest of the analysis
    logging.info(f"Analyzed {len(df)} simulations.")
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

    # Calculate and display dominance distributions as percentages
    logging.info("\nDominance distribution (percentages):")

    #! Need a better way to execute analysis types, or is it needed since its only logging?
    # Population dominance
    pop_counts = df["population_dominance"].value_counts()
    pop_percentages = (pop_counts / pop_counts.sum() * 100).round(1)
    logging.info("Population dominance:")
    for agent_type, percentage in pop_percentages.items():
        logging.info(f"  {agent_type}: {percentage}%")

    # Survival dominance
    surv_counts = df["survival_dominance"].value_counts()
    surv_percentages = (surv_counts / surv_counts.sum() * 100).round(1)
    logging.info("Survival dominance:")
    for agent_type, percentage in surv_percentages.items():
        logging.info(f"  {agent_type}: {percentage}%")

    # Comprehensive dominance
    if "comprehensive_dominance" in df.columns:
        comp_counts = df["comprehensive_dominance"].value_counts()
        comp_percentages = (comp_counts / comp_counts.sum() * 100).round(1)
        logging.info("Comprehensive dominance:")
        for agent_type, percentage in comp_percentages.items():
            logging.info(f"  {agent_type}: {percentage}%")

    # Dominance switching statistics
    if "total_switches" in df.columns:
        logging.info("\nDominance switching statistics:")
        logging.info(
            f"Average switches per simulation: {df['total_switches'].mean():.2f}"
        )
        logging.info(f"Average switches per step: {df['switches_per_step'].mean():.4f}")

        # Average dominance period by agent type
        logging.info("\nAverage dominance period duration (steps):")
        for agent_type in ["system", "independent", "control"]:
            avg_period = df[f"{agent_type}_avg_dominance_period"].mean()
            logging.info(f"  {agent_type}: {avg_period:.2f}")

        # Phase-specific switching
        if "early_phase_switches" in df.columns:
            logging.info("\nAverage switches by simulation phase:")
            for phase in ["early", "middle", "late"]:
                avg_switches = df[f"{phase}_phase_switches"].mean()
                logging.info(f"  {phase}: {avg_switches:.2f}")

        # Transition probabilities
        if all(
            f"{from_type}_to_{to_type}" in df.columns
            for from_type in ["system", "independent", "control"]
            for to_type in ["system", "independent", "control"]
        ):
            logging.info("\nDominance transition probabilities:")
            for from_type in ["system", "independent", "control"]:
                logging.info(f"  From {from_type}:")
                for to_type in ["system", "independent", "control"]:
                    if from_type != to_type:  # Skip self-transitions
                        prob = df[f"{from_type}_to_{to_type}"].mean()
                        logging.info(f"    To {to_type}: {prob:.2f}")

    # Plot dominance distribution
    #! make this cleaner and easier to execute
    plot_dominance_distribution(df, dominance_output_path)

    # Generate comprehensive score breakdown chart
    plot_comprehensive_score_breakdown(df, dominance_output_path)

    # Plot dominance switching patterns
    if "total_switches" in df.columns:  #! make this mandatory
        plot_dominance_switches(df, dominance_output_path)

        # Analyze factors related to dominance switching
        analyze_dominance_switch_factors(df, dominance_output_path)

        # Analyze relationship between reproduction and dominance switching
        analyze_reproduction_dominance_switching(df, dominance_output_path)

    # Plot resource proximity vs dominance
    plot_resource_proximity_vs_dominance(df, dominance_output_path)

    # Plot reproduction metrics vs dominance
    plot_reproduction_vs_dominance(df, dominance_output_path)

    # Plot correlation matrices
    for label in ["population_dominance", "survival_dominance"]:
        if label in df.columns and df[label].nunique() > 1:
            plot_correlation_matrix(df, label, dominance_output_path)

    # Plot comparison of different dominance measures
    plot_dominance_comparison(df, dominance_output_path)

    # Prepare features for classification
    #! make a seperate module for ml training feature importances
    # Exclude non-feature columns and outcome variables
    exclude_cols = [
        "iteration",
        "population_dominance",
        "survival_dominance",
        "system_agents",
        "independent_agents",
        "control_agents",
        "total_agents",
        "final_step",
    ]

    # Also exclude derived statistics columns that are outcomes, not predictors
    for prefix in ["system_", "independent_", "control_"]:
        for suffix in ["count", "alive", "dead", "avg_survival", "dead_ratio"]:
            exclude_cols.append(f"{prefix}{suffix}")

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Check if we have enough data for classification
    if len(df) > 10 and len(feature_cols) > 0:
        # Create an explicit copy to avoid SettingWithCopyWarning
        X = df[feature_cols].copy()

        # Handle missing values - separate numeric and non-numeric columns
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(exclude=["number"]).columns

        # Fill numeric columns with mean
        if not numeric_cols.empty:
            for col in numeric_cols:
                X.loc[:, col] = X[col].fillna(X[col].mean())

        # Fill categorical columns with mode (most frequent value)
        if not categorical_cols.empty:
            for col in categorical_cols:
                X.loc[:, col] = X[col].fillna(
                    X[col].mode()[0] if not X[col].mode().empty else "unknown"
                )

        # Train classifiers for each dominance type
        for label in ["population_dominance", "survival_dominance"]:
            if df[label].nunique() > 1:  # Only if we have multiple classes
                logging.info(f"\nTraining classifier for {label}...")
                y = df[label]
                clf, feat_imp = train_classifier(X, y, label)

                # Plot feature importance
                plot_feature_importance(feat_imp, label, dominance_output_path)

    logging.info("\nAnalysis complete. Results saved to CSV and PNG files.")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {dominance_output_path}")


if __name__ == "__main__":
    main()
