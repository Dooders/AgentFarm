import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from farm.analysis.common.context import AnalysisContext


def plot_dominance_distribution(df, output_path=None, ctx: AnalysisContext = None):
    """
    Plot the distribution of dominance types as percentages.
    """
    # Determine how many plots we need
    dominance_measures = [
        "population_dominance",
        "survival_dominance",
        "comprehensive_dominance",
    ]
    available_measures = [m for m in dominance_measures if m in df.columns]
    n_plots = len(available_measures)

    if n_plots == 0:
        return

    # Create a figure with the appropriate number of subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # If there's only one measure, axes will not be an array
    if n_plots == 1:
        axes = [axes]

    # Define consistent agent order and color scheme
    agent_types = ["system", "independent", "control"]
    colors = {
        "system": "blue",
        "independent": "red",
        "control": "#DAA520",
    }  # Goldenrod for control

    # Define mapping between different agent type naming formats
    agent_type_mapping = {
        "SystemAgent": "system",
        "IndependentAgent": "independent",
        "ControlAgent": "control",
        "system": "system",
        "independent": "independent",
        "control": "control",
    }

    # Plot each dominance measure
    for i, measure in enumerate(available_measures):
        # Get counts and convert to percentages
        counts = df[measure].value_counts()
        total = counts.sum()
        percentages = (counts / total) * 100

        # Log the percentages for debugging
        logging.info(f"Percentages for {measure}: {percentages}")

        # Create a consistent DataFrame with all agent types
        # This ensures the same order and includes all agent types even if some have 0%
        ordered_percentages = pd.Series(0, index=agent_types)

        # Map agent types to consistent format
        for agent in percentages.index:
            # Map the agent type to our standard format if possible
            standard_agent_type = agent_type_mapping.get(agent)

            if standard_agent_type in agent_types:
                ordered_percentages[standard_agent_type] = percentages[agent]
            else:
                logging.warning(f"Unknown agent type in {measure}: {agent}")

        # Log the ordered percentages for debugging
        logging.info(f"Ordered percentages for {measure}: {ordered_percentages}")

        # For survival dominance, ensure we have data
        if measure == "survival_dominance" and ordered_percentages.sum() == 0:
            logging.warning(f"No data for survival_dominance, using placeholder data")
            # If there's no data, check if we can derive it from the DataFrame
            if "survival_rate" in df.columns:
                # Try to derive dominance from survival rates
                for agent_type in agent_types:
                    col = f"{agent_type}_survival_rate"
                    if col in df.columns:
                        avg_survival = df[col].mean()
                        ordered_percentages[agent_type] = avg_survival * 100

                # Normalize to ensure percentages sum to 100
                if ordered_percentages.sum() > 0:
                    ordered_percentages = (
                        ordered_percentages / ordered_percentages.sum()
                    ) * 100
                    logging.info(f"Derived survival dominance: {ordered_percentages}")

        # Plot percentages with consistent colors
        bars = axes[i].bar(
            ordered_percentages.index,
            ordered_percentages.values,
            color=[colors.get(agent, "gray") for agent in ordered_percentages.index],
        )

        axes[i].set_title(f"{measure.replace('_', ' ').title()} Distribution")
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_xlabel("")

        # Add percentage labels on top of each bar
        for j, p in enumerate(ordered_percentages):
            if p > 0:  # Only add label if percentage is greater than 0
                axes[i].annotate(f"{p:.1f}%", (j, p), ha="center", va="bottom")

        # Set y-axis limit to slightly above 100% to make room for annotations
        axes[i].set_ylim(0, 105)

        # Set consistent x-tick labels with capitalized agent types
        axes[i].set_xticks(range(len(agent_types)))
        axes[i].set_xticklabels([agent.capitalize() for agent in agent_types])

    # Add caption
    caption = (
        "Percentage distribution of dominant agent types across simulations. "
        "Each bar represents the percentage of simulations where a particular agent type "
        "was dominant according to different dominance measures."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    if ctx is not None and hasattr(ctx, "output_path") and ctx.output_path:
        output_dir = ctx.output_path
    else:
        output_dir = output_path or ""
    output_file = os.path.join(output_dir, "dominance_distribution.png")
    plt.savefig(output_file)
    logging.info(f"Saved dominance distribution plot to {output_file}")
    plt.close()


def plot_feature_importance(df=None, output_path=None, feat_imp=None, label_name=None, ctx: AnalysisContext = None):
    """
    Plot feature importance for a classifier.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        DataFrame with analysis results (not used directly but required for compatibility)
    output_path : str, optional
        Directory to save the plot to. If None, the plot will be displayed but not saved.
    feat_imp : list or array, optional
        Feature importance values
    label_name : str, optional
        Name of the target label
    """
    # Check required parameters
    if feat_imp is None:
        raise ValueError("feat_imp parameter is required")
    if label_name is None:
        raise ValueError("label_name parameter is required")
    if (ctx is None or not getattr(ctx, "output_path", None)) and output_path is None:
        raise ValueError("output_path parameter is required")

    top_features = feat_imp[:15]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances, align="center")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importances for {label_name}")

    # Add caption
    caption = (
        f"Top 15 most important features that predict {label_name}. "
        f"Features with higher importance values have a stronger influence on determining "
        f"which agent type becomes dominant in the simulation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_dir = ctx.output_path if (ctx and ctx.output_path) else output_path
    output_file = os.path.join(output_dir, f"{label_name}_feature_importance.png")
    plt.savefig(output_file)
    logging.info(f"Saved feature importance plot to {output_file}")
    plt.close()


def plot_resource_proximity_vs_dominance(df, output_path=None, ctx: AnalysisContext = None):
    """
    Plot the relationship between initial resource proximity and dominance.
    """
    resource_metrics = [
        col
        for col in df.columns
        if "resource_distance" in col or "resource_proximity" in col
    ]

    if not resource_metrics:
        return

    # Create a figure with subplots for each metric
    n_metrics = len(resource_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(resource_metrics):
        if metric in df.columns:
            sns.boxplot(x="population_dominance", y=metric, data=df, ax=axes[i])
            axes[i].set_title(f"{metric} vs Population Dominance")
            axes[i].set_xlabel("Dominant Agent Type")
            axes[i].set_ylabel(metric)

    # Add caption
    caption = (
        "Relationship between initial resource proximity/distance and which agent type "
        "becomes dominant. The boxplots display the distribution of resource metrics for each dominant agent type, "
        "helping identify if certain agent types tend to dominate when resources are closer or further away."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    output_dir = ctx.output_path if (ctx and ctx.output_path) else (output_path or "")
    output_file = os.path.join(output_dir, "resource_proximity_vs_dominance.png")
    plt.savefig(output_file)
    logging.info(f"Saved resource proximity plot to {output_file}")
    plt.close()


def plot_reproduction_vs_dominance(df, output_path):
    """
    Plot reproduction metrics vs dominance.
    """
    # Get all reproduction metrics
    all_reproduction_metrics = [
        col for col in df.columns if "reproduction" in col or "offspring" in col
    ]

    if not all_reproduction_metrics:
        return

    # Filter to only include the most important metrics
    # Focus on main reproduction rates and success metrics, not derived or complex metrics
    important_metrics = []

    # Include base reproduction metrics for each agent type
    for agent_type in ["system", "independent", "control"]:
        for base_metric in [
            "reproduction_rate",
            "reproduction_success_rate",
            "offspring_count",
        ]:
            metric = f"{agent_type}_{base_metric}"
            if metric in all_reproduction_metrics:
                important_metrics.append(metric)

    # If we don't have enough basic metrics, add some of the most informative derived metrics
    if len(important_metrics) < 10:
        for metric in all_reproduction_metrics:
            # Add metrics that compare agent types directly
            if "_vs_" in metric and metric not in important_metrics:
                if len(important_metrics) < 10:
                    important_metrics.append(metric)

    # Limit to at most 10 metrics
    reproduction_metrics = important_metrics[:10]

    logging.info(
        f"Plotting {len(reproduction_metrics)} out of {len(all_reproduction_metrics)} reproduction metrics"
    )

    # Create a figure with subplots for each metric
    n_metrics = len(reproduction_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, min(5 * n_metrics, 40)))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(reproduction_metrics):
        if metric in df.columns:
            # Check if there's enough data for each category
            if df["population_dominance"].nunique() > 0 and not df[metric].isna().all():
                try:
                    # Check if each category has data
                    category_counts = df.groupby("population_dominance")[metric].count()
                    if (category_counts > 0).all():
                        sns.boxplot(
                            x="population_dominance", y=metric, data=df, ax=axes[i]
                        )
                        axes[i].set_title(f"{metric} vs Population Dominance")
                        axes[i].set_xlabel("Dominant Agent Type")
                        axes[i].set_ylabel(metric)
                    else:
                        logging.warning(
                            f"Some categories in population_dominance have no data for {metric}"
                        )
                        axes[i].text(
                            0.5,
                            0.5,
                            f"Insufficient data for {metric} boxplot",
                            ha="center",
                            va="center",
                            transform=axes[i].transAxes,
                        )
                except Exception as e:
                    logging.warning(f"Error creating boxplot for {metric}: {str(e)}")
                    axes[i].text(
                        0.5,
                        0.5,
                        f"Error creating boxplot for {metric}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
            else:
                logging.warning(f"Insufficient data for {metric} boxplot")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Insufficient data for {metric} boxplot",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

    # Add caption
    caption = (
        "Relationship between key reproduction metrics and population dominance. "
        "The boxplots show how reproduction rates and offspring counts differ across simulations where "
        "different agent types became dominant, revealing how reproductive success correlates with dominance."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    # Save a single figure with the most important metrics
    output_file = os.path.join(output_path, "reproduction_metrics_boxplots.png")
    plt.savefig(output_file)
    logging.info(f"Saved key reproduction metrics boxplots to {output_file}")
    plt.close()


def plot_correlation_matrix(df, label_name, output_path=None):
    """
    Plot correlation matrix between features and the target label.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results
    label_name : str
        Column name of the target label
    output_path : str, optional
        Directory to save the plot to. If None, the plot will be displayed but not saved.
    """
    if label_name is None or label_name not in df.columns:
        raise ValueError(
            f"label_name parameter must be a valid column name. Provided: {label_name}"
        )
    if output_path is None:
        raise ValueError("output_path parameter is required")

    # Convert categorical target to numeric for correlation
    target_numeric = pd.get_dummies(df[label_name])

    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number])

    # Remove the target if it's in the numeric features
    if label_name in numeric_features.columns:
        numeric_features = numeric_features.drop(label_name, axis=1)

    # Filter out columns with zero standard deviation to avoid division by zero warnings
    std_dev = numeric_features.std()
    valid_columns = std_dev[std_dev > 0].index
    numeric_features = numeric_features[valid_columns]

    if numeric_features.empty:
        logging.warning(
            f"No valid numeric features with non-zero standard deviation for {label_name}"
        )
        return

    # Calculate correlation with each target class
    correlations = {}
    for target_class in target_numeric.columns:
        # Filter out any NaN values in the target
        valid_mask = ~target_numeric[target_class].isna()
        if valid_mask.sum() > 0:
            correlations[target_class] = numeric_features[valid_mask].corrwith(
                target_numeric[target_class][valid_mask]
            )

    if not correlations:
        logging.warning(f"Could not calculate correlations for {label_name}")
        return

    # Combine correlations into a single DataFrame
    corr_df = pd.DataFrame(correlations)

    # Sort by absolute correlation
    corr_df["max_abs_corr"] = corr_df.abs().max(axis=1)
    corr_df = corr_df.sort_values("max_abs_corr", ascending=False).drop(
        "max_abs_corr", axis=1
    )

    # Plot top correlations
    top_n = min(20, len(corr_df))  # Ensure we don't try to plot more rows than we have
    if top_n == 0:
        logging.warning(f"No correlations to plot for {label_name}")
        return

    top_corr = corr_df.head(top_n)

    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Top {top_n} Feature Correlations with {label_name}")

    # Add caption
    caption = (
        f"Top {top_n} features most correlated with {label_name}. "
        f"Red cells indicate positive correlation (as the feature increases, the likelihood of that agent type "
        f"being dominant increases), while blue cells indicate negative correlation. "
        f"The intensity of color represents the strength of correlation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_file = os.path.join(output_path, f"{label_name}_correlation_matrix.png")
    plt.savefig(output_file)
    logging.info(f"Saved correlation matrix plot to {output_file}")
    plt.close()


def plot_dominance_comparison(df, output_path=None, ctx: AnalysisContext = None):
    """
    Create visualizations to compare different dominance measures.

    This function creates:
    1. A comparison of how often each agent type is dominant according to different measures (as percentages)
    2. A correlation heatmap between different dominance metrics
    3. A scatter plot showing the relationship between AUC and composite dominance scores

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the dominance metrics for each simulation iteration
    """
    plt.figure(figsize=(15, 10))

    # 1. Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Define dominance measures and agent types
    dominance_measures = [
        "population_dominance",
        "survival_dominance",
        "comprehensive_dominance",
    ]
    agent_types = ["system", "independent", "control"]
    colors = {"system": "blue", "independent": "green", "control": "red"}

    # Define mapping between different agent type naming formats
    agent_type_mapping = {
        "SystemAgent": "system",
        "IndependentAgent": "independent",
        "ControlAgent": "control",
        "system": "system",
        "independent": "independent",
        "control": "control",
    }

    # 2. Dominance distribution comparison (as percentages)
    ax = axes[0, 0]
    comparison_data = []

    # Calculate percentages for each measure
    for measure in dominance_measures:
        if measure in df.columns:
            counts = df[measure].value_counts()
            total = counts.sum()

            # Log the raw counts for debugging
            logging.info(f"Raw counts for {measure}: {counts}")

            # Ensure all agent types are represented, even if they have 0%
            for agent_type in agent_types:
                percentage = 0.0  # Default to 0% if not in counts

                # Check for this agent type in counts (using the mapping)
                for raw_type, count in counts.items():
                    # Map the raw agent type to our standard format if possible
                    standard_type = agent_type_mapping.get(raw_type, raw_type)
                    if standard_type == agent_type:
                        percentage = (count / total) * 100
                        break

                comparison_data.append(
                    {
                        "Measure": measure.replace("_", " ").title(),
                        "Agent Type": agent_type,
                        "Percentage": percentage,
                    }
                )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Log the processed data for debugging
        logging.info(f"Processed comparison data: {comparison_df}")

        sns.barplot(
            x="Agent Type",
            y="Percentage",
            hue="Measure",
            data=comparison_df,
            ax=ax,
        )
        ax.set_title("Dominance by Different Measures")
        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Dominance Measure")

        # Add percentage labels on top of each bar
        for p in ax.patches:
            # Only add label if percentage is greater than 0
            if p.get_height() > 0:
                ax.annotate(
                    f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                )

        # Set y-axis limit to slightly above 100% to make room for annotations
        ax.set_ylim(0, 105)

    # 3. Correlation between dominance scores
    ax = axes[0, 1]
    score_cols = [col for col in df.columns if col.endswith("_dominance_score")]

    if score_cols:
        corr = df[score_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Between Dominance Scores")

    # 4. AUC vs Dominance Score
    ax = axes[1, 0]
    for agent_type in agent_types:
        auc_col = f"{agent_type}_auc"
        score_col = f"{agent_type}_dominance_score"

        if auc_col in df.columns and score_col in df.columns:
            ax.scatter(
                df[auc_col],
                df[score_col],
                label=agent_type,
                color=colors[agent_type],
                alpha=0.6,
            )

    ax.set_xlabel("Total Agent-Steps (AUC)")
    ax.set_ylabel("Composite Dominance Score")
    ax.set_title("Relationship Between AUC and Composite Dominance Score")
    ax.legend()

    # 5. Population vs Dominance Score
    ax = axes[1, 1]
    for agent_type in agent_types:
        pop_col = f"{agent_type}_agents"  # Final population count
        score_col = f"{agent_type}_dominance_score"

        if pop_col in df.columns and score_col in df.columns:
            ax.scatter(
                df[pop_col],
                df[score_col],
                label=agent_type,
                color=colors[agent_type],
                alpha=0.6,
            )

    ax.set_xlabel("Final Population Count")
    ax.set_ylabel("Composite Dominance Score")
    ax.set_title("Final Population vs Composite Dominance Score")
    ax.legend()

    # Add caption
    caption = (
        "Top left: Percentage of simulations where each agent type is dominant according to different measures. "
        "Top right: Correlation between different dominance scores. "
        "Bottom left: Relationship between total agent-steps (AUC) and composite dominance score. "
        "Bottom right: Relationship between final population count and composite dominance score."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout and save
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_file = os.path.join(output_path, "dominance_comparison.png")
    plt.savefig(output_file, dpi=300)
    logging.info(f"Saved dominance comparison plot to {output_file}")
    plt.close()


def plot_dominance_switches(df, output_path=None, ctx: AnalysisContext = None):
    """
    Create visualizations for dominance switching patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Path to the directory where output files will be saved
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for plotting")
        return

    # 1. Distribution of total switches
    plt.figure(figsize=(10, 6))
    sns.histplot(df["total_switches"], kde=True)
    plt.title("Distribution of Dominance Switches Across Simulations")
    plt.xlabel("Number of Dominance Switches")
    plt.ylabel("Count")

    # Add caption for the first plot
    caption = (
        "How many dominance switches occurred in each simulation. "
        "A dominance switch happens when the dominant agent type changes during the simulation. "
        "The distribution reveals whether simulations typically have few or many switches in dominance."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_file = os.path.join(output_path, "dominance_switches_distribution.png")
    plt.savefig(output_file)
    plt.close()

    # 2. Average dominance period duration by agent type
    plt.figure(figsize=(10, 6))
    agent_types = ["system", "independent", "control"]
    avg_periods = [
        df[f"{agent_type}_avg_dominance_period"].mean() for agent_type in agent_types
    ]

    bars = plt.bar(agent_types, avg_periods)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.title("Average Dominance Period Duration by Agent Type")
    plt.xlabel("Agent Type")
    plt.ylabel("Average Steps")

    # Add caption for the second plot
    caption = (
        "How long each agent type typically remains dominant before being "
        "replaced by another type. Longer bars indicate that when this agent type becomes dominant, "
        "it tends to maintain dominance for more simulation steps."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_file = os.path.join(output_path, "avg_dominance_period.png")
    plt.savefig(output_file)
    plt.close()

    # 3. Phase-specific switch frequency
    if "early_phase_switches" in df.columns:
        plt.figure(figsize=(10, 6))
        phases = ["early", "middle", "late"]
        phase_data = [df[f"{phase}_phase_switches"].mean() for phase in phases]

        bars = plt.bar(phases, phase_data)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.title("Average Dominance Switches by Simulation Phase")
        plt.xlabel("Simulation Phase")
        plt.ylabel("Average Number of Switches")

        # Add caption for the third plot
        caption = (
            "How dominance switching behavior changes throughout the simulation. "
            "It displays the average number of dominance switches that occur during each phase "
            "(early, middle, and late) of the simulations, revealing when dominance is most volatile."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=(0, 0.07, 1, 0.95))
        output_file = os.path.join(output_path, "phase_switches.png")
        plt.savefig(output_file)
        plt.close()

    # 4. Transition matrix heatmap (average across all simulations)
    if all(
        f"{from_type}_to_{to_type}" in df.columns
        for from_type in agent_types
        for to_type in agent_types
    ):
        plt.figure(figsize=(10, 8))
        transition_data = np.zeros((3, 3))

        for i, from_type in enumerate(agent_types):
            for j, to_type in enumerate(agent_types):
                transition_data[i, j] = df[f"{from_type}_to_{to_type}"].mean()

        # Normalize rows to show probabilities
        row_sums = transition_data.sum(axis=1, keepdims=True)
        transition_probs = np.divide(
            transition_data,
            row_sums,
            out=np.zeros_like(transition_data),
            where=row_sums != 0,
        )

        sns.heatmap(
            transition_probs,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            xticklabels=agent_types,
            yticklabels=agent_types,
        )
        plt.title("Dominance Transition Probabilities")
        plt.xlabel("To Agent Type")
        plt.ylabel("From Agent Type")

        # Add caption for the fourth plot
        caption = (
            "Probability of transitioning from one dominant agent type to another. "
            "Each cell represents the probability that when the row agent type loses dominance, "
            "it will be replaced by the column agent type. Higher values (darker colors) indicate "
            "more common transitions between those agent types."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=(0, 0.05, 1, 0.95))
        output_file = os.path.join(output_path, "dominance_transitions.png")
        plt.savefig(output_file)
        plt.close()

    # 5. Plot dominance stability vs dominance score
    plot_dominance_stability(df, output_path)


def plot_dominance_stability(df, output_path=None, ctx: AnalysisContext = None):
    """
    Create a scatter plot showing the relationship between dominance stability
    (inverse of switches per step) and dominance score for different agent types.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Path to the directory where output files will be saved

    Returns
    -------
    str
        Path to the saved plot file
    """
    if df.empty or "switches_per_step" not in df.columns:
        logging.warning("No dominance stability data available for plotting")
        return

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

    return output_file


def plot_reproduction_advantage_vs_stability(df, output_path):
    """
    Create a visualization showing the relationship between reproduction advantage
    and dominance stability.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Path to the directory where output files will be saved

    Returns
    -------
    str
        Path to the saved plot file
    """
    if df.empty or "switches_per_step" not in df.columns:
        logging.warning("No dominance stability data available for plotting")
        return None

    # Find reproduction advantage columns
    advantage_cols = [
        col
        for col in df.columns
        if "reproduction_rate_advantage" in col
        or "reproduction_efficiency_advantage" in col
    ]

    if not advantage_cols:
        logging.warning("No reproduction advantage data available for plotting")
        return None

    # Calculate stability metric if not already present
    if "dominance_stability" not in df.columns:
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

    plt.figure(figsize=(10, 6))

    # Count how many valid columns we have for plotting
    valid_advantage_cols = []
    for col in advantage_cols:
        # Check if column has enough non-NaN values
        valid_count = df[col].notna().sum()
        if valid_count > 5 and "_vs_" in col:
            valid_advantage_cols.append(col)

    if not valid_advantage_cols:
        logging.warning("No valid advantage columns for plotting")
        plt.close()
        return None

    for i, col in enumerate(valid_advantage_cols):
        try:
            if "_vs_" in col:
                types = (
                    col.split("_vs_")[0],
                    col.split("_vs_")[1].split("_reproduction")[0],
                )
                label = f"{types[0]} vs {types[1]}"

                # Filter out NaN values
                valid_data = df[df[col].notna() & df["dominance_stability"].notna()]

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
                        logging.warning(f"Error creating trend line for {col}: {e}")
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

    # Add caption
    caption = (
        "Relationship between reproduction advantage and dominance stability. "
        "Each point represents a simulation, with reproduction advantage on the x-axis and dominance stability "
        "on the y-axis. Higher dominance stability values indicate fewer changes in which agent type is dominant. "
        "The dashed trend lines show the general relationship between reproductive advantage and stability for "
        "each agent type comparison. This visualization helps identify whether reproductive advantages correlate "
        "with more stable dominance patterns in the simulation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    output_file = os.path.join(output_path, "reproduction_advantage_stability.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(
        f"Saved reproduction advantage vs. stability analysis to {output_file}"
    )

    return output_file


def plot_comprehensive_score_breakdown(df, output_path):
    """
    Generate a chart showing the breakdown of average comprehensive dominance scores for each agent type.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results for each simulation
    output_path : str
        Path to the directory where output files will be saved
    """
    logging.info("Generating comprehensive score breakdown chart...")

    # Extract the components of the comprehensive dominance score for each agent type
    agent_types = ["system", "independent", "control"]
    components = [
        "auc",
        "recency_weighted_auc",
        "dominance_duration",
        "growth_trend",
        "final_ratio",
    ]

    # Weights used in the comprehensive dominance calculation
    weights = {
        "auc": 0.2,  # Basic population persistence
        "recency_weighted_auc": 0.3,  # Emphasize later simulation stages
        "dominance_duration": 0.2,  # Reward consistent dominance
        "growth_trend": 0.1,  # Reward positive growth in latter half
        "final_ratio": 0.2,  # Reward final state
    }

    # Create a DataFrame to store the average normalized scores for each component and agent type
    avg_scores = pd.DataFrame(index=agent_types, columns=components)

    # Calculate the average normalized score for each component and agent type
    for agent_type in agent_types:
        for component in components:
            # Get the raw component values for this agent type
            col_name = f"{agent_type}_{component}"
            if col_name in df.columns:
                # For growth_trend, we only count positive growth in the comprehensive score
                if component == "growth_trend":
                    values = df[col_name].apply(lambda x: max(0, x))
                else:
                    values = df[col_name]

                # Calculate the average
                avg_scores.loc[agent_type, component] = values.mean()
            else:
                logging.warning(f"Column {col_name} not found in DataFrame")
                avg_scores.loc[agent_type, component] = 0

    # Normalize the components for each metric across agent types
    # This is necessary because the comprehensive score uses normalized values
    for component in components:
        component_sum = avg_scores[component].sum()
        if component_sum > 0:
            avg_scores[component] = avg_scores[component] / component_sum

    # Calculate the weighted contribution of each component to the final score
    weighted_scores = pd.DataFrame(index=agent_types, columns=components)
    for component in components:
        weighted_scores[component] = avg_scores[component] * weights[component]

    # Calculate the total score for each agent type
    weighted_scores["total"] = weighted_scores.sum(axis=1)

    # Create the stacked bar chart
    plt.figure(figsize=(12, 8))

    # Set up the bar positions
    bar_width = 0.6
    index = np.arange(len(agent_types))

    # Create a colormap for the components
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 0.8, len(components)))

    # Create the stacked bars
    bottom = np.zeros(len(agent_types))
    for i, component in enumerate(components):
        plt.bar(
            index,
            weighted_scores[component],
            bar_width,
            bottom=bottom,
            label=f"{component.replace('_', ' ').title()} ({weights[component]*100:.0f}%)",
            color=colors[i],
        )
        bottom += weighted_scores[component]

    # Add the total score as text on top of each bar
    for i, agent_type in enumerate(agent_types):
        plt.text(
            i,
            bottom[i] + 0.01,
            f'Total: {weighted_scores.loc[agent_type, "total"]:.3f}',
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Customize the chart
    plt.xlabel("Agent Type", fontsize=14)
    plt.ylabel("Weighted Score Contribution", fontsize=14)
    plt.title(
        "Breakdown of Average Comprehensive Dominance Score by Agent Type", fontsize=16
    )
    plt.xticks(
        index, [agent_type.capitalize() for agent_type in agent_types], fontsize=12
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add caption
    caption = (
        "Breakdown of the comprehensive dominance score for each agent type. "
        "The comprehensive score is calculated using five weighted components: Area Under the Curve (AUC), "
        "Recency-weighted AUC, Dominance Duration, Growth Trend, and Final Population Ratio. "
        "Each component is normalized across agent types before applying weights. "
        "The height of each colored segment represents that component's contribution to the total score."
    )
    plt.figtext(
        0.5,
        0.01,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        style="italic",
        bbox=dict(facecolor="#f0f0f0", alpha=0.5, pad=5),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.95))  # Adjust layout to make room for caption

    # Save the chart
    output_file = os.path.join(output_path, "comprehensive_score_breakdown.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logging.info(f"Comprehensive score breakdown chart saved to {output_file}")

    # Also save the data to a CSV file
    csv_path = os.path.join(output_path, "comprehensive_score_breakdown.csv")
    weighted_scores.to_csv(csv_path)
    logging.info(f"Comprehensive score breakdown data saved to {csv_path}")

    return weighted_scores


def plot_reproduction_success_vs_switching(df, output_path):
    """
    Create a visualization showing the relationship between reproduction success rate and dominance switching.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    output_path : str
        Path to the directory where output files will be saved

    Returns
    -------
    str
        Path to the saved plot file
    """
    # Check if we have the necessary data
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for visualization")
        return None

    # Find reproduction success rate columns
    success_rate_cols = [
        col for col in df.columns if "reproduction_success_rate" in col
    ]

    if not success_rate_cols:
        logging.warning("No reproduction success rate data available for visualization")
        return None

    plt.figure(figsize=(12, 8))

    for i, col in enumerate(success_rate_cols):
        try:
            agent_type = col.split("_reproduction")[0]

            # Filter out NaN values before plotting
            valid_data = df.dropna(subset=[col, "total_switches"])

            if len(valid_data) < 5:  # Skip if not enough valid data
                logging.warning(f"Not enough valid data points for {col} visualization")
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
                        logging.info(f"Not enough variation in {col} for trend line")
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
        "Relationship between reproduction success rates and dominance switching "
        "for different agent types. Each panel displays a scatter plot of reproduction success rate (x-axis) versus "
        "the total number of dominance switches (y-axis) for a specific agent type. The dashed trend lines indicate "
        "the general relationship between reproductive success and dominance stability. This visualization helps identify "
        "whether higher reproduction success correlates with more or fewer changes in dominance throughout the simulation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_file = os.path.join(output_path, "reproduction_vs_switching.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved reproduction vs. switching analysis to {output_file}")

    return output_file
