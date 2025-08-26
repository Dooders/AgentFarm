from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import signal
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from farm.database.data_retrieval import DataRetriever
from farm.database.database import SimulationDatabase

#! SIMULATION LEVEL ANALYSIS


def fetch_health_resource_data(db: SimulationDatabase) -> pd.DataFrame:
    """
    Fetch health and resource data from the database.

    Returns DataFrame with columns:
    - step_number
    - agent_id
    - current_health
    - resource_level
    - age
    - agent_type
    """
    query = """
    SELECT 
        as1.step_number,
        as1.agent_id,
        as1.current_health,
        as1.resource_level,
        as1.age,
        a.agent_type,
        (
            SELECT COUNT(*)
            FROM AgentActions aa 
            WHERE aa.agent_id = as1.agent_id 
            AND aa.step_number BETWEEN as1.step_number - 10 AND as1.step_number
        ) as recent_actions
    FROM AgentStates as1
    JOIN Agents a ON as1.agent_id = a.agent_id
    ORDER BY as1.step_number, as1.agent_id
    """
    return pd.read_sql_query(query, db.engine)


def calculate_cross_correlation(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Figure]:
    """
    Calculate cross-correlation between resource levels and health.
    """
    # Group by step number and calculate means
    time_series = (
        data.groupby("step_number")
        .agg({"current_health": "mean", "resource_level": "mean"})
        .reset_index()
    )

    # Calculate cross-correlation
    correlation, lags = signal.correlate(
        time_series["resource_level"], time_series["current_health"], mode="full"
    ), np.arange(-(len(time_series) - 1), len(time_series))

    # Normalize correlation
    correlation = correlation / np.max(np.abs(correlation))

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lags, correlation)
    ax.set_title("Cross-correlation: Resources vs Health")
    ax.set_xlabel("Lag (time steps)")
    ax.set_ylabel("Correlation coefficient")
    ax.grid(True)

    # Find peak correlation and lag
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_idx]
    peak_corr = correlation[peak_idx]

    results = pd.DataFrame({"peak_lag": [peak_lag], "peak_correlation": [peak_corr]})

    return results, fig


def perform_fourier_analysis(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Figure]:
    """
    Perform Fourier analysis on health and resource levels.
    """
    # Group by step number and calculate means
    time_series = (
        data.groupby("step_number")
        .agg({"current_health": "mean", "resource_level": "mean"})
        .reset_index()
    )

    # Perform FFT
    health_fft = np.fft.fft(time_series["current_health"])
    resource_fft = np.fft.fft(time_series["resource_level"])
    freq = np.fft.fftfreq(len(time_series))

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot health FFT
    ax1.plot(freq[1 : len(freq) // 2], np.abs(health_fft)[1 : len(freq) // 2])
    ax1.set_title("Health Level Frequency Components")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True)

    # Plot resource FFT
    ax2.plot(freq[1 : len(freq) // 2], np.abs(resource_fft)[1 : len(freq) // 2])
    ax2.set_title("Resource Level Frequency Components")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    plt.tight_layout()

    # Find dominant frequencies
    health_peak_freq = freq[np.argmax(np.abs(health_fft)[1 : len(freq) // 2]) + 1]
    resource_peak_freq = freq[np.argmax(np.abs(resource_fft)[1 : len(freq) // 2]) + 1]

    results = pd.DataFrame(
        {
            "metric": ["health", "resources"],
            "dominant_frequency": [health_peak_freq, resource_peak_freq],
            "period": [
                1 / health_peak_freq if health_peak_freq != 0 else np.inf,
                1 / resource_peak_freq if resource_peak_freq != 0 else np.inf,
            ],
        }
    )

    return results, fig


def build_health_prediction_model(
    data: pd.DataFrame,
) -> Tuple[LinearRegression, float, pd.DataFrame]:
    """
    Build and evaluate a multivariate regression model to predict health.
    """
    # Prepare features
    X = data[["age", "resource_level", "recent_actions"]].values
    y = data["current_health"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and fit model
    model = LinearRegression()
    model.fit(X_scaled, np.array(y))

    # Calculate R-squared
    r2_score = float(model.score(X_scaled, np.array(y)))

    # Create feature importance summary
    feature_importance = pd.DataFrame(
        {
            "feature": ["age", "resource_level", "recent_actions"],
            "coefficient": model.coef_,
        }
    )
    feature_importance["abs_importance"] = abs(feature_importance["coefficient"])
    feature_importance = feature_importance.sort_values(
        "abs_importance", ascending=False
    )

    return model, r2_score, feature_importance


def create_health_resource_visualizations(
    data: pd.DataFrame,
) -> Dict[str, Figure]:
    """
    Create comprehensive visualizations of health and resource dynamics.

    Returns:
        Dict[str, plt.matplotlib.figure.Figure]: Dictionary of named matplotlib figures
    """
    figures = {}

    # 1. Time Series Overview
    fig_time = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Main time series
    ax1 = plt.subplot(gs[0])
    time_series = (
        data.groupby("step_number")
        .agg({"current_health": "mean", "resource_level": "mean"})
        .reset_index()
    )

    ax1.plot(
        time_series["step_number"],
        time_series["current_health"],
        label="Average Health",
        color="blue",
        alpha=0.7,
    )
    ax1.plot(
        time_series["step_number"],
        time_series["resource_level"],
        label="Average Resources",
        color="green",
        alpha=0.7,
    )
    ax1.set_title("Health and Resource Levels Over Time")
    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Level")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rolling correlation
    ax2 = plt.subplot(gs[1])
    rolling_corr = (
        time_series["current_health"]
        .rolling(window=50)
        .corr(time_series["resource_level"])
    )
    ax2.plot(time_series["step_number"], rolling_corr, color="purple", alpha=0.7)
    ax2.set_title("Rolling Correlation (window=50)")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Correlation")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    figures["time_series"] = fig_time

    # 2. Health-Resource Distribution
    fig_dist = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2)

    # Health distribution by agent type
    ax1 = plt.subplot(gs[0, 0])
    sns.boxplot(data=data, x="agent_type", y="current_health", ax=ax1)
    ax1.set_title("Health Distribution by Agent Type")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Resource distribution by agent type
    ax2 = plt.subplot(gs[0, 1])
    sns.boxplot(data=data, x="agent_type", y="resource_level", ax=ax2)
    ax2.set_title("Resource Distribution by Agent Type")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # Health vs Resources scatter
    ax3 = plt.subplot(gs[1, :])
    sns.scatterplot(
        data=data.sample(n=min(1000, len(data))),
        x="resource_level",
        y="current_health",
        hue="agent_type",
        alpha=0.5,
        ax=ax3,
    )
    ax3.set_title("Health vs Resources by Agent Type")

    plt.tight_layout()
    figures["distributions"] = fig_dist

    # 3. Age Impact Analysis
    fig_age = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)

    # Health vs Age
    ax1 = plt.subplot(gs[0])
    sns.regplot(
        data=data,
        x="age",
        y="current_health",
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "red"},
        ax=ax1,
    )
    ax1.set_title("Health vs Age")

    # Resources vs Age
    ax2 = plt.subplot(gs[1])
    sns.regplot(
        data=data,
        x="age",
        y="resource_level",
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "red"},
        ax=ax2,
    )
    ax2.set_title("Resources vs Age")

    # Action frequency vs Age
    ax3 = plt.subplot(gs[2])
    sns.regplot(
        data=data,
        x="age",
        y="recent_actions",
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "red"},
        ax=ax3,
    )
    ax3.set_title("Recent Actions vs Age")

    plt.tight_layout()
    figures["age_analysis"] = fig_age

    return figures


def analyze_health_strategies(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Figure]:
    """
    Analyze agent health strategies using clustering.

    Args:
        data: DataFrame containing agent health data

    Returns:
        Tuple[pd.DataFrame, plt.Figure]: Tuple containing cluster analysis results and visualization
    """
    # Prepare time series data for clustering - handle duplicates by taking mean
    pivot_health = (
        data.groupby(["agent_id", "step_number"])["current_health"]
        .mean()  # Handle duplicates by taking mean
        .reset_index()
        .pivot(index="agent_id", columns="step_number", values="current_health")
        .ffill()
        .bfill()
    )

    # Standardize the data
    scaler = StandardScaler()
    health_scaled = scaler.fit_transform(pivot_health)

    # Perform elbow analysis
    inertias = []
    max_clusters = 10
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(health_scaled)
        inertias.append(kmeans.inertia_)

    # Find optimal k using elbow method
    # Calculate the angle between lines to find the "elbow"
    angles = []
    for i in range(1, len(inertias) - 1):
        p1 = np.array([i - 1, inertias[i - 1]])
        p2 = np.array([i, inertias[i]])
        p3 = np.array([i + 1, inertias[i + 1]])

        v1 = p2 - p1
        v2 = p3 - p2

        angle = np.abs(np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))))
        angles.append(angle)

    optimal_k = (
        angles.index(max(angles)) + 2
    )  # +2 because we start at k=1 and index starts at 0

    # Create visualization with added elbow plot
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2)

    # Plot elbow curve
    ax0 = plt.subplot(gs[0, :])
    ax0.plot(range(1, max_clusters + 1), inertias, "bo-")
    ax0.axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal k={optimal_k}")
    ax0.set_title("Elbow Method for Optimal k")
    ax0.set_xlabel("Number of Clusters (k)")
    ax0.set_ylabel("Inertia")
    ax0.grid(True, alpha=0.3)

    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(health_scaled)

    # Plot cluster centroids
    ax1 = plt.subplot(gs[1, :])
    for i in range(optimal_k):
        cluster_mean = np.mean(health_scaled[clusters == i], axis=0)
        ax1.plot(cluster_mean, label=f"Strategy {i+1}")
    ax1.set_title("Health Strategy Patterns (Cluster Centroids)")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Standardized Health")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot resource usage by cluster
    ax2 = plt.subplot(gs[2, 0])
    cluster_data = data.copy()
    cluster_data["strategy"] = pd.Series(clusters).reindex(data["agent_id"]).values
    sns.boxplot(data=cluster_data, x="strategy", y="resource_level", ax=ax2)
    ax2.set_title("Resource Usage by Strategy")
    ax2.set_xlabel("Strategy Cluster")
    ax2.set_ylabel("Resource Level")

    # Plot action frequency by cluster
    ax3 = plt.subplot(gs[2, 1])
    sns.boxplot(data=cluster_data, x="strategy", y="recent_actions", ax=ax3)
    ax3.set_title("Action Frequency by Strategy")
    ax3.set_xlabel("Strategy Cluster")
    ax3.set_ylabel("Recent Actions")

    plt.tight_layout()

    # Prepare summary statistics
    cluster_stats = (
        cluster_data.groupby("strategy")
        .agg(
            {
                "current_health": ["mean", "std"],
                "resource_level": ["mean", "std"],
                "recent_actions": ["mean", "std"],
                "agent_id": "count",
            }
        )
        .round(2)
    )

    return cluster_stats, fig


def analyze_health_resource_dynamics(db_path: str):
    """
    Main function to analyze health and resource dynamics.
    """
    # Initialize database and retriever
    db = SimulationDatabase(db_path)
    retriever = DataRetriever(db.session_manager)

    try:
        # Fetch data using DataRetriever
        print("Fetching data...")
        data = retriever.get_simulation_data()  # Get all steps

        # Extract agent states
        agent_states = pd.DataFrame(
            data["agent_states"],
            columns=[
                "agent_id",
                "agent_type",
                "position_x",
                "position_y",
                "resource_level",
                "current_health",
                "is_defending",
            ],
        )

        # Create visualizations
        print("\nGenerating visualizations...")
        figures = create_health_resource_visualizations(agent_states)

        # Save all figures
        for name, fig in figures.items():
            fig.savefig(f"{name}_analysis.png")
            plt.close(fig)

        # Perform cross-correlation analysis
        print("\nPerforming cross-correlation analysis...")
        corr_results, corr_fig = calculate_cross_correlation(agent_states)
        print("Cross-correlation results:")
        print(corr_results)
        corr_fig.savefig("cross_correlation.png")
        plt.close(corr_fig)

        # Perform Fourier analysis
        print("\nPerforming Fourier analysis...")
        fourier_results, fourier_fig = perform_fourier_analysis(agent_states)
        print("Fourier analysis results:")
        print(fourier_results)
        fourier_fig.savefig("fourier_analysis.png")
        plt.close(fourier_fig)

        # Build prediction model
        print("\nBuilding health prediction model...")
        model, r2_score, feature_importance = build_health_prediction_model(
            agent_states
        )
        print(f"Model R-squared: {r2_score:.4f}")
        print("\nFeature importance:")
        print(feature_importance)

        # Calculate basic correlations
        correlations = agent_states[
            ["current_health", "resource_level", "age", "recent_actions"]
        ].corr()
        print("\nBasic correlations with health:")
        print(correlations["current_health"])

        # Generate summary statistics
        summary = agent_states.groupby("agent_type").agg(
            {
                "current_health": ["mean", "std", "min", "max"],
                "resource_level": ["mean", "std", "min", "max"],
                "age": ["mean", "max"],
                "recent_actions": "mean",
            }
        )
        print("\nSummary statistics by agent type:")
        print(summary)

        # Add clustering analysis
        print("\nAnalyzing health strategies...")
        cluster_stats, cluster_fig = analyze_health_strategies(agent_states)
        print("\nCluster statistics:")
        print(cluster_stats)
        cluster_fig.savefig("health_strategies.png")
        plt.close(cluster_fig)

    finally:
        db.close()


if __name__ == "__main__":
    db_path = "simulations/simulation.db"
    analyze_health_resource_dynamics(db_path)
