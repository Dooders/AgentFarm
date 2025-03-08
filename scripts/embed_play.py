import os

# Set environment variable to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

"""
Configuration Embedding and Analysis Script

This script analyzes simulation configurations by embedding them into a lower-dimensional space
and performing various analyses to understand relationships between different configurations.

Key steps in the embedding process:

1. Feature Extraction:
   - Extracts numerical features from each configuration including:
     * Learning parameters (learning_rate, gamma)
     * Agent ratios (system, independent, control)
     * Behavior parameters (gather_efficiency, share_weight, attack_weight, etc.)

2. Feature Processing:
   - Normalizes features to [0,1] range
   - Applies importance weights to balance feature influence
   - Converts to numerical vectors using DictVectorizer

3. Dimensionality Reduction:
   - Uses PCA to reduce to 3 dimensions while preserving key variations
   - Explains configuration relationships through principal components
   - Visualizes configurations in 3D space

4. Analysis Methods:
   - Similarity analysis using cosine similarity
   - Cluster analysis using K-means
   - Parameter sensitivity analysis
   - Cross-validation for stability assessment

Visualizations Generated:
- Configuration similarity heatmap
- 3D PCA visualization (static and interactive)
- Cluster visualization
- Parameter correlation heatmap

The embedding reveals:
- Primary strategic dimensions (aggressive vs cooperative)
- Learning and control variations
- Resource management patterns
- Key parameter relationships and sensitivities

Output files:
- config_similarity_heatmap.png
- config_pca_visualization_3d.png
- interactive_pca_visualization.html
- cluster_visualization.html
- parameter_correlation_heatmap.png
"""


# Set up logging configuration
logging.basicConfig(
    filename="config_embedding.log",
    level=logging.INFO,
    format="%(message)s",  # Simple format since we're just logging analysis results
)


# Load and prepare simulation settings from config variations
def create_config_variation(base_config, variation_name, **changes):
    config = base_config.copy()
    config["variation_name"] = variation_name
    for key, value in changes.items():
        config[key] = value
    return config


# Load base config
with open("simulations/config.json", "r") as f:
    base_config = json.load(f)

# Create meaningful variations of the config with more distinct behavioral differences
simulation_settings = [
    create_config_variation(
        base_config,
        "Baseline",
        learning_rate=0.001,
        gamma=0.95,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.4,
                "gather_cost_multiplier": 0.4,
                "min_resource_threshold": 0.2,
                "share_weight": 0.3,
                "attack_weight": 0.05,
            }
        },
        agent_type_ratios={
            "SystemAgent": 1.0,
            "IndependentAgent": 0.0,
            "ControlAgent": 0.0,
        },
    ),
    create_config_variation(
        base_config,
        "Aggressive",
        learning_rate=0.002,
        gamma=0.98,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.3,
                "gather_cost_multiplier": 0.5,
                "min_resource_threshold": 0.1,
                "share_weight": 0.1,
                "attack_weight": 0.4,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.3,
            "IndependentAgent": 0.7,
            "ControlAgent": 0.0,
        },
    ),
    create_config_variation(
        base_config,
        "Cooperative",
        learning_rate=0.001,
        gamma=0.99,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.5,
                "gather_cost_multiplier": 0.3,
                "min_resource_threshold": 0.3,
                "share_weight": 0.6,
                "attack_weight": 0.02,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.8,
            "IndependentAgent": 0.0,
            "ControlAgent": 0.2,
        },
    ),
    create_config_variation(
        base_config,
        "Fast Learners",
        learning_rate=0.005,
        gamma=0.90,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.6,
                "gather_cost_multiplier": 0.2,
                "min_resource_threshold": 0.15,
                "share_weight": 0.2,
                "attack_weight": 0.15,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.5,
            "IndependentAgent": 0.3,
            "ControlAgent": 0.2,
        },
    ),
    create_config_variation(
        base_config,
        "Resource Focused",
        learning_rate=0.001,
        gamma=0.97,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.8,
                "gather_cost_multiplier": 0.2,
                "min_resource_threshold": 0.25,
                "share_weight": 0.3,
                "attack_weight": 0.1,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.6,
            "IndependentAgent": 0.2,
            "ControlAgent": 0.2,
        },
    ),
    create_config_variation(
        base_config,
        "Hybrid Learner",
        learning_rate=0.003,
        gamma=0.96,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.5,
                "gather_cost_multiplier": 0.3,
                "min_resource_threshold": 0.2,
                "share_weight": 0.4,
                "attack_weight": 0.2,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.4,
            "IndependentAgent": 0.3,
            "ControlAgent": 0.3,
        },
    ),
    create_config_variation(
        base_config,
        "Extreme Specialist",
        learning_rate=0.001,
        gamma=0.99,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.9,
                "gather_cost_multiplier": 0.1,
                "min_resource_threshold": 0.4,
                "share_weight": 0.8,
                "attack_weight": 0.01,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.9,
            "IndependentAgent": 0.0,
            "ControlAgent": 0.1,
        },
    ),
    create_config_variation(
        base_config,
        "Ultra Aggressive",
        learning_rate=0.003,
        gamma=0.99,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.2,  # Low gathering focus
                "gather_cost_multiplier": 0.6,  # High cost to gather
                "min_resource_threshold": 0.05,  # Very low threshold before attacking
                "share_weight": 0.05,  # Minimal sharing
                "attack_weight": 0.8,  # Very high attack tendency
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.2,  # Mostly independent agents
            "IndependentAgent": 0.8,
            "ControlAgent": 0.0,
        },
    ),
    create_config_variation(
        base_config,
        "Pure Resource",
        learning_rate=0.001,
        gamma=0.99,
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 1.0,  # Maximum gathering efficiency
                "gather_cost_multiplier": 0.1,  # Very low gathering cost
                "min_resource_threshold": 0.6,  # High threshold for resource security
                "share_weight": 0.3,  # Moderate sharing
                "attack_weight": 0.0,  # No attacking
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.7,
            "IndependentAgent": 0.0,
            "ControlAgent": 0.3,  # More control agents for stability
        },
    ),
    create_config_variation(
        base_config,
        "Learning Specialist",
        learning_rate=0.01,  # Very high learning rate
        gamma=0.99,  # Strong future focus
        agent_parameters={
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.5,
                "gather_cost_multiplier": 0.3,
                "min_resource_threshold": 0.3,
                "share_weight": 0.4,
                "attack_weight": 0.1,
            }
        },
        agent_type_ratios={
            "SystemAgent": 0.4,
            "IndependentAgent": 0.1,
            "ControlAgent": 0.5,  # Heavy emphasis on control agents
        },
    ),
]


# Extract relevant features for comparison
def extract_features(config):
    agent_ratios = config["agent_type_ratios"]
    system_params = config["agent_parameters"]["SystemAgent"]

    return {
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "system_agent_ratio": agent_ratios["SystemAgent"],
        "independent_agent_ratio": agent_ratios.get("IndependentAgent", 0),
        "control_agent_ratio": agent_ratios.get("ControlAgent", 0),
        "gather_efficiency": system_params["gather_efficiency_multiplier"],
        "share_weight": system_params["share_weight"],
        "attack_weight": system_params["attack_weight"],
        "resource_threshold": system_params["min_resource_threshold"],
    }


# Prepare feature vectors
feature_vectors = [extract_features(config) for config in simulation_settings]

# Modify feature weights to better balance importance
feature_weights = {
    "learning_rate": 1.0,
    "gamma": 0.9,
    "system_agent_ratio": 0.8,  # Reduced from 1.0
    "independent_agent_ratio": 0.7,  # Reduced from 0.8
    "control_agent_ratio": 0.7,  # Reduced from 0.8
    "gather_efficiency": 0.9,  # Increased from 0.7
    "share_weight": 1.0,  # Increased from 0.9
    "attack_weight": 1.0,  # Increased from 0.9
    "resource_threshold": 0.8,  # Increased from 0.6
}

# Normalize numerical features
for feature in feature_vectors[0].keys():
    max_val = max(config[feature] for config in feature_vectors)
    for config in feature_vectors:
        config[feature] = config[feature] / max_val

# Convert to numerical vectors
vectorizer = DictVectorizer(sparse=False)
settings_matrix = vectorizer.fit_transform(feature_vectors)

# Apply feature weights
feature_names = vectorizer.get_feature_names_out()
weight_vector = np.array([feature_weights[feature] for feature in feature_names])
weighted_settings_matrix = settings_matrix * weight_vector

# Create similarity matrix and visualizations
similarity_matrix = cosine_similarity(weighted_settings_matrix)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    similarity_matrix,
    annot=True,
    cmap="viridis",
    xticklabels=[config["variation_name"] for config in simulation_settings],
    yticklabels=[config["variation_name"] for config in simulation_settings],
)
plt.title("Configuration Similarity Heatmap")
plt.tight_layout()
plt.savefig("config_similarity_heatmap.png")
plt.close()

# PCA visualization
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(weighted_settings_matrix)
explained_variance = pca.explained_variance_ratio_

# Create 3D visualization
plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
scatter = ax.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    reduced_embeddings[:, 2],
    c=range(len(simulation_settings)),
    cmap="viridis",
    s=100,
)

# Annotate points in 3D
for i, (x, y, z) in enumerate(reduced_embeddings):
    ax.text(x, y, z, simulation_settings[i]["variation_name"])

plt.colorbar(scatter, label="Configuration Index")
ax.set_xlabel(f"PCA1 ({explained_variance[0]:.1%} variance explained)")
ax.set_ylabel(f"PCA2 ({explained_variance[1]:.1%} variance explained)")
ax.set_zlabel(f"PCA3 ({explained_variance[2]:.1%} variance explained)")
ax.set_title("Configuration Space Visualization (PCA 3D)")
plt.tight_layout()
plt.savefig("config_pca_visualization_3d.png")
plt.show()

# Update feature importance analysis to include PC3
pca_components_df = pd.DataFrame(
    pca.components_.T, columns=["PC1", "PC2", "PC3"], index=feature_names
)
logging.info("\nFeature Importance Analysis:")
logging.info(pca_components_df.abs().sort_values(by="PC1", ascending=False))

# Add these metrics to the analysis output
logging.info("\nConfiguration Summary:")
for setting in simulation_settings:
    name = setting["variation_name"]
    params = setting["agent_parameters"]["SystemAgent"]
    logging.info(f"\n{name}:")
    logging.info(
        f"  Strategy Balance: Attack={params['attack_weight']:.2f}, Share={params['share_weight']:.2f}"
    )
    logging.info(
        f"  Resource Focus: Efficiency={params['gather_efficiency_multiplier']:.2f}, Threshold={params['min_resource_threshold']:.2f}"
    )
    logging.info(
        f"  Agent Mix: System={setting['agent_type_ratios']['SystemAgent']:.1f}, "
        + f"Independent={setting['agent_type_ratios'].get('IndependentAgent', 0):.1f}, "
        + f"Control={setting['agent_type_ratios'].get('ControlAgent', 0):.1f}"
    )

# Interactive 3D Plotly visualization
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            mode="markers+text",
            text=[config["variation_name"] for config in simulation_settings],
            hovertext=[
                f"""
            {config["variation_name"]}
            Attack: {config["agent_parameters"]["SystemAgent"]["attack_weight"]:.2f}
            Share: {config["agent_parameters"]["SystemAgent"]["share_weight"]:.2f}
            Efficiency: {config["agent_parameters"]["SystemAgent"]["gather_efficiency_multiplier"]:.2f}
        """
                for config in simulation_settings
            ],
            hoverinfo="text",
            marker=dict(
                size=10,
                color=list(range(len(simulation_settings))),
                colorscale="Viridis",
                showscale=True,
            ),
        )
    ]
)

fig.update_layout(
    title="Interactive Configuration Space (PCA 3D)",
    scene=dict(
        xaxis_title=f"PCA1 ({explained_variance[0]:.1%} variance)",
        yaxis_title=f"PCA2 ({explained_variance[1]:.1%} variance)",
        zaxis_title=f"PCA3 ({explained_variance[2]:.1%} variance)",
    ),
    width=1000,
    height=800,
)

fig.write_html("interactive_pca_visualization.html")

# Cluster Analysis
n_clusters_range = range(2, 6)
silhouette_scores = []
kmeans_models = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",  # Changed from 10 to 'auto' to suppress warning
    )
    cluster_labels = kmeans.fit_predict(weighted_settings_matrix)
    score = silhouette_score(weighted_settings_matrix, cluster_labels)
    silhouette_scores.append(score)
    kmeans_models.append(kmeans)

# Find optimal number of clusters
optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
optimal_kmeans = kmeans_models[np.argmax(silhouette_scores)]
cluster_labels = optimal_kmeans.fit_predict(weighted_settings_matrix)

# Visualize clusters in PCA space
fig_clusters = go.Figure()

# Add traces for each cluster
for i in range(optimal_n_clusters):
    mask = cluster_labels == i
    fig_clusters.add_trace(
        go.Scatter3d(
            x=reduced_embeddings[mask, 0],
            y=reduced_embeddings[mask, 1],
            z=reduced_embeddings[mask, 2],
            mode="markers+text",
            text=[
                simulation_settings[j]["variation_name"]
                for j in range(len(simulation_settings))
                if mask[j]
            ],
            name=f"Cluster {i}",
            marker=dict(size=10),
        )
    )

# Add cluster centers
centers_pca = pca.transform(optimal_kmeans.cluster_centers_)
fig_clusters.add_trace(
    go.Scatter3d(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        z=centers_pca[:, 2],
        mode="markers",
        marker=dict(color="black", size=15, symbol="x"),
        name="Cluster Centers",
    )
)

fig_clusters.update_layout(
    title=f"Configuration Clusters (k={optimal_n_clusters})",
    scene=dict(
        xaxis_title=f"PCA1 ({explained_variance[0]:.1%} variance)",
        yaxis_title=f"PCA2 ({explained_variance[1]:.1%} variance)",
        zaxis_title=f"PCA3 ({explained_variance[2]:.1%} variance)",
    ),
    width=1000,
    height=800,
)

fig_clusters.write_html("cluster_visualization.html")

# Print cluster analysis
logging.info("\nCluster Analysis:")
logging.info(f"Optimal number of clusters: {optimal_n_clusters}")
logging.info(f"Silhouette score: {max(silhouette_scores):.3f}")

logging.info("\nCluster Memberships:")
for i in range(optimal_n_clusters):
    members = [
        simulation_settings[j]["variation_name"]
        for j in range(len(simulation_settings))
        if cluster_labels[j] == i
    ]
    logging.info(f"\nCluster {i}:")
    logging.info("  " + ", ".join(members))

# Calculate and visualize parameter gradients in PCA space
# This shows how parameters change across the configuration space
param_gradients = {}
for feature in feature_names:
    feature_idx = vectorizer.vocabulary_[feature]
    # Calculate gradient magnitude across PCA dimensions
    gradients = []
    for dim in range(3):
        # Sort points by PCA dimension and calculate gradient
        sorted_idx = np.argsort(reduced_embeddings[:, dim])
        feature_values = weighted_settings_matrix[sorted_idx, feature_idx]
        grad = np.gradient(feature_values)
        gradients.append(np.mean(np.abs(grad)))

    # Store average gradient magnitude across dimensions
    param_gradients[feature] = np.mean(gradients)

# Print parameter gradients
logging.info("\nParameter Gradients (average rate of change across PCA space):")
for param, gradient in sorted(
    param_gradients.items(), key=lambda x: x[1], reverse=True
):
    logging.info(f"{param}: {gradient:.3f}")

logging.info("\n=== Extended Analysis Metrics ===")

# Cross-validation of PCA stability
logging.info("\nPCA Cross-validation Analysis:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_explained_variance = []
cv_feature_importance = []

for train_idx, test_idx in kf.split(weighted_settings_matrix):
    # Fit PCA on training data
    pca_cv = PCA(n_components=3)
    train_data = weighted_settings_matrix[train_idx]
    test_data = weighted_settings_matrix[test_idx]

    # Standardize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Fit and transform
    pca_cv.fit(train_data_scaled)
    cv_explained_variance.append(pca_cv.explained_variance_ratio_)
    cv_feature_importance.append(np.abs(pca_cv.components_))

# Print PCA stability results
cv_explained_variance = np.array(cv_explained_variance)
logging.info("\nExplained Variance Stability:")
for i in range(3):
    mean = cv_explained_variance[:, i].mean()
    std = cv_explained_variance[:, i].std()
    logging.info(f"PC{i+1}: {mean:.3f} ± {std:.3f}")

# Extended Silhouette Analysis
logging.info("\nDetailed Cluster Quality Analysis:")
for n_clusters in n_clusters_range:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",  # Changed from default to 'auto'
    )
    cluster_labels = kmeans.fit_predict(weighted_settings_matrix)

    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_score(
        weighted_settings_matrix, cluster_labels, sample_size=len(simulation_settings)
    )

    # Calculate within-cluster distances
    within_cluster_distances = []
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 1:  # Need at least 2 points for distance
            cluster_points = weighted_settings_matrix[mask]
            distances = np.mean(
                [
                    np.linalg.norm(p1 - p2)
                    for p1, p2 in itertools.combinations(cluster_points, 2)
                ]
            )
            within_cluster_distances.append(distances)

    logging.info(f"\nNumber of clusters: {n_clusters}")
    logging.info(f"Average silhouette score: {sample_silhouette_values:.3f}")
    logging.info(
        f"Average within-cluster distance: {np.mean(within_cluster_distances):.3f}"
    )

# Parameter Sensitivity Analysis
logging.info("\nParameter Sensitivity Analysis:")

# Calculate correlation matrix between parameters
param_matrix = weighted_settings_matrix
correlation_matrix = np.zeros((len(feature_names), len(feature_names)))
p_values_matrix = np.zeros((len(feature_names), len(feature_names)))

for i, param1 in enumerate(feature_names):
    for j, param2 in enumerate(feature_names):
        corr, p_value = spearmanr(param_matrix[:, i], param_matrix[:, j])
        correlation_matrix[i, j] = corr
        p_values_matrix[i, j] = p_value

# Create correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="RdBu",
    center=0,
    xticklabels=feature_names,
    yticklabels=feature_names,
)
plt.title("Parameter Correlation Matrix")
plt.tight_layout()
plt.savefig("parameter_correlation_heatmap.png")
plt.close()

# Calculate parameter impact scores
impact_scores = {}
for i, param in enumerate(feature_names):
    # Combine correlation magnitude with PCA importance
    correlation_impact = np.mean(np.abs(correlation_matrix[i, :]))
    pca_impact = np.mean(
        np.abs([pca.components_[j][i] * explained_variance[j] for j in range(3)])
    )
    impact_scores[param] = (correlation_impact + pca_impact) / 2

logging.info("\nParameter Impact Scores (combined correlation and PCA importance):")
for param, score in sorted(impact_scores.items(), key=lambda x: x[1], reverse=True):
    logging.info(f"{param}: {score:.3f}")

# Identify key parameter interactions
logging.info("\nSignificant Parameter Interactions (|correlation| > 0.5, p < 0.05):")
for i, param1 in enumerate(feature_names):
    for j, param2 in enumerate(feature_names[i + 1 :], i + 1):
        corr = correlation_matrix[i, j]
        p_val = p_values_matrix[i, j]
        if abs(corr) > 0.5 and p_val < 0.05:
            logging.info(f"{param1} ~ {param2}: r={corr:.3f}, p={p_val:.3f}")
