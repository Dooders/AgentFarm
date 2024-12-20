from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from farm.database.data_types import AgentActionData, BehaviorClustering
from farm.database.repositories.action_repository import ActionRepository


class BehaviorClusteringAnalyzer:
    """
    Analyzes agent behaviors and clusters them based on their action patterns using
    dynamic clustering algorithms (DBSCAN, Spectral, and Hierarchical Clustering).

    This analyzer processes agent action data to identify distinct behavioral patterns
    and group agents with similar strategies. It supports multiple clustering algorithms
    and dimensionality reduction techniques for visualization.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data
        clustering_method (str): The clustering algorithm to use ('dbscan', 'spectral', or 'hierarchical')
        n_hierarchical_clusters (int): Number of top-level clusters for hierarchical clustering
        dim_reduction_method (str): Dimensionality reduction method ('pca', 'tsne', or None)
        n_components (int): Number of dimensions to reduce to for visualization
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the analyzer with a repository for accessing agent action data.

        Args:
            repository (AgentActionRepository): Repository to fetch agent action data.
        """
        self.repository = repository
        self.clustering_method = "dbscan"  # or "spectral" or "hierarchical"
        self.n_hierarchical_clusters = 3  # Default number of top-level clusters
        self.dim_reduction_method = None  # 'pca' or 'tsne' or None
        self.n_components = 2  # Number of dimensions to reduce to

    def analyze(
        self,
        scope: str,
        agent_id: int = None,
        step: int = None,
        step_range: tuple = None,
    ) -> BehaviorClustering:
        """
        Analyze agent behaviors and create behavioral clusters based on action patterns.

        This method processes agent actions, calculates behavioral metrics, performs
        clustering, and generates visualization data if dimensionality reduction is enabled.

        Args:
            scope (str): The scope of analysis (e.g., 'episode', 'training')
            agent_id (int, optional): Specific agent ID to analyze. Defaults to None.
            step (int, optional): Specific step to analyze. Defaults to None.
            step_range (tuple, optional): Range of steps to analyze (start, end). Defaults to None.

        Returns:
            BehaviorClustering: A data structure containing:
                - clusters: Dict mapping cluster names to lists of agent IDs
                - cluster_characteristics: Dict of behavioral metrics for each cluster
                - cluster_performance: Dict of performance metrics for each cluster
                - reduced_features: Visualization data if dimensionality reduction is enabled

        Example:
            >>> analyzer = BehaviorClusteringAnalyzer(repository)
            >>> results = analyzer.analyze(scope='episode', step_range=(0, 1000))
            >>> print(results.clusters.keys())
            dict_keys(['aggressive', 'cooperative', 'efficient'])
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        agent_metrics = self._calculate_agent_metrics(actions)

        # Get features and perform dimensionality reduction
        features, agent_ids = self._prepare_features(agent_metrics)
        if not features:
            return BehaviorClustering(
                clusters={"unclustered": []},
                cluster_characteristics={},
                cluster_performance={},
                reduced_features=None,
            )

        X = StandardScaler().fit_transform(features)
        reduced_features = (
            self._reduce_dimensions(X) if self.dim_reduction_method else None
        )

        # Perform clustering
        clusters = self._cluster_agents(X, agent_ids)
        characteristics, performance = self._calculate_cluster_characteristics(
            clusters, agent_metrics
        )

        return BehaviorClustering(
            clusters=clusters,
            cluster_characteristics=characteristics,
            cluster_performance=performance,
            reduced_features=(
                self._create_visualization_data(reduced_features, agent_ids, clusters)
                if reduced_features is not None
                else None
            ),
        )

    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the dimensionality of the feature matrix using the specified method (PCA or t-SNE).

        This method applies dimensionality reduction to make high-dimensional behavioral data
        suitable for visualization and analysis. It supports both PCA for linear reduction
        and t-SNE for non-linear manifold learning.

        Args:
            X (np.ndarray): Standardized feature matrix of shape (n_samples, n_features)
                containing behavioral metrics for each agent

        Returns:
            np.ndarray: Reduced dimensional representation of shape (n_samples, n_components)
                where n_components is specified in self.n_components

        Example:
            >>> features = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> analyzer.dim_reduction_method = 'pca'
            >>> analyzer.n_components = 2
            >>> reduced = analyzer._reduce_dimensions(features)
            >>> print(reduced.shape)
            (2, 2)
        """
        if self.dim_reduction_method == "pca":
            reducer = PCA(n_components=self.n_components)
        elif self.dim_reduction_method == "tsne":
            reducer = TSNE(
                n_components=self.n_components,
                perplexity=min(30, len(X) - 1),
                random_state=42,
            )
        else:
            return None

        return reducer.fit_transform(X)

    def _create_visualization_data(
        self,
        reduced_features: np.ndarray,
        agent_ids: List[int],
        clusters: Dict[str, List[int]],
    ) -> Dict[str, Dict]:
        """
        Create a structured visualization data format containing reduced dimensional
        representations and cluster assignments.

        This method organizes reduced dimensional data with cluster assignments into
        a format suitable for visualization tools and front-end rendering.

        Args:
            reduced_features (np.ndarray): Reduced dimensional feature matrix of shape
                (n_samples, n_components)
            agent_ids (List[int]): List of agent identifiers corresponding to each row
                in reduced_features
            clusters (Dict[str, List[int]]): Mapping of cluster names to lists of agent IDs

        Returns:
            Dict[str, Dict]: Visualization data structure containing:
                - method: The dimensionality reduction method used
                - n_components: Number of dimensions in the reduced space
                - points: List of dictionaries, each containing:
                    - agent_id: The ID of the agent
                    - cluster: The assigned cluster name
                    - coordinates: The reduced dimensional coordinates

        Example:
            >>> viz_data = analyzer._create_visualization_data(
            ...     reduced_features=np.array([[1, 2], [3, 4]]),
            ...     agent_ids=[1, 2],
            ...     clusters={'cluster1': [1], 'cluster2': [2]}
            ... )
            >>> print(viz_data['points'][0])
            {'agent_id': 1, 'cluster': 'cluster1', 'coordinates': [1, 2]}
        """
        viz_data = {
            "method": self.dim_reduction_method,
            "n_components": self.n_components,
            "points": [],
        }

        # Create mapping from agent_id to cluster
        agent_to_cluster = {}
        for cluster_name, cluster_agents in clusters.items():
            for agent_id in cluster_agents:
                agent_to_cluster[agent_id] = cluster_name

        # Create point data
        for idx, agent_id in enumerate(agent_ids):
            point_data = {
                "agent_id": agent_id,
                "cluster": agent_to_cluster.get(agent_id, "unclustered"),
                "coordinates": reduced_features[idx].tolist(),
            }
            viz_data["points"].append(point_data)

        return viz_data

    def _cluster_agents(
        self, X: np.ndarray, agent_ids: List[int]
    ) -> Dict[str, List[int]]:
        """
        Cluster agents using selected clustering algorithm.
        """
        if self.clustering_method == "hierarchical":
            return self._perform_hierarchical_clustering(X, agent_ids)
        elif self.clustering_method == "dbscan":
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
            return self._process_flat_clustering(clustering.labels_, X, agent_ids)
        else:  # spectral
            n_clusters = min(len(X), 4)
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="nearest_neighbors"
            ).fit(X)
            return self._process_flat_clustering(clustering.labels_, X, agent_ids)

    def _calculate_agent_metrics(
        self, actions: List[AgentActionData]
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate behavioral metrics for each agent based on their action history.

        Processes raw action data to compute various behavioral indicators including
        attack tendencies, cooperation levels, and success rates.

        Args:
            actions (List[AgentActionData]): List of agent actions to analyze, containing
                action types, rewards, and interaction data.

        Returns:
            Dict[int, Dict[str, float]]: A nested dictionary structure where:
                - Outer key: agent ID
                - Inner keys: metric names including:
                    - attack_count: Number of attack actions
                    - share_count: Number of resource sharing actions
                    - interaction_count: Number of agent interactions
                    - success_count: Number of successful actions
                    - total_reward: Cumulative rewards
                    - total_actions: Total number of actions taken

        Example:
            >>> metrics = analyzer._calculate_agent_metrics(actions)
            >>> print(metrics[agent_id]['success_count'] / metrics[agent_id]['total_actions'])
            0.75  # 75% success rate
        """
        agent_metrics = {}
        for action in actions:
            agent_id = action.agent_id
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {
                    "attack_count": 0,
                    "share_count": 0,
                    "interaction_count": 0,
                    "success_count": 0,
                    "total_reward": 0,
                    "total_actions": 0,
                }
            metrics = agent_metrics[agent_id]
            metrics["total_actions"] += 1
            metrics["total_reward"] += action.reward or 0
            if action.action_type == "attack":
                metrics["attack_count"] += 1
            if action.action_type == "share":
                metrics["share_count"] += 1
            if action.action_target_id:
                metrics["interaction_count"] += 1
            if action.reward and action.reward > 0:
                metrics["success_count"] += 1
        return agent_metrics

    def _prepare_features(
        self, agent_metrics: Dict[int, Dict[str, float]]
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Prepare feature matrix from agent metrics.
        """
        agent_ids = list(agent_metrics.keys())
        features = []

        for agent_id in agent_ids:
            metrics = agent_metrics[agent_id]
            if metrics["total_actions"] == 0:
                continue

            feature_vector = [
                metrics["attack_count"] / metrics["total_actions"],
                metrics["share_count"] / metrics["total_actions"],
                metrics["interaction_count"] / metrics["total_actions"],
                metrics["success_count"] / metrics["total_actions"],
                metrics["total_reward"] / metrics["total_actions"],
            ]
            features.append(feature_vector)

        return features, agent_ids

    def _perform_hierarchical_clustering(
        self, X: np.ndarray, agent_ids: List[int]
    ) -> Dict[str, List[int]]:
        """
        Perform hierarchical clustering to identify nested behavioral patterns among agents.

        This method implements agglomerative hierarchical clustering with Ward's linkage
        criterion to create a tree-like structure of behavioral clusters. It automatically
        determines optimal subclusters using the elbow method on the linkage matrix.

        Args:
            X (np.ndarray): Standardized feature matrix of shape (n_samples, n_features)
                agent_ids (List[int]): List of agent identifiers corresponding to each row in X

        Returns:
            Dict[str, List[int]]: Hierarchical cluster assignments where:
                - Keys are cluster names (e.g., 'cooperative_high_success')
                - Values are lists of agent IDs belonging to each cluster
                - Includes both top-level clusters and meaningful subclusters

        Notes:
            - Uses Ward's method for minimum variance clustering
            - Stores linkage matrix in self.last_linkage_matrix for later analysis
            - Automatically determines optimal number of subclusters for large clusters
            - Skips subcluster creation for clusters with fewer than 3 agents

        Example:
            >>> features = np.array([[1, 2], [3, 4], [10, 12], [11, 13]])
            >>> agent_ids = [1, 2, 3, 4]
            >>> clusters = analyzer._perform_hierarchical_clustering(features, agent_ids)
            >>> print(clusters.keys())
            dict_keys(['cooperative_high_success', 'aggressive_standard'])
        """
        # Compute linkage matrix for hierarchical structure
        linkage_matrix = linkage(X, method="ward")

        # Store linkage matrix for visualization or further analysis
        self.last_linkage_matrix = linkage_matrix

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_hierarchical_clusters,
            linkage="ward",
            compute_distances=True,  # Enable distance computation
        ).fit(X)

        # Get top-level clusters
        clusters = self._process_flat_clustering(clustering.labels_, X, agent_ids)

        # Add subclusters for each main cluster using the linkage information
        hierarchical_clusters = {}
        for cluster_name, cluster_agents in clusters.items():
            if len(cluster_agents) < 3:  # Skip small clusters
                hierarchical_clusters[cluster_name] = cluster_agents
                continue

            # Get features for current cluster
            cluster_mask = np.isin(agent_ids, cluster_agents)
            subcluster_X = X[cluster_mask]
            subcluster_agents = np.array(agent_ids)[cluster_mask]

            # Use linkage matrix to determine optimal number of subclusters
            n_subclusters = self._get_optimal_subclusters(
                linkage_matrix,
                cluster_mask,
                min_clusters=2,
                max_clusters=min(len(cluster_agents) // 2, 3),
            )

            if n_subclusters > 1:
                subclustering = AgglomerativeClustering(
                    n_clusters=n_subclusters, linkage="ward"
                ).fit(subcluster_X)

                # Process subclusters
                for i in range(n_subclusters):
                    subcluster_mask = subclustering.labels_ == i
                    subcluster_features = subcluster_X[subcluster_mask].mean(axis=0)
                    subcluster_name = f"{cluster_name}_{self._get_subcluster_name(subcluster_features)}"
                    hierarchical_clusters[subcluster_name] = subcluster_agents[
                        subcluster_mask
                    ].tolist()
            else:
                hierarchical_clusters[cluster_name] = cluster_agents

        return hierarchical_clusters

    def _get_optimal_subclusters(
        self,
        linkage_matrix: np.ndarray,
        cluster_mask: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 3,
    ) -> int:
        """
        Determine the optimal number of subclusters using the elbow method on the linkage matrix.

        This method analyzes the hierarchical structure to find natural divisions in the data
        by identifying significant jumps in the clustering distances.

        Args:
            linkage_matrix (np.ndarray): The linkage matrix from hierarchical clustering,
                containing distances between merged clusters
            cluster_mask (np.ndarray): Boolean mask indicating which samples belong to
                the current cluster
            min_clusters (int, optional): Minimum number of subclusters to consider.
                Defaults to 2.
            max_clusters (int, optional): Maximum number of subclusters to consider.
                Defaults to 3.

        Returns:
            int: Optimal number of subclusters based on the elbow method

        Notes:
            - Uses the elbow method to find the point of diminishing returns in cluster creation
            - Respects minimum and maximum cluster constraints
            - Returns 1 if the cluster is too small to subdivide

        Example:
            >>> linkage_mat = np.array([[0, 1, 0.1, 2], [2, 3, 0.5, 3], [4, 5, 2.0, 4]])
            >>> mask = np.array([True, True, True, True])
            >>> n_clusters = analyzer._get_optimal_subclusters(linkage_mat, mask)
            >>> print(n_clusters)
            2
        """
        # Get relevant part of linkage matrix for this cluster
        cluster_size = np.sum(cluster_mask)
        if cluster_size <= min_clusters:
            return 1

        # Calculate distances between merges
        distances = linkage_matrix[-(cluster_size - 1) :, 2]

        # Simple elbow method: find largest distance gap
        distance_diffs = np.diff(distances)
        if len(distance_diffs) < max_clusters:
            return min(len(distance_diffs) + 1, max_clusters)

        # Find the elbow point within our constraints
        n_clusters = min(
            np.argmax(distance_diffs[: max_clusters - 1]) + 2, max_clusters
        )
        return max(n_clusters, min_clusters)

    def _get_subcluster_name(self, features: np.ndarray) -> str:
        """
        Generate descriptive names for subclusters based on their behavioral characteristics.

        This method analyzes the feature vector of a subcluster to determine its dominant
        behavioral pattern and assigns an appropriate descriptive name.

        Args:
            features (np.ndarray): Feature vector containing behavioral metrics:
                [attack_rate, share_rate, interaction_rate, success_rate, avg_reward]

        Returns:
            str: Descriptive name for the subcluster ('high_success', 'high_reward',
                'social', or 'standard')

        Example:
            >>> features = np.array([0.2, 0.3, 0.4, 0.8, 0.6])
            >>> name = analyzer._get_subcluster_name(features)
            >>> print(name)
            'high_success'
        """
        if features[3] > 0.7:  # High success rate
            return "high_success"
        elif features[4] > 0.6:  # High reward
            return "high_reward"
        elif features[1] + features[2] > 0.5:  # High cooperation
            return "social"
        else:
            return "standard"

    def _process_flat_clustering(
        self, labels: np.ndarray, X: np.ndarray, agent_ids: List[int]
    ) -> Dict[str, List[int]]:
        """
        Process clustering labels into named clusters with assigned agents.

        This method converts numerical cluster labels into meaningful cluster names based
        on the behavioral characteristics of each cluster, and maps agents to their
        respective clusters.

        Args:
            labels (np.ndarray): Cluster labels assigned by the clustering algorithm
            X (np.ndarray): Feature matrix used for clustering
            agent_ids (List[int]): List of agent IDs corresponding to the rows in X

        Returns:
            Dict[str, List[int]]: Dictionary mapping descriptive cluster names to lists
                of agent IDs belonging to each cluster. Special cluster 'outliers' is
                used for agents with label -1.

        Example:
            >>> labels = np.array([0, 0, 1, -1])
            >>> X = np.array([[1, 2], [1, 3], [5, 6], [10, 10]])
            >>> agent_ids = [1, 2, 3, 4]
            >>> clusters = analyzer._process_flat_clustering(labels, X, agent_ids)
            >>> print(clusters)
            {'cooperative': [1, 2], 'aggressive': [3], 'outliers': [4]}
        """
        clusters = defaultdict(list)
        for agent_idx, cluster_label in enumerate(labels):
            if cluster_label == -1:
                cluster_name = "outliers"
            else:
                cluster_features = X[labels == cluster_label].mean(axis=0)
                cluster_name = self._get_cluster_name(cluster_features)
            clusters[cluster_name].append(agent_ids[agent_idx])
        return dict(clusters)

    def _get_cluster_name(self, cluster_features: np.ndarray) -> str:
        """
        Determine cluster name based on average feature characteristics.

        This method analyzes the mean feature values of a cluster to identify its
        predominant behavioral pattern and assigns an appropriate descriptive name.

        Args:
            cluster_features (np.ndarray): Average feature values for the cluster:
                [attack_rate, share_rate, interaction_rate, success_rate, avg_reward]

        Returns:
            str: Descriptive name for the cluster ('aggressive', 'cooperative',
                'efficient', or 'balanced')

        Notes:
            - 'aggressive': High attack rate (> 0.5)
            - 'cooperative': High share + interaction rates (> 0.6)
            - 'efficient': High success + reward rates (> 0.6, > 0.5)
            - 'balanced': No dominant characteristics

        Example:
            >>> features = np.array([0.6, 0.2, 0.2, 0.5, 0.4])
            >>> name = analyzer._get_cluster_name(features)
            >>> print(name)
            'aggressive'
        """
        # Features order: [attack_rate, share_rate, interaction_rate, success_rate, avg_reward]
        if cluster_features[0] > 0.5:  # High attack rate
            return "aggressive"
        elif (
            cluster_features[1] + cluster_features[2] > 0.6
        ):  # High share + interaction
            return "cooperative"
        elif (
            cluster_features[3] > 0.6 and cluster_features[4] > 0.5
        ):  # High success + reward
            return "efficient"
        else:
            return "balanced"

    def _calculate_cluster_characteristics(
        self, clusters: Dict[str, List[int]], agent_metrics: Dict[int, Dict[str, float]]
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Calculate characteristic metrics and performance statistics for each cluster.

        This method computes aggregate behavioral metrics and performance indicators
        for each cluster based on the individual agent metrics within the cluster.

        Args:
            clusters (Dict[str, List[int]]): Dictionary mapping cluster names to lists
                of agent IDs
            agent_metrics (Dict[int, Dict[str, float]]): Dictionary mapping agent IDs
                to their individual behavioral metrics

        Returns:
            tuple[Dict[str, Dict[str, float]], Dict[str, float]]: A tuple containing:
                - characteristics: Dictionary mapping cluster names to their behavioral
                  metrics including:
                    - attack_rate: Average rate of aggressive actions
                    - cooperation: Average rate of cooperative actions
                    - risk_taking: Average risk-taking behavior
                    - success_rate: Average success rate
                    - resource_efficiency: Average resource utilization
                - performance: Dictionary mapping cluster names to their average
                  reward performance

        Example:
            >>> clusters = {'aggressive': [1, 2], 'cooperative': [3, 4]}
            >>> agent_metrics = {
            ...     1: {'attack_count': 10, 'share_count': 2, 'total_actions': 20},
            ...     2: {'attack_count': 8, 'share_count': 3, 'total_actions': 20},
            ...     3: {'attack_count': 2, 'share_count': 10, 'total_actions': 20},
            ...     4: {'attack_count': 3, 'share_count': 8, 'total_actions': 20}
            ... }
            >>> chars, perf = analyzer._calculate_cluster_characteristics(
            ...     clusters, agent_metrics)
            >>> print(chars['aggressive']['attack_rate'])
            0.45  # 45% attack rate for aggressive cluster
        """
        characteristics = {}
        performance = {}
        for cluster_name, agent_ids in clusters.items():
            if not agent_ids:
                continue
            cluster_metrics = {
                "attack_rate": 0,
                "cooperation": 0,
                "risk_taking": 0,
                "success_rate": 0,
                "resource_efficiency": 0,
            }
            total_reward = 0
            for agent_id in agent_ids:
                metrics = agent_metrics[agent_id]
                total_actions = metrics["total_actions"]
                cluster_metrics["attack_rate"] += (
                    metrics["attack_count"] / total_actions
                )
                cluster_metrics["cooperation"] += (
                    metrics["share_count"] + metrics["interaction_count"]
                ) / total_actions
                cluster_metrics["success_rate"] += (
                    metrics["success_count"] / total_actions
                )
                cluster_metrics["resource_efficiency"] += (
                    metrics["total_reward"] / total_actions
                )
                total_reward += metrics["total_reward"] / total_actions
            n_agents = len(agent_ids)
            for metric in cluster_metrics:
                cluster_metrics[metric] /= n_agents
            characteristics[cluster_name] = cluster_metrics
            performance[cluster_name] = total_reward / n_agents
        return characteristics, performance
