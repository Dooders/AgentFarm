#!/usr/bin/env python3

"""
simulation_comparison.py

A script for comparing results across multiple simulations and performing
cross-simulation analysis to identify patterns and trends.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import Simulation

from .simulation_analysis import SimulationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimulationComparator:
    """Class for comparing multiple simulations."""

    def __init__(self, db_path: str):
        """Initialize the comparator with database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.analyzer = SimulationAnalyzer(db_path)

    def load_simulation_data(self, simulation_ids: List[int]) -> pd.DataFrame:
        """Load data for multiple simulations.

        Args:
            simulation_ids: List of simulation IDs to analyze

        Returns:
            DataFrame containing combined simulation data
        """
        logger.info(f"Loading data for simulations: {simulation_ids}")

        simulations = []
        for sim_id in simulation_ids:
            sim = self.session.query(Simulation).get(sim_id)
            if sim:
                sim_data = {
                    "simulation_id": sim_id,
                    **sim.parameters,
                    **sim.results_summary,
                }
                simulations.append(sim_data)

        return pd.DataFrame(simulations)

    def cluster_simulations(self, df: pd.DataFrame, max_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Cluster simulations based on their features and outcomes with statistical validation.

        Args:
            df: DataFrame containing simulation data
            max_clusters: Maximum number of clusters to test (default: min(10, len(df)//2))

        Returns:
            Dictionary containing clustering results with validation metrics
        """
        logger.info("Clustering simulations with statistical validation")

        if len(df) < 4:
            logger.warning(f"Insufficient data for clustering: {len(df)} samples")
            return {"error": "Insufficient data for clustering", "min_samples_required": 4}

        # Select features for clustering
        feature_cols = [
            col
            for col in df.columns
            if col
            not in ["simulation_id", "population_dominance", "survival_dominance"]
        ]
        
        if not feature_cols:
            logger.warning("No features available for clustering")
            return {"error": "No features available for clustering"}
        
        features = df[feature_cols]

        # Check for missing values
        if features.isnull().any().any():
            logger.warning("Missing values found in features, filling with median")
            features = features.fillna(features.median())

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Find optimal number of clusters using multiple methods
        max_clusters = max_clusters or min(10, len(df) // 2)
        k_range = range(2, max_clusters + 1)
        
        # Elbow method
        wcss = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:
                sil_score = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

        # Find optimal k using silhouette score (more reliable than elbow)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette_score = max(silhouette_scores)
        
        logger.info(f"Optimal number of clusters: {optimal_k} (silhouette score: {best_silhouette_score:.3f})")

        # Apply clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        df["cluster"] = cluster_labels

        # Validate clustering quality
        final_silhouette_score = silhouette_score(scaled_features, cluster_labels)
        
        # Calculate cluster separation (between-cluster sum of squares / within-cluster sum of squares)
        cluster_centers = kmeans.cluster_centers_
        within_cluster_ss = 0
        between_cluster_ss = 0
        
        for i in range(optimal_k):
            cluster_points = scaled_features[cluster_labels == i]
            if len(cluster_points) > 0:
                within_cluster_ss += np.sum((cluster_points - cluster_centers[i])**2)
        
        overall_mean = np.mean(scaled_features, axis=0)
        for i in range(optimal_k):
            cluster_size = np.sum(cluster_labels == i)
            between_cluster_ss += cluster_size * np.sum((cluster_centers[i] - overall_mean)**2)
        
        separation_ratio = between_cluster_ss / (within_cluster_ss + 1e-10)

        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_data = df[df["cluster"] == cluster_id]
            cluster_analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df) * 100,
                "dominant_population": cluster_data["population_dominance"].mode().iloc[0] if "population_dominance" in cluster_data.columns else "unknown",
                "dominant_survival": cluster_data["survival_dominance"].mode().iloc[0] if "survival_dominance" in cluster_data.columns else "unknown"
            }

        # Create enhanced elbow curve plot with silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax1.set_xlabel("Number of clusters")
        ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax1.set_title("Elbow Method for Optimal k")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax2.set_xlabel("Number of clusters")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("clustering_validation.png", dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "cluster_assignments": df["cluster"].to_dict(),
            "cluster_analysis": cluster_analysis,
            "clustering_validation": {
                "optimal_k": optimal_k,
                "silhouette_score": final_silhouette_score,
                "separation_ratio": separation_ratio,
                "wcss_values": wcss,
                "silhouette_scores": silhouette_scores,
                "clustering_quality": "good" if final_silhouette_score > 0.5 else "fair" if final_silhouette_score > 0.3 else "poor"
            },
            "feature_importance": dict(
                zip(feature_cols, np.abs(kmeans.cluster_centers_).mean(axis=0))
            ),
        }

    def build_predictive_model(self, df: pd.DataFrame, target_column: str = "population_dominance") -> Dict[str, Any]:
        """Build and evaluate a predictive model for simulation outcomes with proper validation.

        Args:
            df: DataFrame containing simulation data
            target_column: Column to use as target variable

        Returns:
            Dictionary containing model evaluation results with cross-validation
        """
        logger.info(f"Building predictive model for {target_column}")

        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return {"error": f"Target column '{target_column}' not found"}

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ["simulation_id", target_column]]
        
        if not feature_cols:
            logger.error("No features available for modeling")
            return {"error": "No features available for modeling"}

        X = df[feature_cols]
        y = df[target_column]

        # Check for missing values
        if X.isnull().any().any():
            logger.warning("Missing values found in features, filling with median")
            X = X.fillna(X.median())

        if y.isnull().any():
            logger.warning("Missing values found in target, removing rows")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]

        if len(X) < 10:
            logger.error(f"Insufficient data for modeling: {len(X)} samples")
            return {"error": f"Insufficient data for modeling: {len(X)} samples (minimum: 10)"}

        # Check if target has enough variation
        unique_targets = y.nunique()
        if unique_targets < 2:
            logger.error(f"Target variable has insufficient variation: {unique_targets} unique values")
            return {"error": f"Target variable has insufficient variation: {unique_targets} unique values"}

        # Split data with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails (e.g., some classes too small), use random split
            logger.warning("Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Train model
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_accuracy = model.score(X_test, y_test)

        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_scores = []
            cv_mean = test_accuracy
            cv_std = 0

        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        # Calculate confidence intervals for accuracy
        n_test = len(y_test)
        if n_test > 0:
            # Wilson score interval for accuracy
            p = test_accuracy
            z = 1.96  # 95% confidence
            ci_lower = (p + z*z/(2*n_test) - z * np.sqrt((p*(1-p) + z*z/(4*n_test))/n_test)) / (1 + z*z/n_test)
            ci_upper = (p + z*z/(2*n_test) + z * np.sqrt((p*(1-p) + z*z/(4*n_test))/n_test)) / (1 + z*z/n_test)
        else:
            ci_lower = ci_upper = 0

        # Plot feature importance with confidence intervals
        plt.figure(figsize=(12, 8))
        
        # Feature importance plot
        importance_df = pd.DataFrame(
            feature_importance.items(), columns=["feature", "importance"]
        )
        importance_df = importance_df.sort_values("importance", ascending=True)
        
        bars = plt.barh(importance_df["feature"], importance_df["importance"])
        plt.title(f"Feature Importance for Predicting {target_column}")
        plt.xlabel("Feature Importance")
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Model performance summary
        performance_summary = {
            "test_accuracy": test_accuracy,
            "test_accuracy_ci": (ci_lower, ci_upper),
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "cv_scores": cv_scores.tolist() if len(cv_scores) > 0 else [],
            "model_quality": "good" if test_accuracy > 0.8 else "fair" if test_accuracy > 0.6 else "poor",
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "n_test_samples": len(X_test)
        }

        return {
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "feature_importance": feature_importance,
            "performance_summary": performance_summary,
            "model_type": "GradientBoostingClassifier",
            "target_column": target_column,
        }

    def compare_resource_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare agent performance across different resource patterns.

        Args:
            df: DataFrame containing simulation data

        Returns:
            Dictionary containing comparison results
        """
        logger.info("Comparing resource patterns")

        # Group by resource pattern
        pattern_analysis = (
            df.groupby("resource_pattern")
            .agg(
                {
                    "population_dominance": lambda x: x.value_counts().to_dict(),
                    "survival_dominance": lambda x: x.value_counts().to_dict(),
                    "simulation_id": "count",
                }
            )
            .rename(columns={"simulation_id": "total_simulations"})
        )

        # Calculate percentages
        for idx in pattern_analysis.index:
            for dominance_type in ["population_dominance", "survival_dominance"]:
                counts = pattern_analysis.at[idx, dominance_type]
                total = sum(counts.values())
                pattern_analysis.at[idx, dominance_type] = {
                    k: v / total * 100 for k, v in counts.items()
                }

        # Create visualization
        plt.figure(figsize=(12, 6))
        patterns = pattern_analysis.index
        agent_types = ["system", "independent", "control"]

        x = np.arange(len(patterns))
        width = 0.25

        for i, agent_type in enumerate(agent_types):
            percentages = [
                pattern_analysis.at[p, "population_dominance"].get(agent_type, 0)
                for p in patterns
            ]
            plt.bar(x + i * width, percentages, width, label=agent_type)

        plt.xlabel("Resource Pattern")
        plt.ylabel("Population Dominance (%)")
        plt.title("Agent Dominance by Resource Pattern")
        plt.xticks(x + width, list(patterns))
        plt.legend()
        plt.savefig("resource_pattern_comparison.png")
        plt.close()

        return {str(k): v for k, v in pattern_analysis.to_dict().items()}

    def analyze_critical_events_across_simulations(
        self, simulation_ids: List[int]
    ) -> Dict[str, Any]:
        """Analyze patterns in critical events across multiple simulations.

        Args:
            simulation_ids: List of simulation IDs to analyze

        Returns:
            Dictionary containing critical events analysis
        """
        logger.info("Analyzing critical events across simulations")

        all_critical_events = []
        for sim_id in simulation_ids:
            events = self.analyzer.identify_critical_events(sim_id)
            for event in events:
                event["simulation_id"] = sim_id
                all_critical_events.append(event)

        if not all_critical_events:
            return {}

        events_df = pd.DataFrame(all_critical_events)

        # Analyze timing of critical events
        timing_analysis = events_df.groupby("simulation_id")["step"].agg(
            ["min", "max", "mean", "count"]
        )

        # Analyze types of changes
        change_types = (
            events_df[["system_change", "independent_change", "control_change"]]
            .abs()
            .mean()
        )

        # Plot distribution of critical events
        plt.figure(figsize=(10, 6))
        plt.hist(events_df["step"], bins=30)
        plt.title("Distribution of Critical Events Over Time")
        plt.xlabel("Simulation Step")
        plt.ylabel("Number of Events")
        plt.savefig("critical_events_distribution.png")
        plt.close()

        return {
            "timing_analysis": timing_analysis.to_dict(),
            "average_changes": change_types.to_dict(),
            "events_per_simulation": events_df.groupby("simulation_id")
            .size()
            .to_dict(),
        }

    def run_comparative_analysis(self, simulation_ids: List[int]) -> Dict[str, Any]:
        """Run comprehensive comparative analysis across multiple simulations.

        Args:
            simulation_ids: List of simulation IDs to analyze

        Returns:
            Dictionary containing all comparative analysis results
        """
        logger.info(f"Running comparative analysis for simulations: {simulation_ids}")

        # Load simulation data
        df = self.load_simulation_data(simulation_ids)

        if df.empty:
            logger.warning("No simulation data found")
            return {}

        # Run analyses
        results = {
            "clustering": self.cluster_simulations(df),
            "predictive_model": self.build_predictive_model(df),
            "resource_pattern_comparison": self.compare_resource_patterns(df),
            "critical_events_analysis": self.analyze_critical_events_across_simulations(
                simulation_ids
            ),
        }

        # Save results
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / "comparative_analysis.json"
        import json

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comparative analysis complete. Results saved to {results_file}")
        return results


def main():
    """Main function to run the comparative analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple simulations")
    parser.add_argument(
        "--db-path", required=True, help="Path to the simulation database"
    )
    parser.add_argument(
        "--simulation-ids",
        required=True,
        nargs="+",
        type=int,
        help="List of simulation IDs to compare",
    )

    args = parser.parse_args()

    comparator = SimulationComparator(args.db_path)
    results = comparator.run_comparative_analysis(args.simulation_ids)

    logger.info("Comparative analysis completed successfully")


if __name__ == "__main__":
    main()
