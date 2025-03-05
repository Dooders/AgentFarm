#!/usr/bin/env python3

"""
simulation_comparison.py

A script for comparing results across multiple simulations and performing
cross-simulation analysis to identify patterns and trends.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import Simulation
from simulation_analysis import SimulationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
                    'simulation_id': sim_id,
                    **sim.parameters,
                    **sim.results_summary
                }
                simulations.append(sim_data)
        
        return pd.DataFrame(simulations)

    def cluster_simulations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster simulations based on their features and outcomes.
        
        Args:
            df: DataFrame containing simulation data
            
        Returns:
            Dictionary containing clustering results
        """
        logger.info("Clustering simulations")
        
        # Select features for clustering
        feature_cols = [col for col in df.columns if col not in 
                       ['simulation_id', 'population_dominance', 'survival_dominance']]
        features = df[feature_cols]
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Find optimal number of clusters
        wcss = []
        max_clusters = min(10, len(df))
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_features)
            wcss.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), wcss)
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('clustering_elbow.png')
        plt.close()
        
        # Apply clustering with optimal k (you might want to adjust this)
        optimal_k = 4  # This could be determined programmatically
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_analysis = df.groupby('cluster').agg({
            'population_dominance': lambda x: x.value_counts().index[0],
            'survival_dominance': lambda x: x.value_counts().index[0],
            'simulation_id': 'count'
        }).rename(columns={'simulation_id': 'count'})
        
        return {
            'cluster_assignments': df['cluster'].to_dict(),
            'cluster_analysis': cluster_analysis.to_dict(),
            'feature_importance': dict(zip(feature_cols, 
                                        np.abs(kmeans.cluster_centers_).mean(axis=0)))
        }

    def build_predictive_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build and evaluate a predictive model for simulation outcomes.
        
        Args:
            df: DataFrame containing simulation data
            
        Returns:
            Dictionary containing model evaluation results
        """
        logger.info("Building predictive model")
        
        # Prepare features and target
        X = df.drop(['simulation_id', 'population_dominance', 'survival_dominance'], axis=1)
        y_pop = df['population_dominance']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_pop, test_size=0.2, random_state=42)
        
        # Train model
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(feature_importance.items(), columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=True)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance for Predicting Population Dominance')
        plt.savefig('feature_importance.png')
        plt.close()
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': feature_importance
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
        pattern_analysis = df.groupby('resource_pattern').agg({
            'population_dominance': lambda x: x.value_counts().to_dict(),
            'survival_dominance': lambda x: x.value_counts().to_dict(),
            'simulation_id': 'count'
        }).rename(columns={'simulation_id': 'total_simulations'})
        
        # Calculate percentages
        for idx in pattern_analysis.index:
            for dominance_type in ['population_dominance', 'survival_dominance']:
                counts = pattern_analysis.loc[idx, dominance_type]
                total = sum(counts.values())
                pattern_analysis.loc[idx, dominance_type] = {
                    k: v/total*100 for k, v in counts.items()
                }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        patterns = pattern_analysis.index
        agent_types = ['system', 'independent', 'control']
        
        x = np.arange(len(patterns))
        width = 0.25
        
        for i, agent_type in enumerate(agent_types):
            percentages = [pattern_analysis.loc[p, 'population_dominance'].get(agent_type, 0) 
                         for p in patterns]
            plt.bar(x + i*width, percentages, width, label=agent_type)
        
        plt.xlabel('Resource Pattern')
        plt.ylabel('Population Dominance (%)')
        plt.title('Agent Dominance by Resource Pattern')
        plt.xticks(x + width, patterns)
        plt.legend()
        plt.savefig('resource_pattern_comparison.png')
        plt.close()
        
        return pattern_analysis.to_dict()

    def analyze_critical_events_across_simulations(self, simulation_ids: List[int]) -> Dict[str, Any]:
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
                event['simulation_id'] = sim_id
                all_critical_events.append(event)
        
        if not all_critical_events:
            return {}
        
        events_df = pd.DataFrame(all_critical_events)
        
        # Analyze timing of critical events
        timing_analysis = events_df.groupby('simulation_id')['step'].agg(['min', 'max', 'mean', 'count'])
        
        # Analyze types of changes
        change_types = events_df[['system_change', 'independent_change', 'control_change']].abs().mean()
        
        # Plot distribution of critical events
        plt.figure(figsize=(10, 6))
        plt.hist(events_df['step'], bins=30)
        plt.title('Distribution of Critical Events Over Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Number of Events')
        plt.savefig('critical_events_distribution.png')
        plt.close()
        
        return {
            'timing_analysis': timing_analysis.to_dict(),
            'average_changes': change_types.to_dict(),
            'events_per_simulation': events_df.groupby('simulation_id').size().to_dict()
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
            'clustering': self.cluster_simulations(df),
            'predictive_model': self.build_predictive_model(df),
            'resource_pattern_comparison': self.compare_resource_patterns(df),
            'critical_events_analysis': self.analyze_critical_events_across_simulations(simulation_ids)
        }
        
        # Save results
        output_dir = Path('analysis_results')
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / 'comparative_analysis.json'
        pd.io.json.to_json(results_file, results)
        
        logger.info(f"Comparative analysis complete. Results saved to {results_file}")
        return results

def main():
    """Main function to run the comparative analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple simulations')
    parser.add_argument('--db-path', required=True, help='Path to the simulation database')
    parser.add_argument('--simulation-ids', required=True, nargs='+', type=int,
                      help='List of simulation IDs to compare')
    
    args = parser.parse_args()
    
    comparator = SimulationComparator(args.db_path)
    results = comparator.run_comparative_analysis(args.simulation_ids)
    
    logger.info("Comparative analysis completed successfully")

if __name__ == "__main__":
    main() 