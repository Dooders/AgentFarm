from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
import json
import numpy as np

from farm.database.database import SimulationDatabase
from farm.database.models import Experiment, ExperimentMetric, ExperimentEvent, Simulation

class ExperimentAnalyzer:
    """Analyzer for experiment results."""

    def __init__(self, db: SimulationDatabase):
        """Initialize the analyzer with a database connection."""
        self.db = db
        self._setup_plotting_defaults()

    def _setup_plotting_defaults(self):
        """Set up default plotting style."""
        try:
            plt.style.use('seaborn')
        except (OSError, ValueError):
            # Fall back to default style if seaborn style is not found
            plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except Exception:
            # Continue if setting palette fails
            pass
        
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100

    def analyze_experiment(self, experiment_id: int, output_path: Path):
        """Analyze experiment data and generate reports.
        
        Args:
            experiment_id: ID of the experiment
            output_path: Directory to save analysis outputs
        """
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get experiment data within a session
        experiment_data = {}
        def _get_experiment(session):
            experiment = session.query(Experiment).get(experiment_id)
            if experiment:
                experiment_data.update({
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "description": experiment.description,
                    "base_config": experiment.base_config,
                    "start_time": experiment.created_at,
                    "end_time": experiment.updated_at,
                    "status": experiment.status,
                    "experiment_metadata": experiment.experiment_metadata
                })
        
        self.db._execute_in_transaction(_get_experiment)
        
        if not experiment_data:
            raise ValueError(f"Experiment with ID {experiment_id} not found")
        
        # Analyze metrics
        self._analyze_metrics(experiment_id, output_path)
        
        # Analyze events
        self._analyze_events(experiment_id, output_path)
        
        # Generate summary
        self._generate_summary(experiment_data, output_path)
        
        # Return path to summary file
        return output_path / "summary.json"

    def _analyze_metrics(self, experiment_id: int, output_path: Path):
        """Analyze experiment metrics.
        
        Args:
            experiment_id: ID of the experiment
            output_path: Directory to save analysis outputs
        """
        # Get all metrics for the experiment
        metrics_df = self._get_metrics_dataframe(experiment_id)
        
        if metrics_df.empty:
            return

        # Save metrics to CSV
        metrics_df.to_csv(output_path / "metrics.csv", index=False)

        # Plot each metric over time
        for metric_name in metrics_df["metric_name"].unique():
            metric_data = metrics_df[metrics_df["metric_name"] == metric_name]
            
            plt.figure()
            # Plot metrics by timestamp without grouping by simulation_id
            # since we're now using experiment-level metrics
            plt.plot(metric_data["timestamp"], metric_data["value"], marker='o')
            
            plt.title(f"{metric_name} over Time")
            plt.xlabel("Time")
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.savefig(output_path / f"{metric_name}_over_time.png")
            plt.close()

    def _analyze_events(self, experiment_id: int, output_path: Path):
        """Analyze experiment events.
        
        Args:
            experiment_id: ID of the experiment
            output_path: Directory to save analysis outputs
        """
        # Get all events for the experiment
        events_df = self._get_events_dataframe(experiment_id)
        
        if events_df.empty:
            return

        # Save event timeline to CSV
        events_df.to_csv(output_path / "event_timeline.csv", index=False)

    def _generate_summary(self, experiment_data: dict, output_path: Path):
        """Generate experiment summary.
        
        Args:
            experiment_data: Dictionary containing experiment data
            output_path: Directory to save analysis outputs
        """
        summary = {
            "experiment_id": experiment_data["experiment_id"],
            "name": experiment_data["name"],
            "description": experiment_data["description"],
            "base_config": experiment_data["base_config"],
            "num_simulations": self._count_simulations(experiment_data["experiment_id"]),
            "start_time": experiment_data["start_time"].isoformat() if experiment_data["start_time"] else None,
            "end_time": experiment_data["end_time"].isoformat() if experiment_data["end_time"] else None,
            "status": experiment_data["status"]
        }

        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def compare_experiments(self, experiment_ids: List[int], output_path: Path):
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            output_path: Directory to save comparison outputs
        """
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Get experiment details
        experiments = []
        for exp_id in experiment_ids:
            experiment = self.db._execute_in_transaction(
                lambda session: session.query(Experiment)
                .filter_by(experiment_id=exp_id)
                .first()
            )
            if experiment is None:
                raise ValueError(f"No experiment found with ID {exp_id}")
            experiments.append(experiment)

        # Compare metrics
        self._compare_metrics(experiment_ids, output_path)
        
        # Generate comparison report
        self._generate_comparison_report(experiments, output_path)

    def _compare_metrics(self, experiment_ids: List[int], output_path: Path):
        """Compare metrics across experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            output_path: Directory to save comparison outputs
        """
        # Get metrics for all experiments
        all_metrics = []
        for exp_id in experiment_ids:
            metrics_df = self._get_metrics_dataframe(exp_id)
            if not metrics_df.empty:
                all_metrics.append(metrics_df)

        if not all_metrics:
            return

        # Combine metrics
        combined_metrics = pd.concat(all_metrics)
        combined_metrics.to_csv(output_path / "combined_metrics.csv", index=False)

        # Plot comparisons for each metric
        for metric_name in combined_metrics["metric_name"].unique():
            metric_data = combined_metrics[combined_metrics["metric_name"] == metric_name]
            
            plt.figure()
            for exp_id in metric_data["experiment_id"].unique():
                exp_data = metric_data[metric_data["experiment_id"] == exp_id]
                exp_name = self._get_experiment_name(exp_id)
                
                # Calculate mean and std across simulations
                grouped = exp_data.groupby("timestamp")["value"]
                mean = grouped.mean()
                std = grouped.std()
                
                plt.plot(mean.index, mean.values, label=exp_name)
                plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
            
            plt.title(f"{metric_name} Comparison")
            plt.xlabel("Time")
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig(output_path / f"{metric_name}_comparison.png")
            plt.close()

    def _generate_comparison_report(self, experiments: List[Experiment], output_path: Path):
        """Generate comparison report.
        
        Args:
            experiments: List of Experiment objects
            output_path: Directory to save comparison outputs
        """
        report = {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "base_config": exp.base_config,
                    "num_simulations": self._count_simulations(exp.experiment_id),
                    "status": exp.status
                }
                for exp in experiments
            ],
            "comparison_time": datetime.now().isoformat()
        }

        with open(output_path / "comparison_report.json", "w") as f:
            json.dump(report, f, indent=2)

    def _get_metrics_dataframe(self, experiment_id: int) -> pd.DataFrame:
        """Get metrics for an experiment as a DataFrame.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            DataFrame containing metrics data
        """
        # Use a fresh session to avoid detached instance errors
        metrics_data = []
        
        def _get_metrics(session):
            metrics = session.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            ).all()
            
            # Process metrics within the session context
            for m in metrics:
                metrics_data.append({
                    "id": m.id,
                    "experiment_id": m.experiment_id,
                    "metric_name": m.metric_name,
                    "value": m.metric_value,  # Note: column is metric_value not value
                    "metric_type": m.metric_type,
                    "timestamp": m.timestamp,
                    "metadata": m.metric_metadata
                })
        
        # Execute query within transaction to ensure session is active
        self.db._execute_in_transaction(_get_metrics)
        
        if not metrics_data:
            # Return empty DataFrame with correct columns if no metrics
            return pd.DataFrame(columns=[
                "id", "experiment_id", "metric_name", "value", 
                "metric_type", "timestamp", "metadata"
            ])
            
        return pd.DataFrame(metrics_data)

    def _get_events_dataframe(self, experiment_id: int) -> pd.DataFrame:
        """Get events for an experiment as a DataFrame.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            DataFrame containing events data
        """
        # Use a fresh session to avoid detached instance errors
        events_data = []
        
        def _get_events(session):
            events = session.query(ExperimentEvent).filter(
                ExperimentEvent.experiment_id == experiment_id
            ).all()
            
            # Process events within the session context
            for e in events:
                events_data.append({
                    "id": e.id,
                    "experiment_id": e.experiment_id,
                    "event_type": e.event_type,
                    "timestamp": e.event_time,
                    "details": e.details
                })
        
        # Execute query within transaction to ensure session is active
        self.db._execute_in_transaction(_get_events)
        
        if not events_data:
            # Return empty DataFrame with correct columns if no events
            return pd.DataFrame(columns=[
                "id", "experiment_id", "event_type", "timestamp", "details"
            ])
            
        return pd.DataFrame(events_data)

    def _count_simulations(self, experiment_id: int) -> int:
        """Count the number of simulations in an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Number of simulations
        """
        return self.db._execute_in_transaction(
            lambda session: session.query(Simulation)
            .filter_by(experiment_id=experiment_id)
            .count()
        )

    def _get_experiment_name(self, experiment_id: int) -> str:
        """Get the name of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Name of the experiment
        """
        experiment_name = None
        
        def _get_name(session):
            nonlocal experiment_name
            experiment = session.query(Experiment).get(experiment_id)
            if experiment:
                experiment_name = experiment.name
        
        self.db._execute_in_transaction(_get_name)
        
        if experiment_name is None:
            raise ValueError(f"No experiment found with ID {experiment_id}")
            
        return experiment_name 