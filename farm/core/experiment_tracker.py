"""
ExperimentTracker: A comprehensive tool for managing and analyzing simulation experiments.

This module provides functionality to track, compare, and visualize multiple simulation runs.
It stores experiment metadata in JSON format and experimental results in SQLite databases,
allowing for efficient storage and retrieval of experiment configurations and metrics.

Key features:
- Experiment registration and metadata management
- Comparative analysis of multiple experiments
- Automated report generation with visualizations
- Statistical summaries of experiment results
- Data export capabilities
- Automatic cleanup of old experiments

Typical usage:
    tracker = ExperimentTracker("experiments")
    exp_id = tracker.register_experiment("test_run", config_dict, "data.db")
    tracker.generate_comparison_report([exp_id1, exp_id2])
"""

import csv
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

# Add logging configuration at the module level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ExperimentTracker:
    """
    A class for tracking and analyzing simulation experiments.

    This class manages experiment metadata, facilitates comparison between different
    experiment runs, and generates detailed reports with visualizations.

    Attributes
    ----------
    experiments_dir (Path):
        Directory where experiment data is stored
    metadata_file (Path):
        Path to the JSON file storing experiment metadata
    metadata (dict):
        Dictionary containing all experiment metadata
    template_env (Environment):
        Jinja2 environment for template rendering

    Methods
    -------
    register_experiment(self, name: str, config: Dict[str, Any], db_path: Path | str) -> str:
        Register a new experiment run with its configuration and database location.
    compare_experiments(self, experiment_ids: List[str], metrics: Optional[List[str]] = None, fill_method: str = 'nan') -> pd.DataFrame:
        Compare metrics across multiple experiments with graceful handling of missing data.
    generate_comparison_report(self, experiment_ids: List[str], output_file: Path | str | None = None):
        Generate a comprehensive HTML report comparing multiple experiments.
    generate_comparison_summary(self, experiment_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        Generate a summary of the comparison including missing data statistics.
    export_experiment_data(self, experiment_id: str, output_path: Path | str):
        Export experiment data to CSV format.
    cleanup_old_experiments(self, days_old: int = 30):
        Remove experiments older than specified days.
    """

    def __init__(self, experiments_dir: str = "experiments") -> None:
        """
        Initialize the ExperimentTracker.

        Parameters
        ----------
            experiments_dir (Path | str): Path to the directory where experiment data will be stored
        """
        self.experiments_dir = experiments_dir
        try:
            os.makedirs(self.experiments_dir, exist_ok=True)
        except PermissionError as e:
            logging.error(f"Failed to create experiments directory: {e}")
            raise
        self.metadata_file = os.path.join(self.experiments_dir, "metadata.json")
        self._load_metadata()

        # Initialize Jinja environment
        self.template_env = Environment(
            loader=FileSystemLoader("templates"), autoescape=True
        )

    def _load_metadata(self) -> None:
        """Load or create experiment metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    logging.debug("Metadata loaded successfully")
            else:
                self.metadata = {"experiments": {}}
                self._save_metadata()
                logging.info("Created new metadata file")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load metadata: {e}")
            raise

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        try:
            # Create a backup of the existing metadata file
            backup_path = self.metadata_file + ".bak"
            if os.path.exists(self.metadata_file):
                os.rename(self.metadata_file, backup_path)
                logging.debug("Created metadata backup")

            # Write new metadata
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
                logging.debug("Metadata saved successfully")

            # Remove backup if save was successful
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except IOError as e:
            logging.error(f"Failed to save metadata: {e}")
            # Restore from backup if available
            if os.path.exists(backup_path):
                os.rename(backup_path, self.metadata_file)
                logging.info("Restored metadata from backup")
            raise

    def register_experiment(
        self, name: str, config: Dict[str, Any], db_path: str
    ) -> str:
        """Register a new experiment run."""
        if not name or not name.strip():
            raise ValueError("Experiment name cannot be empty")
        if not db_path:
            raise ValueError("Database path must be provided")

        try:
            # Generate unique ID and timestamp
            experiment_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            # Store experiment metadata
            self.metadata["experiments"][experiment_id] = {
                "name": name.strip(),
                "timestamp": timestamp,
                "config": config,
                "db_path": db_path,
                "status": "registered",
            }

            self._save_metadata()
            logging.info(f"Registered new experiment: {name} (ID: {experiment_id})")
            return experiment_id

        except Exception as e:
            logging.error(f"Failed to register experiment '{name}': {e}")
            raise

    def export_experiment_data(self, experiment_id: str, output_path: str) -> None:
        """Export experiment data to CSV."""
        if experiment_id not in self.metadata["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")

        try:
            db_path = self.metadata["experiments"][experiment_id]["db_path"]
            with sqlite3.connect(db_path) as conn, open(
                output_path, "w", encoding="utf-8", newline=""
            ) as csv_file:

                writer = csv.writer(csv_file)
                cursor = conn.cursor()

                # Write headers
                cursor.execute("SELECT * FROM SimulationMetrics LIMIT 1")
                headers = [description[0] for description in cursor.description]
                writer.writerow(headers)

                # Write data in chunks
                chunk_size = 1000
                for offset in range(0, self._get_row_count(cursor), chunk_size):
                    cursor.execute(
                        "SELECT * FROM SimulationMetrics LIMIT ? OFFSET ?",
                        (chunk_size, offset),
                    )
                    writer.writerows(cursor.fetchall())

        except Exception as e:
            logging.error(f"Error exporting data: {e}")
            raise

    def cleanup_old_experiments(self, days_old: int = 30) -> None:
        """Remove experiments older than specified days."""
        current_time = datetime.now(timezone.utc)
        experiments_to_remove = []

        try:
            for exp_id, exp_data in self.metadata["experiments"].items():
                exp_date = datetime.fromisoformat(exp_data["timestamp"])
                if (current_time - exp_date).days > days_old:
                    experiments_to_remove.append(exp_id)

            for exp_id in experiments_to_remove:
                exp_data = self.metadata["experiments"][exp_id]
                db_path = exp_data["db_path"]

                # Remove database file
                if os.path.exists(db_path):
                    os.remove(db_path)
                    logging.info(f"Removed database for experiment {exp_id}")

                # Remove from metadata
                del self.metadata["experiments"][exp_id]
                logging.info(f"Removed metadata for experiment {exp_id}")

            self._save_metadata()
            logging.info(f"Cleaned up {len(experiments_to_remove)} old experiments")

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise
