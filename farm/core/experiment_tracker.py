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
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

from farm.utils.identity import Identity
from farm.utils.logging_config import get_logger

# Shared Identity instance for efficiency
_shared_identity = Identity()

logger = get_logger(__name__)


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
            logger.error(
                "failed_to_create_experiments_directory", error=str(e), exc_info=True
            )
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
                    logger.debug("metadata_loaded_successfully")
            else:
                self.metadata = {"experiments": {}}
                self._save_metadata()
                logger.info("created_new_metadata_file")
        except (json.JSONDecodeError, IOError) as e:
            logger.error("failed_to_load_metadata", error=str(e), exc_info=True)
            raise

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        try:
            # Create a backup of the existing metadata file
            backup_path = self.metadata_file + ".bak"
            if os.path.exists(self.metadata_file):
                os.rename(self.metadata_file, backup_path)
                logger.debug("created_metadata_backup")

            # Write new metadata
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
                logger.debug("metadata_saved_successfully")

            # Remove backup if save was successful
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except IOError as e:
            logger.error("failed_to_save_metadata", error=str(e), exc_info=True)
            # Restore from backup if available
            if os.path.exists(backup_path):
                os.rename(backup_path, self.metadata_file)
                logger.info("restored_metadata_from_backup")
            raise

    def _get_row_count(self, cursor) -> int:
        """Get the total number of rows in the SimulationMetrics table."""
        cursor.execute("SELECT COUNT(*) FROM SimulationMetrics")
        return cursor.fetchone()[0]

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
            experiment_id = _shared_identity.experiment_id()
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
            logger.info(
                "registered_new_experiment", name=name, experiment_id=experiment_id
            )
            return experiment_id

        except Exception as e:
            logger.error(
                "failed_to_register_experiment", name=name, error=str(e), exc_info=True
            )
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
            logger.error("error_exporting_data", error=str(e), exc_info=True)
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
                    logger.info("removed_database_for_experiment", exp_id=exp_id)

                # Remove from metadata
                del self.metadata["experiments"][exp_id]
                logger.info("removed_metadata_for_experiment", exp_id=exp_id)

            self._save_metadata()
            logger.info("cleaned_up_old_experiments", count=len(experiments_to_remove))

        except Exception as e:
            logger.error("error_during_cleanup", error=str(e), exc_info=True)
            raise
