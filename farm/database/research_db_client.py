"""Client for interacting with research database tables."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from farm.database.models import Base
from farm.database.research_models import ExperimentStats, IterationStats, Research

logger = logging.getLogger(__name__)


class ResearchDBClient:
    """Client for managing research database operations."""

    def __init__(self, db_path: str):
        """Initialize database client.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    def get_or_create_research(
        self,
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
    ) -> Research:
        """Get existing research project or create new one.

        Parameters
        ----------
        name : str
            Name of the research project
        description : str, optional
            Description of the research project
        parameters : dict, optional
            Research parameters/configuration

        Returns
        -------
        Research
            Research project record
        """
        with self.Session() as session:
            research = session.query(Research).filter_by(name=name).first()

            if not research:
                research = Research(
                    name=name,
                    description=description,
                    parameters=parameters or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(research)
                session.commit()
                logger.info(f"Created new research project: {name}")

            return research

    def add_experiment_stats(
        self,
        research_id: int,
        experiment_id: str,
        num_iterations: int,
        population_stats: Dict[str, float],
        resource_stats: Dict[str, float],
        reproduction_stats: Dict[str, float],
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
    ) -> ExperimentStats:
        """Add statistics for an experiment.

        Parameters
        ----------
        research_id : int
            ID of the research project
        experiment_id : str
            Unique identifier for the experiment
        num_iterations : int
            Number of iterations in the experiment
        population_stats : dict
            Population statistics (mean, std, max, min)
        resource_stats : dict
            Resource statistics (means and std devs)
        reproduction_stats : dict
            Reproduction statistics
        description : str, optional
            Description of the experiment
        parameters : dict, optional
            Experiment-specific parameters

        Returns
        -------
        ExperimentStats
            Created experiment statistics record
        """
        with self.Session() as session:
            exp_stats = ExperimentStats(
                research_id=research_id,
                experiment_id=experiment_id,
                timestamp=datetime.now(),
                description=description,
                parameters=parameters or {},
                num_iterations=num_iterations,
                # Population metrics
                mean_population=population_stats["mean"],
                std_population=population_stats["std"],
                max_population=population_stats["max"],
                min_population=population_stats["min"],
                # Resource metrics
                mean_resources=resource_stats["mean_resources"],
                std_resources=resource_stats["std_resources"],
                mean_efficiency=resource_stats["mean_efficiency"],
                std_efficiency=resource_stats["std_efficiency"],
                # Reproduction metrics
                mean_success_rate=reproduction_stats["mean_success_rate"],
                std_success_rate=reproduction_stats["std_success_rate"],
                total_reproduction_attempts=reproduction_stats["total_attempts"],
                total_successful_reproductions=reproduction_stats["total_successes"],
            )

            session.add(exp_stats)
            session.commit()
            logger.info(f"Added stats for experiment: {experiment_id}")

            return exp_stats

    def add_iteration_stats(
        self,
        experiment_id: int,
        iteration_id: str,
        population_stats: Dict[str, float],
        resource_stats: Dict[str, float],
        reproduction_stats: Dict[str, float],
    ) -> IterationStats:
        """Add statistics for an experiment iteration.

        Parameters
        ----------
        experiment_id : int
            ID of the experiment
        iteration_id : str
            Identifier for this iteration
        population_stats : dict
            Population statistics for the iteration
        resource_stats : dict
            Resource statistics for the iteration
        reproduction_stats : dict
            Reproduction statistics for the iteration

        Returns
        -------
        IterationStats
            Created iteration statistics record
        """
        with self.Session() as session:
            iter_stats = IterationStats(
                experiment_id=experiment_id,
                iteration_id=iteration_id,
                # Population stats
                avg_population=population_stats["avg"],
                max_population=population_stats["max"],
                min_population=population_stats["min"],
                # Resource stats
                avg_resources=resource_stats["avg_resources"],
                resource_efficiency=resource_stats["efficiency"],
                # Reproduction stats
                reproduction_attempts=reproduction_stats["attempts"],
                successful_reproductions=reproduction_stats["successes"],
                reproduction_rate=reproduction_stats["success_rate"],
            )

            session.add(iter_stats)
            session.commit()
            logger.info(f"Added stats for iteration: {iteration_id}")

            return iter_stats

    def get_experiment_stats(self, research_id: int) -> List[ExperimentStats]:
        """Get all experiment statistics for a research project.

        Parameters
        ----------
        research_id : int
            ID of the research project

        Returns
        -------
        List[ExperimentStats]
            List of experiment statistics records
        """
        with self.Session() as session:
            return (
                session.query(ExperimentStats).filter_by(research_id=research_id).all()
            )

    def get_iteration_stats(self, experiment_id: int) -> List[IterationStats]:
        """Get all iteration statistics for an experiment.

        Parameters
        ----------
        experiment_id : int
            ID of the experiment

        Returns
        -------
        List[IterationStats]
            List of iteration statistics records
        """
        with self.Session() as session:
            return (
                session.query(IterationStats)
                .filter_by(experiment_id=experiment_id)
                .all()
            )
