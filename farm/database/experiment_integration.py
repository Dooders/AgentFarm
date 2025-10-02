"""
Helper functions for integrating ExperimentDatabase with existing simulation code.

This module provides utilities to make it easy to switch from separate database
files to centralized experiment databases without major code changes.
"""

import os
from typing import Optional, Union
from pathlib import Path
from datetime import datetime

from farm.database.database import SimulationDatabase
from farm.database.experiment_database import ExperimentDatabase, SimulationContext
from farm.config import SimulationConfig


class DatabaseFactory:
    """Factory for creating database instances with centralized storage support.
    
    This factory makes it easy to switch between separate database files
    (original behavior) and centralized experiment databases (new behavior)
    based on configuration or environment variables.
    
    Examples
    --------
    Using environment variable to enable centralized storage:
    
    >>> os.environ['USE_EXPERIMENT_DB'] = '1'
    >>> os.environ['EXPERIMENT_ID'] = 'my_experiment'
    >>> db = DatabaseFactory.create(config=config, simulation_id='sim_001')
    
    Explicit centralized storage:
    
    >>> db = DatabaseFactory.create(
    ...     config=config,
    ...     simulation_id='sim_001',
    ...     use_experiment_db=True,
    ...     experiment_id='exp_001'
    ... )
    
    Traditional separate database files:
    
    >>> db = DatabaseFactory.create(
    ...     config=config,
    ...     db_path='simulation.db'
    ... )
    """
    
    @staticmethod
    def create(
        config: Optional[SimulationConfig] = None,
        db_path: Optional[str] = None,
        simulation_id: Optional[str] = None,
        use_experiment_db: Optional[bool] = None,
        experiment_id: Optional[str] = None,
        experiment_db_path: Optional[str] = None,
    ) -> Union[SimulationDatabase, SimulationContext]:
        """Create a database instance.
        
        This method intelligently creates either a traditional SimulationDatabase
        or a SimulationContext for centralized storage based on the provided
        parameters and environment variables.
        
        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation configuration
        db_path : str, optional
            Path for traditional separate database (ignored if using experiment DB)
        simulation_id : str, optional
            Unique identifier for this simulation (required for experiment DB)
        use_experiment_db : bool, optional
            Whether to use centralized experiment database.
            If None, checks USE_EXPERIMENT_DB environment variable.
        experiment_id : str, optional
            Experiment identifier (required if use_experiment_db is True).
            If None, uses EXPERIMENT_ID environment variable or generates one.
        experiment_db_path : str, optional
            Path to experiment database file.
            If None, uses EXPERIMENT_DB_PATH environment variable or generates one.
        
        Returns
        -------
        Union[SimulationDatabase, SimulationContext]
            Database instance or simulation context with same interface
        
        Raises
        ------
        ValueError
            If required parameters are missing for experiment database
        """
        # Check if we should use experiment database
        if use_experiment_db is None:
            use_experiment_db = os.environ.get('USE_EXPERIMENT_DB', '0') == '1'
        
        if not use_experiment_db:
            # Traditional behavior: separate database file
            if db_path is None:
                # Generate a default path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                db_path = f"simulations/simulation_{timestamp}.db"
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            return SimulationDatabase(
                db_path=db_path,
                config=config,
                simulation_id=simulation_id
            )
        
        # Centralized storage: experiment database
        if not simulation_id:
            raise ValueError(
                "simulation_id is required when using experiment database"
            )
        
        # Get or generate experiment_id
        if experiment_id is None:
            experiment_id = os.environ.get('EXPERIMENT_ID')
            if experiment_id is None:
                # Generate default experiment ID with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_id = f"exp_{timestamp}"
        
        # Get or generate experiment database path
        if experiment_db_path is None:
            experiment_db_path = os.environ.get('EXPERIMENT_DB_PATH')
            if experiment_db_path is None:
                # Generate default path
                os.makedirs("experiments", exist_ok=True)
                experiment_db_path = f"experiments/{experiment_id}.db"
        
        # Create or connect to experiment database
        experiment_db = ExperimentDatabase(
            db_path=experiment_db_path,
            experiment_id=experiment_id,
            config=config
        )
        
        # Create and return simulation context
        sim_context = experiment_db.create_simulation_context(
            simulation_id=simulation_id,
            parameters=_extract_parameters_from_config(config)
        )
        
        # Store reference to parent for cleanup
        sim_context._parent_db = experiment_db
        
        return sim_context


def _extract_parameters_from_config(config: Optional[SimulationConfig]) -> dict:
    """Extract relevant parameters from config for storage.
    
    Parameters
    ----------
    config : SimulationConfig, optional
        Configuration to extract from
    
    Returns
    -------
    dict
        Dictionary of key parameters
    """
    if config is None:
        return {}
    
    params = {}
    
    # Extract key parameters if they exist
    if hasattr(config, 'seed'):
        params['seed'] = config.seed
    
    if hasattr(config, 'population'):
        pop = config.population
        if hasattr(pop, 'system_agents'):
            params['system_agents'] = pop.system_agents
        if hasattr(pop, 'independent_agents'):
            params['independent_agents'] = pop.independent_agents
        if hasattr(pop, 'control_agents'):
            params['control_agents'] = pop.control_agents
    
    if hasattr(config, 'environment'):
        params['environment'] = str(config.environment)
    
    return params


def get_experiment_db_from_context(
    context: Union[SimulationDatabase, SimulationContext]
) -> Optional[ExperimentDatabase]:
    """Get the parent ExperimentDatabase from a context.
    
    This is useful for updating experiment-level status or querying
    across simulations.
    
    Parameters
    ----------
    context : Union[SimulationDatabase, SimulationContext]
        Database or context instance
    
    Returns
    -------
    ExperimentDatabase or None
        Parent experiment database if using centralized storage
    """
    if isinstance(context, SimulationContext):
        return context.parent_db
    elif hasattr(context, '_parent_db'):
        return context._parent_db
    return None


def close_database(db: Union[SimulationDatabase, SimulationContext]):
    """Close a database instance created by DatabaseFactory.
    
    This handles cleanup for both traditional databases and experiment contexts.
    
    Parameters
    ----------
    db : Union[SimulationDatabase, SimulationContext]
        Database or context to close
    """
    if isinstance(db, SimulationContext):
        # Flush any remaining data
        db.flush_all_buffers()
        
        # Close parent database if we own it
        if hasattr(db, '_parent_db'):
            db._parent_db.close()
    else:
        # Traditional database
        db.close()


class ExperimentManager:
    """High-level manager for running experiments with centralized storage.
    
    This class provides a simple interface for running multiple simulations
    as part of an experiment, handling all database setup and cleanup.
    
    Examples
    --------
    >>> with ExperimentManager(experiment_id='exp_001') as manager:
    ...     for i in range(10):
    ...         sim_context = manager.create_simulation(f'sim_{i}')
    ...         # Run simulation with sim_context
    ...         manager.complete_simulation(f'sim_{i}')
    """
    
    def __init__(
        self,
        experiment_id: str,
        db_path: Optional[str] = None,
        config: Optional[SimulationConfig] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize the experiment manager.
        
        Parameters
        ----------
        experiment_id : str
            Unique identifier for this experiment
        db_path : str, optional
            Path to database file (default: experiments/{experiment_id}.db)
        config : SimulationConfig, optional
            Configuration to use for database settings
        name : str, optional
            Human-readable experiment name
        description : str, optional
            Experiment description
        """
        self.experiment_id = experiment_id
        
        if db_path is None:
            os.makedirs("experiments", exist_ok=True)
            db_path = f"experiments/{experiment_id}.db"
        
        self.db_path = db_path
        self.config = config
        
        # Create experiment database
        self.experiment_db = ExperimentDatabase(
            db_path=db_path,
            experiment_id=experiment_id,
            config=config
        )
        
        # Update metadata if provided
        if name or description:
            self.experiment_db.update_experiment_status(
                status="running",
                results_summary={
                    "name": name or experiment_id,
                    "description": description or ""
                }
            )
        
        self._simulation_contexts = {}
    
    def create_simulation(
        self,
        simulation_id: str,
        parameters: Optional[dict] = None
    ) -> SimulationContext:
        """Create a new simulation in this experiment.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for this simulation
        parameters : dict, optional
            Simulation parameters to store
        
        Returns
        -------
        SimulationContext
            Context for logging simulation data
        """
        sim_context = self.experiment_db.create_simulation_context(
            simulation_id=simulation_id,
            parameters=parameters or {}
        )
        
        self._simulation_contexts[simulation_id] = sim_context
        
        return sim_context
    
    def complete_simulation(
        self,
        simulation_id: str,
        results_summary: Optional[dict] = None
    ):
        """Mark a simulation as completed.
        
        Parameters
        ----------
        simulation_id : str
            Simulation to mark as completed
        results_summary : dict, optional
            Summary of results
        """
        # Flush any remaining data
        if simulation_id in self._simulation_contexts:
            self._simulation_contexts[simulation_id].flush_all_buffers()
        
        # Update status
        self.experiment_db.update_simulation_status(
            simulation_id=simulation_id,
            status="completed",
            results_summary=results_summary
        )
    
    def get_simulation_ids(self) -> list:
        """Get all simulation IDs in this experiment.
        
        Returns
        -------
        list
            List of simulation IDs
        """
        return self.experiment_db.get_simulation_ids()
    
    def close(self):
        """Close the experiment database."""
        # Flush all simulation contexts
        for sim_context in self._simulation_contexts.values():
            sim_context.flush_all_buffers()
        
        # Close database
        self.experiment_db.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Mark experiment status based on exception
        if exc_type is not None:
            self.experiment_db.update_experiment_status(
                status="failed",
                results_summary={"error": str(exc_val)}
            )
        else:
            self.experiment_db.update_experiment_status(
                status="completed"
            )
        
        self.close()
        return False
