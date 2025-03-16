"""
Base module for analysis modules.
This provides a common interface for all analysis modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


class AnalysisModule(ABC):
    """
    Base class for analysis modules.

    This class defines the interface that all analysis modules should implement.
    It provides methods for registering analysis functions, retrieving them,
    and getting module information.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the analysis module.

        Parameters
        ----------
        name : str
            Name of the module
        description : str
            Description of the module
        """
        self.name = name
        self.description = description
        self._analysis_functions = {}
        self._analysis_groups = {}
        self._registered = False

    @abstractmethod
    def register_analysis(self) -> None:
        """
        Register all analysis functions for this module.
        This method should populate self._analysis_functions and self._analysis_groups.
        """
        pass

    @abstractmethod
    def get_data_processor(self) -> Callable:
        """
        Get the data processor function for this module.

        Returns
        -------
        Callable
            The data processor function
        """
        pass

    @abstractmethod
    def get_db_loader(self) -> Optional[Callable]:
        """
        Get the database loader function for this module.

        Returns
        -------
        Optional[Callable]
            The database loader function, or None if not using a database
        """
        pass

    @abstractmethod
    def get_db_filename(self) -> Optional[str]:
        """
        Get the database filename for this module.

        Returns
        -------
        Optional[str]
            The database filename, or None if not using a database
        """
        pass

    def get_analysis_function(self, name: str) -> Optional[Callable]:
        """
        Get a specific analysis function by name.

        Parameters
        ----------
        name : str
            Name of the analysis function

        Returns
        -------
        Optional[Callable]
            The requested analysis function, or None if not found
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return self._analysis_functions.get(name)

    def get_analysis_functions(self, group: str = "all") -> List[Callable]:
        """
        Get a list of analysis functions by group.

        Parameters
        ----------
        group : str
            Name of the function group

        Returns
        -------
        List[Callable]
            List of analysis functions in the requested group
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return self._analysis_groups.get(group, [])

    def get_function_groups(self) -> List[str]:
        """
        Get a list of available function groups.

        Returns
        -------
        List[str]
            List of function group names
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return list(self._analysis_groups.keys())

    def get_function_names(self) -> List[str]:
        """
        Get a list of all available function names.

        Returns
        -------
        List[str]
            List of function names
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return list(self._analysis_functions.keys())

    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about this analysis module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing module information
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True

        return {
            "name": self.name,
            "description": self.description,
            "data_processor": self.get_data_processor(),
            "db_loader": self.get_db_loader(),
            "db_filename": self.get_db_filename(),
            "function_groups": self.get_function_groups(),
            "functions": self.get_function_names(),
        }

    def run_analysis(
        self,
        experiment_path: str,
        output_path: str,
        group: str = "all",
        processor_kwargs: Optional[Dict[str, Any]] = None,
        analysis_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Run analysis functions for this module.

        Parameters
        ----------
        experiment_path : str
            Path to the experiment folder
        output_path : str
            Path to save analysis results
        group : str
            Name of the function group to run
        processor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments for the data processor
        analysis_kwargs : Optional[Dict[str, Dict[str, Any]]]
            Dictionary mapping function names to their keyword arguments

        Returns
        -------
        Tuple[str, Optional[pd.DataFrame]]
            Tuple containing the output path and the processed DataFrame
        """
        import logging
        import os

        # Initialize default kwargs if not provided
        if processor_kwargs is None:
            processor_kwargs = {}

        if analysis_kwargs is None:
            analysis_kwargs = {}

        # Get the data processor and analysis functions
        data_processor = self.get_data_processor()
        analysis_functions = self.get_analysis_functions(group)

        if not analysis_functions:
            logging.warning(f"No analysis functions found for group '{group}'")
            return output_path, None

        # Process data
        db_filename = self.get_db_filename()
        if db_filename:
            # If using a database, set up the database path
            db_path = os.path.join(output_path, db_filename)
            db_uri = f"sqlite:///{db_path}"
            logging.info(f"Processing data and inserting directly into {db_uri}")

            # Add database parameters to processor kwargs
            db_processor_kwargs = {
                "save_to_db": True,
                "db_path": db_uri,
                **processor_kwargs,
            }

            # Process data and save to database
            df = data_processor(experiment_path, **db_processor_kwargs)

            # Load data from database if processor doesn't return it
            if df is None:
                db_loader = self.get_db_loader()
                if db_loader:
                    logging.info("Loading data from database for visualization...")
                    df = db_loader(db_uri)
        else:
            # Process data without database
            df = data_processor(experiment_path, **processor_kwargs)

        if df is None or df.empty:
            logging.warning("No simulation data found.")
            return output_path, None

        # Log summary statistics
        logging.info(f"Analyzed {len(df)} simulations.")
        logging.info("\nSummary statistics:")
        logging.info(df.describe().to_string())

        # Run each analysis function
        for func in analysis_functions:
            try:
                # Get function name for logging
                func_name = func.__name__

                # Get kwargs for this function if available
                func_kwargs = analysis_kwargs.get(func_name, {})

                logging.info(f"Running {func_name}...")
                func(df, output_path, **func_kwargs)
            except Exception as e:
                logging.error(f"Error running {func.__name__}: {e}")
                import traceback

                logging.error(traceback.format_exc())

        # Log completion
        logging.info("\nAnalysis complete. Results saved.")
        if db_filename:
            logging.info(f"Database saved to: {db_path}")
        logging.info(f"All analysis files saved to: {output_path}")

        return output_path, df
