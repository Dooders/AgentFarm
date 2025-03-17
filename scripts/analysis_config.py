# Add the project root to the Python path
import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


EXPERIMENT_PATH = "results/one_of_a_kind_500x2000/"
DATA_PATH = EXPERIMENT_PATH + "experiments/data/"
OUTPUT_PATH = EXPERIMENT_PATH + "experiments/analysis/"


def find_latest_experiment_path():
    """
    Find the most recent experiment folder and verify it contains iteration folders.

    Returns
    -------
    str
        Path to the most recent experiment folder containing iteration folders
    """
    # Find the most recent experiment folder in DATA_PATH
    experiment_folders = [
        d for d in glob.glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(d)
    ]
    if not experiment_folders:
        logging.error(f"No experiment folders found in {DATA_PATH}")
        return None

    # Sort by modification time (most recent first)
    experiment_folders.sort(key=os.path.getmtime, reverse=True)
    experiment_path = experiment_folders[0]

    # Check if experiment_path contains iteration folders directly
    iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
    if not iteration_folders:
        # If no iteration folders found directly, look for subdirectories that might contain them
        subdirs = [
            d for d in glob.glob(os.path.join(experiment_path, "*")) if os.path.isdir(d)
        ]
        if subdirs:
            # Sort by modification time (most recent first)
            subdirs.sort(key=os.path.getmtime, reverse=True)
            experiment_path = subdirs[0]
            logging.info(f"Using subdirectory as experiment path: {experiment_path}")

            # Verify that this subdirectory contains iteration folders
            iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
            if not iteration_folders:
                logging.error(f"No iteration folders found in {experiment_path}")
                return None
        else:
            logging.error(f"No subdirectories found in {experiment_path}")
            return None

    logging.info(f"Using experiment path: {experiment_path}")
    return experiment_path


def save_analysis_data(df, output_path, filename):
    """
    Save analysis dataframe to CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing analysis data
    output_path : str
        Directory path where the CSV file will be saved
    filename : str
        Name of the CSV file (without extension)

    Returns
    -------
    str
        Full path to the saved CSV file
    """
    # Ensure filename has .csv extension
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    output_csv = os.path.join(output_path, filename)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

    return output_csv


def setup_analysis_directory(analysis_type):
    """
    Set up an analysis output directory for a specific analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., 'dominance', 'reproduction', etc.)

    Returns
    -------
    tuple
        (output_path, log_file) where output_path is the path to the created directory
        and log_file is the path to the log file
    """
    # Create analysis output directory
    analysis_output_path = os.path.join(OUTPUT_PATH, analysis_type)

    # Clear the directory if it exists
    if os.path.exists(analysis_output_path):
        logging.info(
            f"Clearing existing {analysis_type} directory: {analysis_output_path}"
        )
        if not safe_remove_directory(analysis_output_path):
            # If we couldn't remove the directory after retries, create a new one with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_output_path = os.path.join(
                OUTPUT_PATH, f"{analysis_type}_{timestamp}"
            )
            logging.info(f"Using alternative directory: {analysis_output_path}")

    # Create the directory
    os.makedirs(analysis_output_path, exist_ok=True)

    # Set up logging to the analysis directory
    log_file = setup_logging(analysis_output_path)

    logging.info(f"Saving {analysis_type} analysis results to {analysis_output_path}")

    return analysis_output_path, log_file


# Setup logging to both console and file
def setup_logging(output_dir):
    """
    Set up logging to both console and file.

    Parameters
    ----------
    output_dir : str
        Directory to save the log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"analysis_log.txt")

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging to {log_file}")
    return log_file


def safe_remove_directory(directory_path, max_retries=3, retry_delay=1):
    """
    Safely remove a directory with retries.

    Parameters
    ----------
    directory_path : str
        Path to the directory to remove
    max_retries : int
        Maximum number of removal attempts
    retry_delay : float
        Delay in seconds between retries

    Returns
    -------
    bool
        True if directory was successfully removed, False otherwise
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
            return True
        except (PermissionError, OSError) as e:
            logging.warning(
                f"Attempt {attempt+1}/{max_retries} to remove directory failed: {e}"
            )
            if attempt < max_retries - 1:
                logging.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)

    return False


def check_db_schema(engine, table_name):
    """
    Check the schema of a specific table in the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the database
    table_name : str
        Name of the table to check

    Returns
    -------
    dict
        Dictionary with table schema information
    """
    try:
        inspector = sqlalchemy.inspect(engine)

        if table_name not in inspector.get_table_names():
            return {"exists": False}

        columns = inspector.get_columns(table_name)
        column_names = [col["name"] for col in columns]

        # Get primary key
        pk = inspector.get_pk_constraint(table_name)

        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)

        # Get indexes
        indexes = inspector.get_indexes(table_name)

        return {
            "exists": True,
            "column_count": len(columns),
            "columns": column_names,
            "primary_key": pk,
            "foreign_keys": fks,
            "indexes": indexes,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


# def check_reproduction_events(experiment_path):
#     #! probably deleting this
#     """
#     Check if reproduction events exist in the simulation databases.

#     Parameters
#     ----------
#     experiment_path : str
#         Path to the experiment folder containing simulation databases

#     Returns
#     -------
#     bool
#         True if any reproduction events are found, False otherwise
#     """
#     logging.info("Checking for reproduction events in databases...")

#     # Find all simulation folders
#     sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
#     total_events = 0
#     checked_dbs = 0

#     for folder in sim_folders[:5]:  # Check first 5 databases
#         # Check if this is a simulation folder with a database
#         db_path = os.path.join(folder, "simulation.db")

#         if not os.path.exists(db_path):
#             continue

#         try:
#             # Connect to the database
#             engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
#             Session = sessionmaker(bind=engine)
#             session = Session()

#             # Check if ReproductionEventModel table exists
#             inspector = sqlalchemy.inspect(engine)
#             if "reproduction_events" not in inspector.get_table_names():
#                 logging.warning(
#                     f"Database {db_path} does not have a reproduction_events table"
#                 )

#                 # Check what tables do exist
#                 tables = inspector.get_table_names()
#                 logging.info(f"Available tables: {', '.join(tables)}")

#                 session.close()
#                 continue

#             # Check the schema of the reproduction_events table
#             schema_info = check_db_schema(engine, "reproduction_events")
#             if schema_info["exists"]:
#                 logging.info(
#                     f"reproduction_events table schema: {len(schema_info['columns'])} columns"
#                 )
#                 required_columns = [
#                     "event_id",
#                     "step_number",
#                     "parent_id",
#                     "offspring_id",
#                     "success",
#                     "parent_resources_before",
#                     "parent_resources_after",
#                 ]
#                 missing_columns = [
#                     col for col in required_columns if col not in schema_info["columns"]
#                 ]
#                 if missing_columns:
#                     logging.warning(
#                         f"Missing required columns in reproduction_events: {', '.join(missing_columns)}"
#                     )

#             # Count reproduction events
#             event_count = session.query(ReproductionEventModel).count()
#             total_events += event_count
#             checked_dbs += 1

#             logging.info(
#                 f"Database {os.path.basename(folder)}: {event_count} reproduction events"
#             )

#             # If we have events, check a sample
#             if event_count > 0:
#                 sample_event = session.query(ReproductionEventModel).first()
#                 logging.info(
#                     f"Sample event: parent_id={sample_event.parent_id}, success={sample_event.success}"
#                 )

#                 # Check if we have resource data
#                 has_resource_data = (
#                     hasattr(sample_event, "parent_resources_before")
#                     and sample_event.parent_resources_before is not None
#                 )
#                 if not has_resource_data:
#                     logging.warning("Sample event is missing resource data")

#             # Close the session
#             session.close()

#         except Exception as e:
#             logging.error(f"Error checking reproduction events in {folder}: {e}")
#             import traceback

#             logging.error(traceback.format_exc())

#     if checked_dbs > 0:
#         logging.info(
#             f"Found {total_events} total reproduction events in {checked_dbs} checked databases"
#         )
#         return total_events > 0
#     else:
#         logging.warning("Could not check any databases for reproduction events")
#         return False


def get_valid_numeric_columns(df, column_list):
    """
    Filter columns to only include numeric ones with sufficient non-zero values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    column_list : list
        List of column names to filter

    Returns
    -------
    list
        List of valid numeric column names
    """
    numeric_cols = []
    for col in column_list:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if column has enough non-zero values
                non_zero_count = (df[col] != 0).sum()
                if non_zero_count > 5:  # Need at least 5 non-zero values for analysis
                    numeric_cols.append(col)
                else:
                    logging.info(
                        f"Skipping column {col} with only {non_zero_count} non-zero values"
                    )
            else:
                logging.info(f"Skipping non-numeric reproduction column: {col}")
    return numeric_cols


def run_analysis(
    analysis_type: str,
    data_processor: Callable = None,
    analysis_functions: List[Callable] = None,
    db_filename: str = None,
    load_data_function: Callable = None,
    processor_kwargs: Dict[str, Any] = None,
    analysis_kwargs: Dict[str, Dict[str, Any]] = None,
    function_group: str = "all",
):
    """
    Generic function to run any type of analysis module.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., 'dominance', 'reproduction', etc.)
    data_processor : Callable, optional
        Function to process raw data and return a DataFrame or save to DB
        If None, will use the module's data processor
    analysis_functions : List[Callable], optional
        List of analysis/visualization functions to run on the processed data
        If None, will use the module's analysis functions for the specified group
    db_filename : str, optional
        Name of the database file if using a database
        If None, will use the module's database filename
    load_data_function : Callable, optional
        Function to load data from database if data_processor saves to DB
        If None, will use the module's database loader
    processor_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the data_processor
    analysis_kwargs : Dict[str, Dict[str, Any]], optional
        Dictionary mapping function names to their keyword arguments
    function_group : str, optional
        Name of the function group to run if analysis_functions is None

    Returns
    -------
    Tuple
        (output_path, df) where output_path is the path to the analysis directory
        and df is the processed DataFrame
    """
    # Set up the analysis directory
    output_path, log_file = setup_analysis_directory(analysis_type)
    logging.info(f"Saving results to {output_path}")

    # Try to use the module system first
    try:
        from farm.analysis.registry import get_module, register_modules

        # Register all modules
        register_modules()

        # Get the module for this analysis type
        module = get_module(analysis_type)

        if module:
            logging.info(f"Using module system for {analysis_type} analysis")

            # Use the module's run_analysis method
            return module.run_analysis(
                experiment_path=find_latest_experiment_path(),
                output_path=output_path,
                group=function_group,
                processor_kwargs=processor_kwargs,
                analysis_kwargs=analysis_kwargs,
            )
    except ImportError:
        logging.warning("Module system not available, falling back to legacy mode")
    except Exception as e:
        logging.error(f"Error using module system: {e}")
        import traceback

        logging.error(traceback.format_exc())
        logging.warning("Falling back to legacy mode")

    # Legacy mode - use the provided functions
    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return output_path, None

    logging.info(f"Analyzing simulations in {experiment_path}...")

    # Initialize default kwargs if not provided
    if processor_kwargs is None:
        processor_kwargs = {}

    if analysis_kwargs is None:
        analysis_kwargs = {}

    # Process data
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
        if df is None and load_data_function:
            logging.info("Loading data from database for visualization...")
            df = load_data_function(db_uri)
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
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {output_path}")

    return output_path, df


def setup_and_process_simulations(
    experiment_path: str,
    process_simulation_func: Callable,
    process_kwargs: Dict[str, Any] = None,
    show_progress: bool = True,
    progress_interval: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find all simulation folders, set up database connections, and process each simulation.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing simulation data
    process_simulation_func : Callable
        Function that processes a single simulation. Should accept:
        - session: SQLAlchemy session
        - iteration: int
        - config: dict
        - **process_kwargs
        And return a dict of data for the simulation or None on error
    process_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the process_simulation_func
    show_progress : bool, optional
        Whether to show progress information, defaults to True
    progress_interval : int, optional
        Interval (number of simulations) at which to log progress, defaults to 10

    Returns
    -------
    List[Dict[str, Any]]
        List of data dictionaries for all successfully processed simulations
    """
    if process_kwargs is None:
        process_kwargs = {}

    data = []

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
    total_simulations = len(sim_folders)
    logging.info(f"Found {total_simulations} simulation folders to analyze")

    # Track progress
    processed_count = 0
    successful_count = 0
    error_count = 0
    start_time = time.time()

    for folder in sim_folders:
        folder_start_time = time.time()
        processed_count += 1

        # Log progress based on specified interval
        if show_progress and (
            processed_count <= 5
            or processed_count % progress_interval == 0
            or processed_count == total_simulations
        ):
            elapsed_time = time.time() - start_time
            avg_time_per_sim = (
                elapsed_time / processed_count if processed_count > 0 else 0
            )
            estimated_remaining = avg_time_per_sim * (
                total_simulations - processed_count
            )

            logging.info(
                f"Processing simulation {processed_count}/{total_simulations} "
                f"({(processed_count/total_simulations)*100:.1f}%) - "
                f"Elapsed: {elapsed_time:.1f}s, "
                f"Est. remaining: {estimated_remaining:.1f}s"
            )

        # Check if this is a simulation folder with a database
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not (os.path.exists(db_path) and os.path.exists(config_path)):
            logging.warning(f"Skipping {folder}: Missing database or config file")
            error_count += 1
            continue

        try:
            # Extract the iteration number from the folder name
            folder_name = os.path.basename(folder)
            if folder_name.startswith("iteration_"):
                iteration = int(folder_name.split("_")[1])
                if show_progress:
                    logging.debug(f"Processing iteration {iteration}")
            else:
                logging.warning(f"Skipping {folder}: Invalid folder name format")
                error_count += 1
                continue

            # Load the configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            # Connect to the database
            if show_progress:
                logging.debug(f"Connecting to database: {db_path}")
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Process the simulation using the provided function
            try:
                sim_data = process_simulation_func(
                    session=session,
                    iteration=iteration,
                    config=config,
                    **process_kwargs,
                )

                if sim_data:
                    data.append(sim_data)
                    successful_count += 1
                else:
                    logging.warning(f"No data returned for iteration {iteration}")
                    error_count += 1
            finally:
                # Ensure session is closed even if processing fails
                session.close()

            # Log time taken for this simulation
            if show_progress:
                sim_duration = time.time() - folder_start_time
                logging.debug(f"Completed iteration {iteration} in {sim_duration:.2f}s")

        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")
            import traceback

            logging.error(traceback.format_exc())
            error_count += 1

    # Log final results
    total_duration = time.time() - start_time
    logging.info(
        f"Completed analysis of {successful_count} simulations "
        f"({error_count} errors) in {total_duration:.2f}s "
        f"(avg: {total_duration/max(successful_count, 1):.2f}s per simulation)"
    )

    return data
