# Add the project root to the Python path
import glob
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import sqlalchemy
from sqlalchemy.orm import sessionmaker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from farm.database.models import ReproductionEventModel

EXPERIMENT_PATH = "results/one_of_a_kind_50x1000/"
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
    if not filename.endswith('.csv'):
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
        logging.info(f"Clearing existing {analysis_type} directory: {analysis_output_path}")
        if not safe_remove_directory(analysis_output_path):
            # If we couldn't remove the directory after retries, create a new one with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_output_path = os.path.join(OUTPUT_PATH, f"{analysis_type}_{timestamp}")
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
