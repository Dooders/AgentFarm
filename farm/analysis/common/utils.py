"""
Common utility functions for analysis modules.

Extracted from database analyzers and scripts for reuse.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for a data array.

    Args:
        data: Numpy array of numeric values

    Returns:
        Dictionary containing mean, median, std, min, max, percentiles
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
    }


def calculate_trend(data: np.ndarray) -> float:
    """Calculate linear trend slope.

    Args:
        data: Time series data

    Returns:
        Slope of linear regression line
    """
    if len(data) < 2:
        return 0.0
    x = np.arange(len(data))
    return float(np.polyfit(x, data, 1)[0])


def calculate_rolling_mean(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Calculate rolling mean with specified window.

    Args:
        data: Input data array
        window: Rolling window size

    Returns:
        Array of rolling means
    """
    return np.convolve(data, np.ones(window) / window, mode="valid")


def normalize_dict(d: Dict[str, int]) -> Dict[str, float]:
    """Normalize dictionary values to proportions summing to 1.0.

    Args:
        d: Dictionary with numeric values

    Returns:
        Dictionary with normalized values
    """
    total = sum(d.values())
    return {k: v / total if total > 0 else 0 for k, v in d.items()}


def create_output_subdirs(output_path: Path, subdirs: List[str]) -> Dict[str, Path]:
    """Create output subdirectories for organized results.

    Args:
        output_path: Base output path
        subdirs: List of subdirectory names

    Returns:
        Dictionary mapping subdir names to paths
    """
    paths = {}
    for subdir in subdirs:
        path = output_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    return paths


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required: List of required column names

    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def align_time_series(data_list: List[np.ndarray], max_length: Optional[int] = None) -> np.ndarray:
    """Align multiple time series to same length by padding.

    Args:
        data_list: List of arrays to align
        max_length: Target length (uses max if None)

    Returns:
        2D array with aligned series
    """
    if not data_list:
        return np.array([])

    if max_length is None:
        max_length = max(len(arr) for arr in data_list)

    aligned = []
    for arr in data_list:
        if len(arr) < max_length:
            padded = np.pad(arr, (0, max_length - len(arr)), mode="edge")
        else:
            padded = arr[:max_length]
        aligned.append(padded)

    return np.array(aligned)


# ============================================================================
# Data Loading Patterns
# ============================================================================


def find_database_path(experiment_path: Path, db_filename: str = "simulation.db") -> Path:
    """Find database file in experiment directory.

    Args:
        experiment_path: Path to experiment directory
        db_filename: Name of database file

    Returns:
        Path to database file

    Raises:
        FileNotFoundError: If database not found
    """
    from farm.utils.logging_config import get_logger

    logger = get_logger(__name__)

    # Try direct location first
    db_path = experiment_path / db_filename
    if db_path.exists():
        logger.debug(f"Found database at: {db_path}")
        return db_path

    # Try in data subdirectory
    db_path = experiment_path / "data" / db_filename
    if db_path.exists():
        logger.debug(f"Found database at: {db_path}")
        return db_path

    # Try experiment_path itself if it's a db file
    if experiment_path.is_file() and experiment_path.name == db_filename:
        logger.debug(f"Found database at: {experiment_path}")
        return experiment_path

    raise FileNotFoundError(f"No database '{db_filename}' found in {experiment_path} or {experiment_path}/data")


def convert_dict_to_dataframe(data: Dict[str, Any], step_column: str = "step") -> pd.DataFrame:
    """Convert dictionary data to DataFrame with consistent structure.

    Args:
        data: Dictionary with numeric values or lists
        step_column: Name for step/iteration column if needed

    Returns:
        DataFrame with data
    """
    if not data:
        return pd.DataFrame()

    # If data contains lists of equal length, create time series DataFrame
    list_values = [v for v in data.values() if isinstance(v, list) and len(v) > 0]
    if list_values and all(len(v) == len(list_values[0]) for v in list_values):
        # Time series data
        df = pd.DataFrame(data)
        if step_column not in df.columns:
            df[step_column] = range(len(df))
        return df

    # Single value per key - create single row DataFrame
    return pd.DataFrame([data])


def load_data_with_csv_fallback(
    experiment_path: Path,
    csv_filename: str,
    db_loader_func: Optional[Callable[[], pd.DataFrame]] = None,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """Load data from database with CSV fallback.

    Attempts to load data from database first, then falls back to CSV files
    if database loading fails or returns empty data.

    Args:
        experiment_path: Path to experiment directory
        csv_filename: Name of CSV file to load as fallback (e.g., "actions.csv")
        db_loader_func: Function to load from database, should return DataFrame or None
        logger: Logger instance for logging messages

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If neither database nor CSV data is found
    """
    df = None

    # Try database loading first
    if db_loader_func is not None:
        try:
            df = db_loader_func()
        except Exception as e:
            if logger:
                logger.warning(f"Database loading failed: {e}. Falling back to CSV files")
            df = None

    # Fallback to CSV loading
    if df is None or df.empty:
        if logger:
            logger.info("Loading data from CSV files")
        data_dir = experiment_path / "data"
        if data_dir.exists():
            csv_path = data_dir / csv_filename
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"No {csv_filename} found in {experiment_path}")
        else:
            raise FileNotFoundError(f"No data directory found in {experiment_path}")

    return df


# ============================================================================
# Analysis Patterns
# ============================================================================


def save_analysis_results(results: Dict[str, Any], filename: str, output_path: Path) -> Path:
    """Save analysis results to JSON file.

    Args:
        results: Dictionary of results to save
        filename: Output filename
        output_path: Directory to save to

    Returns:
        Path to saved file
    """
    output_file = output_path / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return output_file


def compute_basic_metrics(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute basic statistics for specified columns.

    Args:
        df: DataFrame with data
        columns: Columns to compute statistics for

    Returns:
        Dictionary mapping column names to statistics
    """
    metrics = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            data = df[col].dropna().values
            if len(data) > 0:
                metrics[col] = calculate_statistics(data)
    return metrics


# ============================================================================
# Plotting Patterns
# ============================================================================


def setup_plot_figure(
    n_plots: int = 1, figsize: Optional[Tuple[int, int]] = None, style: str = "default"
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """Set up matplotlib figure with consistent styling.

    Args:
        n_plots: Number of subplots
        figsize: Figure size (auto-calculated if None)
        style: Matplotlib style

    Returns:
        Tuple of (figure, axes)
    """
    plt.style.use(style)

    if figsize is None:
        figsize = (5 * n_plots, 5) if n_plots > 1 else (8, 6)

    if n_plots == 1:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        return fig, axes


def save_plot_figure(
    fig: plt.Figure, output_path: Path, filename: str, dpi: int = 300, bbox_inches: str = "tight"
) -> Path:
    """Save matplotlib figure with consistent parameters.

    Args:
        fig: Matplotlib figure
        output_path: Directory to save to
        filename: Output filename
        dpi: Resolution
        bbox_inches: Bounding box setting

    Returns:
        Path to saved file
    """
    output_file = output_path / filename
    fig.savefig(output_file, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return output_file


def get_agent_type_colors() -> Dict[str, str]:
    """Get consistent color scheme for agent types.

    Returns:
        Dictionary mapping agent types to colors
    """
    return {
        "system": "blue",
        "independent": "red",
        "control": "#DAA520",  # Goldenrod
    }


def normalize_agent_type_names(agent_types: List[str]) -> List[str]:
    """Normalize agent type names to consistent format.

    Args:
        agent_types: List of agent type names

    Returns:
        List of normalized names
    """
    mapping = {
        "SystemAgent": "system",
        "IndependentAgent": "independent",
        "ControlAgent": "control",
        "system": "system",
        "independent": "independent",
        "control": "control",
    }

    return [mapping.get(at, at.lower()) for at in agent_types]


# ============================================================================
# Validation Patterns
# ============================================================================


def validate_data_quality(
    df: pd.DataFrame, min_rows: int = 1, required_numeric_cols: Optional[List[str]] = None
) -> None:
    """Validate DataFrame meets quality requirements.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_numeric_cols: Columns that must be numeric

    Raises:
        ValueError: If validation fails
    """
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")

    if required_numeric_cols:
        for col in required_numeric_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")


def handle_missing_data(df: pd.DataFrame, strategy: str = "drop", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Handle missing data in DataFrame.

    Args:
        df: DataFrame with potential missing data
        strategy: How to handle missing data ('drop', 'fill_mean', 'fill_zero')
        columns: Specific columns to handle (all numeric if None)

    Returns:
        DataFrame with missing data handled
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == "drop":
        return df.dropna(subset=columns)
    elif strategy == "fill_mean":
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        return df
    elif strategy == "fill_zero":
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df
    else:
        return df
