from typing import Any, Dict, List

import numpy as np


def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient between two lists of numeric values.

    Computes the correlation coefficient to measure the linear correlation between two
    variables. Returns a value between -1 and 1, where:
    * 1 indicates perfect positive correlation
    * -1 indicates perfect negative correlation
    * 0 indicates no linear correlation

    Args:
        x (List[float]): First list of numeric values
        y (List[float]): Second list of numeric values to correlate with x

    Returns:
        float: Correlation coefficient between -1 and 1. Returns 0 if either list is
            empty or contains only identical values (zero standard deviation).

    Example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> _calculate_correlation(x, y)
        1.0  # Perfect positive correlation
    """
    if not x or not y or len(x) != len(y):
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
    std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return covariance / (std_x * std_y)


def _normalize_dict(d: Dict[str, int]) -> Dict[str, float]:
    """Normalize dictionary values to proportions that sum to 1.0.

    Converts raw counts or frequencies in a dictionary to proportions by dividing
    each value by the sum of all values. Useful for converting counts to
    percentages or probability distributions.

    Args:
        d (Dict[str, int]): Dictionary with string keys and integer values representing
            counts or frequencies

    Returns:
        Dict[str, float]: Dictionary with the same keys but values normalized to
            proportions between 0 and 1 that sum to 1.0. Returns dictionary with
            all zeros if input sum is 0.

    Example:
        >>> counts = {"a": 2, "b": 3, "c": 5}
        >>> _normalize_dict(counts)
        {"a": 0.2, "b": 0.3, "c": 0.5}
    """
    total = sum(d.values())
    return {k: v / total if total > 0 else 0 for k, v in d.items()}


def calculate_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling mean with the specified window size.

    Args:
        data (np.ndarray): Input data array
        window (int): Size of rolling window

    Returns:
        np.ndarray: Array containing rolling means
    """
    return np.convolve(data, np.ones(window) / window, mode="valid")


def calculate_trend(data: np.ndarray) -> float:
    """Calculate the overall trend using linear regression.

    Args:
        data (np.ndarray): Input data array

    Returns:
        float: Slope of trend line, 0.0 if insufficient data
    """
    if len(data) < 2:
        return 0.0
    x = np.arange(len(data))
    slope = np.polyfit(x, data, 1)[0]
    return slope


def find_peaks(data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Find peaks in the data above the threshold.

    Args:
        data (np.ndarray): Input data array
        threshold (float): Minimum peak height

    Returns:
        np.ndarray: Array of peak values
    """
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(data[i])
    return np.array(peaks)


def get_recent_trend(data: np.ndarray, window: int) -> str:
    """Determine the trend direction in the recent window.

    Args:
        data (np.ndarray): Input data array
        window (int): Size of window to analyze

    Returns:
        str: Trend direction ('increasing', 'decreasing', or 'stable')
    """
    if len(data) < window:
        return "stable"

    recent_data = data[-window:]
    trend = calculate_trend(recent_data)

    if trend > 0.1:
        return "increasing"
    elif trend < -0.1:
        return "decreasing"
    return "stable"


def calculate_reward_stats(rewards: List[float]) -> dict:
    """Calculate comprehensive reward statistics.

    Args:
        rewards (List[float]): List of reward values for a specific action type.

    Returns:
        dict: Dictionary containing various reward statistics including:
            - average: Mean reward value
            - median: Median reward value
            - min/max: Minimum/maximum rewards
            - variance: Variance of rewards
            - std_dev: Standard deviation
            - percentiles: 25th, 50th, 75th percentiles
    """
    if not rewards:
        return {
            "average": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "variance": 0,
            "std_dev": 0,
            "percentile_25": 0,
            "percentile_50": 0,
            "percentile_75": 0,
        }

    rewards_array = np.array(rewards)
    return {
        "average": float(np.mean(rewards_array)),
        "median": float(np.median(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "variance": float(np.var(rewards_array)),
        "std_dev": float(np.std(rewards_array)),
        "percentile_25": float(np.percentile(rewards_array, 25)),
        "percentile_50": float(np.percentile(rewards_array, 50)),
        "percentile_75": float(np.percentile(rewards_array, 75)),
    }


def calculate_consistency(frequencies: np.ndarray) -> float:
    """Calculate the consistency of action frequency.

    Args:
        frequencies (np.ndarray): Array of frequency values

    Returns:
        float: Consistency score between 0 and 1
    """
    if len(frequencies) < 2:
        return 1.0
    return 1.0 - np.std(frequencies) / (np.mean(frequencies) + 1e-10)


def calculate_periodicity(data: np.ndarray) -> float:
    """Calculate the periodicity of action frequency using autocorrelation.

    Args:
        data (np.ndarray): Array of frequency or time series values

    Returns:
        float: Periodicity score between 0 and 1
    """
    if len(data) < 2:
        return 0.0

    # Calculate mean and variance
    mean = np.mean(data)
    var = np.var(data)

    # Handle zero variance case
    if var == 0:
        return 0.0

    # Normalize the data
    normalized = data - mean

    # Calculate autocorrelation
    correlation = np.correlate(normalized, normalized, mode="full")
    correlation = correlation[len(correlation) // 2 :]  # Take only the positive lags
    correlation = correlation / (var * len(data))  # Normalize

    # Find peaks in autocorrelation
    peaks = find_peaks(correlation)

    # Return mean of peaks if any found, otherwise 0
    return float(np.mean(peaks)) if len(peaks) > 0 else 0.0
