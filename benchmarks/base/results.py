"""
Benchmark results class for AgentFarm.
"""

import os
import json
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None


class BenchmarkResults:
    """
    Class for storing and analyzing benchmark results.
    
    This class provides methods for storing, analyzing, and visualizing
    benchmark results.
    
    Attributes
    ----------
    name : str
        Name of the benchmark
    parameters : Dict[str, Any]
        Parameters used for the benchmark
    metadata : Dict[str, Any]
        Metadata about the benchmark run
    iteration_results : List[Dict[str, Any]]
        Results from each iteration of the benchmark
    """
    
    def __init__(self, name: str):
        """
        Initialize a new benchmark results object.
        
        Parameters
        ----------
        name : str
            Name of the benchmark
        """
        self.name = name
        self.parameters = {}
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
        }
        self.iteration_results = []
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the parameters used for the benchmark.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            Parameters used for the benchmark
        """
        self.parameters = parameters
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set metadata about the benchmark run.
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata about the benchmark run
        """
        self.metadata.update(metadata)
    
    def add_iteration_result(self, iteration: int, results: Dict[str, Any], duration: float) -> None:
        """
        Add results from a benchmark iteration.
        
        Parameters
        ----------
        iteration : int
            Iteration number
        results : Dict[str, Any]
            Raw results from the benchmark run
        duration : float
            Duration of the benchmark run in seconds
        """
        self.iteration_results.append({
            "iteration": iteration,
            "duration": duration,
            "results": results,
        })
    
    def get_durations(self) -> List[float]:
        """
        Get the durations of all benchmark iterations.
        
        Returns
        -------
        List[float]
            List of durations in seconds
        """
        return [result["duration"] for result in self.iteration_results]
    
    def get_mean_duration(self) -> float:
        """
        Get the mean duration of all benchmark iterations.
        
        Returns
        -------
        float
            Mean duration in seconds
        """
        durations = self.get_durations()
        return statistics.mean(durations) if durations else 0.0
    
    def get_median_duration(self) -> float:
        """
        Get the median duration of all benchmark iterations.
        
        Returns
        -------
        float
            Median duration in seconds
        """
        durations = self.get_durations()
        return statistics.median(durations) if durations else 0.0
    
    def get_std_duration(self) -> float:
        """
        Get the standard deviation of durations of all benchmark iterations.
        
        Returns
        -------
        float
            Standard deviation of durations in seconds
        """
        durations = self.get_durations()
        return statistics.stdev(durations) if len(durations) > 1 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the benchmark results.
        
        Returns
        -------
        Dict[str, Any]
            Summary of the benchmark results
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "iterations": len(self.iteration_results),
            "mean_duration": self.get_mean_duration(),
            "median_duration": self.get_median_duration(),
            "std_duration": self.get_std_duration(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the benchmark results to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the benchmark results
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "iteration_results": self.iteration_results,
        }
    
    def save(self, output_dir: str) -> str:
        """
        Save the benchmark results to a file.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the results
            
        Returns
        -------
        str
            Path to the saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename based on the benchmark name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save the results to a JSON file
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BenchmarkResults':
        """
        Load benchmark results from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the results file
            
        Returns
        -------
        BenchmarkResults
            Loaded benchmark results
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        results = cls(data["name"])
        results.set_parameters(data["parameters"])
        results.set_metadata(data["metadata"])
        
        for iteration_result in data["iteration_results"]:
            results.iteration_results.append(iteration_result)
        
        return results
    
    def plot_durations(self, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plot the durations of all benchmark iterations.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot
        """
        if plt is None:
            print("matplotlib not available; skipping plot_durations()")
            return

        durations = self.get_durations()
        iterations = [result["iteration"] for result in self.iteration_results]

        plt.figure(figsize=(10, 6))
        plt.bar(iterations, durations)
        plt.xlabel("Iteration")
        plt.ylabel("Duration (seconds)")
        plt.title(title or f"{self.name} Benchmark Durations")
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_path:
            plt.savefig(save_path)

        plt.show()
    
    def compare_with(self, other: 'BenchmarkResults', 
                    title: Optional[str] = None, 
                    save_path: Optional[str] = None) -> None:
        """
        Compare this benchmark results with another.
        
        Parameters
        ----------
        other : BenchmarkResults
            Other benchmark results to compare with
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot
        """
        if plt is None:
            print("matplotlib not available; skipping compare_with()")
            return

        this_mean = self.get_mean_duration()
        other_mean = other.get_mean_duration()

        labels = [self.name, other.name]
        means = [this_mean, other_mean]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, means)
        plt.ylabel("Mean Duration (seconds)")
        plt.title(title or "Benchmark Comparison")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add values on top of bars
        for i, v in enumerate(means):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

        if save_path:
            plt.savefig(save_path)

        plt.show()