"""
Base benchmark class for AgentFarm.
"""

import abc
import time
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from benchmarks.base.results import BenchmarkResults


class Benchmark(abc.ABC):
    """
    Abstract base class for all benchmarks.
    
    This class defines the interface that all benchmarks must implement.
    It provides common functionality for setting up, running, and analyzing
    benchmark results.
    
    Attributes
    ----------
    name : str
        Name of the benchmark
    description : str
        Description of what the benchmark measures
    parameters : Dict[str, Any]
        Parameters for the benchmark
    results : BenchmarkResults
        Results of the benchmark run
    """
    
    def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a new benchmark.
        
        Parameters
        ----------
        name : str
            Name of the benchmark
        description : str
            Description of what the benchmark measures
        parameters : Dict[str, Any], optional
            Parameters for the benchmark
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.results = BenchmarkResults(self.name)
        self._start_time = None
        self._end_time = None
    
    @abc.abstractmethod
    def setup(self) -> None:
        """
        Set up the benchmark environment.
        
        This method should prepare any resources needed for the benchmark.
        """
        pass
    
    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Returns
        -------
        Dict[str, Any]
            Raw results from the benchmark run
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """
        Clean up after the benchmark.
        
        This method should release any resources used by the benchmark.
        """
        pass
    
    def execute(self, iterations: int = 1) -> BenchmarkResults:
        """
        Execute the benchmark for the specified number of iterations.
        
        Parameters
        ----------
        iterations : int
            Number of iterations to run
            
        Returns
        -------
        BenchmarkResults
            Results of the benchmark run
        """
        self.results = BenchmarkResults(self.name)
        self.results.set_parameters(self.parameters)
        self.results.set_metadata({
            "description": self.description,
            "iterations": iterations,
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            self.setup()
            
            for i in range(iterations):
                print(f"Running iteration {i+1}/{iterations} of benchmark '{self.name}'")
                
                # Time the execution
                self._start_time = time.time()
                raw_results = self.run()
                self._end_time = time.time()
                
                # Calculate duration
                duration = self._end_time - self._start_time
                
                # Store results for this iteration
                self.results.add_iteration_result(i, raw_results, duration)
                
            return self.results
        finally:
            self.cleanup()
    
    def save_results(self, output_dir: str) -> str:
        """
        Save benchmark results to a file.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
            
        Returns
        -------
        str
            Path to the saved results file
        """
        return self.results.save(output_dir)
    
    @classmethod
    def load_results(cls, results_file: str) -> BenchmarkResults:
        """
        Load benchmark results from a file.
        
        Parameters
        ----------
        results_file : str
            Path to the results file
            
        Returns
        -------
        BenchmarkResults
            Loaded benchmark results
        """
        return BenchmarkResults.load(results_file) 