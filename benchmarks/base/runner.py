"""
Benchmark runner class for AgentFarm.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Type, Union

from benchmarks.base.benchmark import Benchmark
from benchmarks.base.results import BenchmarkResults


class BenchmarkRunner:
    """
    Class for running benchmarks.
    
    This class provides methods for running benchmarks and managing their results.
    
    Attributes
    ----------
    output_dir : str
        Directory to save benchmark results
    benchmarks : Dict[str, Benchmark]
        Dictionary of registered benchmarks
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """
        Initialize a new benchmark runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.benchmarks = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def register_benchmark(self, benchmark: Benchmark) -> None:
        """
        Register a benchmark with the runner.
        
        Parameters
        ----------
        benchmark : Benchmark
            Benchmark to register
        """
        self.benchmarks[benchmark.name] = benchmark
    
    def run_benchmark(self, name: str, iterations: int = 1) -> BenchmarkResults:
        """
        Run a registered benchmark.
        
        Parameters
        ----------
        name : str
            Name of the benchmark to run
        iterations : int
            Number of iterations to run
            
        Returns
        -------
        BenchmarkResults
            Results of the benchmark run
            
        Raises
        ------
        KeyError
            If the benchmark is not registered
        """
        if name not in self.benchmarks:
            raise KeyError(f"Benchmark '{name}' not registered")
        
        benchmark = self.benchmarks[name]
        
        # Create a directory for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"{name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run the benchmark
        print(f"Running benchmark '{name}'...")
        start_time = time.time()
        results = benchmark.execute(iterations)
        end_time = time.time()
        
        # Save the results
        results_file = results.save(run_dir)
        
        # Print summary
        print(f"Benchmark '{name}' completed in {end_time - start_time:.2f} seconds")
        print(f"Results saved to {results_file}")
        
        return results
    
    def run_all_benchmarks(self, iterations: int = 1) -> Dict[str, BenchmarkResults]:
        """
        Run all registered benchmarks.
        
        Parameters
        ----------
        iterations : int
            Number of iterations to run for each benchmark
            
        Returns
        -------
        Dict[str, BenchmarkResults]
            Dictionary of benchmark results
        """
        results = {}
        
        for name in self.benchmarks:
            results[name] = self.run_benchmark(name, iterations)
        
        return results
    
    def compare_benchmarks(self, benchmark1: str, benchmark2: str, 
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Compare the results of two benchmarks.
        
        Parameters
        ----------
        benchmark1 : str
            Name of the first benchmark
        benchmark2 : str
            Name of the second benchmark
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot
            
        Raises
        ------
        KeyError
            If either benchmark is not registered
        """
        if benchmark1 not in self.benchmarks:
            raise KeyError(f"Benchmark '{benchmark1}' not registered")
        if benchmark2 not in self.benchmarks:
            raise KeyError(f"Benchmark '{benchmark2}' not registered")
        
        results1 = self.benchmarks[benchmark1].results
        results2 = self.benchmarks[benchmark2].results
        
        results1.compare_with(results2, title, save_path)
    
    def load_results(self, results_file: str) -> BenchmarkResults:
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
    
    def find_results(self, benchmark_name: Optional[str] = None) -> List[str]:
        """
        Find all results files for a benchmark.
        
        Parameters
        ----------
        benchmark_name : str, optional
            Name of the benchmark to find results for.
            If None, find results for all benchmarks.
            
        Returns
        -------
        List[str]
            List of paths to results files
        """
        results_files = []
        
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith(".json"):
                    if benchmark_name is None or file.startswith(f"{benchmark_name}_"):
                        results_files.append(os.path.join(root, file))
        
        return results_files 