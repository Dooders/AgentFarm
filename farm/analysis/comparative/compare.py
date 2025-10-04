"""
Comparative analysis comparison functions.
"""

import os
from pathlib import Path
from typing import Optional

from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService


def compare_simulations(search_path: str, analysis_path: str) -> None:
    """Compare simulations found in search_path and save analysis to analysis_path.

    This function performs comparative analysis between multiple simulations
    by running the comparative analysis module on all simulation data found
    in the search path.

    Args:
        search_path: Path to directory containing simulation data
        analysis_path: Path to directory where analysis results should be saved
    """
    search_path = Path(search_path)
    analysis_path = Path(analysis_path)

    if not search_path.exists():
        raise ValueError(f"Search path does not exist: {search_path}")

    analysis_path.mkdir(parents=True, exist_ok=True)

    # Initialize analysis service
    config_service = EnvConfigService()
    service = AnalysisService(config_service)

    # Find all simulation directories
    simulation_dirs = []
    if search_path.is_dir():
        # Look for simulation subdirectories
        for item in search_path.iterdir():
            if item.is_dir() and (item / "simulation.db").exists():
                simulation_dirs.append(item)
    else:
        # Single simulation directory
        if (search_path / "simulation.db").exists():
            simulation_dirs.append(search_path)

    if len(simulation_dirs) < 2:
        raise ValueError(f"Need at least 2 simulations to compare, found {len(simulation_dirs)}")

    # Create comparative analysis request
    # For now, we'll run basic comparative analysis
    # In a full implementation, this would collect data from all simulations
    # and create comparative datasets

    print(f"Found {len(simulation_dirs)} simulations to compare")
    print(f"Analysis results will be saved to: {analysis_path}")

    # TODO: Implement full comparative analysis logic
    # For now, create a placeholder result file
    result_file = analysis_path / "comparison_summary.txt"
    with open(result_file, 'w') as f:
        f.write(f"Comparative analysis of {len(simulation_dirs)} simulations\\n")
        f.write(f"Simulation directories: {[str(d) for d in simulation_dirs]}\\n")
        f.write("\\nNote: Full comparative analysis implementation pending\\n")

    print(f"Comparative analysis placeholder created at: {result_file}")


def compare_simulations_by_id(exp1_path: str, exp2_path: str) -> None:
    """Compare two specific experiments by their paths.

    Args:
        exp1_path: Path to first experiment
        exp2_path: Path to second experiment
    """
    # This is a simpler version that just calls the main function
    # In practice, this might have different logic for pairwise comparison
    compare_simulations(exp1_path, exp2_path)
