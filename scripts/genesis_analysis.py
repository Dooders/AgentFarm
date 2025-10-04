#!/usr/bin/env python3
"""
Genesis Analysis Script

This script analyzes how initial states and conditions impact simulation outcomes
using the unified analysis framework.

Usage:
    python genesis_analysis.py

The script automatically finds the most recent experiment in the DATA_PATH
defined in analysis_config.py and saves results to the OUTPUT_PATH.
"""

import glob
import os
from datetime import datetime
from pathlib import Path

# Import analysis framework
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Import analysis configuration
from analysis_config import DATA_PATH, OUTPUT_PATH, safe_remove_directory, setup_logging


def main():
    """Main function to run the Genesis analysis."""
    start_time = datetime.now()
    print("Starting genesis analysis script...")

    try:
        # Create genesis output directory
        genesis_output_path = os.path.join(OUTPUT_PATH, "genesis")

        # Clear the genesis directory if it exists
        if os.path.exists(genesis_output_path):
            print(f"Clearing existing genesis directory: {genesis_output_path}")
            if not safe_remove_directory(genesis_output_path):
                # Create timestamped alternative
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                genesis_output_path = os.path.join(OUTPUT_PATH, f"genesis_{timestamp}")
                print(f"Using alternative directory: {genesis_output_path}")

        # Create the directory
        os.makedirs(genesis_output_path, exist_ok=True)

        # Set up logging
        log_file = setup_logging(genesis_output_path)
        print(f"Saving results to {genesis_output_path}")

        # Find the most recent experiment folder
        if not os.path.exists(DATA_PATH):
            print(f"DATA_PATH does not exist: {DATA_PATH}")
            return

        experiment_folders = [
            d for d in glob.glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(d)
        ]
        if not experiment_folders:
            print(f"No experiment folders found in {DATA_PATH}")
            return

        # Sort by modification time (most recent first)
        experiment_folders.sort(key=os.path.getmtime, reverse=True)
        experiment_path = experiment_folders[0]
        print(f"Found most recent experiment folder: {experiment_path}")

        # Verify experiment structure
        iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
        if not iteration_folders:
            # Check subdirectories
            subdirs = [
                d for d in glob.glob(os.path.join(experiment_path, "*")) if os.path.isdir(d)
            ]
            if subdirs:
                subdirs.sort(key=os.path.getmtime, reverse=True)
                experiment_path = subdirs[0]
                iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
                if not iteration_folders:
                    print(f"No iteration folders found in {experiment_path}")
                    return
            else:
                print(f"No valid experiment structure found")
                return

        print(f"Found {len(iteration_folders)} iteration folders")

        # Initialize analysis service
        config_service = EnvConfigService()
        service = AnalysisService(config_service)

        # Run comprehensive genesis analysis
        print("Running genesis analysis...")
        request = AnalysisRequest(
            module_name="genesis",
            experiment_path=Path(experiment_path),
            output_path=Path(genesis_output_path),
            group="all"  # Run all analysis functions
        )

        result = service.run(request)

        if result.success:
            duration = datetime.now() - start_time
            print(f"✅ Analysis complete in {duration.total_seconds():.2f} seconds")
            print(f"📁 Results saved to: {result.output_path}")
            print(f"📊 Generated {len(result.generated_files)} files:")
            for file_path in result.generated_files:
                print(f"   - {file_path}")
        else:
            print(f"❌ Analysis failed: {result.error}")

        # Log completion
        total_duration = datetime.now() - start_time
        print(f"\nAnalysis completed. Total execution time: {total_duration.total_seconds():.2f} seconds")
        print(f"Log file saved to: {log_file}")
        print(f"All results saved to: {genesis_output_path}")

    except Exception as e:
        print(f"❌ Unhandled exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
