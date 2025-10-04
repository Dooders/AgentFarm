#!/usr/bin/env python
"""
Advantage Analysis Script

This script analyzes advantages between agent types across simulations using
the unified analysis framework.

Usage:
    python advantage_analysis.py

The script will:
1. Find the most recent experiment
2. Run comprehensive advantage analysis
3. Generate visualizations and reports
4. Save results to organized output directory
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
    """Main function to run advantage analysis."""
    start_time = datetime.now()
    print("Starting advantage analysis script...")

    try:
        # Create advantage output directory
        adv_output_path = os.path.join(OUTPUT_PATH, "advantage")

        # Clear the advantage directory if it exists
        if os.path.exists(adv_output_path):
            print(f"Clearing existing advantage directory: {adv_output_path}")
            if not safe_remove_directory(adv_output_path):
                # Create timestamped alternative
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                adv_output_path = os.path.join(OUTPUT_PATH, f"advantage_{timestamp}")
                print(f"Using alternative directory: {adv_output_path}")

        # Create the directory
        os.makedirs(adv_output_path, exist_ok=True)

        # Set up logging
        log_file = setup_logging(adv_output_path)
        print(f"Saving results to {adv_output_path}")

        # Find the most recent experiment folder
        print(f"Searching for experiment folders in {DATA_PATH}")
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

        # Run comprehensive advantage analysis
        print("Running advantage analysis...")
        request = AnalysisRequest(
            module_name="advantage",
            experiment_path=Path(experiment_path),
            output_path=Path(adv_output_path),
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
        print(f"All results saved to: {adv_output_path}")

    except Exception as e:
        print(f"❌ Unhandled exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
