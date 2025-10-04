#!/usr/bin/env python
"""
Comprehensive examples of using the farm.analysis module system.

This file demonstrates 7 complete examples of how to use the new analysis
modules for different types of simulation analysis.
"""

import time
from pathlib import Path

from farm.analysis.registry import get_module, get_module_names
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.core.services import EnvConfigService


def example_1_basic_population_analysis():
    """Example 1: Basic population analysis on a single experiment."""
    print("=== Example 1: Basic Population Analysis ===")

    # Initialize service
    config_service = EnvConfigService()
    service = AnalysisService(config_service)

    # Assume we have an experiment directory
    experiment_path = Path("docs/sample")
    if not experiment_path.exists():
        print("Note: This example assumes experiment data exists at docs/sample")
        print("Creating mock directory for demonstration...")
        experiment_path.mkdir(parents=True, exist_ok=True)
        # In real usage, this would contain simulation.db and data files

    # Create analysis request
    request = AnalysisRequest(
        module_name="population",
        experiment_path=experiment_path,
        output_path=Path("results/population_basic"),
        group="basic",  # Run basic analysis (stats + plots)
    )

    # Run analysis
    print("Running population analysis...")
    start_time = time.time()
    result = service.run(request)
    end_time = time.time()

    if result.success:
        print("✅ Analysis completed successfully!")
        print(f"⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"📁 Results saved to: {result.output_path}")

        # List generated files
        if result.output_path.exists():
            files = list(result.output_path.glob("*"))
            print(f"📄 Generated files: {len(files)}")
            for file in sorted(files):
                print(f"   - {file.name}")

    else:
        print(f"❌ Analysis failed: {result.error}")

    print()


def example_2_comprehensive_multi_module_analysis():
    """Example 2: Run multiple analysis modules on the same experiment."""
    print("=== Example 2: Comprehensive Multi-Module Analysis ===")

    service = AnalysisService(EnvConfigService())
    experiment_path = Path("docs/sample")

    # Define modules to run
    modules = ["population", "resources", "actions", "agents"]

    print(f"Running {len(modules)} analysis modules...")

    requests = []
    for module in modules:
        request = AnalysisRequest(
            module_name=module,
            experiment_path=experiment_path,
            output_path=Path(f"results/comprehensive/{module}"),
            group="all",  # Run all functions for each module
        )
        requests.append(request)

    # Run batch analysis
    start_time = time.time()
    results = service.run_batch(requests)
    end_time = time.time()

    print(f"Batch analysis completed in {end_time - start_time:.2f}s")

    # Report results
    success_count = sum(1 for r in results if r.success)
    print(f"✅ {success_count}/{len(results)} analyses completed successfully")

    for result in results:
        status = "✅" if result.success else "❌"
        print(f"   {status} {result.module_name}: {result.execution_time:.2f}s")

        if not result.success:
            print(f"      Error: {result.error}")

    print()


def example_3_custom_analysis_parameters():
    """Example 3: Analysis with custom parameters and progress tracking."""
    print("=== Example 3: Custom Analysis Parameters ===")

    service = AnalysisService(EnvConfigService())
    experiment_path = Path("docs/sample")

    # Define progress callback
    def progress_callback(message: str, progress: float):
        print(f"[{progress:>5.1%}] {message}")

    # Run population analysis with custom parameters
    request = AnalysisRequest(
        module_name="population",
        experiment_path=experiment_path,
        output_path=Path("results/custom_params"),
        group="plots",  # Only run plotting functions
        progress_callback=progress_callback,
        analysis_kwargs={
            "plot_population": {
                "figsize": (12, 8),
                "dpi": 300,
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],  # Custom colors
            },
            "plot_births_deaths": {
                "window": 20,  # Rolling window for smoothing
                "alpha": 0.8,
            },
        },
    )

    print("Running population analysis with custom parameters...")
    result = service.run(request)

    if result.success:
        print("✅ Custom analysis completed!")
        print(f"📁 Results: {result.output_path}")
    else:
        print(f"❌ Analysis failed: {result.error}")

    print()


def example_4_caching_and_performance():
    """Example 4: Demonstrate caching for performance improvement."""
    print("=== Example 4: Caching and Performance ===")

    service = AnalysisService(EnvConfigService())
    experiment_path = Path("docs/sample")

    request = AnalysisRequest(
        module_name="population",
        experiment_path=experiment_path,
        output_path=Path("results/caching_demo"),
        enable_caching=True,
    )

    print("Running analysis with caching enabled...")

    # First run
    print("First run (computing)...")
    start_time = time.time()
    result1 = service.run(request)
    first_run_time = time.time() - start_time

    if result1.success:
        print(f"First run completed in {first_run_time:.2f}s")
        # Second run (cached)
        print("Second run (cached)...")
        start_time = time.time()
        result2 = service.run(request)
        second_run_time = time.time() - start_time

        print(f"Second run completed in {second_run_time:.2f}s")
        if result2.cache_hit:
            speedup = first_run_time / second_run_time if second_run_time > 0 else float("inf")
            print(f"Speedup: {speedup:.1f}x")
        else:
            print("Note: Cache hit not detected (may be due to implementation details)")

        print(f"Cache status - First: {result1.cache_hit}, Second: {result2.cache_hit}")

    else:
        print(f"❌ Analysis failed: {result1.error}")

    print()


def example_5_batch_experiment_analysis():
    """Example 5: Analyze multiple experiments in batch."""
    print("=== Example 5: Batch Experiment Analysis ===")

    service = AnalysisService(EnvConfigService())

    # Simulate multiple experiments
    experiment_names = [f"experiment_{i:03d}" for i in range(1, 6)]  # 5 experiments

    print(f"Analyzing {len(experiment_names)} experiments...")

    requests = []
    for exp_name in experiment_names:
        exp_path = Path(f"data/{exp_name}")
        # Note: In real usage, ensure these directories exist

        request = AnalysisRequest(
            module_name="population",
            experiment_path=exp_path,
            output_path=Path(f"results/batch/{exp_name}"),
            group="basic",
        )
        requests.append(request)

    # Run batch
    start_time = time.time()
    results = service.run_batch(requests)
    batch_time = time.time() - start_time

    print(f"Batch completed in {batch_time:.2f}s")
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"Average time per analysis: {batch_time / len(results):.2f}s")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for result in failed:
            print(f"   - {result.module_name}: {result.error}")

    print()


def example_6_specialized_module_analysis():
    """Example 6: Use specialized analysis modules."""
    print("=== Example 6: Specialized Module Analysis ===")

    service = AnalysisService(EnvConfigService())
    experiment_path = Path("docs/sample")

    # Specialized modules with different analysis focuses
    specialized_analyses = [
        ("learning", "performance", "Analyze learning curves and performance"),
        ("spatial", "movement", "Analyze movement patterns and spatial behavior"),
        ("temporal", "analysis", "Analyze time series patterns"),
        ("combat", "all", "Analyze combat metrics and outcomes"),
    ]

    print("Running specialized analyses...")

    for module, group, description in specialized_analyses:
        print(f"\n📊 {description}")

        request = AnalysisRequest(
            module_name=module,
            experiment_path=experiment_path,
            output_path=Path(f"results/specialized/{module}"),
            group=group,
        )

        start_time = time.time()
        result = service.run(request)
        analysis_time = time.time() - start_time

        if result.success:
            print(f"   Completed in {analysis_time:.2f}s")
            print(f"   📁 Results: {result.output_path}")
        else:
            print(f"   ❌ Failed: {result.error}")

    print()


def example_7_error_handling_and_recovery():
    """Example 7: Error handling and recovery patterns."""
    print("=== Example 7: Error Handling and Recovery ===")

    from farm.analysis.exceptions import DataProcessingError, DataValidationError, ModuleNotFoundError

    service = AnalysisService(EnvConfigService())

    # Test cases for error handling
    error_scenarios = [
        ("nonexistent_module", "Test invalid module name"),
        ("population", "Test with invalid experiment path"),
        ("population", "Test with valid setup"),
    ]

    for module_name, description in error_scenarios:
        print(f"\n🔍 {description}")
        print(f"   Module: {module_name}")

        try:
            if "invalid experiment" in description:
                exp_path = Path("nonexistent/experiment/path")
            else:
                exp_path = Path("docs/sample")

            request = AnalysisRequest(
                module_name=module_name, experiment_path=exp_path, output_path=Path(f"results/error_test/{module_name}")
            )

            result = service.run(request)

            if result.success:
                print("   ✅ Analysis completed successfully")
            else:
                print(f"   ⚠️  Analysis completed with issues: {result.error}")

        except ModuleNotFoundError as e:
            print(f"   ❌ Module not found: {e.module_name}")
            print(f"      Available: {', '.join(e.available_modules[:5])}...")
        except DataValidationError as e:
            print("   ❌ Data validation error:")
            print(f"      Missing columns: {e.missing_columns}")
            print(f"      Invalid columns: {e.invalid_columns}")
        except DataProcessingError as e:
            print(f"   ❌ Data processing error at step: {e.step}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")

    print()


def example_8_module_discovery_and_introspection():
    """Example 8: Discover and inspect available analysis modules."""
    print("=== Example 8: Module Discovery and Introspection ===")

    service = AnalysisService(EnvConfigService())

    # List all available modules
    print("📋 Available Analysis Modules:")
    modules = service.list_modules()

    for module_info in modules:
        print(f"\n📦 {module_info['name']}")
        print(f"   {module_info['description']}")

        # Get detailed module info
        try:
            details = service.get_module_info(module_info["name"])
            print(f"   Functions: {len(details['functions'])}")
            print(f"   Groups: {list(details['function_groups'].keys())}")

        except Exception as e:
            print(f"   Could not get details: {e}")

    # Demonstrate getting a specific module
    print("\n🔍 Detailed info for 'population' module:")
    try:
        pop_info = service.get_module_info("population")
        print(f"   Description: {pop_info['description']}")
        print(f"   Functions: {pop_info['functions']}")
        print(f"   Groups: {pop_info['function_groups']}")

    except Exception as e:
        print(f"   Error getting module info: {e}")

    print()


def main():
    """Run all analysis examples."""
    print("AgentFarm Analysis Module Examples")
    print("=" * 50)
    print("This script demonstrates various ways to use the new analysis module system.")
    print("Make sure you have experiment data in the 'data/' directory for full functionality.\n")

    examples = [
        ("Basic Population Analysis", example_1_basic_population_analysis),
        ("Multi-Module Analysis", example_2_comprehensive_multi_module_analysis),
        ("Custom Parameters", example_3_custom_analysis_parameters),
        ("Caching & Performance", example_4_caching_and_performance),
        ("Batch Experiment Analysis", example_5_batch_experiment_analysis),
        ("Specialized Modules", example_6_specialized_module_analysis),
        ("Error Handling", example_7_error_handling_and_recovery),
        ("Module Discovery", example_8_module_discovery_and_introspection),
    ]

    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n{'=' * 20} Example {i}: {name} {'=' * 20}")
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in example {i}: {e}")
            import traceback

            traceback.print_exc()

        # Brief pause between examples
        if i < len(examples):
            input("\nPress Enter to continue to next example...")

    print("\n" + "=" * 50)
    print("🎉 All examples completed!")
    print("\nKey takeaways:")
    print("• Use AnalysisService for running analyses")
    print("• AnalysisRequest configures what to run")
    print("• Results are automatically saved to output_path")
    print("• Caching improves performance for repeated analyses")
    print("• Batch processing handles multiple analyses efficiently")
    print("• Error handling provides detailed feedback")
    print("• All modules follow the same API pattern")


if __name__ == "__main__":
    main()
