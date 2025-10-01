"""
Example: Using the Analysis Module System

This example demonstrates how to use the modern analysis module system
to run analysis on simulation data.
"""

from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService


def basic_analysis_example():
    """Run a basic analysis."""
    print("=" * 60)
    print("Basic Analysis Example")
    print("=" * 60)
    
    # Initialize service
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    # List available modules
    print("\nAvailable modules:")
    for module in service.list_modules():
        print(f"  - {module['name']}: {module['description']}")
    
    # Create analysis request
    request = AnalysisRequest(
        module_name="dominance",  # Use dominance analysis
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/dominance"),
        group="basic"  # Run only basic plots
    )
    
    # Run analysis
    print(f"\nRunning {request.module_name} analysis...")
    result = service.run(request)
    
    # Check results
    if result.success:
        print(f"✓ Analysis complete in {result.execution_time:.2f}s")
        print(f"  Results saved to: {result.output_path}")
        if result.dataframe is not None:
            print(f"  Processed {len(result.dataframe)} records")
    else:
        print(f"✗ Analysis failed: {result.error}")


def analysis_with_progress():
    """Run analysis with progress tracking."""
    print("\n" + "=" * 60)
    print("Analysis with Progress Tracking")
    print("=" * 60)
    
    # Progress callback
    def show_progress(message: str, progress: float):
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {progress:>5.1%} - {message}", end="", flush=True)
    
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/dominance_progress"),
        progress_callback=show_progress
    )
    
    result = service.run(request)
    print()  # New line after progress bar
    
    if result.success:
        print(f"✓ Analysis complete")
    else:
        print(f"✗ Failed: {result.error}")


def batch_analysis_example():
    """Run batch analysis on multiple experiments."""
    print("\n" + "=" * 60)
    print("Batch Analysis Example")
    print("=" * 60)
    
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    # Create batch of requests
    requests = []
    for i in range(1, 6):
        request = AnalysisRequest(
            module_name="dominance",
            experiment_path=Path(f"data/experiment_{i:03d}"),
            output_path=Path(f"results/batch/exp_{i:03d}"),
            group="basic",
            enable_caching=True
        )
        requests.append(request)
    
    print(f"\nRunning batch analysis on {len(requests)} experiments...")
    
    # Run batch
    results = service.run_batch(requests)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    print(f"\nBatch complete: {successful}/{len(results)} successful")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result.success else "✗"
        time_str = f"{result.execution_time:.2f}s" if result.success else "failed"
        cache_str = " (cached)" if result.cache_hit else ""
        print(f"  {status} Experiment {i}: {time_str}{cache_str}")


def analysis_with_caching():
    """Demonstrate caching functionality."""
    print("\n" + "=" * 60)
    print("Analysis with Caching")
    print("=" * 60)
    
    config_service = EnvConfigService()
    service = AnalysisService(
        config_service,
        cache_dir=Path(".analysis_cache")
    )
    
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/cached"),
        enable_caching=True
    )
    
    # First run
    print("\nFirst run (no cache):")
    result1 = service.run(request)
    print(f"  Execution time: {result1.execution_time:.2f}s")
    print(f"  Cache hit: {result1.cache_hit}")
    
    # Second run (should use cache)
    print("\nSecond run (with cache):")
    result2 = service.run(request)
    print(f"  Execution time: {result2.execution_time:.2f}s")
    print(f"  Cache hit: {result2.cache_hit}")
    
    # Force refresh
    print("\nThird run (force refresh):")
    request.force_refresh = True
    result3 = service.run(request)
    print(f"  Execution time: {result3.execution_time:.2f}s")
    print(f"  Cache hit: {result3.cache_hit}")
    
    # Clear cache
    print("\nClearing cache...")
    count = service.clear_cache()
    print(f"  Cleared {count} cache entries")


def custom_analysis_parameters():
    """Run analysis with custom parameters."""
    print("\n" + "=" * 60)
    print("Analysis with Custom Parameters")
    print("=" * 60)
    
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/custom"),
        group="all",
        
        # Custom parameters for data processor
        processor_kwargs={
            "min_population": 10,
            "include_metadata": True
        },
        
        # Custom parameters for specific analysis functions
        analysis_kwargs={
            "plot_dominance_distribution": {
                "bins": 50,
                "figsize": (12, 8)
            },
            "run_dominance_classification": {
                "test_size": 0.3,
                "random_state": 42
            }
        },
        
        # Add metadata
        metadata={
            "experiment_name": "Test Experiment 001",
            "researcher": "John Doe",
            "date": "2025-01-15"
        }
    )
    
    result = service.run(request)
    
    if result.success:
        print(f"✓ Analysis complete")
        print(f"\nMetadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
        
        # Save summary
        summary_path = result.save_summary()
        print(f"\nSummary saved to: {summary_path}")
    else:
        print(f"✗ Failed: {result.error}")


def module_introspection():
    """Explore module capabilities."""
    print("\n" + "=" * 60)
    print("Module Introspection")
    print("=" * 60)
    
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    # Get module info
    module_name = "dominance"
    info = service.get_module_info(module_name)
    
    print(f"\nModule: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Supports database: {info['supports_database']}")
    
    print(f"\nFunction groups:")
    for group in info['function_groups']:
        print(f"  - {group}")
    
    print(f"\nAvailable functions:")
    for func in info['functions']:
        print(f"  - {func}")


def error_handling_example():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("Error Handling Example")
    print("=" * 60)
    
    config_service = EnvConfigService()
    service = AnalysisService(config_service)
    
    # Test 1: Invalid module
    print("\nTest 1: Invalid module name")
    request = AnalysisRequest(
        module_name="nonexistent_module",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/error1")
    )
    result = service.run(request)
    print(f"  Result: {result.error if not result.success else 'Success'}")
    
    # Test 2: Missing experiment path
    print("\nTest 2: Missing experiment path")
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("data/nonexistent"),
        output_path=Path("results/error2")
    )
    result = service.run(request)
    print(f"  Result: {result.error if not result.success else 'Success'}")
    
    # Test 3: Invalid function group
    print("\nTest 3: Invalid function group")
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("data/experiment_001"),
        output_path=Path("results/error3"),
        group="invalid_group"
    )
    result = service.run(request)
    print(f"  Result: {result.error if not result.success else 'Success'}")


if __name__ == "__main__":
    # Run all examples
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Analysis Module System Examples" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        basic_analysis_example()
        analysis_with_progress()
        batch_analysis_example()
        analysis_with_caching()
        custom_analysis_parameters()
        module_introspection()
        error_handling_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
