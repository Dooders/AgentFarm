"""
Example usage of the AnalysisController.

This demonstrates how to use the AnalysisController to run analysis modules
with progress tracking, callbacks, and lifecycle management.
"""

import time
from pathlib import Path

from farm.analysis.service import AnalysisRequest
from farm.api.analysis_controller import AnalysisController
from farm.core.services import EnvConfigService


def basic_usage():
    """Basic example: Run a single analysis module."""
    print("\n=== Basic Usage Example ===\n")
    
    # Create controller
    config_service = EnvConfigService()
    controller = AnalysisController(config_service)
    
    # Create analysis request
    request = AnalysisRequest(
        module_name="genesis",  # Change to your analysis module
        experiment_path=Path("results/experiment_1"),
        output_path=Path("results/analysis/genesis"),
        group="all"
    )
    
    try:
        # Initialize and run
        controller.initialize_analysis(request)
        controller.start()
        
        # Wait for completion using the new method (recommended)
        print("Waiting for analysis to complete...")
        if controller.wait_for_completion(timeout=300):  # 5 minute timeout
            # Get results
            result = controller.get_result()
            if result and result.success:
                print(f"\n✓ Analysis complete!")
                print(f"  Output: {result.output_path}")
                print(f"  Rows: {len(result.dataframe) if result.dataframe is not None else 0}")
                print(f"  Time: {result.execution_time:.2f}s")
                print(f"  Cache hit: {result.cache_hit}")
            else:
                print(f"\n✗ Analysis failed: {result.error if result else 'Unknown error'}")
        else:
            print("\n✗ Analysis timed out after 5 minutes")
    
    finally:
        controller.cleanup()


def with_callbacks():
    """Example with progress and status callbacks."""
    print("\n=== Callbacks Example ===\n")
    
    # Define callbacks
    def on_progress(message: str, progress: float):
        """Handle progress updates."""
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {progress*100:.1f}% - {message}", end="", flush=True)
    
    def on_status(status: str):
        """Handle status changes."""
        print(f"\n>> Status changed: {status}")
    
    # Create controller
    config_service = EnvConfigService()
    controller = AnalysisController(config_service)
    
    # Register callbacks
    controller.register_progress_callback("progress_bar", on_progress)
    controller.register_status_callback("status_logger", on_status)
    
    # Create and run analysis
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("results/experiment_1"),
        output_path=Path("results/analysis/dominance"),
        group="all"
    )
    
    try:
        controller.initialize_analysis(request)
        controller.start()
        
        # Wait for completion using new method
        controller.wait_for_completion()
        
        print()  # New line after progress bar
        
        # Get results
        result = controller.get_result()
        if result and result.success:
            print(f"\n✓ Analysis complete in {result.execution_time:.2f}s")
    
    finally:
        controller.cleanup()


def with_context_manager():
    """Example using context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===\n")
    
    config_service = EnvConfigService()
    
    with AnalysisController(config_service) as controller:
        request = AnalysisRequest(
            module_name="advantage",
            experiment_path=Path("results/experiment_1"),
            output_path=Path("results/analysis/advantage"),
            group="all"
        )
        
        controller.initialize_analysis(request)
        controller.start()
        
        # Wait for completion with timeout
        if controller.wait_for_completion(timeout=600):  # 10 minute timeout
            result = controller.get_result()
            if result and result.success:
                print(f"✓ Complete! Output: {result.output_path}")
        else:
            print("✗ Analysis timed out")
    
    # Cleanup happens automatically


def pause_resume_example():
    """Example demonstrating pause/resume functionality."""
    print("\n=== Pause/Resume Example ===\n")
    
    config_service = EnvConfigService()
    controller = AnalysisController(config_service)
    
    request = AnalysisRequest(
        module_name="genesis",
        experiment_path=Path("results/experiment_1"),
        output_path=Path("results/analysis/genesis"),
        group="all"
    )
    
    try:
        controller.initialize_analysis(request)
        controller.start()
        
        # Run for a bit
        time.sleep(2)
        
        # Pause
        print("Pausing analysis...")
        controller.pause()
        
        state = controller.get_state()
        print(f"Paused at {state['progress']*100:.1f}%")
        
        # Wait while paused
        time.sleep(2)
        print("Resuming...")
        
        # Resume
        controller.start()
        
        # Wait for completion
        controller.wait_for_completion()
        
        result = controller.get_result()
        if result and result.success:
            print(f"✓ Completed after pause/resume")
    
    finally:
        controller.cleanup()


def list_modules_example():
    """Example showing how to list available modules."""
    print("\n=== List Modules Example ===\n")
    
    config_service = EnvConfigService()
    controller = AnalysisController(config_service)
    
    # List all modules
    modules = controller.list_available_modules()
    
    print("Available analysis modules:")
    print("-" * 60)
    for module in modules:
        print(f"  • {module['name']}")
        print(f"    {module['description']}")
        print(f"    Database support: {module.get('supports_database', False)}")
        print()
    
    # Get detailed info for specific module
    if modules:
        module_name = modules[0]['name']
        print(f"\nDetailed info for '{module_name}':")
        print("-" * 60)
        info = controller.get_module_info(module_name)
        print(f"  Function groups: {info.get('function_groups', [])}")
        print(f"  Functions: {info.get('functions', [])}")


def batch_analysis_example():
    """Example running multiple analyses sequentially."""
    print("\n=== Batch Analysis Example ===\n")
    
    config_service = EnvConfigService()
    
    # Define multiple analyses to run
    analyses = [
        ("genesis", "all"),
        ("dominance", "plots"),
        ("advantage", "metrics"),
    ]
    
    for module_name, group in analyses:
        print(f"\nRunning {module_name} ({group})...")
        
        with AnalysisController(config_service) as controller:
            request = AnalysisRequest(
                module_name=module_name,
                experiment_path=Path("results/experiment_1"),
                output_path=Path(f"results/analysis/{module_name}"),
                group=group
            )
            
            controller.initialize_analysis(request)
            controller.start()
            
            # Use wait_for_completion with timeout
            if controller.wait_for_completion(timeout=300):
                result = controller.get_result()
                if result and result.success:
                    print(f"  ✓ {module_name} complete ({result.execution_time:.1f}s)")
                else:
                    print(f"  ✗ {module_name} failed")
            else:
                print(f"  ✗ {module_name} timed out")


def custom_parameters_example():
    """Example with custom analysis parameters."""
    print("\n=== Custom Parameters Example ===\n")
    
    config_service = EnvConfigService()
    controller = AnalysisController(config_service)
    
    # Create request with custom parameters
    request = AnalysisRequest(
        module_name="dominance",
        experiment_path=Path("results/experiment_1"),
        output_path=Path("results/analysis/dominance_custom"),
        group="all",
        processor_kwargs={
            "save_to_db": True,
            "verbose": True,
        },
        analysis_kwargs={
            "plot_distribution": {
                "bins": 50,
                "figsize": (12, 8),
            },
            "compute_statistics": {
                "include_median": True,
            }
        },
        enable_caching=True,
        force_refresh=False
    )
    
    try:
        controller.initialize_analysis(request)
        controller.start()
        
        # Wait for completion
        controller.wait_for_completion()
        
        result = controller.get_result()
        if result and result.success:
            print(f"✓ Analysis complete with custom parameters")
            print(f"  Cache hit: {result.cache_hit}")
    
    finally:
        controller.cleanup()


if __name__ == "__main__":
    # Run examples
    # Note: Modify paths to match your actual experiment data
    
    print("=" * 60)
    print("AnalysisController Examples")
    print("=" * 60)
    
    # List available modules first
    list_modules_example()
    
    # Uncomment the examples you want to run:
    
    # basic_usage()
    # with_callbacks()
    # with_context_manager()
    # pause_resume_example()
    # batch_analysis_example()
    # custom_parameters_example()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
