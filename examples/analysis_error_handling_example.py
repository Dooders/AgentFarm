"""
Example: Using the New Error Handling Features

This example demonstrates the new error handling modes and configuration system
for analysis modules.
"""

from pathlib import Path
from farm.analysis.registry import get_module
from farm.analysis.core import ErrorHandlingMode
from farm.analysis.config import genesis_config, spatial_config, reset_to_defaults


def example_1_fail_fast():
    """Example 1: Fail fast mode - stops on first error."""
    print("=" * 60)
    print("Example 1: FAIL_FAST Mode")
    print("=" * 60)

    module = get_module("genesis")
    module.set_error_mode(ErrorHandlingMode.FAIL_FAST)

    try:
        output_path, df, errors = module.run_analysis(
            experiment_path=Path("experiments/exp001"),
            output_path=Path("analysis/exp001")
        )
        print(f"✅ Analysis completed successfully")
        print(f"   Records: {len(df) if df is not None else 0}")
        print(f"   Errors: {len(errors)}")
    except Exception as e:
        print(f"❌ Analysis stopped on error: {e}")


def example_2_collect_errors():
    """Example 2: Collect mode - continues and collects all errors."""
    print("\n" + "=" * 60)
    print("Example 2: COLLECT Mode")
    print("=" * 60)

    module = get_module("agents")

    output_path, df, errors = module.run_analysis(
        experiment_path=Path("experiments/exp001"),
        output_path=Path("analysis/exp001"),
        error_mode=ErrorHandlingMode.COLLECT  # Override module default
    )

    if errors:
        print(f"⚠️  Analysis completed with {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error.function_name}: {error.original_error}")
    else:
        print(f"✅ Analysis completed without errors")

    print(f"   Records: {len(df) if df is not None else 0}")


def example_3_continue_default():
    """Example 3: Continue mode - skip errors and continue (default)."""
    print("\n" + "=" * 60)
    print("Example 3: CONTINUE Mode (Default)")
    print("=" * 60)

    module = get_module("population")
    # CONTINUE is the default mode, no need to set it

    output_path, df, errors = module.run_analysis(
        experiment_path=Path("experiments/exp001"),
        output_path=Path("analysis/exp001")
    )

    print(f"✅ Analysis completed")
    print(f"   Records: {len(df) if df is not None else 0}")
    print(f"   Errors collected: {len(errors)}")
    print(f"   Note: Errors were skipped, analysis continued")


def example_4_custom_config():
    """Example 4: Using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)

    # View default config
    print(f"Default critical period: {genesis_config.critical_period_end}")
    print(f"Default resource threshold: {genesis_config.resource_proximity_threshold}")

    # Customize for this experiment
    genesis_config.critical_period_end = 200
    genesis_config.resource_proximity_threshold = 50.0

    print(f"\nCustomized:")
    print(f"  Critical period: {genesis_config.critical_period_end}")
    print(f"  Resource threshold: {genesis_config.resource_proximity_threshold}")

    # Run analysis with custom config
    module = get_module("genesis")
    output_path, df, errors = module.run_analysis(
        experiment_path=Path("experiments/exp001"),
        output_path=Path("analysis/exp001")
    )

    print(f"\n✅ Analysis completed with custom config")

    # Reset to defaults for next run
    reset_to_defaults()
    print(f"\nReset to defaults:")
    print(f"  Critical period: {genesis_config.critical_period_end}")


def example_5_spatial_config():
    """Example 5: Configuring spatial analysis."""
    print("\n" + "=" * 60)
    print("Example 5: Spatial Analysis Configuration")
    print("=" * 60)

    print("Default spatial config:")
    print(f"  Max clusters: {spatial_config.max_clusters}")
    print(f"  Density bins: {spatial_config.density_bins}")
    print(f"  Gathering range: {spatial_config.gathering_range}")

    # Adjust for high-density simulation
    spatial_config.max_clusters = 20
    spatial_config.density_bins = 40
    spatial_config.gathering_range = 25.0

    print(f"\nAdjusted for high-density simulation:")
    print(f"  Max clusters: {spatial_config.max_clusters}")
    print(f"  Density bins: {spatial_config.density_bins}")
    print(f"  Gathering range: {spatial_config.gathering_range}")

    module = get_module("spatial")
    output_path, df, errors = module.run_analysis(
        experiment_path=Path("experiments/exp001"),
        output_path=Path("analysis/exp001")
    )

    print(f"\n✅ Spatial analysis completed")


def example_6_production_vs_debug():
    """Example 6: Production vs Debug configurations."""
    print("\n" + "=" * 60)
    print("Example 6: Production vs Debug Modes")
    print("=" * 60)

    module = get_module("learning")

    # Production: continue on errors
    print("Production mode (CONTINUE):")
    module.set_error_mode(ErrorHandlingMode.CONTINUE)
    output_path, df, errors = module.run_analysis(
        experiment_path=Path("experiments/exp001"),
        output_path=Path("analysis/exp001_prod")
    )
    print(f"  ✅ Completed: {len(errors)} errors skipped")

    # Debug: fail fast to catch issues
    print("\nDebug mode (FAIL_FAST):")
    module.set_error_mode(ErrorHandlingMode.FAIL_FAST)
    try:
        output_path, df, errors = module.run_analysis(
            experiment_path=Path("experiments/exp001"),
            output_path=Path("analysis/exp001_debug")
        )
        print(f"  ✅ Completed without errors")
    except Exception as e:
        print(f"  ❌ Stopped on first error (as expected in debug mode)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" Analysis Module Error Handling & Configuration Examples")
    print("=" * 70)

    # Note: These examples will fail if experiments don't exist
    # They're meant to show the API usage
    print("\nNote: These examples show API usage. Some may fail if experiment")
    print("      paths don't exist - that's expected for demonstration.")

    try:
        example_1_fail_fast()
    except Exception as e:
        print(f"(Expected demo failure: {type(e).__name__})")

    try:
        example_2_collect_errors()
    except Exception as e:
        print(f"(Expected demo failure: {type(e).__name__})")

    try:
        example_3_continue_default()
    except Exception as e:
        print(f"(Expected demo failure: {type(e).__name__})")

    example_4_custom_config()
    example_5_spatial_config()

    try:
        example_6_production_vs_debug()
    except Exception as e:
        print(f"(Expected demo failure: {type(e).__name__})")

    print("\n" + "=" * 70)
    print(" Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
