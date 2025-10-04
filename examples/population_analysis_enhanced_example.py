"""
Enhanced Population Analysis Example

Demonstrates the new and improved features of the population analysis module.
"""

from pathlib import Path
from farm.analysis.population import (
    population_module,
    compute_growth_rate_analysis,
    compute_demographic_metrics,
    compute_population_stability,
)


def example_1_basic_usage():
    """Example 1: Basic population analysis (backward compatible)."""
    print("=" * 80)
    print("Example 1: Basic Population Analysis (Backward Compatible)")
    print("=" * 80)
    
    # This works exactly as before
    experiment_path = Path("path/to/your/experiment")
    
    try:
        output_path, df = population_module.run_analysis(
            experiment_path=experiment_path,
            function_names=["analyze_dynamics", "plot_population"]
        )
        print(f"✓ Analysis completed successfully!")
        print(f"  Output: {output_path}")
        print(f"  Data shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"⚠ Skipping: Experiment path not found")
        print(f"  Please provide a valid experiment path")


def example_2_comprehensive_analysis():
    """Example 2: Run comprehensive analysis with all new features."""
    print("\n" + "=" * 80)
    print("Example 2: Comprehensive Analysis with Dashboard")
    print("=" * 80)
    
    experiment_path = Path("path/to/your/experiment")
    
    try:
        # Run comprehensive analysis group
        output_path, df = population_module.run_analysis(
            experiment_path=experiment_path,
            function_groups=["comprehensive"]  # New function group!
        )
        
        print(f"✓ Comprehensive analysis completed!")
        print(f"\n📊 Generated files:")
        print(f"  • comprehensive_population_analysis.json - All metrics")
        print(f"  • population_report.txt - Human-readable report")
        print(f"  • population_dashboard.png - Multi-panel visualization")
        
    except FileNotFoundError:
        print(f"⚠ Skipping: Experiment path not found")


def example_3_growth_rate_analysis():
    """Example 3: Detailed growth rate analysis."""
    print("\n" + "=" * 80)
    print("Example 3: Growth Rate Analysis")
    print("=" * 80)
    
    # Assume we have a DataFrame from previous analysis
    import pandas as pd
    import numpy as np
    
    # Create sample data for demonstration
    steps = np.arange(0, 100)
    # Exponential growth followed by stabilization
    population = 100 * np.exp(0.03 * steps[:50]).tolist()
    population += [population[-1] + np.random.randint(-5, 6) for _ in range(50)]
    
    df = pd.DataFrame({
        'step': steps,
        'total_agents': population,
    })
    
    # Compute growth analysis
    growth = compute_growth_rate_analysis(df)
    
    print(f"📈 Growth Metrics:")
    print(f"  Average growth rate: {growth['average_growth_rate']:.2f}%")
    print(f"  Max growth rate: {growth['max_growth_rate']:.2f}%")
    print(f"  Min growth rate: {growth['min_growth_rate']:.2f}%")
    
    if growth['doubling_time']:
        print(f"  Population doubling time: {growth['doubling_time']:.1f} steps")
    
    if growth['exponential_fit']:
        fit = growth['exponential_fit']
        print(f"\n  Exponential fit (early phase):")
        print(f"    Rate: {fit['rate']:.4f}")
        print(f"    R²: {fit['r_squared']:.4f}")
    
    print(f"\n⏱ Time Distribution:")
    print(f"  Growth: {growth['time_in_growth']} steps")
    print(f"  Decline: {growth['time_in_decline']} steps")
    print(f"  Stable: {growth['time_stable']} steps")
    
    print(f"\n📋 Growth Phases:")
    for i, phase in enumerate(growth['growth_phases'][:3], 1):
        print(f"  {i}. {phase['phase'].upper()}: steps {phase['start_step']}-{phase['end_step']} ({phase['duration']} steps)")


def example_4_demographic_analysis():
    """Example 4: Demographic composition analysis."""
    print("\n" + "=" * 80)
    print("Example 4: Demographic Analysis")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Create sample data with agent types
    steps = 100
    df = pd.DataFrame({
        'step': np.arange(steps),
        'total_agents': np.linspace(100, 150, steps),
        'system_agents': np.linspace(50, 60, steps),
        'independent_agents': np.linspace(30, 50, steps),
        'control_agents': np.linspace(20, 40, steps),
    })
    
    # Compute demographic metrics
    demographics = compute_demographic_metrics(df)
    
    if demographics:
        print(f"🧬 Demographic Metrics:")
        
        div = demographics['diversity_index']
        print(f"\n  Shannon Diversity Index:")
        print(f"    Mean: {div['mean']:.4f}")
        print(f"    Range: [{div['min']:.4f}, {div['max']:.4f}]")
        
        dom = demographics['dominance_index']
        print(f"\n  Simpson's Dominance Index:")
        print(f"    Mean: {dom['mean']:.4f}")
        print(f"    (Higher = more dominated by single type)")
        
        print(f"\n  Average Type Proportions:")
        for agent_type, prop in demographics['type_proportions'].items():
            print(f"    {agent_type.replace('_', ' ').title()}: {prop*100:.1f}%")
        
        print(f"\n  Type Stability Scores:")
        for agent_type, score in demographics['type_stability'].items():
            print(f"    {agent_type.replace('_', ' ').title()}: {score:.3f}")
        
        if demographics['composition_changes']:
            print(f"\n  ⚠ Detected {demographics['num_significant_changes']} significant composition changes")
            print(f"  Top change at step {demographics['composition_changes'][0]['step']}")


def example_5_stability_analysis():
    """Example 5: Enhanced stability metrics."""
    print("\n" + "=" * 80)
    print("Example 5: Enhanced Stability Analysis")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Create sample data with varying stability
    steps = 200
    # Add noise that increases over time
    noise = np.random.normal(0, np.linspace(5, 20, steps))
    population = 100 + np.cumsum(np.random.randn(steps)) + noise
    population = np.maximum(population, 10)  # Ensure positive
    
    df = pd.DataFrame({
        'step': np.arange(steps),
        'total_agents': population,
    })
    
    # Compute stability with custom window
    stability = compute_population_stability(df, window=50)
    
    print(f"⚖ Stability Metrics:")
    print(f"  Stability Score: {stability['stability_score']:.4f} (0-1, higher is more stable)")
    print(f"  Mean CV: {stability['mean_cv']:.4f} (coefficient of variation)")
    print(f"  Volatility: {stability['volatility']:.2f} (std of changes)")
    print(f"  Max Fluctuation: {stability['max_fluctuation']:.0f} agents")
    print(f"  Mean Relative Change: {stability['mean_relative_change']*100:.2f}%")
    print(f"  Max Relative Change: {stability['max_relative_change']*100:.2f}%")


def example_6_custom_visualization():
    """Example 6: Using the new dashboard visualization."""
    print("\n" + "=" * 80)
    print("Example 6: Dashboard Visualization")
    print("=" * 80)
    
    experiment_path = Path("path/to/your/experiment")
    
    try:
        # Run just the dashboard plot
        output_path, df = population_module.run_analysis(
            experiment_path=experiment_path,
            function_names=["plot_dashboard"]
        )
        
        print(f"✓ Dashboard created!")
        print(f"\n📊 The dashboard includes:")
        print(f"  1. Population trends over time")
        print(f"  2. Growth rate with smoothing")
        print(f"  3. Agent composition stacked area")
        print(f"  4. Rolling statistics with confidence bands")
        print(f"  5. Population distribution histogram")
        print(f"\n  File: {output_path}/population_dashboard.png")
        
    except FileNotFoundError:
        print(f"⚠ Skipping: Experiment path not found")
        print(f"\n  The dashboard provides a comprehensive single-view")
        print(f"  visualization of all population dynamics!")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ENHANCED POPULATION ANALYSIS EXAMPLES" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    examples = [
        example_1_basic_usage,
        example_2_comprehensive_analysis,
        example_3_growth_rate_analysis,
        example_4_demographic_analysis,
        example_5_stability_analysis,
        example_6_custom_visualization,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✨ Examples completed! Check the documentation for more details.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
