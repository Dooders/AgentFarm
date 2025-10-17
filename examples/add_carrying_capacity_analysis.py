"""
Example: Adding Carrying Capacity Analysis to Population Module

This demonstrates how to extend the population analysis module with
a new analysis capability - in this case, carrying capacity estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.optimize import curve_fit
import json

from farm.analysis.common.context import AnalysisContext


# ============================================================================
# Step 1: Create the core computation function
# ============================================================================

def compute_carrying_capacity(df: pd.DataFrame, use_logistic: bool = True) -> Dict[str, Any]:
    """Estimate population carrying capacity.

    Estimates the maximum sustainable population using logistic growth model
    or empirical methods.

    Args:
        df: Population data with 'step' and 'total_agents' columns
        use_logistic: If True, fit logistic model; if False, use empirical method

    Returns:
        Dictionary containing:
        - estimated_capacity: Estimated carrying capacity (K)
        - growth_rate: Intrinsic growth rate (r)
        - initial_population: Population at t=0 (N0)
        - model_fit_quality: R² value for fit
        - current_percent_of_capacity: How full the population is
        - time_to_90_percent: Estimated steps to reach 90% capacity
        - method: Which method was used
    """
    steps = df['step'].values
    population = df['total_agents'].values

    if use_logistic:
        try:
            return _fit_logistic_model(steps, population)
        except Exception as e:
            print(f"Logistic fit failed: {e}, falling back to empirical method")
            return _estimate_empirical_capacity(steps, population)
    else:
        return _estimate_empirical_capacity(steps, population)


def _fit_logistic_model(steps: np.ndarray, population: np.ndarray) -> Dict[str, Any]:
    """Fit logistic growth model to estimate carrying capacity."""

    def logistic_function(t, r, K, N0):
        """
        Logistic growth equation:
        N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))

        Where:
        - K: carrying capacity
        - r: intrinsic growth rate
        - N0: initial population
        """
        return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    # Initial parameter guesses
    K_guess = max(population) * 1.2  # Assume capacity is 20% above observed max
    r_guess = 0.1
    N0_guess = population[0]

    # Fit the model
    params, covariance = curve_fit(
        logistic_function,
        steps,
        population,
        p0=[r_guess, K_guess, N0_guess],
        bounds=([0, population[0], population[0] * 0.5],  # Lower bounds
                [1, population.max() * 10, population[0] * 2]),  # Upper bounds
        maxfev=10000
    )

    r, K, N0 = params

    # Calculate fit quality (R²)
    predicted = logistic_function(steps, r, K, N0)
    residuals = population - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((population - np.mean(population))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate current status
    current_pop = population[-1]
    percent_of_capacity = (current_pop / K) * 100

    # Estimate time to reach 90% capacity
    target = 0.9 * K
    if current_pop < target:
        # Solve for t when N(t) = 0.9*K
        # 0.9*K = K / (1 + ((K-N0)/N0) * exp(-r*t))
        # Solving: t = (1/r) * ln(((K-N0)/N0) / ((K/0.9K) - 1))
        if current_pop > N0:
            time_to_90 = (1/r) * np.log(((K - N0) / N0) / ((K / target) - 1))
            time_to_90 = max(0, time_to_90 - steps[-1])  # Remaining time
        else:
            time_to_90 = None
    else:
        time_to_90 = 0  # Already there

    return {
        'method': 'logistic_fit',
        'estimated_capacity': float(K),
        'growth_rate': float(r),
        'initial_population': float(N0),
        'model_fit_quality': float(r_squared),
        'current_population': float(current_pop),
        'current_percent_of_capacity': float(percent_of_capacity),
        'time_to_90_percent': float(time_to_90) if time_to_90 is not None else None,
        'is_approaching_capacity': percent_of_capacity > 70,
        'is_at_capacity': percent_of_capacity > 95,
        'confidence': 'high' if r_squared > 0.9 else 'medium' if r_squared > 0.7 else 'low'
    }


def _estimate_empirical_capacity(steps: np.ndarray, population: np.ndarray) -> Dict[str, Any]:
    """Estimate carrying capacity using empirical methods (without model fitting)."""

    # Method 1: Use maximum observed + buffer
    max_observed = population.max()

    # Method 2: Look at recent plateau
    window = min(50, len(population) // 4)
    recent_mean = population[-window:].mean()
    recent_std = population[-window:].std()
    recent_cv = recent_std / recent_mean if recent_mean > 0 else 1.0

    # If recent CV is low, we might be at capacity
    is_stable = recent_cv < 0.1

    if is_stable:
        # Use recent plateau as capacity estimate
        K_estimate = recent_mean * 1.05  # Add 5% buffer
    else:
        # Use max observed + growth allowance
        K_estimate = max_observed * 1.3

    current_pop = population[-1]
    percent_of_capacity = (current_pop / K_estimate) * 100

    return {
        'method': 'empirical',
        'estimated_capacity': float(K_estimate),
        'max_observed': float(max_observed),
        'recent_mean': float(recent_mean),
        'recent_stability': float(1 - recent_cv),
        'current_population': float(current_pop),
        'current_percent_of_capacity': float(percent_of_capacity),
        'is_stable': is_stable,
        'confidence': 'medium' if is_stable else 'low'
    }


# ============================================================================
# Step 2: Create the analysis function (uses AnalysisContext)
# ============================================================================

def analyze_carrying_capacity(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze population carrying capacity and save results.

    Args:
        df: Population data
        ctx: Analysis context for output
        **kwargs: Additional options
            - use_logistic: bool (default True)
            - save_plot: bool (default True)
    """
    ctx.logger.info("Analyzing carrying capacity...")

    use_logistic = kwargs.get('use_logistic', True)
    save_plot = kwargs.get('save_plot', True)

    # Compute carrying capacity
    results = compute_carrying_capacity(df, use_logistic=use_logistic)

    # Save results to JSON
    output_file = ctx.get_output_file("carrying_capacity_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ctx.logger.info(f"Saved carrying capacity analysis to {output_file}")

    # Generate visualization if requested
    if save_plot:
        _plot_carrying_capacity(df, results, ctx)

    # Log key findings
    ctx.logger.info(f"  Estimated capacity: {results['estimated_capacity']:.0f} agents")
    ctx.logger.info(f"  Current: {results['current_percent_of_capacity']:.1f}% of capacity")
    ctx.logger.info(f"  Confidence: {results.get('confidence', 'unknown')}")

    ctx.report_progress("Carrying capacity analysis complete", 1.0)


def _plot_carrying_capacity(df: pd.DataFrame, results: Dict[str, Any], ctx: AnalysisContext) -> None:
    """Create visualization of carrying capacity analysis."""

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = df['step'].values
    population = df['total_agents'].values

    # Left plot: Population with capacity line
    ax1.plot(steps, population, 'b-', linewidth=2, label='Actual Population')

    K = results['estimated_capacity']
    ax1.axhline(y=K, color='r', linestyle='--', linewidth=2, label=f'Carrying Capacity (K={K:.0f})')
    ax1.axhline(y=K * 0.9, color='orange', linestyle=':', linewidth=1.5, label='90% Capacity')

    # If logistic fit, show the fitted curve
    if results['method'] == 'logistic_fit':
        def logistic(t, r, K, N0):
            return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

        fitted = logistic(steps, results['growth_rate'], K, results['initial_population'])
        ax1.plot(steps, fitted, 'g--', linewidth=2, alpha=0.7, label='Logistic Fit')

    ax1.fill_between(steps, 0, K, alpha=0.1, color='red', label='Capacity Region')

    ax1.set_xlabel('Simulation Step', fontsize=11)
    ax1.set_ylabel('Population', fontsize=11)
    ax1.set_title('Population vs Carrying Capacity', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Right plot: Percent of capacity over time
    ax2.plot(steps, (population / K) * 100, 'b-', linewidth=2)
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Capacity')
    ax2.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, label='90% Capacity')
    ax2.fill_between(steps, 0, 100, alpha=0.1, color='green', where=(population / K * 100) < 100)
    ax2.fill_between(steps, 100, (population / K) * 100, alpha=0.1, color='red',
                     where=(population / K * 100) >= 100)

    ax2.set_xlabel('Simulation Step', fontsize=11)
    ax2.set_ylabel('Percent of Capacity (%)', fontsize=11)
    ax2.set_title('Capacity Utilization Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(110, (population / K * 100).max() * 1.1))

    # Add text annotation with key stats
    textstr = f"Method: {results['method']}\n"
    textstr += f"K = {K:.0f}\n"
    textstr += f"Current: {results['current_percent_of_capacity']:.1f}%\n"
    if results['method'] == 'logistic_fit':
        textstr += f"R² = {results['model_fit_quality']:.3f}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("carrying_capacity_analysis.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved carrying capacity plot to {output_file}")


# ============================================================================
# Step 3: Register with the module (how to add to module.py)
# ============================================================================

def register_carrying_capacity_analysis():
    """
    Example of how to register this analysis with the population module.

    Add this to farm/analysis/population/module.py:

    1. Import the function:
       from farm.analysis.population.capacity import analyze_carrying_capacity

    2. Add to __init__ or register_functions():
       self._functions['analyze_capacity'] = make_analysis_function(analyze_carrying_capacity)

    3. Add to appropriate groups:
       self._groups['comprehensive'].append(self._functions['analyze_capacity'])
    """
    pass


# ============================================================================
# Step 4: Usage examples
# ============================================================================

def example_direct_usage():
    """Example: Use the analysis function directly."""
    print("=" * 80)
    print("Example 1: Direct Usage of Carrying Capacity Analysis")
    print("=" * 80)

    # Create sample data
    steps = np.arange(0, 200)
    # Logistic growth: starts exponential, then plateaus
    r, K, N0 = 0.05, 500, 50
    population = K / (1 + ((K - N0) / N0) * np.exp(-r * steps))
    # Add some noise
    population = population + np.random.normal(0, 10, len(population))
    population = np.maximum(population, 10)  # Keep positive

    df = pd.DataFrame({
        'step': steps,
        'total_agents': population
    })

    # Compute carrying capacity
    results = compute_carrying_capacity(df, use_logistic=True)

    print(f"\n📊 Carrying Capacity Analysis Results:")
    print(f"  Method: {results['method']}")
    print(f"  Estimated Capacity (K): {results['estimated_capacity']:.0f}")
    print(f"  Growth Rate (r): {results['growth_rate']:.4f}")
    print(f"  Model Fit (R²): {results['model_fit_quality']:.4f}")
    print(f"  Current Population: {results['current_population']:.0f}")
    print(f"  Capacity Usage: {results['current_percent_of_capacity']:.1f}%")

    if results['time_to_90_percent']:
        print(f"  Steps to 90% Capacity: {results['time_to_90_percent']:.0f}")

    print(f"  Status: {'At capacity' if results['is_at_capacity'] else 'Approaching' if results['is_approaching_capacity'] else 'Below capacity'}")
    print(f"  Confidence: {results['confidence']}")


def example_with_module():
    """Example: How to use it with the population module."""
    print("\n" + "=" * 80)
    print("Example 2: Integration with Population Module")
    print("=" * 80)

    print("""
After adding to the module, you can use it like:

```python
from farm.analysis.population import population_module

output_path, df = population_module.run_analysis(
    experiment_path="path/to/experiment",
    function_names=["analyze_capacity"]
)

# Or include in comprehensive analysis
output_path, df = population_module.run_analysis(
    experiment_path="path/to/experiment",
    function_groups=["comprehensive"]  # If added to this group
)
```

This will generate:
- carrying_capacity_analysis.json (numerical results)
- carrying_capacity_analysis.png (visualization)
""")


def main():
    """Run examples."""
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ADDING CARRYING CAPACITY ANALYSIS" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝\n")

    example_direct_usage()
    example_with_module()

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("\nNext steps:")
    print("1. Copy compute_carrying_capacity() to farm/analysis/population/capacity.py")
    print("2. Add analyze_carrying_capacity() to the same file")
    print("3. Register in module.py")
    print("4. Update __init__.py exports")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
