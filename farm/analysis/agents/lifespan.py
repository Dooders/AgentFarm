"""
Agent lifespan analysis functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def analyze_agent_lifespans(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent lifespans in detail.

    Args:
        df: Agent data with lifespan columns
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent lifespans...")

    if df.empty or 'lifespan' not in df.columns:
        ctx.logger.warning("No lifespan data available")
        return

    lifespans = df['lifespan'].dropna()

    # Calculate detailed statistics
    stats = {
        'count': len(lifespans),
        'mean': float(lifespans.mean()),
        'median': float(lifespans.median()),
        'std': float(lifespans.std()),
        'min': float(lifespans.min()),
        'max': float(lifespans.max()),
        'q25': float(lifespans.quantile(0.25)),
        'q75': float(lifespans.quantile(0.75)),
    }

    # Save statistics
    output_file = ctx.get_output_file("detailed_lifespan_stats.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    ctx.logger.info(f"Saved detailed lifespan stats to {output_file}")


def plot_lifespan_histogram(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create detailed lifespan histogram.

    Args:
        df: Agent data with lifespan columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating detailed lifespan histogram...")

    if df.empty or 'lifespan' not in df.columns:
        ctx.logger.warning("No lifespan data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    lifespans = df['lifespan'].dropna()

    # Create histogram with density curve
    counts, bins, patches = ax.hist(lifespans, bins=50, alpha=0.7,
                                   color='skyblue', edgecolor='black',
                                   density=True, label='Histogram')

    # Add density curve
    from scipy import stats
    density = stats.gaussian_kde(lifespans)
    x_vals = np.linspace(lifespans.min(), lifespans.max(), 100)
    ax.plot(x_vals, density(x_vals), 'r-', linewidth=2, label='Density')

    ax.set_xlabel('Lifespan')
    ax.set_ylabel('Density')
    ax.set_title('Agent Lifespan Distribution with Density Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("lifespan_histogram.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved histogram to {output_file}")
