"""Command-line tool for comparing simulation results.

This tool provides a command-line interface for comparing two simulation runs,
showing differences in their parameters, metrics, and results.

Examples
--------
Basic comparison with text output:
    $ python -m farm.tools.compare_sims 1 2
    Metadata Differences:
      status:
        Simulation 1: completed
        Simulation 2: completed
    
    Parameter Changes:
      Changed parameters:
        environment.resource_density:
          From: 0.1
          To:   0.2
    
    Metric Differences:
      total_agents:
        Mean difference: +5.321
        Max difference:  +8.000
        Min difference:  +2.000
        Std deviation difference: +1.234

Compare with JSON output:
    $ python -m farm.tools.compare_sims 1 2 --format json
    {
      "metadata": {
        "status": {
          "simulation_1": "completed",
          "simulation_2": "completed"
        }
      },
      "parameters": {
        "changed": {
          "environment.resource_density": {
            "old_value": 0.1,
            "new_value": 0.2
          }
        }
      }
      ...
    }

Show only significant differences:
    $ python -m farm.tools.compare_sims 1 2 --significant-only --threshold 0.2
    Metric Differences:
      total_agents:
        Mean difference: +5.321
        Max difference:  +8.000
        Min difference:  +2.000
        Std deviation difference: +1.234

Show only metrics:
    $ python -m farm.tools.compare_sims 1 2 --metrics-only
    Metric Differences:
      total_agents:
        Mean difference: +5.321
        Max difference:  +8.000
        Min difference:  +2.000
        Std deviation difference: +1.234

Save comparison to file:
    $ python -m farm.tools.compare_sims 1 2 --output comparison.txt

Use custom database path:
    $ python -m farm.tools.compare_sims 1 2 --db-path sqlite:///custom_db.sqlite

Arguments
---------
sim1_id : int
    ID of first simulation to compare
sim2_id : int
    ID of second simulation to compare
--db-path : str
    Path to simulation database (default: farm_simulation.db)
--format : str
    Output format, either 'text' or 'json' (default: text)
--output : str
    Output file path (default: stdout)
--threshold : float
    Threshold for significant changes (default: 0.1)
--metrics-only : bool
    Show only metric differences
--significant-only : bool
    Show only significant differences

Notes
-----
- The tool compares various aspects of simulations including:
  * Metadata (status, timestamps, etc.)
  * Parameters (configuration differences)
  * Results summary
  * Step-by-step metrics (statistical differences)
- Use --significant-only to focus on meaningful changes
- Use --metrics-only to see only numerical differences
- JSON output is useful for programmatic processing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.simulation_comparison import (
    compare_simulations,
    summarize_comparison,
    get_significant_changes
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two simulation runs and display their differences"
    )
    
    parser.add_argument(
        "sim1_id",
        type=int,
        help="ID of first simulation to compare"
    )
    
    parser.add_argument(
        "sim2_id",
        type=int,
        help="ID of second simulation to compare"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="sqlite:///farm_simulation.db",
        help="Path to simulation database (default: farm_simulation.db)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for significant changes (default: 0.1)"
    )
    
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Show only metric differences"
    )
    
    parser.add_argument(
        "--significant-only",
        action="store_true",
        help="Show only significant differences"
    )
    
    return parser.parse_args()

def format_json_output(
    differences: dict,
    significant_only: bool = False,
    threshold: float = 0.1
) -> dict:
    """Format comparison results as JSON."""
    if significant_only:
        # Filter to only show significant metric differences
        metrics = differences["metrics"]
        filtered_metrics = {
            metric: stats for metric, stats in metrics.items()
            if abs(stats["mean_difference"]) > threshold
        }
        differences["metrics"] = filtered_metrics
        
        # Remove empty sections
        return {k: v for k, v in differences.items() if v}
    
    return differences

def format_text_output(
    differences: dict,
    significant_only: bool = False,
    threshold: float = 0.1,
    metrics_only: bool = False
) -> List[str]:
    """Format comparison results as text."""
    lines = []
    
    if not metrics_only:
        # Metadata differences
        if differences["metadata"]:
            lines.append("Metadata Differences:")
            for field, values in differences["metadata"].items():
                lines.append(f"  {field}:")
                lines.append(f"    Simulation 1: {values['simulation_1']}")
                lines.append(f"    Simulation 2: {values['simulation_2']}")
            lines.append("")
        
        # Parameter differences
        if any(differences["parameters"].values()):
            lines.append("Parameter Changes:")
            if differences["parameters"]["added"]:
                lines.append("  Added parameters:")
                for param in differences["parameters"]["added"]:
                    lines.append(f"    + {param}")
            if differences["parameters"]["removed"]:
                lines.append("  Removed parameters:")
                for param in differences["parameters"]["removed"]:
                    lines.append(f"    - {param}")
            if differences["parameters"]["changed"]:
                lines.append("  Changed parameters:")
                for path, change in differences["parameters"]["changed"].items():
                    lines.append(f"    {path}:")
                    lines.append(f"      From: {change['old_value']}")
                    lines.append(f"      To:   {change['new_value']}")
            lines.append("")
    
    # Metric differences
    if differences["metrics"]:
        lines.append("Metric Differences:")
        for metric, stats in differences["metrics"].items():
            mean_diff = stats["mean_difference"]
            if not significant_only or abs(mean_diff) > threshold:
                lines.append(f"  {metric}:")
                lines.append(f"    Mean difference: {mean_diff:+.3f}")
                lines.append(f"    Max difference:  {stats['max_difference']:+.3f}")
                lines.append(f"    Min difference:  {stats['min_difference']:+.3f}")
                lines.append(f"    Std deviation difference: {stats['std_dev_difference']:+.3f}")
                lines.append("")
    
    return lines

def main():
    """Main entry point for the comparison tool."""
    args = parse_args()
    
    # Set up database connection
    engine = create_engine(args.db_path)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get comparison data
        differences = compare_simulations(session, args.sim1_id, args.sim2_id)
        
        # Format output
        if args.format == "json":
            output = format_json_output(
                differences,
                significant_only=args.significant_only,
                threshold=args.threshold
            )
            formatted_output = json.dumps(output, indent=2)
        else:
            lines = format_text_output(
                differences,
                significant_only=args.significant_only,
                threshold=args.threshold,
                metrics_only=args.metrics_only
            )
            formatted_output = "\n".join(lines)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
        else:
            print(formatted_output)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    main() 