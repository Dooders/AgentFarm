"""Script to diagnose reproduction issues in the simulation."""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)

class ReproductionDiagnostics:
    def __init__(self, db_path: str):
        """Initialize diagnostics with database path."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        
    def analyze_reproduction_patterns(self) -> Dict:
        """Analyze reproduction patterns and failure reasons."""
        # Query reproduction events
        repro_events = pd.read_sql(
            "SELECT * FROM reproduction_events ORDER BY step_number",
            self.engine
        )
        
        # Query agent states
        agent_states = pd.read_sql(
            """
            SELECT step_number, 
                   COUNT(*) as total_agents,
                   AVG(resource_level) as avg_resources,
                   AVG(current_health) as avg_health
            FROM agent_states 
            GROUP BY step_number
            ORDER BY step_number
            """,
            self.engine
        )
        
        # Calculate reproduction metrics
        metrics = {
            "total_attempts": len(repro_events),
            "successful_attempts": len(repro_events[repro_events["success"] == True]),
            "failed_attempts": len(repro_events[repro_events["success"] == False]),
            "failure_reasons": repro_events[repro_events["success"] == False]["failure_reason"].value_counts().to_dict(),
            "success_rate_over_time": self._calculate_success_rate_over_time(repro_events),
            "resource_levels": self._analyze_resource_levels(agent_states),
            "last_successful_reproduction": self._get_last_successful_reproduction(repro_events),
            "current_state": self._get_current_state()
        }
        
        return metrics

    def _calculate_success_rate_over_time(self, events_df: pd.DataFrame) -> Dict:
        """Calculate reproduction success rate over time."""
        if events_df.empty:
            return {"error": "No reproduction events found"}
            
        # Group by time periods (e.g., every 100 steps)
        period = 100
        events_df['time_period'] = events_df['step_number'] // period
        
        success_rates = events_df.groupby('time_period').agg({
            'success': ['count', 'sum']
        })
        
        success_rates['rate'] = success_rates[('success', 'sum')] / success_rates[('success', 'count')]
        
        return {
            "time_periods": success_rates.index.tolist(),
            "rates": success_rates['rate'].tolist()
        }

    def _analyze_resource_levels(self, states_df: pd.DataFrame) -> Dict:
        """Analyze resource levels in relation to reproduction."""
        return {
            "avg_resources_trend": states_df['avg_resources'].tolist(),
            "steps": states_df['step_number'].tolist(),
            "correlation_with_population": np.corrcoef(
                states_df['avg_resources'],
                states_df['total_agents']
            )[0,1]
        }

    def _get_last_successful_reproduction(self, events_df: pd.DataFrame) -> Dict:
        """Get details about the last successful reproduction."""
        if events_df.empty:
            return {"error": "No reproduction events found"}
            
        last_success = events_df[events_df['success'] == True].iloc[-1] if any(events_df['success']) else None
        
        if last_success is not None:
            return {
                "step": int(last_success['step_number']),
                "parent_resources": float(last_success['parent_resources_before']),
                "offspring_resources": float(last_success['offspring_initial_resources'])
            }
        return {"error": "No successful reproductions found"}

    def _get_current_state(self) -> Dict:
        """Get current state of the simulation regarding reproduction capability."""
        # Query latest agent states
        latest_states = pd.read_sql(
            """
            WITH LatestStates AS (
                SELECT agent_id, MAX(step_number) as max_step
                FROM agent_states
                GROUP BY agent_id
            )
            SELECT as1.*
            FROM agent_states as1
            JOIN LatestStates ls 
                ON as1.agent_id = ls.agent_id 
                AND as1.step_number = ls.max_step
            """,
            self.engine
        )
        
        return {
            "active_agents": len(latest_states),
            "agents_with_sufficient_resources": len(
                latest_states[latest_states['resource_level'] >= 10]  # Assuming minimum reproduction threshold
            ),
            "average_resources": float(latest_states['resource_level'].mean()),
            "average_health": float(latest_states['current_health'].mean())
        }

    def plot_diagnostics(self, metrics: Dict):
        """Create diagnostic plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Success Rate Over Time
        success_rates = metrics['success_rate_over_time']
        if 'error' not in success_rates:
            ax1.plot(success_rates['time_periods'], success_rates['rates'])
            ax1.set_title('Reproduction Success Rate Over Time')
            ax1.set_xlabel('Time Period (100 steps)')
            ax1.set_ylabel('Success Rate')
            
        # Plot 2: Resource Levels
        resource_data = metrics['resource_levels']
        ax2.plot(resource_data['steps'], resource_data['avg_resources_trend'])
        ax2.set_title('Average Resource Levels Over Time')
        ax2.set_xlabel('Step Number')
        ax2.set_ylabel('Average Resources')
        
        # Plot 3: Failure Reasons
        failure_reasons = metrics['failure_reasons']
        if failure_reasons:
            reasons = list(failure_reasons.keys())
            counts = list(failure_reasons.values())
            x = np.arange(len(reasons))  # Create x positions for bars
            ax3.bar(x, counts)
            ax3.set_title('Reproduction Failure Reasons')
            ax3.set_xticks(x)  # Set the tick positions
            ax3.set_xticklabels(reasons, rotation=45, ha='right')  # Now set the labels
            ax3.set_ylabel('Count')
        
        # Plot 4: Current State
        current = metrics['current_state']
        state_metrics = ['active_agents', 'agents_with_sufficient_resources']
        values = [current[m] for m in state_metrics]
        x = np.arange(len(state_metrics))  # Create x positions for bars
        ax4.bar(x, values)
        ax4.set_title('Current Population State')
        ax4.set_xticks(x)  # Set the tick positions
        ax4.set_xticklabels(state_metrics, rotation=45, ha='right')  # Now set the labels
        
        plt.tight_layout()
        plt.savefig('reproduction_diagnostics.png')
        plt.close()

def main():
    """Main function to run reproduction diagnostics."""
    logger.info("Starting reproduction diagnostics...")
    
    diagnostics = ReproductionDiagnostics("simulations/simulation.db")
    metrics = diagnostics.analyze_reproduction_patterns()
    
    # Print key findings
    print("\n=== Reproduction Diagnostics Report ===")
    print(f"\nOverall Statistics:")
    print(f"Total reproduction attempts: {metrics['total_attempts']}")
    print(f"Successful attempts: {metrics['successful_attempts']}")
    print(f"Failed attempts: {metrics['failed_attempts']}")
    
    print("\nFailure Reasons:")
    for reason, count in metrics['failure_reasons'].items():
        print(f"- {reason}: {count}")
        
    print("\nLast Successful Reproduction:")
    last_success = metrics['last_successful_reproduction']
    if 'error' not in last_success:
        print(f"Step: {last_success['step']}")
        print(f"Parent resources: {last_success['parent_resources']:.2f}")
        print(f"Offspring resources: {last_success['offspring_resources']:.2f}")
    else:
        print(f"Error: {last_success['error']}")
        
    print("\nCurrent State:")
    current = metrics['current_state']
    print(f"Active agents: {current['active_agents']}")
    print(f"Agents with sufficient resources: {current['agents_with_sufficient_resources']}")
    print(f"Average resources: {current['average_resources']:.2f}")
    print(f"Average health: {current['average_health']:.2f}")
    
    # Generate plots
    diagnostics.plot_diagnostics(metrics)
    logger.info("Diagnostics complete. Check reproduction_diagnostics.png for visualizations.")

if __name__ == "__main__":
    main() 