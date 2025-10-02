#!/usr/bin/env python3
"""
Query and analyze data from centralized experiment databases.

This script provides utilities for exploring and analyzing simulation data
stored in ExperimentDatabase files.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    ExperimentModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
)


class ExperimentQueryTool:
    """Tool for querying experiment databases."""
    
    def __init__(self, db_path: str):
        """Initialize the query tool.
        
        Parameters
        ----------
        db_path : str
            Path to the experiment database
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def close(self):
        """Close the database connection."""
        self.session.close()
        self.engine.dispose()
    
    def get_experiment_info(self) -> dict:
        """Get information about the experiment.
        
        Returns
        -------
        dict
            Experiment metadata
        """
        experiment = self.session.query(ExperimentModel).first()
        if not experiment:
            return {"error": "No experiment found"}
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status,
            "creation_date": experiment.creation_date,
            "last_updated": experiment.last_updated,
            "results_summary": experiment.results_summary
        }
    
    def list_simulations(self) -> pd.DataFrame:
        """List all simulations in the experiment.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with simulation information
        """
        simulations = self.session.query(Simulation).all()
        
        data = []
        for sim in simulations:
            data.append({
                "simulation_id": sim.simulation_id,
                "status": sim.status,
                "start_time": sim.start_time,
                "end_time": sim.end_time,
                "parameters": str(sim.parameters) if sim.parameters else "",
                "results_summary": str(sim.results_summary) if sim.results_summary else ""
            })
        
        return pd.DataFrame(data)
    
    def get_simulation_summary(self, simulation_id: Optional[str] = None) -> pd.DataFrame:
        """Get summary statistics for one or all simulations.
        
        Parameters
        ----------
        simulation_id : str, optional
            Specific simulation ID, or None for all simulations
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        query = self.session.query(
            SimulationStepModel.simulation_id,
            func.count(SimulationStepModel.step_number).label("total_steps"),
            func.avg(SimulationStepModel.total_agents).label("avg_agents"),
            func.max(SimulationStepModel.total_agents).label("max_agents"),
            func.min(SimulationStepModel.total_agents).label("min_agents"),
            func.avg(SimulationStepModel.total_resources).label("avg_resources"),
            func.sum(SimulationStepModel.births).label("total_births"),
            func.sum(SimulationStepModel.deaths).label("total_deaths")
        )
        
        if simulation_id:
            query = query.filter(SimulationStepModel.simulation_id == simulation_id)
        
        query = query.group_by(SimulationStepModel.simulation_id)
        
        results = query.all()
        
        data = []
        for row in results:
            data.append({
                "simulation_id": row.simulation_id,
                "total_steps": row.total_steps,
                "avg_agents": round(row.avg_agents, 2) if row.avg_agents else 0,
                "max_agents": row.max_agents or 0,
                "min_agents": row.min_agents or 0,
                "avg_resources": round(row.avg_resources, 2) if row.avg_resources else 0,
                "total_births": row.total_births or 0,
                "total_deaths": row.total_deaths or 0
            })
        
        return pd.DataFrame(data)
    
    def get_agent_counts_over_time(self, simulation_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Get agent counts over time for specified simulations.
        
        Parameters
        ----------
        simulation_ids : List[str], optional
            List of simulation IDs, or None for all simulations
        
        Returns
        -------
        pd.DataFrame
            Agent counts by step and simulation
        """
        query = self.session.query(
            SimulationStepModel.simulation_id,
            SimulationStepModel.step_number,
            SimulationStepModel.total_agents
        )
        
        if simulation_ids:
            query = query.filter(SimulationStepModel.simulation_id.in_(simulation_ids))
        
        query = query.order_by(
            SimulationStepModel.simulation_id,
            SimulationStepModel.step_number
        )
        
        return pd.read_sql(query.statement, self.engine)
    
    def get_agent_types_distribution(self, simulation_id: str) -> pd.DataFrame:
        """Get distribution of agent types for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Simulation ID
        
        Returns
        -------
        pd.DataFrame
            Agent type counts
        """
        query = self.session.query(
            AgentModel.agent_type,
            func.count(AgentModel.agent_id).label("count")
        ).filter(
            AgentModel.simulation_id == simulation_id
        ).group_by(
            AgentModel.agent_type
        )
        
        return pd.read_sql(query.statement, self.engine)
    
    def get_final_state(self, simulation_id: str) -> dict:
        """Get final state of a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Simulation ID
        
        Returns
        -------
        dict
            Final state information
        """
        # Get last step
        last_step = self.session.query(SimulationStepModel).filter(
            SimulationStepModel.simulation_id == simulation_id
        ).order_by(
            SimulationStepModel.step_number.desc()
        ).first()
        
        if not last_step:
            return {"error": "No steps found"}
        
        # Get agent type distribution at last step
        agent_states = self.session.query(
            AgentModel.agent_type,
            func.count(AgentStateModel.agent_id).label("count")
        ).join(
            AgentStateModel,
            AgentModel.agent_id == AgentStateModel.agent_id
        ).filter(
            AgentStateModel.simulation_id == simulation_id,
            AgentStateModel.step_number == last_step.step_number
        ).group_by(
            AgentModel.agent_type
        ).all()
        
        agent_distribution = {row.agent_type: row.count for row in agent_states}
        
        return {
            "simulation_id": simulation_id,
            "final_step": last_step.step_number,
            "total_agents": last_step.total_agents,
            "total_resources": last_step.total_resources,
            "average_health": last_step.average_agent_health,
            "average_resources": last_step.average_agent_resources,
            "total_births": last_step.births,
            "total_deaths": last_step.deaths,
            "agent_distribution": agent_distribution
        }
    
    def compare_simulations(self, metric: str = "total_agents") -> pd.DataFrame:
        """Compare a metric across all simulations.
        
        Parameters
        ----------
        metric : str
            Metric to compare (column name from SimulationStepModel)
        
        Returns
        -------
        pd.DataFrame
            Pivot table with simulations as columns and steps as rows
        """
        query = self.session.query(
            SimulationStepModel.simulation_id,
            SimulationStepModel.step_number,
            getattr(SimulationStepModel, metric)
        ).order_by(
            SimulationStepModel.simulation_id,
            SimulationStepModel.step_number
        )
        
        df = pd.read_sql(query.statement, self.engine)
        
        # Pivot to wide format
        pivot_df = df.pivot(
            index="step_number",
            columns="simulation_id",
            values=metric
        )
        
        return pivot_df
    
    def export_simulation_data(self, simulation_id: str, output_path: str):
        """Export all data for a simulation to CSV files.
        
        Parameters
        ----------
        simulation_id : str
            Simulation ID
        output_path : str
            Directory to save CSV files
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Export simulation steps
        steps_df = pd.read_sql(
            self.session.query(SimulationStepModel).filter(
                SimulationStepModel.simulation_id == simulation_id
            ).statement,
            self.engine
        )
        steps_df.to_csv(f"{output_path}/steps.csv", index=False)
        
        # Export agents
        agents_df = pd.read_sql(
            self.session.query(AgentModel).filter(
                AgentModel.simulation_id == simulation_id
            ).statement,
            self.engine
        )
        agents_df.to_csv(f"{output_path}/agents.csv", index=False)
        
        # Export agent states
        states_df = pd.read_sql(
            self.session.query(AgentStateModel).filter(
                AgentStateModel.simulation_id == simulation_id
            ).statement,
            self.engine
        )
        states_df.to_csv(f"{output_path}/agent_states.csv", index=False)
        
        # Export actions
        actions_df = pd.read_sql(
            self.session.query(ActionModel).filter(
                ActionModel.simulation_id == simulation_id
            ).statement,
            self.engine
        )
        actions_df.to_csv(f"{output_path}/actions.csv", index=False)
        
        # Export resources
        resources_df = pd.read_sql(
            self.session.query(ResourceModel).filter(
                ResourceModel.simulation_id == simulation_id
            ).statement,
            self.engine
        )
        resources_df.to_csv(f"{output_path}/resources.csv", index=False)
        
        print(f"Exported simulation {simulation_id} data to {output_path}")


def main():
    """Main entry point for the query tool."""
    parser = argparse.ArgumentParser(
        description="Query and analyze experiment databases"
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="Path to the experiment database"
    )
    parser.add_argument(
        "--command",
        type=str,
        default="info",
        choices=[
            "info",
            "list",
            "summary",
            "compare",
            "final",
            "export",
            "agents-over-time"
        ],
        help="Command to run"
    )
    parser.add_argument(
        "--simulation-id",
        type=str,
        help="Specific simulation ID (for commands that support it)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="total_agents",
        help="Metric to compare (for compare command)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (for export command)"
    )
    
    args = parser.parse_args()
    
    # Create query tool
    tool = ExperimentQueryTool(args.db_path)
    
    try:
        if args.command == "info":
            # Show experiment info
            info = tool.get_experiment_info()
            print("\n=== EXPERIMENT INFORMATION ===")
            for key, value in info.items():
                print(f"{key:20s}: {value}")
        
        elif args.command == "list":
            # List all simulations
            df = tool.list_simulations()
            print("\n=== SIMULATIONS ===")
            print(df.to_string(index=False))
        
        elif args.command == "summary":
            # Show summary statistics
            df = tool.get_simulation_summary(args.simulation_id)
            print("\n=== SIMULATION SUMMARY ===")
            print(df.to_string(index=False))
        
        elif args.command == "compare":
            # Compare metric across simulations
            df = tool.compare_simulations(args.metric)
            print(f"\n=== COMPARISON: {args.metric.upper()} ===")
            print(df.head(20).to_string())
            print(f"\n... showing first 20 rows of {len(df)} total rows")
        
        elif args.command == "final":
            # Show final state
            if not args.simulation_id:
                print("ERROR: --simulation-id required for final command")
                sys.exit(1)
            
            final = tool.get_final_state(args.simulation_id)
            print(f"\n=== FINAL STATE: {args.simulation_id} ===")
            for key, value in final.items():
                print(f"{key:20s}: {value}")
        
        elif args.command == "export":
            # Export simulation data
            if not args.simulation_id:
                print("ERROR: --simulation-id required for export command")
                sys.exit(1)
            if not args.output:
                print("ERROR: --output required for export command")
                sys.exit(1)
            
            tool.export_simulation_data(args.simulation_id, args.output)
        
        elif args.command == "agents-over-time":
            # Show agent counts over time
            df = tool.get_agent_counts_over_time(
                [args.simulation_id] if args.simulation_id else None
            )
            print("\n=== AGENT COUNTS OVER TIME ===")
            print(df.head(20).to_string(index=False))
            print(f"\n... showing first 20 rows of {len(df)} total rows")
        
    finally:
        tool.close()


if __name__ == "__main__":
    main()
