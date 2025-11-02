"""SQLAlchemy models for the simulation database.

This module defines the database schema using SQLAlchemy ORM models.
Each class represents a table in the database and defines its structure and relationships.

Main Models:
- Agent: Represents simulation agents with their core attributes
- AgentState: Tracks agent state changes over time
- ResourceState: Tracks resource states in the environment
- SimulationStep: Stores simulation-wide metrics per step
- AgentAction: Records actions taken by agents
- LearningExperience: Stores agent learning data
- HealthIncident: Tracks changes in agent health
- SimulationConfig: Stores simulation configuration data
- ExperimentModel: Stores experiment metadata and groups related simulations
- Simulation: Stores simulation metadata

Each model includes appropriate indexes for query optimization and relationships
between related tables.
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from deepdiff import DeepDiff
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    func,
    text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

from farm.utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# Define SQLAlchemy Models
class AgentModel(Base):
    """Represents a simulation agent and its core attributes.

    This model stores the fundamental properties of agents in the simulation,
    including their lifecycle data, physical attributes, and genetic information.

    Attributes
    ----------
    agent_id : str
        Unique identifier for the agent
    birth_time : int
        Step number when the agent was created
    death_time : Optional[int]
        Step number when the agent died (None if still alive)
    agent_type : str
        Type/category of the agent (e.g., 'system', 'independent', 'control')
    position_x : float
        X-coordinate of agent's position
    position_y : float
        Y-coordinate of agent's position
    initial_resources : float
        Starting resource level of the agent
    starting_health : float
        Maximum health capacity of the agent
    genome_id : str
        Unique identifier for agent's genetic code
    generation : int
        Generational number in evolutionary lineage
    action_weights : Dict[str, float]
        Dictionary of action names to their weights/probabilities

    Relationships
    ------------
    states : List[AgentState]
        History of agent states over time
    actions : List[AgentAction]
        History of actions taken by the agent
    health_incidents : List[HealthIncident]
        Record of health-affecting events
    targeted_actions : List[AgentAction]
        Actions where this agent is the target
    """

    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_agent_type", "agent_type"),
        Index("idx_agents_birth_time", "birth_time"),
        Index("idx_agents_death_time", "death_time"),
    )

    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    agent_id = Column(String(64), primary_key=True)
    birth_time = Column(Integer)
    death_time = Column(Integer)
    agent_type = Column(String(50))
    position_x = Column(Float(precision=6))
    position_y = Column(Float(precision=6))
    initial_resources = Column(Float(precision=6))
    starting_health = Column(Float(precision=4))
    genome_id = Column(String(64))
    generation = Column(Integer)
    action_weights = Column(JSON, nullable=True)

    # Relationships
    states = relationship("AgentStateModel", back_populates="agent")
    actions = relationship(
        "ActionModel",
        back_populates="agent",
        foreign_keys="[ActionModel.agent_id]",
        primaryjoin="AgentModel.agent_id==ActionModel.agent_id",
    )
    health_incidents = relationship("HealthIncident", back_populates="agent")
    targeted_actions = relationship(
        "ActionModel",
        foreign_keys="[ActionModel.action_target_id]",
        primaryjoin="AgentModel.agent_id==ActionModel.action_target_id",
        backref="target",
        overlaps="targeted_by",
    )


class AgentStateModel(Base):
    """Tracks the state of an agent at a specific simulation step.

    This model captures the complete state of an agent at each time step,
    including position, resources, health, and cumulative metrics.

    Attributes
    ----------
    id : int
        Unique identifier for the state record
    step_number : int
        Simulation step this state represents
    agent_id : str
        ID of the agent this state belongs to
    position_x : float
        Current X-coordinate position
    position_y : float
        Current Y-coordinate position
    position_z : float
        Current Z-coordinate position
    resource_level : float
        Current resource amount held by agent
    current_health : float
        Current health level
    starting_health : float
        Maximum possible health
    starvation_counter : int
        Consecutive steps with zero resources (for starvation tracking)
    is_defending : bool
        Whether agent is in defensive stance
    total_reward : float
        Cumulative reward received
    age : int
        Number of steps agent has existed

    Relationships
    ------------
    agent : Agent
        The agent this state belongs to

    Methods
    -------
    as_dict() -> Dict[str, Any]
        Convert state to dictionary format for serialization
    """

    __tablename__ = "agent_states"
    __table_args__ = (
        Index("idx_agent_states_agent_id", "agent_id"),
        Index("idx_agent_states_step_number", "step_number"),
        Index("idx_agent_states_agent_step", "agent_id", "step_number"),
        {"sqlite_autoincrement": False},
    )

    id = Column(
        String(128), primary_key=True, nullable=False
    )  # Stores "<simulation_id>:<agent_id>-<step_number>" when simulation_id is present, else "agent_id-<step_number>"
    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    step_number = Column(Integer)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"))
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    resource_level = Column(Float)
    current_health = Column(Float)
    starting_health = Column(Float(precision=4))
    starvation_counter = Column(Integer)
    is_defending = Column(Boolean)
    total_reward = Column(Float)
    age = Column(Integer)

    agent = relationship("AgentModel", back_populates="states")

    def __init__(self, **kwargs):
        # Generate id before initializing other attributes
        if "agent_id" in kwargs and "step_number" in kwargs:
            sim_id = kwargs.get("simulation_id")
            if sim_id:
                kwargs["id"] = f"{sim_id}:{kwargs['agent_id']}-{kwargs['step_number']}"
            else:
                kwargs["id"] = f"{kwargs['agent_id']}-{kwargs['step_number']}"
        elif "id" not in kwargs:
            raise ValueError(
                "Both agent_id and step_number are required to create AgentStateModel"
            )
        super().__init__(**kwargs)

    @staticmethod
    def generate_id(agent_id: str, step_number: int) -> str:
        """Generate a unique ID for an agent state."""
        # Centralize via Identity for consistency without instantiation
        from farm.utils.identity import Identity

        return str(Identity.agent_state_id(agent_id, step_number))

    def as_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "step_number": self.step_number,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "resource_level": self.resource_level,
            "current_health": self.current_health,
            "starting_health": self.starting_health,
            "starvation_counter": self.starvation_counter,
            "is_defending": self.is_defending,
            "total_reward": self.total_reward,
            "age": self.age,
        }


class ResourceModel(Base):
    """Tracks the state of resources in the environment.

    This model records the amount and location of resources at each simulation step,
    enabling analysis of resource distribution and movement patterns.

    Attributes
    ----------
    id : int
        Unique identifier for the resource state record
    step_number : int
        Simulation step this state represents
    resource_id : str
        Identifier for the specific resource (format: resource_{shortid})
    amount : float
        Quantity of resource available
    position_x : float
        X-coordinate of resource location
    position_y : float
        Y-coordinate of resource location

    Methods
    -------
    as_dict() -> Dict[str, Any]
        Convert resource state to dictionary format
    """

    __tablename__ = "resource_states"
    __table_args__ = (
        Index("idx_resource_states_step_number", "step_number"),
        Index("idx_resource_states_resource_id", "resource_id"),
    )

    id = Column(Integer, primary_key=True)
    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    step_number = Column(Integer)
    resource_id = Column(String(64))
    amount = Column(Float)
    position_x = Column(Float)
    position_y = Column(Float)

    def as_dict(self) -> Dict[str, Any]:
        """Convert resource state to dictionary."""
        return {
            "resource_id": self.resource_id,
            "amount": self.amount,
            "position": (self.position_x, self.position_y),
        }


class SimulationStepModel(Base):
    """Records simulation-wide metrics for each time step.

    This model captures aggregate statistics and metrics about the entire simulation
    state at each step, including population counts, resource metrics, and various
    performance indicators.

    Attributes
    ----------
    step_number : int
        Unique step identifier
    total_agents : int
        Total number of living agents
    agent_type_counts : dict
        JSON column containing counts by agent type (e.g., {"system": 10, "independent": 5, "control": 3})
    total_resources : float
        Total resources in environment
    average_agent_resources : float
        Mean resources per agent
    births : int
        Number of new agents created this step
    deaths : int
        Number of agents that died this step
    current_max_generation : int
        Highest generation number present
    resource_efficiency : float
        Measure of resource utilization efficiency
    resource_distribution_entropy : float
        Measure of resource distribution evenness
    average_agent_health : float
        Mean health across all agents
    average_agent_age : int
        Mean age of all agents
    average_reward : float
        Mean reward received by agents
    genetic_diversity : float
        Measure of genetic variation in population
    dominant_genome_ratio : float
        Proportion of agents sharing most common genome
    resources_consumed : float
        Total resources consumed by the simulation

    Methods
    -------
    as_dict() -> Dict[str, Any]
        Convert step metrics to dictionary format
    """

    __tablename__ = "simulation_steps"

    __table_args__ = (
        PrimaryKeyConstraint("step_number", "simulation_id"),
        Index("idx_simulation_steps_step_number", "step_number"),
        Index("idx_simulation_steps_simulation_id", "simulation_id"),
    )

    step_number = Column(Integer, primary_key=False)
    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    total_agents = Column(Integer)
    agent_type_counts = Column(JSON)
    total_resources = Column(Float)
    average_agent_resources = Column(Float)
    births = Column(Integer)
    deaths = Column(Integer)
    current_max_generation = Column(Integer)
    resource_efficiency = Column(Float)
    resource_distribution_entropy = Column(Float)
    average_agent_health = Column(Float)
    average_agent_age = Column(Integer)
    average_reward = Column(Float)
    genetic_diversity = Column(Float)
    dominant_genome_ratio = Column(Float)
    resources_consumed = Column(Float, default=0.0)

    def as_dict(self) -> Dict[str, Any]:
        """Convert simulation step to dictionary."""
        # Extract individual counts from JSON for backwards compatibility
        agent_counts = self.agent_type_counts or {}
        return {
            "total_agents": self.total_agents,
            "agent_type_counts": agent_counts,
            "system_agents": agent_counts.get("system", 0),
            "independent_agents": agent_counts.get("independent", 0),
            "control_agents": agent_counts.get("control", 0),
            "total_resources": self.total_resources,
            "average_agent_resources": self.average_agent_resources,
            "births": self.births,
            "deaths": self.deaths,
            "current_max_generation": self.current_max_generation,
            "resource_efficiency": self.resource_efficiency,
            "resource_distribution_entropy": self.resource_distribution_entropy,
            "average_agent_health": self.average_agent_health,
            "average_agent_age": self.average_agent_age,
            "average_reward": self.average_reward,
            "genetic_diversity": self.genetic_diversity,
            "dominant_genome_ratio": self.dominant_genome_ratio,
            "resources_consumed": self.resources_consumed,
        }


class ActionModel(Base):
    """Record of an action taken by an agent during simulation.

    This model tracks individual actions performed by agents, including the type of action,
    target (if any), and resulting rewards. Resource and state information can be derived
    from the agent_states table.

    Attributes
    ----------
    action_id : int
        Unique identifier for the action
    step_number : int
        Simulation step when the action occurred
    agent_id : str
        ID of the agent that performed the action
    action_type : str
        Type of action performed (e.g., 'move', 'attack', 'share')
    action_target_id : Optional[str]
        ID of the target (agent_id for agent-to-agent actions, resource_id for resource gathering)
    reward : float
        Reward received for the action
    details : Optional[str]
        JSON string containing additional action details
    module_type : Optional[str]
        Type of learning module that generated this action (e.g., 'dqn', 'ppo')
    module_id : Optional[str]
        Unique identifier for the specific learning module instance

    Relationships
    ------------
    agent : Agent
        The agent that performed the action
    """

    __tablename__ = "agent_actions"
    __table_args__ = (
        Index("idx_agent_actions_step_number", "step_number"),
        Index("idx_agent_actions_agent_id", "agent_id"),
        Index("idx_agent_actions_action_type", "action_type"),
        Index("idx_agent_actions_module_type", "module_type"),
    )

    action_id = Column(Integer, primary_key=True)
    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    step_number = Column(Integer, nullable=False)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=False)
    action_type = Column(String(20), nullable=False)
    action_target_id = Column(String(64), nullable=True)
    reward = Column(Float(precision=6), nullable=True)
    details = Column(String(1024), nullable=True)
    module_type = Column(String(50), nullable=True)
    module_id = Column(String(64), nullable=True)

    agent = relationship(
        "AgentModel", back_populates="actions", foreign_keys=[agent_id]
    )


class HealthIncident(Base):
    """Health incident records."""

    __tablename__ = "health_incidents"
    __table_args__ = (
        Index("idx_health_incidents_step_number", "step_number"),
        Index("idx_health_incidents_agent_id", "agent_id"),
    )

    incident_id = Column(Integer, primary_key=True)
    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    step_number = Column(Integer, nullable=False)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=False)
    health_before = Column(Float(precision=4))
    health_after = Column(Float(precision=4))
    cause = Column(String(50), nullable=False)
    details = Column(String(512))

    agent = relationship("AgentModel", back_populates="health_incidents")


class SimulationConfig(Base):
    """Simulation configuration records."""

    __tablename__ = "simulation_config"

    simulation_id = Column(String(64), ForeignKey("simulations.simulation_id"))
    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(String(4096), nullable=False)


class ExperimentModel(Base):
    """Represents a research experiment that groups related simulations.

    This model stores metadata about an experiment, including its purpose,
    hypothesis, and parameters varied across simulations.

    Attributes
    ----------
    experiment_id : str
        Unique identifier for the experiment
    name : str
        Human-readable name of the experiment
    description : str
        Detailed description of the experiment's purpose
    hypothesis : str
        The research hypothesis being tested
    creation_date : DateTime
        When the experiment was created
    last_updated : DateTime
        When the experiment was last modified
    status : str
        Current status (e.g., 'planned', 'running', 'completed', 'analyzed')
    tags : list
        List of keywords/tags for categorization
    variables : dict
        Dictionary of variables being manipulated across simulations
    results_summary : dict
        High-level findings from the experiment
    notes : str
        Additional research notes or observations

    Relationships
    ------------
    simulations : List[Simulation]
        All simulations that are part of this experiment
    """

    __tablename__ = "experiments"

    experiment_id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String(4096), nullable=True)
    hypothesis = Column(String(2048), nullable=True)
    creation_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    status = Column(String(50), default="planned")
    tags = Column(JSON, nullable=True)
    variables = Column(JSON, nullable=True)
    results_summary = Column(JSON, nullable=True)
    notes = Column(String(4096), nullable=True)

    # Relationships
    simulations = relationship("Simulation", back_populates="experiment")

    def __repr__(self):
        return f"<Experiment(experiment_id={self.experiment_id}, name={self.name}, status={self.status})>"


class Simulation(Base):
    """Simulation records."""

    __tablename__ = "simulations"

    simulation_id = Column(String(64), primary_key=True)
    experiment_id = Column(
        String(64), ForeignKey("experiments.experiment_id"), nullable=True
    )
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime, nullable=True)
    status = Column(String(50), default="pending")
    parameters = Column(JSON, nullable=False)
    results_summary = Column(JSON, nullable=True)
    simulation_db_path = Column(String(255), nullable=False)

    # Relationships
    experiment = relationship("ExperimentModel", back_populates="simulations")

    def __repr__(self):
        return f"<Simulation(simulation_id={self.simulation_id}, status={self.status})>"


@dataclass
class SimulationDifference:
    """Represents differences between two simulations.

    Attributes
    ----------
    metadata_diff : Dict[str, tuple]
        Differences in basic metadata fields, with (sim1_value, sim2_value) tuples
    parameter_diff : Dict
        Differences in simulation parameters (from DeepDiff)
    results_diff : Dict
        Differences in results summary (from DeepDiff)
    step_metrics_diff : Dict[str, Dict[str, float]]
        Statistical differences in step metrics (min, max, mean, etc.)
    """

    metadata_diff: Dict[str, tuple]
    parameter_diff: Dict
    results_diff: Dict
    step_metrics_diff: Dict[str, Dict[str, float]]


class SimulationComparison:
    """Utility class for comparing two simulations.

    This class provides methods to compare different aspects of two simulations,
    including metadata, parameters, results, and step metrics.
    """

    def __init__(self, sim1: Simulation, sim2: Simulation):
        """Initialize with two simulations to compare.

        Parameters
        ----------
        sim1 : Simulation
            First simulation to compare
        sim2 : Simulation
            Second simulation to compare
        """
        self.sim1 = sim1
        self.sim2 = sim2

    def _compare_metadata(self) -> Dict[str, tuple]:
        """Compare basic metadata fields between simulations."""
        metadata_fields = ["status", "simulation_db_path"]
        differences = {}

        for field in metadata_fields:
            val1 = getattr(self.sim1, field)
            val2 = getattr(self.sim2, field)
            if val1 != val2:
                differences[field] = (val1, val2)

        # Compare timestamps
        start_time1: Any = self.sim1.start_time
        start_time2: Any = self.sim2.start_time
        if start_time1 != start_time2:
            differences["start_time"] = (start_time1, start_time2)

        end_time1: Any = self.sim1.end_time
        end_time2: Any = self.sim2.end_time
        if end_time1 != end_time2:
            differences["end_time"] = (end_time1, end_time2)

        return differences

    def _compare_parameters(self) -> Dict:
        """Compare simulation parameters using DeepDiff."""
        return DeepDiff(self.sim1.parameters, self.sim2.parameters, ignore_order=True)

    def _compare_results(self) -> Dict:
        """Compare results summaries using DeepDiff."""
        return DeepDiff(
            self.sim1.results_summary, self.sim2.results_summary, ignore_order=True
        )

    def _compare_step_metrics(self, session) -> Dict[str, Dict[str, float]]:
        """Compare statistical summaries of step metrics.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping metric names to their statistical differences
        """
        metrics = {
            "total_agents": [],
            "births": [],
            "deaths": [],
            "average_agent_health": [],
            "average_reward": [],
            "combat_encounters": [],
            "resources_consumed": [],
        }

        differences = {}

        # Get step data for both simulations
        for sim_id, metric_list in [
            (self.sim1.simulation_id, "_sim1_metrics"),
            (self.sim2.simulation_id, "_sim2_metrics"),
        ]:
            steps = (
                session.query(SimulationStepModel)
                .filter(SimulationStepModel.simulation_id == sim_id)
                .all()
            )

            setattr(
                self,
                metric_list,
                {
                    metric: [getattr(step, metric) for step in steps]
                    for metric in metrics
                },
            )

        # Compare statistics for each metric
        for metric in metrics:
            sim1_values = getattr(self, "_sim1_metrics")[metric]
            sim2_values = getattr(self, "_sim2_metrics")[metric]

            if sim1_values and sim2_values:  # Only compare if both have data
                differences[metric] = {
                    "mean_diff": statistics.mean(sim1_values)
                    - statistics.mean(sim2_values),
                    "max_diff": max(sim1_values) - max(sim2_values),
                    "min_diff": min(sim1_values) - min(sim2_values),
                    "std_diff": statistics.stdev(sim1_values)
                    - statistics.stdev(sim2_values),
                }

        return differences

    def compare(self, session) -> SimulationDifference:
        """Perform full comparison between simulations.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for database queries

        Returns
        -------
        SimulationDifference
            Object containing all differences between the simulations
        """
        return SimulationDifference(
            metadata_diff=self._compare_metadata(),
            parameter_diff=self._compare_parameters(),
            results_diff=self._compare_results(),
            step_metrics_diff=self._compare_step_metrics(session),
        )
