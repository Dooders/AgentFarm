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
- Simulation: Stores simulation metadata

Each model includes appropriate indexes for query optimization and relationships
between related tables.
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from deepdiff import DeepDiff
import statistics

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

logger = logging.getLogger(__name__)

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
    starvation_threshold : int
        Resource level below which agent begins to starve
    genome_id : str
        Unique identifier for agent's genetic code
    generation : int
        Generational number in evolutionary lineage

    Relationships
    ------------
    states : List[AgentState]
        History of agent states over time
    actions : List[AgentAction]
        History of actions taken by the agent
    health_incidents : List[HealthIncident]
        Record of health-affecting events
    learning_experiences : List[LearningExperience]
        History of learning events and outcomes
    targeted_actions : List[AgentAction]
        Actions where this agent is the target
    """

    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_agent_type", "agent_type"),
        Index("idx_agents_birth_time", "birth_time"),
        Index("idx_agents_death_time", "death_time"),
    )

    agent_id = Column(String(64), primary_key=True)
    birth_time = Column(Integer)
    death_time = Column(Integer)
    agent_type = Column(String(50))
    position_x = Column(Float(precision=6))
    position_y = Column(Float(precision=6))
    initial_resources = Column(Float(precision=6))
    starting_health = Column(Float(precision=4))
    starvation_threshold = Column(Integer)
    genome_id = Column(String(64))
    generation = Column(Integer)

    # Relationships
    states = relationship("AgentStateModel", back_populates="agent")
    actions = relationship(
        "ActionModel",
        back_populates="agent",
        foreign_keys="[ActionModel.agent_id]",
        primaryjoin="AgentModel.agent_id==ActionModel.agent_id",
    )
    health_incidents = relationship("HealthIncident", back_populates="agent")
    learning_experiences = relationship(
        "LearningExperienceModel", back_populates="agent"
    )
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
    starvation_threshold : int
        Resource level that triggers starvation
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
    )

    id = Column(
        String(128), primary_key=True, nullable=False
    )  # Will store "agent_id-step_number"
    step_number = Column(Integer)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"))
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    resource_level = Column(Float)
    current_health = Column(Float)
    is_defending = Column(Boolean)
    total_reward = Column(Float)
    age = Column(Integer)

    agent = relationship("AgentModel", back_populates="states")

    def __init__(self, **kwargs):
        # Generate id before initializing other attributes
        if "agent_id" in kwargs and "step_number" in kwargs:
            kwargs["id"] = f"{kwargs['agent_id']}-{kwargs['step_number']}"
        elif not "id" in kwargs:
            raise ValueError(
                "Both agent_id and step_number are required to create AgentStateModel"
            )
        super().__init__(**kwargs)

    @staticmethod
    def generate_id(agent_id: str, step_number: int) -> str:
        """Generate a unique ID for an agent state."""
        return f"{agent_id}-{step_number}"

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
    resource_id : int
        Identifier for the specific resource
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
    step_number = Column(Integer)
    resource_id = Column(Integer)
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
    system_agents : int
        Number of system-type agents
    independent_agents : int
        Number of independent-type agents
    control_agents : int
        Number of control-type agents
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
    combat_encounters : int
        Number of combat interactions
    successful_attacks : int
        Number of successful attack actions
    resources_shared : float
        Amount of resources transferred between agents
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
    __table_args__ = (Index("idx_simulation_steps_step_number", "step_number"),)

    step_number = Column(Integer, primary_key=True)
    total_agents = Column(Integer)
    system_agents = Column(Integer)
    independent_agents = Column(Integer)
    control_agents = Column(Integer)
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
    combat_encounters = Column(Integer)
    successful_attacks = Column(Integer)
    resources_shared = Column(Float)
    genetic_diversity = Column(Float)
    dominant_genome_ratio = Column(Float)
    resources_consumed = Column(Float, default=0.0)

    def as_dict(self) -> Dict[str, Any]:
        """Convert simulation step to dictionary."""
        return {
            "total_agents": self.total_agents,
            "system_agents": self.system_agents,
            "independent_agents": self.independent_agents,
            "control_agents": self.control_agents,
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
            "combat_encounters": self.combat_encounters,
            "successful_attacks": self.successful_attacks,
            "resources_shared": self.resources_shared,
            "genetic_diversity": self.genetic_diversity,
            "dominant_genome_ratio": self.dominant_genome_ratio,
            "resources_consumed": self.resources_consumed,
        }


class ActionModel(Base):
    """Record of an action taken by an agent during simulation.

    This model tracks individual actions performed by agents, including the type of action,
    target (if any), position changes, resource changes, and resulting rewards.

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
    action_target_id : Optional[int]
        ID of the target agent, if the action involved another agent
    state_before_id : Optional[int]
        Reference to agent's state before the action
    state_after_id : Optional[int]
        Reference to agent's state after the action
    resources_before : float
        Agent's resource level before the action
    resources_after : float
        Agent's resource level after the action
    reward : float
        Reward received for the action
    details : Optional[str]
        JSON string containing additional action details

    Relationships
    ------------
    agent : Agent
        The agent that performed the action
    state_before : Optional[AgentState]
        The agent's state before the action
    state_after : Optional[AgentState]
        The agent's state after the action
    """

    __tablename__ = "agent_actions"
    __table_args__ = (
        Index("idx_agent_actions_step_number", "step_number"),
        Index("idx_agent_actions_agent_id", "agent_id"),
        Index("idx_agent_actions_action_type", "action_type"),
    )

    action_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=False)
    action_type = Column(String(20), nullable=False)
    action_target_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=True)
    state_before_id = Column(String(128), ForeignKey("agent_states.id"), nullable=True)
    state_after_id = Column(String(128), ForeignKey("agent_states.id"), nullable=True)
    resources_before = Column(Float(precision=6), nullable=True)
    resources_after = Column(Float(precision=6), nullable=True)
    reward = Column(Float(precision=6), nullable=True)
    details = Column(String(1024), nullable=True)

    agent = relationship(
        "AgentModel", back_populates="actions", foreign_keys=[agent_id]
    )
    state_before = relationship("AgentStateModel", foreign_keys=[state_before_id])
    state_after = relationship("AgentStateModel", foreign_keys=[state_after_id])


class LearningExperienceModel(Base):
    """Learning experience records."""

    __tablename__ = "learning_experiences"
    __table_args__ = (
        Index("idx_learning_experiences_step_number", "step_number"),
        Index("idx_learning_experiences_agent_id", "agent_id"),
        Index("idx_learning_experiences_module_type", "module_type"),
    )

    experience_id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    agent_id = Column(String(64), ForeignKey("agents.agent_id"))
    module_type = Column(String(50))
    module_id = Column(String(64))
    action_taken = Column(Integer)
    action_taken_mapped = Column(String(20))
    reward = Column(Float(precision=6))

    agent = relationship("AgentModel", back_populates="learning_experiences")


class HealthIncident(Base):
    """Health incident records."""

    __tablename__ = "health_incidents"
    __table_args__ = (
        Index("idx_health_incidents_step_number", "step_number"),
        Index("idx_health_incidents_agent_id", "agent_id"),
    )

    incident_id = Column(Integer, primary_key=True)
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

    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(String(4096), nullable=False)


class Simulation(Base):
    """Simulation records."""

    __tablename__ = "simulations"

    simulation_id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime, nullable=True)
    status = Column(String(50), default="pending")
    parameters = Column(JSON, nullable=False)
    results_summary = Column(JSON, nullable=True)
    simulation_db_path = Column(String(255), nullable=False)

    # Relationships (optional, for future extensions)
    # e.g., could relate to logs, individual agent data, etc.
    # logs = relationship("Log", back_populates="simulation")

    def __repr__(self):
        return f"<Simulation(simulation_id={self.simulation_id}, status={self.status})>"


class ReproductionEventModel(Base):
    """Records reproduction attempts and outcomes in the simulation.
    
    This model tracks both successful and failed reproduction attempts,
    including details about the parent agent, resources involved, and
    any offspring created.

    Attributes
    ----------
    event_id : int
        Unique identifier for the reproduction event
    step_number : int
        Simulation step when the reproduction attempt occurred
    parent_id : str
        ID of the agent attempting reproduction
    offspring_id : Optional[str]
        ID of the created offspring (if successful)
    success : bool
        Whether the reproduction attempt succeeded
    parent_resources_before : float
        Parent's resource level before reproduction
    parent_resources_after : float
        Parent's resource level after reproduction
    offspring_initial_resources : float
        Resources given to offspring (if successful)
    failure_reason : Optional[str]
        Reason for failed reproduction attempt
    parent_generation : int
        Generation number of parent agent
    offspring_generation : Optional[int]
        Generation number of offspring (if successful)
    parent_position_x : float
        X-coordinate where reproduction occurred
    parent_position_y : float
        Y-coordinate where reproduction occurred
    timestamp : DateTime
        When the event occurred

    Relationships
    ------------
    parent : Agent
        The agent that attempted reproduction
    offspring : Optional[Agent]
        The newly created agent (if successful)
    """

    __tablename__ = "reproduction_events"
    __table_args__ = (
        Index("idx_reproduction_events_step_number", "step_number"),
        Index("idx_reproduction_events_parent_id", "parent_id"),
        Index("idx_reproduction_events_success", "success"),
    )

    event_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    parent_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=False)
    offspring_id = Column(String(64), ForeignKey("agents.agent_id"), nullable=True)
    success = Column(Boolean, nullable=False)
    parent_resources_before = Column(Float(precision=6), nullable=False)
    parent_resources_after = Column(Float(precision=6), nullable=False)
    offspring_initial_resources = Column(Float(precision=6), nullable=True)
    failure_reason = Column(String(255), nullable=True)
    parent_generation = Column(Integer, nullable=False)
    offspring_generation = Column(Integer, nullable=True)
    parent_position_x = Column(Float, nullable=False)
    parent_position_y = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    parent = relationship(
        "AgentModel",
        foreign_keys=[parent_id],
        backref="reproduction_attempts"
    )
    offspring = relationship(
        "AgentModel",
        foreign_keys=[offspring_id],
        backref="creation_event"
    )

    def as_dict(self) -> Dict[str, Any]:
        """Convert reproduction event to dictionary format."""
        return {
            "step_number": self.step_number,
            "parent_id": self.parent_id,
            "offspring_id": self.offspring_id,
            "success": self.success,
            "parent_resources_before": self.parent_resources_before,
            "parent_resources_after": self.parent_resources_after,
            "offspring_initial_resources": self.offspring_initial_resources,
            "failure_reason": self.failure_reason,
            "parent_generation": self.parent_generation,
            "offspring_generation": self.offspring_generation,
            "parent_position": (self.parent_position_x, self.parent_position_y),
            "timestamp": self.timestamp,
        }


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
        metadata_fields = ['status', 'simulation_db_path']
        differences = {}
        
        for field in metadata_fields:
            val1 = getattr(self.sim1, field)
            val2 = getattr(self.sim2, field)
            if val1 != val2:
                differences[field] = (val1, val2)
                
        # Compare timestamps
        if self.sim1.start_time != self.sim2.start_time:
            differences['start_time'] = (self.sim1.start_time, self.sim2.start_time)
        if self.sim1.end_time != self.sim2.end_time:
            differences['end_time'] = (self.sim1.end_time, self.sim2.end_time)
            
        return differences
    
    def _compare_parameters(self) -> Dict:
        """Compare simulation parameters using DeepDiff."""
        return DeepDiff(self.sim1.parameters, self.sim2.parameters, ignore_order=True)
    
    def _compare_results(self) -> Dict:
        """Compare results summaries using DeepDiff."""
        return DeepDiff(self.sim1.results_summary, self.sim2.results_summary, ignore_order=True)
    
    def _compare_step_metrics(self, session) -> Dict[str, Dict[str, float]]:
        """Compare statistical summaries of step metrics.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping metric names to their statistical differences
        """
        metrics = {
            'total_agents': [], 'births': [], 'deaths': [],
            'average_agent_health': [], 'average_reward': [],
            'combat_encounters': [], 'resources_consumed': []
        }
        
        differences = {}
        
        # Get step data for both simulations
        for sim_id, metric_list in [
            (self.sim1.simulation_id, '_sim1_metrics'),
            (self.sim2.simulation_id, '_sim2_metrics')
        ]:
            steps = session.query(SimulationStepModel).filter(
                SimulationStepModel.simulation_id == sim_id
            ).all()
            
            setattr(self, metric_list, {
                metric: [getattr(step, metric) for step in steps]
                for metric in metrics.keys()
            })
        
        # Compare statistics for each metric
        for metric in metrics.keys():
            sim1_values = getattr(self, '_sim1_metrics')[metric]
            sim2_values = getattr(self, '_sim2_metrics')[metric]
            
            if sim1_values and sim2_values:  # Only compare if both have data
                differences[metric] = {
                    'mean_diff': statistics.mean(sim1_values) - statistics.mean(sim2_values),
                    'max_diff': max(sim1_values) - max(sim2_values),
                    'min_diff': min(sim1_values) - min(sim2_values),
                    'std_diff': statistics.stdev(sim1_values) - statistics.stdev(sim2_values)
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
            step_metrics_diff=self._compare_step_metrics(session)
        )
