from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Simulation(Base):
    """Base simulation information"""

    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True)
    iteration = Column(Integer, nullable=False, unique=True, index=True)

    # Relationships
    dominance_metrics = relationship(
        "DominanceMetrics", back_populates="simulation", uselist=False
    )
    agent_populations = relationship(
        "AgentPopulation", back_populates="simulation", uselist=False
    )
    reproduction_stats = relationship(
        "ReproductionStats", back_populates="simulation", uselist=False
    )
    dominance_switching = relationship(
        "DominanceSwitching", back_populates="simulation", uselist=False
    )
    resource_distribution = relationship(
        "ResourceDistribution", back_populates="simulation", uselist=False
    )
    high_low_switching = relationship(
        "HighLowSwitchingComparison", back_populates="simulation", uselist=False
    )
    correlation_analysis = relationship(
        "CorrelationAnalysis", back_populates="simulation", uselist=False
    )


class DominanceMetrics(Base):
    """Core dominance metrics for each simulation"""

    __tablename__ = "dominance_metrics"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Dominance type results
    population_dominance = Column(String, nullable=False)
    survival_dominance = Column(String, nullable=False)
    comprehensive_dominance = Column(String, nullable=False)

    # Dominance scores
    system_dominance_score = Column(Float, nullable=False)
    independent_dominance_score = Column(Float, nullable=False)
    control_dominance_score = Column(Float, nullable=False)

    # Area under curve metrics
    system_auc = Column(Float, nullable=False)
    independent_auc = Column(Float, nullable=False)
    control_auc = Column(Float, nullable=False)

    # Recency weighted area under curve
    system_recency_weighted_auc = Column(Float, nullable=False)
    independent_recency_weighted_auc = Column(Float, nullable=False)
    control_recency_weighted_auc = Column(Float, nullable=False)

    # Dominance duration metrics
    system_dominance_duration = Column(Float, nullable=False)
    independent_dominance_duration = Column(Float, nullable=False)
    control_dominance_duration = Column(Float, nullable=False)

    # Growth trend metrics
    system_growth_trend = Column(Float, nullable=False)
    independent_growth_trend = Column(Float, nullable=False)
    control_growth_trend = Column(Float, nullable=False)

    # Final population ratio metrics
    system_final_ratio = Column(Float, nullable=False)
    independent_final_ratio = Column(Float, nullable=False)
    control_final_ratio = Column(Float, nullable=False)

    # Relationship
    simulation = relationship("Simulation", back_populates="dominance_metrics")


class AgentPopulation(Base):
    """Agent population and survival statistics"""

    __tablename__ = "agent_populations"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Final population counts
    system_agents = Column(Integer)
    independent_agents = Column(Integer)
    control_agents = Column(Integer)
    total_agents = Column(Integer)
    final_step = Column(Integer)

    # System agent survival statistics
    system_count = Column(Integer)
    system_alive = Column(Integer)
    system_dead = Column(Integer)
    system_avg_survival = Column(Float)
    system_dead_ratio = Column(Float)

    # Independent agent survival statistics
    independent_count = Column(Integer)
    independent_alive = Column(Integer)
    independent_dead = Column(Integer)
    independent_avg_survival = Column(Float)
    independent_dead_ratio = Column(Float)

    # Control agent survival statistics
    control_count = Column(Integer)
    control_alive = Column(Integer)
    control_dead = Column(Integer)
    control_avg_survival = Column(Float)
    control_dead_ratio = Column(Float)

    # Initial counts
    initial_system_count = Column(Integer)
    initial_independent_count = Column(Integer)
    initial_control_count = Column(Integer)
    initial_resource_count = Column(Integer)
    initial_resource_amount = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="agent_populations")


class ReproductionStats(Base):
    """Reproduction statistics for each agent type"""

    __tablename__ = "reproduction_stats"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # System agent reproduction
    system_reproduction_attempts = Column(Integer)
    system_reproduction_successes = Column(Integer)
    system_reproduction_failures = Column(Integer)
    system_reproduction_success_rate = Column(Float)
    system_first_reproduction_time = Column(Float)
    system_reproduction_efficiency = Column(Float)
    system_avg_resources_per_reproduction = Column(Float)
    system_avg_offspring_resources = Column(Float)

    # Independent agent reproduction
    independent_reproduction_attempts = Column(Integer)
    independent_reproduction_successes = Column(Integer)
    independent_reproduction_failures = Column(Integer)
    independent_reproduction_success_rate = Column(Float)
    independent_first_reproduction_time = Column(Float)
    independent_reproduction_efficiency = Column(Float)
    independent_avg_resources_per_reproduction = Column(Float)
    independent_avg_offspring_resources = Column(Float)

    # Control agent reproduction
    control_reproduction_attempts = Column(Integer)
    control_reproduction_successes = Column(Integer)
    control_reproduction_failures = Column(Integer)
    control_reproduction_success_rate = Column(Float)
    control_first_reproduction_time = Column(Float)
    control_reproduction_efficiency = Column(Float)
    control_avg_resources_per_reproduction = Column(Float)
    control_avg_offspring_resources = Column(Float)

    # Reproduction advantage metrics
    independent_vs_control_first_reproduction_advantage = Column(Float)
    independent_vs_control_reproduction_efficiency_advantage = Column(Float)
    independent_vs_control_reproduction_rate_advantage = Column(Float)
    system_vs_independent_reproduction_rate_advantage = Column(Float)
    system_vs_control_reproduction_rate_advantage = Column(Float)
    system_vs_independent_reproduction_efficiency_advantage = Column(Float)
    system_vs_control_first_reproduction_advantage = Column(Float)
    system_vs_independent_first_reproduction_advantage = Column(Float)
    system_vs_control_reproduction_efficiency_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="reproduction_stats")


class DominanceSwitching(Base):
    """Dominance switching metrics and transition probabilities"""

    __tablename__ = "dominance_switching"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Switching metrics
    total_switches = Column(Integer)
    switches_per_step = Column(Float)
    dominance_stability = Column(Float)

    # Average dominance periods
    system_avg_dominance_period = Column(Float)
    independent_avg_dominance_period = Column(Float)
    control_avg_dominance_period = Column(Float)

    # Phase-specific switch counts
    early_phase_switches = Column(Integer)
    middle_phase_switches = Column(Integer)
    late_phase_switches = Column(Integer)

    # Average switches by agent type
    control_avg_switches = Column(Float)
    independent_avg_switches = Column(Float)
    system_avg_switches = Column(Float)

    # Transition probabilities
    system_to_system = Column(Float)
    system_to_independent = Column(Float)
    system_to_control = Column(Float)
    independent_to_system = Column(Float)
    independent_to_independent = Column(Float)
    independent_to_control = Column(Float)
    control_to_system = Column(Float)
    control_to_independent = Column(Float)
    control_to_control = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="dominance_switching")


class ResourceDistribution(Base):
    """Resource distribution metrics"""

    __tablename__ = "resource_distribution"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # System agent resource metrics
    systemagent_avg_resource_dist = Column(Float)
    systemagent_weighted_resource_dist = Column(Float)
    systemagent_nearest_resource_dist = Column(Float)
    systemagent_resources_in_range = Column(Float)
    systemagent_resource_amount_in_range = Column(Float)

    # Independent agent resource metrics
    independentagent_avg_resource_dist = Column(Float)
    independentagent_weighted_resource_dist = Column(Float)
    independentagent_nearest_resource_dist = Column(Float)
    independentagent_resources_in_range = Column(Float)
    independentagent_resource_amount_in_range = Column(Float)

    # Control agent resource metrics
    controlagent_avg_resource_dist = Column(Float)
    controlagent_weighted_resource_dist = Column(Float)
    controlagent_nearest_resource_dist = Column(Float)
    controlagent_resources_in_range = Column(Float)
    controlagent_resource_amount_in_range = Column(Float)

    # Resource correlation metrics
    positive_corr_controlagent_resource_amount_in_range = Column(Float)
    positive_corr_systemagent_avg_resource_dist = Column(Float)
    positive_corr_systemagent_weighted_resource_dist = Column(Float)
    positive_corr_independentagent_avg_resource_dist = Column(Float)
    positive_corr_independentagent_weighted_resource_dist = Column(Float)
    negative_corr_systemagent_resource_amount_in_range = Column(Float)
    negative_corr_systemagent_nearest_resource_dist = Column(Float)
    negative_corr_independentagent_resource_amount_in_range = Column(Float)
    negative_corr_controlagent_avg_resource_dist = Column(Float)
    negative_corr_controlagent_nearest_resource_dist = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="resource_distribution")


class HighLowSwitchingComparison(Base):
    """Comparison metrics between high and low switching simulations"""

    __tablename__ = "high_low_switching_comparison"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # System agent reproduction comparisons
    system_reproduction_attempts_high_switching_mean = Column(Float)
    system_reproduction_attempts_low_switching_mean = Column(Float)
    system_reproduction_attempts_difference = Column(Float)
    system_reproduction_attempts_percent_difference = Column(Float)

    system_reproduction_successes_high_switching_mean = Column(Float)
    system_reproduction_successes_low_switching_mean = Column(Float)
    system_reproduction_successes_difference = Column(Float)
    system_reproduction_successes_percent_difference = Column(Float)

    system_reproduction_failures_high_switching_mean = Column(Float)
    system_reproduction_failures_low_switching_mean = Column(Float)
    system_reproduction_failures_difference = Column(Float)
    system_reproduction_failures_percent_difference = Column(Float)

    system_reproduction_success_rate_high_switching_mean = Column(Float)
    system_reproduction_success_rate_low_switching_mean = Column(Float)
    system_reproduction_success_rate_difference = Column(Float)
    system_reproduction_success_rate_percent_difference = Column(Float)

    system_first_reproduction_time_high_switching_mean = Column(Float)
    system_first_reproduction_time_low_switching_mean = Column(Float)
    system_first_reproduction_time_difference = Column(Float)
    system_first_reproduction_time_percent_difference = Column(Float)

    system_reproduction_efficiency_high_switching_mean = Column(Float)
    system_reproduction_efficiency_low_switching_mean = Column(Float)
    system_reproduction_efficiency_difference = Column(Float)
    system_reproduction_efficiency_percent_difference = Column(Float)

    system_avg_resources_per_reproduction_high_switching_mean = Column(Float)
    system_avg_resources_per_reproduction_low_switching_mean = Column(Float)
    system_avg_resources_per_reproduction_difference = Column(Float)
    system_avg_resources_per_reproduction_percent_difference = Column(Float)

    # Independent agent reproduction comparisons
    independent_reproduction_attempts_high_switching_mean = Column(Float)
    independent_reproduction_attempts_low_switching_mean = Column(Float)
    independent_reproduction_attempts_difference = Column(Float)
    independent_reproduction_attempts_percent_difference = Column(Float)

    independent_reproduction_successes_high_switching_mean = Column(Float)
    independent_reproduction_successes_low_switching_mean = Column(Float)
    independent_reproduction_successes_difference = Column(Float)
    independent_reproduction_successes_percent_difference = Column(Float)

    independent_reproduction_failures_high_switching_mean = Column(Float)
    independent_reproduction_failures_low_switching_mean = Column(Float)
    independent_reproduction_failures_difference = Column(Float)
    independent_reproduction_failures_percent_difference = Column(Float)

    independent_reproduction_success_rate_high_switching_mean = Column(Float)
    independent_reproduction_success_rate_low_switching_mean = Column(Float)
    independent_reproduction_success_rate_difference = Column(Float)
    independent_reproduction_success_rate_percent_difference = Column(Float)

    independent_first_reproduction_time_high_switching_mean = Column(Float)
    independent_first_reproduction_time_low_switching_mean = Column(Float)
    independent_first_reproduction_time_difference = Column(Float)
    independent_first_reproduction_time_percent_difference = Column(Float)

    independent_reproduction_efficiency_high_switching_mean = Column(Float)
    independent_reproduction_efficiency_low_switching_mean = Column(Float)
    independent_reproduction_efficiency_difference = Column(Float)
    independent_reproduction_efficiency_percent_difference = Column(Float)

    independent_avg_resources_per_reproduction_high_switching_mean = Column(Float)
    independent_avg_resources_per_reproduction_low_switching_mean = Column(Float)
    independent_avg_resources_per_reproduction_difference = Column(Float)
    independent_avg_resources_per_reproduction_percent_difference = Column(Float)

    # Control agent reproduction comparisons
    control_reproduction_attempts_high_switching_mean = Column(Float)
    control_reproduction_attempts_low_switching_mean = Column(Float)
    control_reproduction_attempts_difference = Column(Float)
    control_reproduction_attempts_percent_difference = Column(Float)

    control_reproduction_successes_high_switching_mean = Column(Float)
    control_reproduction_successes_low_switching_mean = Column(Float)
    control_reproduction_successes_difference = Column(Float)
    control_reproduction_successes_percent_difference = Column(Float)

    control_reproduction_failures_high_switching_mean = Column(Float)
    control_reproduction_failures_low_switching_mean = Column(Float)
    control_reproduction_failures_difference = Column(Float)
    control_reproduction_failures_percent_difference = Column(Float)

    control_reproduction_success_rate_high_switching_mean = Column(Float)
    control_reproduction_success_rate_low_switching_mean = Column(Float)
    control_reproduction_success_rate_difference = Column(Float)
    control_reproduction_success_rate_percent_difference = Column(Float)

    control_first_reproduction_time_high_switching_mean = Column(Float)
    control_first_reproduction_time_low_switching_mean = Column(Float)
    control_first_reproduction_time_difference = Column(Float)
    control_first_reproduction_time_percent_difference = Column(Float)

    control_reproduction_efficiency_high_switching_mean = Column(Float)
    control_reproduction_efficiency_low_switching_mean = Column(Float)
    control_reproduction_efficiency_difference = Column(Float)
    control_reproduction_efficiency_percent_difference = Column(Float)

    control_avg_resources_per_reproduction_high_switching_mean = Column(Float)
    control_avg_resources_per_reproduction_low_switching_mean = Column(Float)
    control_avg_resources_per_reproduction_difference = Column(Float)
    control_avg_resources_per_reproduction_percent_difference = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="high_low_switching")


class CorrelationAnalysis(Base):
    """Correlation analysis metrics"""

    __tablename__ = "correlation_analysis"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Reproduction correlations
    repro_corr_system_reproduction_success_rate = Column(Float)
    repro_corr_independent_avg_resources_per_reproduction = Column(Float)
    repro_corr_independent_reproduction_success_rate = Column(Float)
    repro_corr_independent_reproduction_failures = Column(Float)
    repro_corr_independent_reproduction_attempts = Column(Float)

    # Timing correlations
    first_reproduction_timing_correlation_system_first_reproduction_time = Column(Float)
    first_reproduction_timing_correlation_independent_first_reproduction_time = Column(
        Float
    )
    first_reproduction_timing_correlation_control_first_reproduction_time = Column(
        Float
    )

    # Efficiency correlations
    reproduction_efficiency_stability_correlation_system_reproduction_efficiency = (
        Column(Float)
    )
    reproduction_efficiency_stability_correlation_independent_reproduction_efficiency = Column(
        Float
    )
    reproduction_efficiency_stability_correlation_control_reproduction_efficiency = (
        Column(Float)
    )

    # Advantage correlations
    reproduction_advantage_stability_correlation_independent_vs_control_reproduction_efficiency_advantage = Column(
        Float
    )
    reproduction_advantage_stability_correlation_independent_vs_control_reproduction_rate_advantage = Column(
        Float
    )
    reproduction_advantage_stability_correlation_system_vs_independent_reproduction_rate_advantage = Column(
        Float
    )
    reproduction_advantage_stability_correlation_system_vs_control_reproduction_rate_advantage = Column(
        Float
    )
    reproduction_advantage_stability_correlation_system_vs_independent_reproduction_efficiency_advantage = Column(
        Float
    )
    reproduction_advantage_stability_correlation_system_vs_control_reproduction_efficiency_advantage = Column(
        Float
    )

    # System dominance correlations
    system_dominance_reproduction_correlations_system_reproduction_attempts = Column(
        Float
    )
    system_dominance_reproduction_correlations_system_reproduction_successes = Column(
        Float
    )
    system_dominance_reproduction_correlations_system_reproduction_failures = Column(
        Float
    )
    system_dominance_reproduction_correlations_system_reproduction_success_rate = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_system_first_reproduction_time = Column(
        Float
    )
    system_dominance_reproduction_correlations_independent_reproduction_attempts = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_independent_reproduction_successes = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_independent_reproduction_failures = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_independent_reproduction_success_rate = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_independent_first_reproduction_time = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_control_reproduction_attempts = Column(
        Float
    )
    system_dominance_reproduction_correlations_control_reproduction_successes = Column(
        Float
    )
    system_dominance_reproduction_correlations_control_reproduction_failures = Column(
        Float
    )
    system_dominance_reproduction_correlations_control_reproduction_success_rate = (
        Column(Float)
    )
    system_dominance_reproduction_correlations_control_first_reproduction_time = Column(
        Float
    )

    # Control dominance correlations
    control_dominance_reproduction_correlations_system_reproduction_attempts = Column(
        Float
    )
    control_dominance_reproduction_correlations_system_reproduction_successes = Column(
        Float
    )
    control_dominance_reproduction_correlations_system_reproduction_failures = Column(
        Float
    )
    control_dominance_reproduction_correlations_system_reproduction_success_rate = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_system_first_reproduction_time = Column(
        Float
    )
    control_dominance_reproduction_correlations_independent_reproduction_attempts = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_independent_reproduction_successes = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_independent_reproduction_failures = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_independent_reproduction_success_rate = Column(
        Float
    )
    control_dominance_reproduction_correlations_independent_first_reproduction_time = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_control_reproduction_attempts = Column(
        Float
    )
    control_dominance_reproduction_correlations_control_reproduction_successes = Column(
        Float
    )
    control_dominance_reproduction_correlations_control_reproduction_failures = Column(
        Float
    )
    control_dominance_reproduction_correlations_control_reproduction_success_rate = (
        Column(Float)
    )
    control_dominance_reproduction_correlations_control_first_reproduction_time = (
        Column(Float)
    )

    # Relationship
    simulation = relationship("Simulation", back_populates="correlation_analysis")


# Database initialization function
def init_db(db_path="sqlite:///dominance.db"):
    """Initialize the database and create tables"""
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


# Session creation function
def get_session(engine):
    """Create a session for database operations"""
    Session = sessionmaker(bind=engine)
    return Session()
