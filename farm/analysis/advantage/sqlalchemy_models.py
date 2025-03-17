from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Simulation(Base):
    """Base simulation information"""

    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True)
    iteration = Column(Integer, nullable=False, unique=True, index=True)

    # Relationships
    resource_acquisition = relationship(
        "ResourceAcquisition", back_populates="simulation", uselist=False
    )
    reproduction_advantage = relationship(
        "ReproductionAdvantage", back_populates="simulation", uselist=False
    )
    survival_advantage = relationship(
        "SurvivalAdvantage", back_populates="simulation", uselist=False
    )
    population_growth = relationship(
        "PopulationGrowth", back_populates="simulation", uselist=False
    )
    combat_advantage = relationship(
        "CombatAdvantage", back_populates="simulation", uselist=False
    )
    initial_positioning = relationship(
        "InitialPositioning", back_populates="simulation", uselist=False
    )
    composite_advantage = relationship(
        "CompositeAdvantage", back_populates="simulation", uselist=False
    )
    advantage_dominance_correlation = relationship(
        "AdvantageDominanceCorrelation", back_populates="simulation", uselist=False
    )


class ResourceAcquisition(Base):
    """Resource acquisition advantages between agent types"""

    __tablename__ = "resource_acquisition"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw resource metrics for each agent type
    system_early_phase = Column(Float)
    system_mid_phase = Column(Float)
    system_late_phase = Column(Float)

    independent_early_phase = Column(Float)
    independent_mid_phase = Column(Float)
    independent_late_phase = Column(Float)

    control_early_phase = Column(Float)
    control_mid_phase = Column(Float)
    control_late_phase = Column(Float)

    # System vs Independent advantages
    system_vs_independent_early_phase_advantage = Column(Float)
    system_vs_independent_mid_phase_advantage = Column(Float)
    system_vs_independent_late_phase_advantage = Column(Float)
    system_vs_independent_advantage_trajectory = Column(Float)

    # System vs Control advantages
    system_vs_control_early_phase_advantage = Column(Float)
    system_vs_control_mid_phase_advantage = Column(Float)
    system_vs_control_late_phase_advantage = Column(Float)
    system_vs_control_advantage_trajectory = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_early_phase_advantage = Column(Float)
    independent_vs_control_mid_phase_advantage = Column(Float)
    independent_vs_control_late_phase_advantage = Column(Float)
    independent_vs_control_advantage_trajectory = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="resource_acquisition")


class ReproductionAdvantage(Base):
    """Reproduction advantages between agent types"""

    __tablename__ = "reproduction_advantage"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw reproduction metrics for each agent type
    system_success_rate = Column(Float)
    system_total_offspring = Column(Integer)
    system_reproduction_efficiency = Column(Float)
    system_first_reproduction_time = Column(Float)

    independent_success_rate = Column(Float)
    independent_total_offspring = Column(Integer)
    independent_reproduction_efficiency = Column(Float)
    independent_first_reproduction_time = Column(Float)

    control_success_rate = Column(Float)
    control_total_offspring = Column(Integer)
    control_reproduction_efficiency = Column(Float)
    control_first_reproduction_time = Column(Float)

    # System vs Independent advantages
    system_vs_independent_success_rate_advantage = Column(Float)
    system_vs_independent_offspring_advantage = Column(Float)
    system_vs_independent_efficiency_advantage = Column(Float)
    system_vs_independent_timing_advantage = Column(Float)

    # System vs Control advantages
    system_vs_control_success_rate_advantage = Column(Float)
    system_vs_control_offspring_advantage = Column(Float)
    system_vs_control_efficiency_advantage = Column(Float)
    system_vs_control_timing_advantage = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_success_rate_advantage = Column(Float)
    independent_vs_control_offspring_advantage = Column(Float)
    independent_vs_control_efficiency_advantage = Column(Float)
    independent_vs_control_timing_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="reproduction_advantage")


class SurvivalAdvantage(Base):
    """Survival advantages between agent types"""

    __tablename__ = "survival_advantage"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw survival metrics for each agent type
    system_survival_rate = Column(Float)
    system_avg_lifespan = Column(Float)
    system_death_rate = Column(Float)

    independent_survival_rate = Column(Float)
    independent_avg_lifespan = Column(Float)
    independent_death_rate = Column(Float)

    control_survival_rate = Column(Float)
    control_avg_lifespan = Column(Float)
    control_death_rate = Column(Float)

    # System vs Independent advantages
    system_vs_independent_survival_rate_advantage = Column(Float)
    system_vs_independent_lifespan_advantage = Column(Float)
    system_vs_independent_death_rate_advantage = Column(Float)

    # System vs Control advantages
    system_vs_control_survival_rate_advantage = Column(Float)
    system_vs_control_lifespan_advantage = Column(Float)
    system_vs_control_death_rate_advantage = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_survival_rate_advantage = Column(Float)
    independent_vs_control_lifespan_advantage = Column(Float)
    independent_vs_control_death_rate_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="survival_advantage")


class PopulationGrowth(Base):
    """Population growth advantages between agent types"""

    __tablename__ = "population_growth"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw population metrics for each agent type
    system_early_growth_rate = Column(Float)
    system_mid_growth_rate = Column(Float)
    system_late_growth_rate = Column(Float)
    system_final_population = Column(Integer)

    independent_early_growth_rate = Column(Float)
    independent_mid_growth_rate = Column(Float)
    independent_late_growth_rate = Column(Float)
    independent_final_population = Column(Integer)

    control_early_growth_rate = Column(Float)
    control_mid_growth_rate = Column(Float)
    control_late_growth_rate = Column(Float)
    control_final_population = Column(Integer)

    # System vs Independent advantages
    system_vs_independent_early_growth_advantage = Column(Float)
    system_vs_independent_mid_growth_advantage = Column(Float)
    system_vs_independent_late_growth_advantage = Column(Float)
    system_vs_independent_final_population_advantage = Column(Float)

    # System vs Control advantages
    system_vs_control_early_growth_advantage = Column(Float)
    system_vs_control_mid_growth_advantage = Column(Float)
    system_vs_control_late_growth_advantage = Column(Float)
    system_vs_control_final_population_advantage = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_early_growth_advantage = Column(Float)
    independent_vs_control_mid_growth_advantage = Column(Float)
    independent_vs_control_late_growth_advantage = Column(Float)
    independent_vs_control_final_population_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="population_growth")


class CombatAdvantage(Base):
    """Combat advantages between agent types"""

    __tablename__ = "combat_advantage"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw combat metrics for each agent type
    system_win_rate = Column(Float)
    system_damage_dealt = Column(Float)
    system_damage_received = Column(Float)

    independent_win_rate = Column(Float)
    independent_damage_dealt = Column(Float)
    independent_damage_received = Column(Float)

    control_win_rate = Column(Float)
    control_damage_dealt = Column(Float)
    control_damage_received = Column(Float)

    # System vs Independent advantages
    system_vs_independent_win_rate_advantage = Column(Float)
    system_vs_independent_damage_advantage = Column(Float)

    # System vs Control advantages
    system_vs_control_win_rate_advantage = Column(Float)
    system_vs_control_damage_advantage = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_win_rate_advantage = Column(Float)
    independent_vs_control_damage_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="combat_advantage")


class InitialPositioning(Base):
    """Initial positioning advantages between agent types"""

    __tablename__ = "initial_positioning"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Raw positioning metrics for each agent type
    system_avg_resource_distance = Column(Float)
    system_nearest_resource_distance = Column(Float)
    system_resources_in_range = Column(Float)

    independent_avg_resource_distance = Column(Float)
    independent_nearest_resource_distance = Column(Float)
    independent_resources_in_range = Column(Float)

    control_avg_resource_distance = Column(Float)
    control_nearest_resource_distance = Column(Float)
    control_resources_in_range = Column(Float)

    # System vs Independent advantages
    system_vs_independent_avg_distance_advantage = Column(Float)
    system_vs_independent_nearest_distance_advantage = Column(Float)
    system_vs_independent_resources_in_range_advantage = Column(Float)

    # System vs Control advantages
    system_vs_control_avg_distance_advantage = Column(Float)
    system_vs_control_nearest_distance_advantage = Column(Float)
    system_vs_control_resources_in_range_advantage = Column(Float)

    # Independent vs Control advantages
    independent_vs_control_avg_distance_advantage = Column(Float)
    independent_vs_control_nearest_distance_advantage = Column(Float)
    independent_vs_control_resources_in_range_advantage = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="initial_positioning")


class CompositeAdvantage(Base):
    """Composite advantages between agent types"""

    __tablename__ = "composite_advantage"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # System vs Independent composite advantage
    system_vs_independent_score = Column(Float)
    system_vs_independent_resource_component = Column(Float)
    system_vs_independent_reproduction_component = Column(Float)
    system_vs_independent_survival_component = Column(Float)
    system_vs_independent_population_component = Column(Float)
    system_vs_independent_combat_component = Column(Float)
    system_vs_independent_positioning_component = Column(Float)

    # System vs Control composite advantage
    system_vs_control_score = Column(Float)
    system_vs_control_resource_component = Column(Float)
    system_vs_control_reproduction_component = Column(Float)
    system_vs_control_survival_component = Column(Float)
    system_vs_control_population_component = Column(Float)
    system_vs_control_combat_component = Column(Float)
    system_vs_control_positioning_component = Column(Float)

    # Independent vs Control composite advantage
    independent_vs_control_score = Column(Float)
    independent_vs_control_resource_component = Column(Float)
    independent_vs_control_reproduction_component = Column(Float)
    independent_vs_control_survival_component = Column(Float)
    independent_vs_control_population_component = Column(Float)
    independent_vs_control_combat_component = Column(Float)
    independent_vs_control_positioning_component = Column(Float)

    # Relationship
    simulation = relationship("Simulation", back_populates="composite_advantage")


class AdvantageDominanceCorrelation(Base):
    """Correlations between advantages and dominance outcomes"""

    __tablename__ = "advantage_dominance_correlation"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(
        Integer, ForeignKey("simulations.id"), nullable=False, unique=True
    )

    # Dominant type in the simulation
    dominant_type = Column(String)

    # Resource acquisition correlations
    resource_early_phase_correlation = Column(Float)
    resource_mid_phase_correlation = Column(Float)
    resource_late_phase_correlation = Column(Float)
    resource_trajectory_correlation = Column(Float)

    # Reproduction correlations
    reproduction_success_rate_correlation = Column(Float)
    reproduction_offspring_correlation = Column(Float)
    reproduction_efficiency_correlation = Column(Float)
    reproduction_timing_correlation = Column(Float)

    # Survival correlations
    survival_rate_correlation = Column(Float)
    lifespan_correlation = Column(Float)
    death_rate_correlation = Column(Float)

    # Population growth correlations
    early_growth_correlation = Column(Float)
    mid_growth_correlation = Column(Float)
    late_growth_correlation = Column(Float)
    final_population_correlation = Column(Float)

    # Combat correlations
    win_rate_correlation = Column(Float)
    damage_correlation = Column(Float)

    # Initial positioning correlations
    avg_distance_correlation = Column(Float)
    nearest_distance_correlation = Column(Float)
    resources_in_range_correlation = Column(Float)

    # Composite advantage correlation
    composite_advantage_correlation = Column(Float)

    # Relationship
    simulation = relationship(
        "Simulation", back_populates="advantage_dominance_correlation"
    )


# Database initialization function
def init_db(db_path="sqlite:///advantage.db"):
    """Initialize the database and create tables"""
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


# Session creation function
def get_session(engine):
    """Create a session for database operations"""
    Session = sessionmaker(bind=engine)
    return Session()
