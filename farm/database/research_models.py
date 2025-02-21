"""Models for storing research and experiment analysis data."""

from datetime import datetime
from typing import Dict, List

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from farm.database.models import Base


class Research(Base):
    """Model for storing research project data."""

    __tablename__ = "research"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(String(1000))
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )
    parameters = Column(JSON)  # Global research parameters/configuration

    # Relationships
    experiments = relationship(
        "ExperimentStats", back_populates="research", cascade="all, delete-orphan"
    )


class ExperimentStats(Base):
    """Model for storing experiment-level statistics within a research project."""

    __tablename__ = "experiment_stats"

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey("research.id"), nullable=False)
    experiment_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    description = Column(String(1000))
    parameters = Column(JSON)  # Experiment-specific parameters
    num_iterations = Column(Integer, nullable=False)

    # Population metrics
    mean_population = Column(Float, nullable=False)
    std_population = Column(Float, nullable=False)
    max_population = Column(Float, nullable=False)
    min_population = Column(Float, nullable=False)

    # Resource metrics
    mean_resources = Column(Float, nullable=False)
    std_resources = Column(Float, nullable=False)
    mean_efficiency = Column(Float, nullable=False)
    std_efficiency = Column(Float, nullable=False)

    # Reproduction metrics
    mean_success_rate = Column(Float, nullable=False)
    std_success_rate = Column(Float, nullable=False)
    total_reproduction_attempts = Column(Integer, nullable=False)
    total_successful_reproductions = Column(Integer, nullable=False)

    # Relationships
    research = relationship("Research", back_populates="experiments")
    iterations = relationship(
        "IterationStats", back_populates="experiment", cascade="all, delete-orphan"
    )


class IterationStats(Base):
    """Statistics for individual experiment iterations."""

    __tablename__ = "iteration_stats"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiment_stats.id"), nullable=False)
    iteration_id = Column(String(255), nullable=False)

    # Population stats
    avg_population = Column(Float, nullable=False)
    max_population = Column(Integer, nullable=False)
    min_population = Column(Integer, nullable=False)

    # Resource stats
    avg_resources = Column(Float, nullable=False)
    resource_efficiency = Column(Float, nullable=False)

    # Reproduction stats
    reproduction_attempts = Column(Integer, nullable=False)
    successful_reproductions = Column(Integer, nullable=False)
    reproduction_rate = Column(Float, nullable=False)

    # Relationship
    experiment = relationship("ExperimentStats", back_populates="iterations")
