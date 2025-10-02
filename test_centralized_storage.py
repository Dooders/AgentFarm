#!/usr/bin/env python3
"""
Simple test of centralized storage without full simulation dependencies.
"""

import os
import sys
from datetime import datetime

# Test imports
try:
    from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON
    from sqlalchemy.orm import declarative_base, sessionmaker
    print("✓ SQLAlchemy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SQLAlchemy: {e}")
    sys.exit(1)

# Create simple test database
print("\n" + "="*70)
print("TESTING CENTRALIZED STORAGE CONCEPT")
print("="*70)

Base = declarative_base()

class TestExperiment(Base):
    """Simple experiment model for testing."""
    __tablename__ = "experiments"
    experiment_id = Column(String(64), primary_key=True)
    name = Column(String(255))
    status = Column(String(50))
    creation_date = Column(DateTime, default=datetime.now)

class TestSimulation(Base):
    """Simple simulation model for testing."""
    __tablename__ = "simulations"
    simulation_id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64))
    status = Column(String(50))
    parameters = Column(JSON)

class TestData(Base):
    """Simple data model for testing."""
    __tablename__ = "data"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(String(64))
    step = Column(Integer)
    value = Column(Integer)

# Create test database
db_path = "experiments/test_centralized.db"
os.makedirs("experiments", exist_ok=True)

engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

print(f"\n✓ Created test database: {db_path}")

# Add experiment
session = Session()

experiment = TestExperiment(
    experiment_id="test_exp_001",
    name="Test Experiment",
    status="running"
)
session.add(experiment)
session.commit()

print("✓ Added experiment: test_exp_001")

# Add multiple simulations to same database
print("\n✓ Adding simulations to centralized database...")
for i in range(5):
    sim_id = f"sim_{i:03d}"
    
    # Create simulation record
    simulation = TestSimulation(
        simulation_id=sim_id,
        experiment_id="test_exp_001",
        status="running",
        parameters={"run": i, "seed": i * 100}
    )
    session.add(simulation)
    
    # Add data for this simulation
    for step in range(10):
        data = TestData(
            simulation_id=sim_id,
            step=step,
            value=i * 100 + step
        )
        session.add(data)
    
    session.commit()
    print(f"  ✓ Added {sim_id} with 10 data points")

# Query data
print("\n" + "="*70)
print("QUERYING CENTRALIZED DATA")
print("="*70)

# Count simulations
sim_count = session.query(TestSimulation).count()
print(f"\n✓ Total simulations: {sim_count}")

# Count data points
data_count = session.query(TestData).count()
print(f"✓ Total data points: {data_count}")

# Show data per simulation
print("\n✓ Data points per simulation:")
for sim_id in [f"sim_{i:03d}" for i in range(5)]:
    count = session.query(TestData).filter(
        TestData.simulation_id == sim_id
    ).count()
    print(f"  {sim_id}: {count} data points")

# Compare final values across simulations
print("\n✓ Comparing final values (step 9) across simulations:")
for sim_id in [f"sim_{i:03d}" for i in range(5)]:
    final = session.query(TestData).filter(
        TestData.simulation_id == sim_id,
        TestData.step == 9
    ).first()
    if final:
        print(f"  {sim_id}: {final.value}")

session.close()

print("\n" + "="*70)
print("SUCCESS! Centralized storage concept verified:")
print("="*70)
print("✓ Single database file stores multiple simulations")
print("✓ Each simulation is tagged with simulation_id")
print("✓ Easy to query data for specific simulations")
print("✓ Easy to compare across simulations")
print(f"\nDatabase file: {db_path}")
print(f"File size: {os.path.getsize(db_path)} bytes")
print("="*70 + "\n")
