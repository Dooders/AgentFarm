from sqlalchemy import Column, Integer, String, Float, DateTime
from .db import Base

class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    result = Column(Float)
    timestamp = Column(DateTime)
