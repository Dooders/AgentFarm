from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models
from ..db import get_db
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
import asyncio
from typing import Optional, List
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationRequest(BaseModel):
    name: str
    steps: Optional[int] = 1000  # Default to 1000 steps
    config: Optional[dict] = None  # Optional configuration parameters

class SimulationHistory(BaseModel):
    id: int
    name: str
    result: float
    timestamp: datetime

    class Config:
        orm_mode = True

# Store active simulations
active_simulations = {}

# Create a lock for thread-safe operations
simulation_lock = threading.Lock()

router = APIRouter(
    prefix="/simulation",
    tags=["simulation"]
)

def run_simulation_steps(name: str, db: Session = Depends(get_db)):
    """Background task to update simulation progress."""
    while active_simulations.get(name, {}).get("running", False):
        with simulation_lock:
            sim_info = active_simulations[name]
            if sim_info["steps"] < sim_info["total_steps"]:
                sim_info["steps"] += 1
                logger.info(f"Simulation {name} step {sim_info['steps']}/{sim_info['total_steps']}")
            else:
                sim_info["running"] = False
                # Update simulation result in database
                result = db.query(models.SimulationResult).filter(
                    models.SimulationResult.id == sim_info["id"]
                ).first()
                if result:
                    result.result = sim_info["steps"] / sim_info["total_steps"] * 100
                    db.commit()
                logger.info(f"Simulation {name} completed")
                break
        time.sleep(0.1)  # Delay between steps

@router.post("/start")
def start_simulation(request: SimulationRequest, db: Session = Depends(get_db)):
    """Start a new simulation."""
    logger.info(f"Starting simulation with name: {request.name}")
    
    with simulation_lock:
        if request.name in active_simulations:
            if active_simulations[request.name]["running"]:
                raise HTTPException(status_code=400, detail="Simulation already running")
            else:
                # Clean up completed simulation
                del active_simulations[request.name]
    
    # Create simulation record
    db_result = models.SimulationResult(
        name=request.name,
        result=0.0,  # Initial result
        timestamp=datetime.now()
    )
    
    try:
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        
        # Store simulation info
        with simulation_lock:
            active_simulations[request.name] = {
                "id": db_result.id,
                "running": True,
                "steps": 0,
                "total_steps": request.steps,
                "start_time": datetime.now()
            }
        
        # Start background simulation thread
        thread = threading.Thread(
            target=run_simulation_steps,
            args=(request.name, db),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Simulation {request.name} started successfully")
        return {
            "status": "started",
            "name": request.name,
            "id": db_result.id,
            "total_steps": request.steps
        }
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start simulation")

@router.post("/{name}/stop")
def stop_simulation(name: str):
    """Stop a running simulation."""
    logger.info(f"Attempting to stop simulation {name}")
    
    with simulation_lock:
        if name not in active_simulations:
            # Instead of 404, return success if simulation is already gone
            return {
                "status": "stopped",
                "name": name,
                "steps_completed": 0,
                "message": "Simulation already completed"
            }
        
        sim_info = active_simulations[name]
        if not sim_info["running"]:
            # Instead of 400, return success with current state
            return {
                "status": "stopped",
                "name": name,
                "steps_completed": sim_info["steps"],
                "message": "Simulation already stopped"
            }
        
        sim_info["running"] = False
        steps_completed = sim_info["steps"]
        logger.info(f"Simulation {name} stopped")
        
        # Clean up completed simulation
        del active_simulations[name]
    
    return {
        "status": "stopped",
        "name": name,
        "steps_completed": steps_completed
    }

@router.get("/status/{name}")
def get_simulation_status(name: str):
    """Get the status of a simulation."""
    with simulation_lock:
        if name not in active_simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
        return active_simulations[name]

@router.get("/list")
def list_simulations():
    """List all simulations."""
    with simulation_lock:
        return active_simulations

@router.get("/history", response_model=List[SimulationHistory])
def get_simulation_history(db: Session = Depends(get_db)):
    """Get history of all simulations."""
    results = db.query(models.SimulationResult).order_by(
        models.SimulationResult.timestamp.desc()
    ).all()
    # Convert SQLAlchemy models to Pydantic models
    return [SimulationHistory.from_orm(result) for result in results]

@router.get("/history/{simulation_id}", response_model=SimulationHistory)
def get_simulation_details(simulation_id: int, db: Session = Depends(get_db)):
    """Get details of a specific simulation."""
    result = db.query(models.SimulationResult).filter(
        models.SimulationResult.id == simulation_id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulationHistory.from_orm(result)

@router.post("/load/{simulation_id}")
def load_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Load a previous simulation."""
    result = db.query(models.SimulationResult).filter(
        models.SimulationResult.id == simulation_id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Start a new simulation with the same name
    sim_request = SimulationRequest(name=f"{result.name}_loaded")
    return start_simulation(sim_request, db)

@router.delete("/history/{simulation_id}")
def delete_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Delete a simulation from history."""
    result = db.query(models.SimulationResult).filter(
        models.SimulationResult.id == simulation_id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        db.delete(result)
        db.commit()
        return {"status": "deleted", "id": simulation_id}
    except Exception as e:
        logger.error(f"Failed to delete simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete simulation")