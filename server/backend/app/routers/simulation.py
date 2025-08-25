import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from .. import models
from ..db import get_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationRequest(BaseModel):
    """
    Request model for starting a new simulation.

    Attributes:
        name (str): Name of the simulation (1-100 characters)
        steps (int, optional): Number of simulation steps (1-1,000,000). Defaults to 1000
        config (dict, optional): Additional configuration parameters. Defaults to empty dict
    """

    name: str = Field(..., min_length=1, max_length=100)
    steps: Optional[int] = Field(
        default=1000, gt=0, le=1000000
    )  # More specific constraints
    config: Optional[dict] = Field(default_factory=dict)

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty or just whitespace")
        return v.strip()


class SimulationHistory(BaseModel):
    id: int
    name: str
    result: float
    timestamp: datetime

    class Config:
        orm_mode = True


class SimulationResponse(BaseModel):
    status: str
    name: str
    id: int
    total_steps: int
    message: Optional[str] = None


class SimulationStatusResponse(BaseModel):
    id: int
    running: bool
    steps: int
    total_steps: int
    start_time: datetime
    progress: float  # Add calculated progress


# Store active simulations
active_simulations = {}

# Create a lock for thread-safe operations
simulation_lock = threading.Lock()

# Rate limiting
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100
request_history = {}


def check_rate_limit(client_id: str = Depends(APIKeyHeader(name="X-Client-ID"))):
    """
    Check if the client has exceeded the rate limit.

    Args:
        client_id (str): Client identifier from X-Client-ID header

    Raises:
        HTTPException: If rate limit is exceeded (429)

    Returns:
        str: The client_id if rate limit check passes
    """
    now = time.time()
    if client_id in request_history:
        requests = [
            t for t in request_history[client_id] if now - t < RATE_LIMIT_DURATION
        ]
        if len(requests) >= MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        request_history[client_id] = requests + [now]
    else:
        request_history[client_id] = [now]
    return client_id


router = APIRouter(prefix="/simulation", tags=["simulation"])


def run_simulation_steps(name: str, db: Session = Depends(get_db)):
    """
    Background task to update simulation progress.

    Runs in a separate thread and updates the simulation state until completion
    or interruption. Updates progress in memory and final result in database.

    Args:
        name (str): Name of the simulation to run
        db (Session): Database session for storing results
    """
    try:
        while active_simulations.get(name, {}).get("running", False):
            with simulation_lock:
                sim_info = active_simulations[name]
                if sim_info["steps"] < sim_info["total_steps"]:
                    sim_info["steps"] += 1
                    sim_info["progress"] = (
                        sim_info["steps"] / sim_info["total_steps"]
                    ) * 100
                    logger.info(
                        f"Simulation {name} progress: {sim_info['progress']:.2f}%"
                    )
                else:
                    break
            time.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in simulation {name}: {str(e)}")
    finally:
        with simulation_lock:
            if name in active_simulations:
                sim_info = active_simulations[name]
                sim_info["running"] = False
                # Update final result in database
                try:
                    db = next(get_db())
                    result = (
                        db.query(models.SimulationResult)
                        .filter(models.SimulationResult.id == sim_info["id"])
                        .first()
                    )
                    if result:
                        result.result = sim_info["progress"]
                        db.commit()
                    db.close()
                except Exception as e:
                    logger.error(f"Failed to update final result: {str(e)}")
                logger.info(f"Simulation {name} completed")


@router.post(
    "/start",
    response_model=SimulationResponse,
    dependencies=[Depends(check_rate_limit)],
)
def start_simulation(request: SimulationRequest, db: Session = Depends(get_db)):
    """
    Start a new simulation.

    Creates a new simulation record and starts a background thread to run it.
    Rate-limited endpoint.

    Args:
        request (SimulationRequest): Parameters for the new simulation
        db (Session): Database session

    Returns:
        SimulationResponse: Details of the started simulation

    Raises:
        HTTPException: If simulation already exists (400) or database error (500)
    """
    logger.info(f"Starting simulation with name: {request.name}")

    with simulation_lock:
        if request.name in active_simulations:
            if active_simulations[request.name]["running"]:
                raise HTTPException(
                    status_code=400, detail="Simulation already running"
                )
            else:
                # Clean up completed simulation
                del active_simulations[request.name]

    # Create simulation record
    db_result = models.SimulationResult(
        name=request.name, result=0.0, timestamp=datetime.now()  # Initial result
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
                "start_time": datetime.now(),
            }

        # Start background simulation thread
        thread = threading.Thread(
            target=run_simulation_steps, args=(request.name, db), daemon=True
        )
        thread.start()

        logger.info(f"Simulation {request.name} started successfully")
        return {
            "status": "started",
            "name": request.name,
            "id": db_result.id,
            "total_steps": request.steps,
        }

    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start simulation")


@router.post("/{name}/stop")
def stop_simulation(name: str):
    """
    Stop a running simulation.

    Args:
        name (str): Name of the simulation to stop

    Returns:
        dict: Status information including:
            - status: "stopped"
            - name: simulation name
            - steps_completed: number of completed steps
            - message: optional status message
    """
    logger.info(f"Attempting to stop simulation {name}")

    with simulation_lock:
        if name not in active_simulations:
            # Instead of 404, return success if simulation is already gone
            return {
                "status": "stopped",
                "name": name,
                "steps_completed": 0,
                "message": "Simulation already completed",
            }

        sim_info = active_simulations[name]
        if not sim_info["running"]:
            # Instead of 400, return success with current state
            return {
                "status": "stopped",
                "name": name,
                "steps_completed": sim_info["steps"],
                "message": "Simulation already stopped",
            }

        sim_info["running"] = False
        steps_completed = sim_info["steps"]
        logger.info(f"Simulation {name} stopped")

        # Clean up completed simulation
        del active_simulations[name]

    return {"status": "stopped", "name": name, "steps_completed": steps_completed}


@router.get("/status/{name}")
def get_simulation_status(name: str):
    """
    Get the current status of a simulation.

    Args:
        name (str): Name of the simulation

    Returns:
        dict: Current simulation state including progress and running status

    Raises:
        HTTPException: If simulation not found (404)
    """
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
def get_simulation_history(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
):
    """
    Get paginated history of all simulations.

    Args:
        db (Session): Database session
        skip (int): Number of records to skip for pagination
        limit (int): Maximum number of records to return (1-100)

    Returns:
        dict: Paginated results containing:
            - total: Total number of records
            - items: List of SimulationHistory objects
            - page: Current page number
            - pages: Total number of pages
    """
    total = db.query(models.SimulationResult).count()
    results = (
        db.query(models.SimulationResult)
        .order_by(models.SimulationResult.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "items": [SimulationHistory.from_orm(result) for result in results],
        "page": skip // limit + 1,
        "pages": (total + limit - 1) // limit,
    }


@router.get("/history/{simulation_id}", response_model=SimulationHistory)
def get_simulation_details(simulation_id: int, db: Session = Depends(get_db)):
    """Get details of a specific simulation."""
    result = (
        db.query(models.SimulationResult)
        .filter(models.SimulationResult.id == simulation_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulationHistory.from_orm(result)


@router.post("/load/{simulation_id}")
def load_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Load a previous simulation."""
    result = (
        db.query(models.SimulationResult)
        .filter(models.SimulationResult.id == simulation_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Start a new simulation with the same name
    sim_request = SimulationRequest(name=f"{result.name}_loaded")
    return start_simulation(sim_request, db)


@router.delete("/history/{simulation_id}")
def delete_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Delete a simulation from history."""
    result = (
        db.query(models.SimulationResult)
        .filter(models.SimulationResult.id == simulation_id)
        .first()
    )

    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")

    try:
        db.delete(result)
        db.commit()
        return {"status": "deleted", "id": simulation_id}
    except Exception as e:
        logger.error(f"Failed to delete simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete simulation")


def cleanup_stale_simulations():
    """
    Clean up simulations that haven't been updated in the last hour.

    Runs periodically to prevent memory leaks from abandoned simulations.
    Marks stale simulations as not running and removes them from active tracking.
    """
    stale_threshold = datetime.now() - timedelta(hours=1)
    with simulation_lock:
        stale_sims = [
            name
            for name, info in active_simulations.items()
            if info["start_time"] < stale_threshold
        ]
        for name in stale_sims:
            logger.warning(f"Cleaning up stale simulation: {name}")
            active_simulations[name]["running"] = False
            del active_simulations[name]


# Add periodic cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_stale_simulations, "interval", minutes=15)
scheduler.start()
