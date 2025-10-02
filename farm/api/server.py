import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.config import SimulationConfig
from farm.core.analysis import analyze_simulation
from farm.core.services import EnvConfigService
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from farm.utils.logging_config import configure_logging, get_logger

# Configure structured logging
configure_logging(
    environment="production",
    log_dir="logs",
    log_level="INFO",
    json_logs=True,
)
logger = get_logger(__name__)

app = FastAPI(title="AgentFarm API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class SimulationCreateRequest(BaseModel):
    simulation_steps: Optional[int] = None
    # Add other config fields as needed


class SimulationResponse(BaseModel):
    status: str
    sim_id: Optional[str] = None
    message: Optional[str] = None


class AnalysisRequestModel(BaseModel):
    experiment_path: str = "results"
    output_path: str = "results/analysis"
    group: str = "all"
    processor_kwargs: Optional[Dict[str, Any]] = None
    analysis_kwargs: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    status: str
    output_path: Optional[str] = None
    rows: Optional[int] = None
    message: Optional[str] = None


class SimulationStatus(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


# Store active simulations
active_simulations = {}
_active_simulations_lock = threading.Lock()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        # Iterate over a snapshot to allow mutation during iteration
        for client_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_text(message)
            except Exception:
                # Remove failed/stale connections to avoid resource leaks
                self.disconnect(client_id)


manager = ConnectionManager()


def _run_simulation_background(sim_id, config, db_path):
    try:
        with _active_simulations_lock:
            if sim_id in active_simulations:
                active_simulations[sim_id]["status"] = "running"

        # Run simulation
        run_simulation(
            num_steps=config.simulation_steps,
            config=config,
            path=os.path.dirname(db_path),
        )

        with _active_simulations_lock:
            if sim_id in active_simulations:
                active_simulations[sim_id]["status"] = "completed"
                active_simulations[sim_id]["ended_at"] = datetime.now().isoformat()
    except Exception as e:
        logger.error(
            "background_simulation_failed",
            simulation_id=sim_id,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        with _active_simulations_lock:
            if sim_id in active_simulations:
                active_simulations[sim_id]["status"] = "error"
                active_simulations[sim_id]["error_message"] = str(e)


@app.post("/api/simulation/new", response_model=SimulationResponse)
async def create_simulation(
    request_data: SimulationCreateRequest, background_tasks: BackgroundTasks
):
    """Create a new simulation with provided configuration."""
    try:
        config_data = request_data.dict(exclude_unset=True)

        # Generate unique simulation ID
        sim_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(
            "api_simulation_create_request",
            simulation_id=sim_id,
            config_keys=list(config_data.keys()),
        )
        db_path = f"results/simulation_{sim_id}.db"

        # Load and update config
        base_config = SimulationConfig.from_centralized_config()
        # Apply overrides without dataclasses.replace to avoid unused import
        for key, value in config_data.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
        config = base_config

        # Create database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Store simulation info (pending)
        with _active_simulations_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": config_data,
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Start background task
        background_tasks.add_task(_run_simulation_background, sim_id, config, db_path)

        return SimulationResponse(
            status="accepted", sim_id=sim_id, message="Simulation started"
        )

    except Exception as e:
        logger.error(
            "api_simulation_create_failed",
            simulation_id=sim_id if "sim_id" in locals() else "unknown",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/{sim_id}/step/{step}")
async def get_step(sim_id: str, step: int):
    """Get simulation state for a specific step."""
    try:
        with _active_simulations_lock:
            if sim_id not in active_simulations:
                raise HTTPException(
                    status_code=404, detail=f"Simulation {sim_id} not found"
                )
            db_path = active_simulations[sim_id]["db_path"]

        db = SimulationDatabase(db_path)
        data = db.query.gui_repository.get_simulation_data(step)

        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_get_step_failed",
            simulation_id=sim_id,
            step=step,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/{sim_id}/analysis")
async def get_analysis(sim_id: str):
    """Get detailed simulation analysis."""
    try:
        with _active_simulations_lock:
            if sim_id not in active_simulations:
                raise HTTPException(
                    status_code=404, detail=f"Simulation {sim_id} not found"
                )
            db_path = active_simulations[sim_id]["db_path"]

        db = SimulationDatabase(db_path)
        analysis_results = analyze_simulation(db)

        return {"status": "success", "data": analysis_results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_analysis_failed",
            simulation_id=sim_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/{module_name}", response_model=AnalysisResponse)
async def run_analysis_module(module_name: str, request_data: AnalysisRequestModel):
    try:
        service = AnalysisService(config_service=EnvConfigService())
        req = AnalysisRequest(
            module_name=module_name,
            experiment_path=request_data.experiment_path,
            output_path=request_data.output_path,
            group=request_data.group,
            processor_kwargs=request_data.processor_kwargs,
            analysis_kwargs=request_data.analysis_kwargs,
        )
        result = service.run(req)
        return AnalysisResponse(
            status="success",
            output_path=str(result.output_path),
            rows=int(result.dataframe.shape[0]) if result.dataframe is not None else 0,
        )
    except Exception as e:
        logger.error(
            "api_analysis_module_failed",
            module_name=module_name,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulations")
async def list_simulations():
    """Get list of active simulations."""
    with _active_simulations_lock:
        data = dict(active_simulations)
    return {"status": "success", "data": data}


@app.get("/api/simulation/{sim_id}/export")
async def export_simulation(sim_id: str):
    """Export simulation data."""
    try:
        with _active_simulations_lock:
            if sim_id not in active_simulations:
                raise HTTPException(
                    status_code=404, detail=f"Simulation {sim_id} not found"
                )
            db_path = active_simulations[sim_id]["db_path"]

        db = SimulationDatabase(db_path)
        export_path = f"results/export_{sim_id}.csv"
        db.export_data(export_path)

        return {
            "status": "success",
            "path": export_path,
            "message": "Data exported successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_export_failed",
            simulation_id=sim_id,
            export_path=export_path if "export_path" in locals() else None,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    logger.info("websocket_client_connected", client_id=client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Parse the message (assuming JSON format)
            import json

            try:
                message = json.loads(data)
                if message.get("type") == "subscribe_simulation":
                    sim_id = message.get("sim_id")
                    with _active_simulations_lock:
                        exists = sim_id in active_simulations

                    if exists:
                        logger.info(
                            "client_subscribed_to_simulation",
                            client_id=client_id,
                            simulation_id=sim_id,
                        )
                        await manager.send_personal_message(
                            json.dumps(
                                {"type": "subscription_success", "sim_id": sim_id}
                            ),
                            websocket,
                        )
                    else:
                        await manager.send_personal_message(
                            json.dumps(
                                {
                                    "type": "subscription_error",
                                    "message": f"Simulation {sim_id} not found",
                                }
                            ),
                            websocket,
                        )
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info("websocket_client_disconnected", client_id=client_id)


@app.get("/api/simulation/{sim_id}/status", response_model=SimulationStatus)
async def get_simulation_status(sim_id: str):
    try:
        with _active_simulations_lock:
            if sim_id not in active_simulations:
                raise HTTPException(
                    status_code=404, detail=f"Simulation {sim_id} not found"
                )
            data = dict(active_simulations[sim_id])
        return SimulationStatus(status="success", data=data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_get_simulation_status_failed",
            simulation_id=sim_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Start FastAPI server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
