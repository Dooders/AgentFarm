import os
from dataclasses import replace
from datetime import datetime
import threading

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

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

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active simulations
active_simulations = {}


def _run_simulation_background(sim_id, config, db_path):
    try:
        active_simulations[sim_id]["status"] = "running"

        # Run simulation
        run_simulation(
            num_steps=config.simulation_steps,
            config=config,
            path=os.path.dirname(db_path),
        )

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
        active_simulations[sim_id]["status"] = "error"
        active_simulations[sim_id]["error_message"] = str(e)


@app.route("/api/simulation/new", methods=["POST"])
def create_simulation():
    """Create a new simulation with provided configuration."""
    try:
        config_data = request.json or {}

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
        config = replace(base_config, **config_data)

        # Create database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Store simulation info
        active_simulations[sim_id] = {
            "db_path": db_path,
            "config": config_data,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }

        # Start background thread
        thread = threading.Thread(
            target=_run_simulation_background, args=(sim_id, config, db_path)
        )
        thread.daemon = True
        thread.start()

        return (
            jsonify(
                {
                    "status": "accepted",
                    "sim_id": sim_id,
                    "message": "Simulation started",
                }
            ),
            202,
        )

    except Exception as e:
        logger.error(
            "api_simulation_create_failed",
            simulation_id=sim_id if "sim_id" in locals() else "unknown",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/<sim_id>/step/<int:step>", methods=["GET"])
def get_step(sim_id, step):
    """Get simulation state for a specific step."""
    try:
        if sim_id not in active_simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        db = SimulationDatabase(active_simulations[sim_id]["db_path"])
        data = db.query.gui_repository.get_simulation_data(step)

        return jsonify({"status": "success", "data": data})

    except Exception as e:
        logger.error(
            "api_get_step_failed",
            simulation_id=sim_id,
            step=step,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/<sim_id>/analysis", methods=["GET"])
def get_analysis(sim_id):
    """Get detailed simulation analysis."""
    try:
        if sim_id not in active_simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        db = SimulationDatabase(active_simulations[sim_id]["db_path"])
        analysis_results = analyze_simulation(db)

        return jsonify({"status": "success", "data": analysis_results})

    except Exception as e:
        logger.error(
            "api_analysis_failed",
            simulation_id=sim_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/<sim_id>/status", methods=["GET"])
def get_simulation_status(sim_id):
    try:
        if sim_id not in active_simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        return jsonify({"status": "success", "data": active_simulations[sim_id]})
    except Exception as e:
        logger.error(
            "api_get_simulation_status_failed",
            simulation_id=sim_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analysis/<module_name>", methods=["POST"])
def run_analysis_module(module_name):
    try:
        payload = request.json or {}
        experiment_path = payload.get("experiment_path", "results")
        output_path = payload.get("output_path", "results/analysis")
        group = payload.get("group", "all")
        processor_kwargs = payload.get("processor_kwargs")
        analysis_kwargs = payload.get("analysis_kwargs")

        service = AnalysisService(config_service=EnvConfigService())
        req = AnalysisRequest(
            module_name=module_name,
            experiment_path=experiment_path,
            output_path=output_path,
            group=group,
            processor_kwargs=processor_kwargs,
            analysis_kwargs=analysis_kwargs,
        )
        result = service.run(req)
        return jsonify(
            {
                "status": "success",
                "output_path": str(result.output_path),
                "rows": (
                    int(result.dataframe.shape[0])
                    if result.dataframe is not None
                    else 0
                ),
            }
        )
    except Exception as e:
        logger.error(
            "api_analysis_module_failed",
            module_name=module_name,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulations", methods=["GET"])
def list_simulations():
    """Get list of active simulations."""
    return jsonify({"status": "success", "data": active_simulations})


@app.route("/api/simulation/<sim_id>/export", methods=["GET"])
def export_simulation(sim_id):
    """Export simulation data."""
    try:
        if sim_id not in active_simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        db = SimulationDatabase(active_simulations[sim_id]["db_path"])
        export_path = f"results/export_{sim_id}.csv"
        db.export_data(export_path)

        return jsonify(
            {
                "status": "success",
                "path": export_path,
                "message": "Data exported successfully",
            }
        )

    except Exception as e:
        logger.error(
            "api_export_failed",
            simulation_id=sim_id,
            export_path=export_path,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return jsonify({"status": "error", "message": str(e)}), 500


# WebSocket events
@socketio.on("connect")
def handle_connect():
    logger.info("websocket_client_connected", client_id=request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("websocket_client_disconnected", client_id=request.sid)


@socketio.on("subscribe_simulation")
def handle_subscribe(sim_id):
    """Subscribe to simulation updates."""
    if sim_id in active_simulations:
        logger.info(
            "client_subscribed_to_simulation",
            client_id=request.sid,
            simulation_id=sim_id,
        )
        emit("subscription_success", {"sim_id": sim_id})
    else:
        emit("subscription_error", {"message": f"Simulation {sim_id} not found"})


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Start SocketIO server
    socketio.run(app, port=5000, debug=True)
