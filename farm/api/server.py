import logging
import os
from dataclasses import replace
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from farm.core.analysis import analyze_simulation
from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active simulations
active_simulations = {}


@app.route("/api/simulation/new", methods=["POST"])
def create_simulation():
    """Create a new simulation with provided configuration."""
    try:
        config_data = request.json or {}
        logger.info(f"Creating new simulation with config: {config_data}")

        # Generate unique simulation ID
        sim_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"results/simulation_{sim_id}.db"

        # Load and update config
        base_config = SimulationConfig.from_yaml("config.yaml")
        config = replace(base_config, **config_data)

        # Create database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Run simulation with progress updates
        def progress_callback(step, total_steps):
            socketio.emit(
                "simulation_progress",
                {
                    "sim_id": sim_id,
                    "step": step,
                    "total_steps": total_steps,
                    "percentage": (step / total_steps) * 100,
                },
            )

        # Run simulation
        run_simulation(
            num_steps=config.simulation_steps,
            config=config,
            path=os.path.dirname(db_path),
        )

        # Store simulation info
        active_simulations[sim_id] = {
            "db_path": db_path,
            "config": config_data,
            "created_at": datetime.now().isoformat(),
        }

        # Get initial state
        db = SimulationDatabase(db_path)
        initial_state = db.query.gui_repository.get_simulation_data(0)

        return jsonify(
            {
                "status": "success",
                "sim_id": sim_id,
                "data": initial_state,
                "message": "Simulation created successfully",
            }
        )

    except Exception as e:
        logger.error(f"Error creating simulation: {str(e)}", exc_info=True)
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
        logger.error(f"Error getting step {step}: {str(e)}")
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
        logger.error(f"Error analyzing simulation: {str(e)}")
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
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# WebSocket events
@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


@socketio.on("subscribe_simulation")
def handle_subscribe(sim_id):
    """Subscribe to simulation updates."""
    if sim_id in active_simulations:
        logger.info(f"Client subscribed to simulation {sim_id}")
        emit("subscription_success", {"sim_id": sim_id})
    else:
        emit("subscription_error", {"message": f"Simulation {sim_id} not found"})


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Start SocketIO server
    socketio.run(app, port=5000, debug=True)
