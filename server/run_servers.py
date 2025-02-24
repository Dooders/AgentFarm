import logging
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_python_executable():
    """Get the correct Python executable path"""
    return sys.executable


def is_port_in_use(port):
    """Check if a port is in use"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def start_servers():
    """Start both frontend and backend servers"""
    # Get the directory containing this script
    base_dir = Path(__file__).parent
    python_exe = get_python_executable()

    # Check if ports are already in use
    if is_port_in_use(8000):
        logger.error(
            "Port 8000 is already in use. Please stop any running backend server."
        )
        return False
    if is_port_in_use(3000):
        logger.error(
            "Port 3000 is already in use. Please stop any running frontend server."
        )
        return False

    try:
        # Start backend server
        backend_cmd = [
            python_exe,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        logger.info("Backend server started on http://localhost:8000")

        # Give the backend a moment to start
        time.sleep(2)

        # Start frontend server
        frontend_script = base_dir / "frontend" / "serve.py"
        frontend_process = subprocess.Popen(
            [python_exe, str(frontend_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        logger.info("Frontend server started on http://localhost:3000")

        # Open the browser
        webbrowser.open("http://localhost:3000")

        def signal_handler(signum, frame):
            """Handle shutdown gracefully"""
            logger.info("Shutting down servers...")
            frontend_process.terminate()
            backend_process.terminate()
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep the script running and monitor the processes
        while True:
            if backend_process.poll() is not None:
                logger.error("Backend server stopped unexpectedly!")
                frontend_process.terminate()
                break
            if frontend_process.poll() is not None:
                logger.error("Frontend server stopped unexpectedly!")
                backend_process.terminate()
                break
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error starting servers: {e}")
        # Cleanup
        if "backend_process" in locals():
            backend_process.terminate()
        if "frontend_process" in locals():
            frontend_process.terminate()
        return False

    return True


if __name__ == "__main__":
    logger.info("Starting servers...")
    start_servers()
