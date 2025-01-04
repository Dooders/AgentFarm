import os
import tkinter as tk
import logging
import sys

from farm.database.session_manager import SessionManager
from farm.gui import SimulationGUI

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the simulation GUI application.
    """
    try:
        # Get the directory where the script is located
        if getattr(sys, 'frozen', False):
            # If running as compiled executable
            app_dir = os.path.dirname(sys.executable)
        else:
            # If running as script
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
        logger.info(f"Application directory: {app_dir}")
        
        # Create simulations directory relative to app directory
        sim_dir = os.path.join(app_dir, "simulations")
        os.makedirs(sim_dir, exist_ok=True)
        logger.info(f"Using simulations directory: {sim_dir}")
        
        # Set up database path
        save_path = os.path.join(sim_dir, "simulation_results.db")
        db_url = f"sqlite:///{save_path}"
        logger.info(f"Database path: {save_path}")

        # Initialize session manager
        session_manager = SessionManager(db_url)

        root = tk.Tk()
        app = SimulationGUI(root, save_path, session_manager)

        # Handle window close event
        def on_closing():
            session_manager.cleanup()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        raise
    finally:
        session_manager.cleanup()

if __name__ == "__main__":
    main()
