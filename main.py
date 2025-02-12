import logging
import os
import sys
import tkinter as tk

from farm.database.session_manager import SessionManager
from farm.gui import SimulationGUI
from farm.database.models import Base

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the simulation GUI application.
    """
    session_manager = None
    try:
        # Get the directory where the script is located
        if getattr(sys, "frozen", False):
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

        # Initialize session manager and create tables
        session_manager = SessionManager(path=sim_dir)
        
        # Create all database tables if they don't exist
        Base.metadata.create_all(bind=session_manager.engine)
        logger.info("Database tables created successfully")

        root = tk.Tk()
        app = SimulationGUI(root, sim_dir, session_manager)

        # Handle window close event
        def on_closing():
            try:
                if session_manager:
                    session_manager.cleanup()
                root.destroy()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        raise
    finally:
        try:
            if session_manager:
                session_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}", exc_info=True)


if __name__ == "__main__":
    main()
