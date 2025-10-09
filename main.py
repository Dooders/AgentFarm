import os
import sys
import tkinter as tk

from farm.database.models import Base
from farm.database.session_manager import SessionManager
from farm.utils.logging import configure_logging, get_logger

# Configure structured logging
configure_logging(
    environment="development",
    log_dir="logs",
    log_level="INFO",
    enable_colors=True,
)
logger = get_logger(__name__)


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

        logger.info("application_directory_set", app_dir=app_dir)

        # Create simulations directory relative to app directory
        sim_dir = os.path.join(app_dir, "simulations")
        os.makedirs(sim_dir, exist_ok=True)
        logger.info("simulations_directory_configured", sim_dir=sim_dir)

        # Initialize session manager and create tables
        session_manager = SessionManager(path=sim_dir)

        # Create all database tables if they don't exist
        Base.metadata.create_all(bind=session_manager.engine)
        logger.info("database_tables_initialized", status="success")

        root = tk.Tk()

        # Handle window close event
        def on_closing():
            try:
                if session_manager:
                    session_manager.cleanup()
                root.destroy()
            except Exception as e:
                logger.error(
                    "cleanup_error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except Exception as e:
        logger.error(
            "application_error",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise
    finally:
        try:
            if session_manager:
                session_manager.cleanup()
        except Exception as e:
            logger.error(
                "final_cleanup_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )


if __name__ == "__main__":
    main()
