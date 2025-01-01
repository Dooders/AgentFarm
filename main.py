import os
import tkinter as tk

from farm.database.session_manager import SessionManager
from farm.gui import SimulationGUI

# Use absolute path and proper SQLite URL format
save_path = "farm/results/simulation_results.db"
db_url = f"sqlite:///{os.path.abspath(save_path)}"


def main():
    """
    Main entry point for the simulation GUI application.
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize session manager
    session_manager = SessionManager(db_url)

    try:
        root = tk.Tk()
        app = SimulationGUI(root, save_path, session_manager)

        # Handle window close event
        def on_closing():
            session_manager.cleanup()  # Cleanup database resources
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except Exception as e:
        print(f"Error running simulation: {e}")
        raise
    finally:
        # Ensure cleanup happens even if an error occurs
        session_manager.cleanup()


if __name__ == "__main__":
    main()
