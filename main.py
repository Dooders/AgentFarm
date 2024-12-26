import os
import tkinter as tk
from farm.gui import SimulationGUI
from farm.database.session_manager import SessionManager

# Use absolute path and proper SQLite URL format
save_path = "farm/results/simulation_results.db"
db_url = f"sqlite:///{os.path.abspath(save_path)}"

def main():
    """
    Main entry point for the simulation GUI application.
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    root = tk.Tk()
    session_manager = SessionManager(db_url)
    app = SimulationGUI(root, save_path, session_manager)
    root.mainloop()

if __name__ == "__main__":
    main()