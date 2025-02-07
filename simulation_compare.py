import os
from farm.analysis.comparative_analysis import compare_simulations, find_simulation_databases

EXPERIMENT_PATH = "research/one_of_a_kind/control_agent/data"

def main():
    # Ensure the experiment directory exists
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)
    
    # Create analysis directory within experiment path
    analysis_path = os.path.join(EXPERIMENT_PATH, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    
    # Find all simulation.db files in subdirectories
    db_paths = []
    for root, dirs, files in os.walk(EXPERIMENT_PATH):
        if "simulation.db" in files:
            db_path = os.path.join(root, "simulation.db")
            db_paths.append(db_path)
    
    if not db_paths:
        print(f"No simulation.db files found in {EXPERIMENT_PATH} or its subdirectories")
        return
        
    # Compare the simulations and save results to analysis folder
    compare_simulations(db_paths, analysis_path)

if __name__ == "__main__":
    main()




