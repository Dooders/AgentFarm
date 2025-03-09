# Add the project root to the Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

EXPERIMENT_PATH = "results/one_of_a_kind/"
DATA_PATH = EXPERIMENT_PATH + "experiments/data/"
OUTPUT_PATH = EXPERIMENT_PATH + "experiments/analysis/"
