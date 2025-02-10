import os
import matplotlib.pyplot as plt

def save_plot(plt, chart_name, save_to_file=True):
    """Helper function to save plot to file and return path."""
    if save_to_file:
        output_dir = "chart_analysis"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{chart_name}.png")
        plt.savefig(file_path)
        plt.close()
        return file_path
    else:
        # Return the current figure for in-memory use
        return plt.gcf() 