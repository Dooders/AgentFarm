import os

import matplotlib.pyplot as plt


def save_plot(plt, chart_name, output_dir: str = ""):
    """Helper function to save plot to file and return path."""
    if output_dir:
        final_dir = os.path.join(output_dir, "charts")
        os.makedirs(final_dir, exist_ok=True)
        file_path = os.path.join(final_dir, f"{chart_name}.png")
        plt.savefig(file_path)
        plt.close()
        return file_path
    else:
        # Return the current figure for in-memory use
        return plt.gcf()
