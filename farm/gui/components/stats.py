import tkinter as tk
from tkinter import ttk
from typing import Dict

from farm.gui.components.tooltips import ToolTip
from farm.gui.utils.styles import CARD_COLORS


class StatsPanel(ttk.Frame):
    """
    Panel for displaying simulation statistics and log.

    Displays:
    - Agent counts by type
    - Resource metrics
    - Simulation log
    - Progress indicators

    Attributes:
        stats_vars (Dict[str, Dict]): Collection of statistics variables and metadata
        log_text (tk.Text): Log display widget
        progress_bar (ttk.Progressbar): Progress indicator
        progress_label (ttk.Label): Progress status text
    """

    def __init__(self, parent):
        super().__init__(parent)

        # Initialize variables
        self.stats_vars = {}
        self.logger = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the statistics panel UI."""
        # Create container for two columns
        columns_frame = ttk.Frame(self)
        columns_frame.pack(fill="both", expand=True, padx=2)

        # Create left and right column frames
        left_column = ttk.Frame(columns_frame)
        right_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=2)
        right_column.pack(side="left", fill="both", expand=True, padx=2)

        # Setup statistics cards
        self._setup_stat_cards(left_column, right_column)

    def _setup_stat_cards(self, left_column: ttk.Frame, right_column: ttk.Frame):
        """Setup statistics display cards."""
        # Stats variables with labels and colors
        stats_config = {
            "total_agents": {
                "label": "Total Agents",
                "color": CARD_COLORS["total_agents"],
                "column": left_column,
                "tooltip": "Total number of agents in the simulation",
            },
            "system_agents": {
                "label": "System Agents",
                "color": CARD_COLORS["system_agents"],
                "column": right_column,
                "tooltip": "Number of system-controlled agents",
            },
            "independent_agents": {
                "label": "Independent Agents",
                "color": CARD_COLORS["independent_agents"],
                "column": left_column,
                "tooltip": "Number of independently-controlled agents",
            },
            "control_agents": {
                "label": "Control Agents",
                "color": CARD_COLORS["control_agents"],
                "column": right_column,
                "tooltip": "Number of control group agents",
            },
            "total_resources": {
                "label": "Total Resources",
                "color": CARD_COLORS["total_resources"],
                "column": left_column,
                "tooltip": "Total resources available in the environment",
            },
            "average_agent_resources": {
                "label": "Avg Resources/Agent",
                "color": CARD_COLORS["average_agent_resources"],
                "column": right_column,
                "tooltip": "Average resources per agent",
            },
        }

        # Create stat cards
        for stat_id, config in stats_config.items():
            self.stats_vars[stat_id] = {"var": tk.StringVar(value="0"), **config}
            self._create_stat_card(stat_id, config)

    def _create_stat_card(self, stat_id: str, config: Dict):
        """Create a single statistics card."""
        # Create card frame
        card = ttk.Frame(config["column"], style="Card.TFrame")
        card.pack(fill="x", padx=3, pady=3)

        # Inner padding frame
        padding_frame = ttk.Frame(card, style="Card.TFrame")
        padding_frame.pack(fill="x", padx=1, pady=1)

        # Label with custom style
        label = ttk.Label(
            padding_frame, text=config["label"], style=f"{stat_id}.CardLabel.TLabel"
        )
        label.pack(anchor="w", padx=8, pady=(5, 0))

        # Value with custom style
        ttk.Label(
            padding_frame,
            textvariable=self.stats_vars[stat_id]["var"],
            style=f"{stat_id}.CardValue.TLabel",
        ).pack(anchor="e", padx=8, pady=(0, 5))

        # Add tooltip
        if "tooltip" in config:
            ToolTip(label, config["tooltip"])

    def update(self, data=None):
        """Update the stats component with new data."""
        if data:
            # Handle metrics data structure
            metrics = data.get("metrics", {})

            # Map incoming metrics to stat variables
            stat_mapping = {
                "total_agents": "total_agents",
                "system_agents": "system_agents",
                "independent_agents": "independent_agents",
                "control_agents": "control_agents",
                "total_resources": "total_resources",
                "average_agent_resources": "average_agent_resources",
            }

            # Update each stat with formatted value
            for stat_id, metric_key in stat_mapping.items():
                if stat_id in self.stats_vars and metric_key in metrics:
                    value = metrics[metric_key]
                    # Format numbers appropriately
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    self.stats_vars[stat_id]["var"].set(formatted_value)

        # Call the widget's update method without arguments
        ttk.Frame.update(self)

    def log_message(self, message: str):
        """Add a message to the log display."""
        pass  # Remove logging functionality

    def clear_log(self):
        """Clear the log display."""
        pass  # Remove logging functionality

    def show_progress(self, message: str = "Working..."):
        """Show progress indicator with message."""
        pass  # Progress functionality removed

    def hide_progress(self):
        """Hide progress indicator."""
        pass  # Progress functionality removed

    def set_progress(self, value: float):
        """Set determinate progress value (0-100)."""
        pass  # Progress functionality removed

    def reset(self):
        """Reset all statistics to zero."""
        for stat_info in self.stats_vars.values():
            stat_info["var"].set("0")

    def set_logger(self, logger):
        """Set the data logger for this component."""
        self.logger = logger
