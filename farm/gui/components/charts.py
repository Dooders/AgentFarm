import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Optional
import logging

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from PIL import ImageGrab
import io
import win32clipboard
from PIL import Image


class SimulationChart(ttk.Frame):
    """Component for displaying simulation metrics charts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.is_dragging = False
        self.was_playing = False
        self.last_click_time = 0  # Track time of last click
        self.double_click_delay = 300  # Milliseconds to detect double click
        
        # Store historical data
        self.history = {
            "steps": [],
            "system_agents": [],
            "independent_agents": [],
            "control_agents": [],
            "total_resources": []
        }
        
        # Store full simulation data
        self.full_data = {
            "steps": [],
            "system_agents": [],
            "independent_agents": [],
            "control_agents": [],
            "total_resources": [],
            "births": [],
            "deaths": []
        }
        
        self.logger = None
        
        self._setup_chart()
        self._setup_interactions()
        self._setup_context_menu()

    def _setup_context_menu(self):
        """Setup right-click context menu."""
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(
            label="Copy Chart",
            command=self._copy_chart_to_clipboard
        )
        
        # Bind right-click event to canvas
        self.canvas.get_tk_widget().bind("<Button-3>", self._show_context_menu)

    def _show_context_menu(self, event):
        """Show the context menu on right-click."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _copy_chart_to_clipboard(self):
        """Copy the current chart to clipboard."""
        try:
            # Get the figure as a PNG image
            buf = io.BytesIO()
            self.fig.savefig(buf, format='PNG', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image
            image = Image.open(buf)
            
            # Convert image to bitmap for clipboard
            output = io.BytesIO()
            image.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove bitmap header
            output.close()
            
            # Copy to clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to copy chart: {str(e)}",
                parent=self.winfo_toplevel()
            )
        finally:
            buf.close()

    def _setup_chart(self):
        """Initialize the chart figure and axes."""
        # Create notebook for tabs
        self.chart_notebook = ttk.Notebook(self)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        metrics_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(metrics_frame, text="Population & Resources")
        
        # Demographics tab
        demographics_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(demographics_frame, text="Births & Deaths")

        # Setup main metrics chart
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Setup demographics chart
        self.demo_fig = Figure(figsize=(8, 4), dpi=100)
        self.demo_ax = self.demo_fig.add_subplot(111)

        # Initialize line objects with reduced opacity for historical data
        self.lines = {
            "system_agents": self.ax1.plot(
                [], [], "b-", label="System Agents", alpha=0.3)[0],
            "independent_agents": self.ax1.plot(
                [], [], "r-", label="Independent Agents", alpha=0.3)[0],
            "control_agents": self.ax1.plot(
                [], [], color="#FFD700", label="Control Agents", alpha=0.3)[0],
            "resources": self.ax2.plot(
                [], [], "g-", label="Resources", alpha=0.3)[0],
            "current_step": self.ax1.axvline(
                x=0, color="gray", linestyle="--", alpha=0.5
            )
        }
        
        # Initialize future data lines with full opacity
        self.future_lines = {
            "system_agents": self.ax1.plot(
                [], [], "b-", alpha=1.0)[0],
            "independent_agents": self.ax1.plot(
                [], [], "r-", alpha=1.0)[0],
            "control_agents": self.ax1.plot(
                [], [], color="#FFD700", alpha=1.0)[0],
            "resources": self.ax2.plot(
                [], [], "g-", alpha=1.0)[0]
        }

        # Setup axis labels and colors for main chart
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Agent Count", color="black")
        self.ax2.set_ylabel("Resource Count", color="green", rotation=270, labelpad=20)

        # Configure axis positions and colors
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.set_ticks_position("right")
        self.ax1.tick_params(axis="y", labelcolor="black")
        self.ax2.tick_params(axis="y", labelcolor="green")

        # Setup legends for main chart
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

        # Setup demographics chart
        self.demo_ax.set_xlabel("Step")
        self.demo_ax.set_ylabel("Count")
        
        # Initialize birth/death bars
        self.birth_bars = None
        self.death_bars = None

        # Create canvases
        self.canvas = FigureCanvasTkAgg(self.fig, master=metrics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.demo_canvas = FigureCanvasTkAgg(self.demo_fig, master=demographics_frame)
        self.demo_canvas.draw()
        self.demo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, data: Optional[Dict] = None):
        """Update chart with new simulation data."""
        if not data or "metrics" not in data:
            return

        try:
            # Get metrics
            metrics = data.get("metrics", {})
            if not metrics:
                return

            # Get current step from history length
            current_step = len(self.history["steps"])
            
            # Update history with current data
            self.history["steps"].append(current_step)
            self.history["system_agents"].append(metrics.get("system_agents", 0))
            self.history["independent_agents"].append(metrics.get("independent_agents", 0))
            self.history["control_agents"].append(metrics.get("control_agents", 0))
            self.history["total_resources"].append(metrics.get("total_resources", 0))

            # Update current data lines
            for key in ["system_agents", "independent_agents", "control_agents"]:
                self.lines[key].set_data(
                    self.history["steps"], 
                    self.history[key]
                )

            # Update current resources line
            self.lines["resources"].set_data(
                self.history["steps"], 
                self.history["total_resources"]
            )

            # Update future data lines using full data from DataRetriever
            if hasattr(self, 'full_data'):
                for key in ["system_agents", "independent_agents", "control_agents"]:
                    self.future_lines[key].set_data(
                        self.full_data["steps"][current_step:], 
                        self.full_data[key][current_step:]
                    )

                self.future_lines["resources"].set_data(
                    self.full_data["steps"][current_step:], 
                    self.full_data["total_resources"][current_step:]
                )

            # Update current step line
            self.lines["current_step"].set_xdata([current_step, current_step])
            
            # Update step line height using stored y-limits
            y_max = self.ax1.get_ylim()[1]
            self.lines["current_step"].set_ydata([0, y_max])

            # Update demographics chart
            self._update_demographics(data)

            # Force redraw
            self.canvas.draw()
            self.demo_canvas.draw()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating chart: {e}")

    def _update_demographics(self, data: Dict):
        """Update the demographics chart with birth/death data."""
        try:
            metrics = data.get("metrics", {})
            current_step = len(self.history["steps"])

            # Get births and deaths for all steps up to current
            births_data = self.full_data["births"][:current_step + 1]
            deaths_data = self.full_data["deaths"][:current_step + 1]
            steps = self.full_data["steps"][:current_step + 1]

            # Clear previous plot
            self.demo_ax.clear()

            # Create mirrored bar chart for all data at once
            self.birth_bars = self.demo_ax.bar(steps, births_data, color='g', alpha=0.6, width=0.8, label='Births')
            self.death_bars = self.demo_ax.bar(steps, [-d for d in deaths_data], color='r', alpha=0.6, width=0.8, label='Deaths')

            # Update axis limits with minimum range
            max_value = max(max(births_data or [1]), max(deaths_data or [1]))
            max_range = max(1, max_value * 1.2)
            self.demo_ax.set_ylim(-max_range, max_range)
            
            # Set x-axis limits
            self.demo_ax.set_xlim(0, self.max_step)
            
            # Add zero line
            self.demo_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Add labels and legend
            self.demo_ax.set_title("Population Changes")
            self.demo_ax.set_ylabel("Count")
            self.demo_ax.set_xlabel("Step")
            self.demo_ax.legend()

            # Force redraw
            self.demo_canvas.draw()

        except Exception as e:
            logging.error(f"Error updating demographics chart: {str(e)}")

    def _setup_interactions(self):
        """Setup mouse interaction handlers."""
        # Connect to matplotlib events
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def _on_click(self, event):
        """Handle mouse click on the chart."""
        if event.inaxes in [self.ax1, self.ax2] and event.xdata is not None:
            try:
                current_time = int(time.time() * 1000)  # Current time in milliseconds
                step = int(round(event.xdata))
                
                if hasattr(self, 'max_step'):
                    step = max(0, min(step, self.max_step))
                    
                    # Check if this is a double click
                    if current_time - self.last_click_time < self.double_click_delay:
                        # Double click - toggle playback
                        if hasattr(self, "on_timeline_click"):
                            self.on_timeline_click(step)
                            if hasattr(self, "on_playback_toggle"):
                                self.on_playback_toggle()  # Will toggle between play/pause
                    else:
                        # Single click - move to step but maintain playback state
                        if hasattr(self, "on_timeline_click"):
                            self.on_timeline_click(step)
                            self.is_dragging = True
                    
                    self.last_click_time = current_time
                    
            except Exception as e:
                pass

    def _on_drag(self, event):
        """Handle mouse drag on the chart."""
        if self.is_dragging and event.inaxes in [self.ax1, self.ax2] and event.xdata is not None:
            try:
                step = int(round(event.xdata))
                if hasattr(self, 'max_step'):
                    step = max(0, min(step, self.max_step))
                    if hasattr(self, "on_timeline_click"):
                        self.on_timeline_click(step)
            except Exception as e:
                pass

    def _on_release(self, event):
        """Handle mouse release."""
        self.is_dragging = False

    def set_timeline_callback(self, callback):
        """Set callback for timeline navigation."""
        self.on_timeline_click = callback
        if hasattr(self, 'full_data') and self.full_data["steps"]:
            self.max_step = len(self.full_data["steps"]) - 1

    def set_playback_callback(self, callback):
        """Set callback for playback resume."""
        self.on_playback_resume = callback

    def clear(self):
        """Clear all chart data."""
        # Clear main chart
        self.history = {key: [] for key in self.history}
        self.full_data = {key: [] for key in self.full_data}
        
        for line in self.lines.values():
            line.set_data([], [])
        for line in self.future_lines.values():
            line.set_data([], [])
        
        # Clear demographics chart
        if self.birth_bars:
            self.birth_bars.remove()
        if self.death_bars:
            self.death_bars.remove()
        self.birth_bars = None
        self.death_bars = None
        
        # Reset axes
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.demo_ax.set_xlim(0, 100)
        self.demo_ax.set_ylim(-1, 1)
        
        # Redraw both canvases
        self.canvas.draw()
        self.demo_canvas.draw()

    def set_full_data(self, data: Dict):
        """Store the full simulation data without displaying it."""
        if not data or "metrics" not in data:
            return
        
        try:
            steps = data["steps"]
            metrics = data["metrics"]
            
            # Store full data with safe defaults for missing metrics
            self.full_data = {
                "steps": steps,
                "system_agents": metrics.get("system_agents", [0] * len(steps)),
                "independent_agents": metrics.get("independent_agents", [0] * len(steps)),
                "control_agents": metrics.get("control_agents", [0] * len(steps)),
                "total_resources": metrics.get("total_resources", [0] * len(steps)),
                "births": metrics.get("births", [0] * len(steps)),
                "deaths": metrics.get("deaths", [0] * len(steps))
            }
            
            # Clear history (current display)
            self.history = {key: [] for key in self.history}
            
            # Calculate and store fixed axis limits
            self.max_step = max(steps) if steps else 100
            
            # Calculate max values for y-axis limits
            max_agents = max(
                max(metrics.get("system_agents", [0]) or [0]),
                max(metrics.get("independent_agents", [0]) or [0]),
                max(metrics.get("control_agents", [0]) or [0])
            )
            max_resources = max(metrics.get("total_resources", [0]) or [0])
            
            # Set fixed y-limits with padding
            self.ax1.set_ylim(0, max(1, max_agents * 1.1))
            self.ax2.set_ylim(0, max(1, max_resources * 1.1))
            
            # Set fixed x-limits for both charts
            self.ax1.set_xlim(0, self.max_step)
            self.demo_ax.set_xlim(0, self.max_step)
            
            # Set initial y-limits for demographics chart with minimum range
            max_population_change = max(
                max(metrics.get("births", [0]) or [0]),
                max(metrics.get("deaths", [0]) or [0])
            )
            # Ensure minimum range of 1 to avoid identical limits
            max_range = max(1, max_population_change * 1.2)
            self.demo_ax.set_ylim(-max_range, max_range)
            
            # Clear all lines
            for line in self.lines.values():
                line.set_data([], [])
            for line in self.future_lines.values():
                line.set_data([], [])
            
            # Show future data in semi-transparent
            for key in ["system_agents", "independent_agents", "control_agents"]:
                self.future_lines[key].set_data(steps, metrics.get(key, [0] * len(steps)))
            
            self.future_lines["resources"].set_data(steps, metrics.get("total_resources", [0] * len(steps)))
            
            # Force redraw both canvases
            self.canvas.draw()
            self.demo_canvas.draw()
            
        except Exception as e:
            logging.error(f"Error setting full data: {str(e)}")

    def reset_history_to_step(self, step: int):
        """Reset the history up to a specific step."""
        if not hasattr(self, 'full_data') or not self.full_data["steps"]:
            return
        
        # Reset history
        self.history = {
            "steps": self.full_data["steps"][:step+1],
            "system_agents": self.full_data["system_agents"][:step+1],
            "independent_agents": self.full_data["independent_agents"][:step+1],
            "control_agents": self.full_data["control_agents"][:step+1],
            "total_resources": self.full_data["total_resources"][:step+1]
        }
        
        # Update current data lines
        for key in ["system_agents", "independent_agents", "control_agents"]:
            self.lines[key].set_data(
                self.history["steps"], 
                self.history[key]
            )

        # Update current resources line
        self.lines["resources"].set_data(
            self.history["steps"], 
            self.history["total_resources"]
        )

        # Update future data lines
        for key in ["system_agents", "independent_agents", "control_agents"]:
            self.future_lines[key].set_data(
                self.full_data["steps"][step+1:], 
                self.full_data[key][step+1:]
            )

        # Update future resources line
        self.future_lines["resources"].set_data(
            self.full_data["steps"][step+1:], 
            self.full_data["total_resources"][step+1:]
        )

        # Update current step line
        self.lines["current_step"].set_xdata([step, step])
        
        # Update step line height
        y_max = self.ax1.get_ylim()[1]
        self.lines["current_step"].set_ydata([0, y_max])

        # Force redraw
        self.canvas.draw()

    def set_playback_stop_callback(self, callback):
        """Set callback for stopping playback."""
        self.on_playback_stop = callback

    def set_playback_toggle_callback(self, callback):
        """Set callback for toggling playback."""
        self.on_playback_toggle = callback

    def set_logger(self, logger):
        """Set the data logger for this component."""
        self.logger = logger