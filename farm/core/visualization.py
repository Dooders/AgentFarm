import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import ttk
from typing import Dict, List

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont, ImageTk

from database.database import SimulationDatabase


class SimulationVisualizer:
    # Visualization constants
    DEFAULT_CANVAS_SIZE = (400, 400)
    PADDING = 20

    # Animation constants
    MAX_ANIMATION_FRAMES = 5
    ANIMATION_MIN_DELAY = 50

    # Resource visualization
    MAX_RESOURCE_AMOUNT = 30
    RESOURCE_GLOW_RED = 50
    RESOURCE_GLOW_GREEN = 255
    RESOURCE_GLOW_BLUE = 50

    # Agent visualization
    AGENT_RADIUS_SCALE = 2
    BIRTH_RADIUS_SCALE = 4
    DEATH_MARK_SCALE = 1.5

    # Font settings
    MIN_FONT_SIZE = 10
    FONT_SCALE_FACTOR = 40

    # Colors
    SYSTEM_AGENT_COLOR = "blue"
    INDEPENDENT_AGENT_COLOR = "red"
    CONTROL_AGENT_COLOR = "#DAA520"  # Changed to goldenrod color
    DEATH_MARK_COLOR = (255, 0, 0)  # Red
    BIRTH_MARK_COLOR = (255, 255, 255)  # White
    BACKGROUND_COLOR = "black"

    # Style constants
    CARD_COLORS = {
        "total_agents": "#4a90e2",  # Blue
        "system_agents": "#50c878",  # Emerald green
        "independent_agents": "#e74c3c",  # Red
        "control_agents": "#DAA520",  # Changed to goldenrod
        "total_resources": "#f39c12",  # Orange
        "average_agent_resources": "#9b59b6",  # Purple
    }

    def __init__(self, parent, db_path="simulation_results.db"):
        """Initialize visualizer with parent frame."""
        self.parent = parent
        self.db = SimulationDatabase(db_path)
        self.current_step = 0
        self.playing = False
        self.was_playing = False

        # Initialize animation tracking
        self.previous_agent_ids = set()
        self.previous_agent_states = []
        self.birth_animations = {}
        self.death_animations = {}

        # Initialize chart interaction state
        self.is_dragging = False

        # Get root window
        self.root = self._get_root(parent)
        if isinstance(self.root, tk.Tk):
            self.root.title("Agent-Based Simulation Visualizer")

        # Initialize variables and setup UI
        self._setup_ui()

    def _get_root(self, widget):
        """Get the root window of a widget."""
        parent = widget.master
        while parent is not None:
            if isinstance(parent, tk.Tk):
                return parent
            parent = parent.master
        return None

    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        self.main_container = ttk.Frame(self.parent)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Stats frame at top
        self.stats_frame = ttk.LabelFrame(
            self.main_container, text="Simulation Statistics", padding=10
        )
        self.stats_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Chart frame on right
        self.chart_frame = ttk.LabelFrame(
            self.main_container, text="Population & Resources", padding=10
        )
        self.chart_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        # Environment frame below stats
        self.env_frame = ttk.LabelFrame(
            self.main_container, text="Environment View", padding=10
        )
        self.env_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Controls frame at bottom
        self.controls_frame = ttk.Frame(self.main_container, padding=10)
        self.controls_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Configure grid weights
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Initialize UI components
        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

        # Get total simulation length
        self.total_steps = self._get_total_steps()

        # Initialize visualization
        self.parent.update_idletasks()
        self._step_to(0)
        self.parent.update()

    def _get_total_steps(self) -> int:
        """Get the total number of steps in the simulation."""
        try:
            self.db.cursor.execute("SELECT MAX(step_number) FROM SimulationSteps")
            result = self.db.cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to get total steps: {str(e)}"
            )
            return 0

    def _setup_stats_panel(self):
        """Setup statistics panel with card-like metrics in two columns."""
        # Create container for two columns using ttk
        columns_frame = ttk.Frame(self.stats_frame)
        columns_frame.pack(fill="both", expand=True, padx=2)

        # Create left and right column frames
        left_column = ttk.Frame(columns_frame)
        right_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=2)
        right_column.pack(side="left", fill="both", expand=True, padx=2)

        # Configure styles for cards
        style = ttk.Style()
        style.configure("Card.TFrame", background="white", relief="solid")
        style.configure("CardLabel.TLabel", background="white", font=("Arial", 10))
        style.configure(
            "CardValue.TLabel", background="white", font=("Arial", 14, "bold")
        )

        # Stats variables with labels and colors
        self.stats_vars = {
            "total_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Total Agents",
                "color": "#4a90e2",  # Blue
                "column": left_column,
            },
            "system_agents": {
                "var": tk.StringVar(value="0"),
                "label": "System Agents",
                "color": "#50c878",  # Emerald green
                "column": right_column,
            },
            "independent_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Independent Agents",
                "color": "#e74c3c",  # Red
                "column": left_column,
            },
            "control_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Control Agents",
                "color": "#DAA520",  # Changed to goldenrod
                "column": right_column,
            },
            "total_resources": {
                "var": tk.StringVar(value="0"),
                "label": "Total Resources",
                "color": "#f39c12",  # Orange
                "column": left_column,
            },
            "average_agent_resources": {
                "var": tk.StringVar(value="0.0"),
                "label": "Avg Resources/Agent",
                "color": "#9b59b6",  # Purple
                "column": right_column,
            },
        }

        # Create card-like frames for each stat
        for stat_id, stat_info in self.stats_vars.items():
            # Create card frame with custom style
            card = ttk.Frame(stat_info["column"], style="Card.TFrame")
            card.pack(fill="x", padx=3, pady=3)

            # Configure specific style for this stat
            style.configure(
                f"{stat_id}.CardLabel.TLabel",
                foreground=stat_info["color"],
                background="white",
                font=("Arial", 10),
            )
            style.configure(
                f"{stat_id}.CardValue.TLabel",
                foreground="black",
                background="white",
                font=("Arial", 14, "bold"),
            )

            # Inner padding frame
            padding_frame = ttk.Frame(card, style="Card.TFrame")
            padding_frame.pack(fill="x", padx=1, pady=1)

            # Label
            ttk.Label(
                padding_frame,
                text=stat_info["label"],
                style=f"{stat_id}.CardLabel.TLabel",
            ).pack(anchor="w", padx=8, pady=(5, 0))

            # Value
            ttk.Label(
                padding_frame,
                textvariable=stat_info["var"],
                style=f"{stat_id}.CardValue.TLabel",
            ).pack(anchor="e", padx=8, pady=(0, 5))

            # Add tooltips
            tooltips = {
                "total_agents": "Total number of agents in the simulation",
                "system_agents": "Number of system-controlled agents",
                "independent_agents": "Number of independently-controlled agents",
                "total_resources": "Total resources available in the environment",
                "average_agent_resources": "Average resources per agent",
            }

            if stat_id in tooltips:
                # Find the label widget in the card's padding frame
                label = padding_frame.winfo_children()[0]  # First child is the label
                ToolTip(label, tooltips[stat_id])

    def _setup_chart(self):
        """Setup the chart with click interaction."""
        self.fig = Figure(figsize=(8, 4))
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Initialize empty line objects with updated colors
        self.lines = {
            "system_agents": self.ax1.plot([], [], "b-", label="System Agents")[0],
            "independent_agents": self.ax1.plot(
                [], [], "r-", label="Independent Agents"
            )[0],
            "control_agents": self.ax1.plot(
                [], [], color="#FFD700", label="Control Agents"
            )[
                0
            ],  # Changed to soft yellow
            "resources": self.ax2.plot([], [], "g-", label="Resources")[0],
            "system_agents_future": self.ax1.plot([], [], "b-", alpha=0.3)[0],
            "independent_agents_future": self.ax1.plot([], [], "r-", alpha=0.3)[0],
            "control_agents_future": self.ax1.plot([], [], color="#FFD700", alpha=0.3)[
                0
            ],  # Changed to soft yellow
            "resources_future": self.ax2.plot([], [], "g-", alpha=0.3)[0],
            "current_step": self.ax1.axvline(
                x=0, color="gray", linestyle="--", alpha=0.5
            ),
        }

        # Setup axis labels and colors
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Agent Count", color="black")
        self.ax2.set_ylabel("Resource Count", color="green", rotation=270, labelpad=20)

        # Configure axis positions and colors
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.set_ticks_position("right")
        self.ax1.tick_params(axis="y", labelcolor="black")
        self.ax2.tick_params(axis="y", labelcolor="green")

        # Setup legends
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

        # Create canvas without toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()

        # Add mouse event handlers
        self.canvas.mpl_connect("button_press_event", self._on_chart_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_chart_motion)
        self.canvas.mpl_connect("button_release_event", self._on_chart_release)

        # Pack canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_chart_press(self, event):
        """Handle mouse press on the chart."""
        if event.inaxes in (self.ax1, self.ax2):
            self.is_dragging = True
            # Store the current playback state
            self.was_playing = self.playing
            # Temporarily pause while dragging
            self.playing = False
            # Update to the clicked position
            self._update_step_from_x(event.xdata)

    def _on_chart_motion(self, event):
        """Handle mouse motion while dragging."""
        if self.is_dragging and event.inaxes in (self.ax1, self.ax2):
            self._update_step_from_x(event.xdata)

    def _on_chart_release(self, event):
        """Handle mouse release to end dragging."""
        self.is_dragging = False
        # Restore playback state if it was playing before
        if hasattr(self, "was_playing") and self.was_playing:
            self.playing = True
            self._play_simulation()

    def _update_step_from_x(self, x_coord):
        """Update the simulation step based on x coordinate."""
        if x_coord is not None:
            # Get the step number from x coordinate
            step = int(round(x_coord))
            # Ensure step is within bounds
            step = max(0, min(step, self.total_steps))
            # Jump to the selected step
            self._step_to(step)

    def _setup_environment_view(self):
        """Setup the environment view with ttk styling."""
        style = ttk.Style()
        style.configure("Environment.TFrame", background="black")

        # Create frame for canvas
        canvas_frame = ttk.Frame(self.env_frame, style="Environment.TFrame")
        canvas_frame.pack(fill="both", expand=True)

        # Create canvas with black background
        self.env_canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        self.env_canvas.pack(fill="both", expand=True)

        # Bind resize event
        self.env_canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas_size = self.DEFAULT_CANVAS_SIZE

    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        try:
            self.canvas_size = (event.width, event.height)
            # Trigger redraw if we have current data
            data = self.db.get_simulation_data(self.current_step)
            if data["agent_states"] or data["resource_states"]:
                self._update_visualization(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle resize: {str(e)}")

    def _setup_controls(self):
        """Setup playback controls with consistent ttk styling."""
        # Configure styles for controls
        style = ttk.Style()
        style.configure("Control.TButton", padding=5)

        # Configure Scale style - need to use the correct style name
        style.layout(
            "Horizontal.TScale",
            [
                (
                    "Horizontal.Scale.trough",
                    {
                        "sticky": "nswe",
                        "children": [
                            ("Horizontal.Scale.slider", {"side": "left", "sticky": ""})
                        ],
                    },
                )
            ],
        )
        style.configure("Horizontal.TScale", background="white")

        # Control buttons frame
        buttons_frame = ttk.Frame(self.controls_frame)
        buttons_frame.pack(side="left", fill="x", expand=True)

        # Playback controls
        self.play_button = ttk.Button(
            buttons_frame,
            text="Play/Pause",
            command=self._toggle_playback,
            style="Control.TButton",
        )
        self.play_button.pack(side="left", padx=5)

        # Step controls with consistent styling
        for text, command in [
            ("<<", lambda: self._step_to(self.current_step - 10)),
            ("<", lambda: self._step_to(self.current_step - 1)),
            (">", lambda: self._step_to(self.current_step + 1)),
            (">>", lambda: self._step_to(self.current_step + 10)),
        ]:
            ttk.Button(
                buttons_frame, text=text, command=command, style="Control.TButton"
            ).pack(side="left", padx=2)

        # Speed control frame
        speed_frame = ttk.Frame(self.controls_frame)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(
            speed_frame,
            text="Playback Speed:",
        ).pack(side="left", padx=5)

        # Create scale without custom style
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=50, orient="horizontal")
        self.speed_scale.set(10)
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Export button
        ttk.Button(
            self.controls_frame,
            text="Export Data",
            command=self._export_data,
            style="Control.TButton",
        ).pack(side="right", padx=5)

    def _toggle_playback(self):
        self.playing = not self.playing
        if self.playing:
            self._play_simulation()

    def _play_simulation(self):
        """Handle simulation playback."""
        if self.playing:
            try:
                data = self.db.get_simulation_data(self.current_step + 1)
                if not data["agent_states"] and not data["resource_states"]:
                    self.playing = False
                    return

                self._step_to(self.current_step + 1)
                delay = int(1000 / self.speed_scale.get())
                if self.birth_animations or self.death_animations:
                    delay = min(delay, self.ANIMATION_MIN_DELAY)
                self.root.after(delay, self._play_simulation)
            except Exception as e:
                messagebox.showerror(
                    "Playback Error", f"Failed to advance simulation: {str(e)}"
                )
                self.playing = False

    def _step_to(self, step_number):
        """Move to a specific step in the simulation."""
        try:
            data = self.db.get_simulation_data(step_number)
            if data["agent_states"] or data["resource_states"]:
                self.current_step = step_number
                self._update_visualization(data)
                # Show message when agents die out, but continue playback
                # if not data["agent_states"] and self.playing:
                #     print(f"Agent states empty at step {step_number}")
        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to load step {step_number}: {str(e)}"
            )
            self.playing = False  # Stop playback on error

    def _update_visualization(self, data):
        """Update all visualization components with new data."""
        # Update statistics
        metrics = data["metrics"]
        for metric_name, value in metrics.items():
            if metric_name in self.stats_vars:
                formatted_value = (
                    f"{value:.1f}" if isinstance(value, float) else str(value)
                )
                self.stats_vars[metric_name]["var"].set(formatted_value)

        # Update environment view
        self._draw_environment(data["agent_states"], data["resource_states"])

        # Update charts
        self._update_charts()

    def _draw_environment(self, agent_states, resource_states):
        """Draw the current state of the environment with auto-scaling."""
        # Get canvas dimensions and create image
        width, height = self.canvas_size
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        # Calculate transformation parameters
        transform_params = self._calculate_transform_params(
            resource_states, width, height
        )

        # Track current agents for birth/death detection
        self._update_animation_states(agent_states)

        # Draw environment elements
        self._draw_resources(draw, resource_states, transform_params)
        self._draw_agents(draw, agent_states, transform_params)
        self._draw_birth_animations(draw, transform_params)
        self._draw_death_animations(draw, transform_params)
        self._draw_step_number(draw, transform_params)

        # Update canvas
        photo = ImageTk.PhotoImage(img)
        self.env_canvas.create_image(0, 0, image=photo, anchor="nw")
        self.env_canvas.image = photo  # Keep reference to prevent garbage collection

    def _calculate_transform_params(self, resource_states, width, height):
        """Calculate scaling and offset parameters for coordinate transformation."""
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])

        scale_x = (width - 2 * self.PADDING) / env_width
        scale_y = (height - 2 * self.PADDING) / env_height
        scale = min(scale_x, scale_y)

        offset_x = max(self.PADDING, (width - (env_width * scale)) / 2)
        offset_y = max(self.PADDING, (height - (env_height * scale)) / 2)

        return {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "padding": self.PADDING,
            "width": width,
            "height": height,
        }

    def _transform_coords(self, x, y, params):
        """Transform environment coordinates to screen coordinates."""
        return (
            params["offset_x"] + x * params["scale"],
            params["offset_y"] + y * params["scale"],
        )

    def _update_animation_states(self, agent_states):
        """Update birth and death animation states."""
        current_agent_ids = {agent[0] for agent in agent_states}

        # Check for new births
        new_births = current_agent_ids - self.previous_agent_ids
        for agent_id in new_births:
            agent_data = next(a for a in agent_states if a[0] == agent_id)
            pos = (agent_data[2], agent_data[3])
            self.birth_animations[agent_id] = (pos, 0)

        # Check for deaths
        deaths = self.previous_agent_ids - current_agent_ids
        for agent_id in deaths:
            if agent_data := next(
                (a for a in self.previous_agent_states if a[0] == agent_id), None
            ):
                pos = (agent_data[2], agent_data[3])
                self.death_animations[agent_id] = (pos, 0)

        self.previous_agent_ids = current_agent_ids
        self.previous_agent_states = agent_states

    def _draw_resources(self, draw, resource_states, params):
        """Draw resources as rounded squares with intensity-based coloring."""
        for resource in resource_states:
            amount = resource[1]
            if amount > 0:
                x, y = self._transform_coords(resource[2], resource[3], params)

                # Calculate color intensity and size
                intensity = amount / self.MAX_RESOURCE_AMOUNT
                resource_color = (
                    int(self.RESOURCE_GLOW_RED * intensity),
                    int(self.RESOURCE_GLOW_GREEN * intensity),
                    int(self.RESOURCE_GLOW_BLUE * intensity),
                )

                size = max(1, int(self.AGENT_RADIUS_SCALE * params["scale"]))
                radius = int(size * 0.2)

                self._draw_rounded_rectangle(draw, x, y, size, radius, resource_color)

    def _draw_rounded_rectangle(self, draw, x, y, size, radius, color):
        """Helper method to draw a rounded rectangle."""
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size

        # Main rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color)

        # Corner circles
        corners = [
            (x1, y1),  # Top-left
            (x2 - radius * 2, y1),  # Top-right
            (x1, y2 - radius * 2),  # Bottom-left
            (x2 - radius * 2, y2 - radius * 2),  # Bottom-right
        ]

        for corner_x, corner_y in corners:
            draw.ellipse(
                [corner_x, corner_y, corner_x + radius * 2, corner_y + radius * 2],
                fill=color,
            )

    def _draw_agents(self, draw, agent_states, params):
        """Draw agents as colored circles."""
        for agent in agent_states:
            x, y = self._transform_coords(agent[2], agent[3], params)
            if agent[1] == "SystemAgent":
                color = self.SYSTEM_AGENT_COLOR
            elif agent[1] == "IndependentAgent":
                color = self.INDEPENDENT_AGENT_COLOR
            else:  # ControlAgent
                color = self.CONTROL_AGENT_COLOR

            radius = max(1, int(self.AGENT_RADIUS_SCALE * params["scale"]))
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)], fill=color
            )

    def _draw_birth_animations(self, draw, params):
        """Draw expanding circle animations for new agents."""
        births_to_remove = []
        for agent_id, (pos, frame) in self.birth_animations.items():
            if frame < self.MAX_ANIMATION_FRAMES:
                x, y = self._transform_coords(pos[0], pos[1], params)
                radius = (
                    max(2, int(self.BIRTH_RADIUS_SCALE * params["scale"]))
                    * (frame + 1)
                    / self.MAX_ANIMATION_FRAMES
                )
                opacity = int(255 * (1 - frame / self.MAX_ANIMATION_FRAMES))
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=(*self.BIRTH_MARK_COLOR, opacity),
                )
                self.birth_animations[agent_id] = (pos, frame + 1)
            else:
                births_to_remove.append(agent_id)

        for agent_id in births_to_remove:
            del self.birth_animations[agent_id]

    def _draw_death_animations(self, draw, params):
        """Draw fading X mark animations for dying agents."""
        deaths_to_remove = []
        for agent_id, (pos, frame) in self.death_animations.items():
            if frame < self.MAX_ANIMATION_FRAMES:
                x, y = self._transform_coords(pos[0], pos[1], params)
                size = max(1, int(self.DEATH_MARK_SCALE * params["scale"]))
                opacity = int(128 * (1 - frame / self.MAX_ANIMATION_FRAMES))
                color = (*self.DEATH_MARK_COLOR, opacity)

                draw.line(
                    [(x - size, y - size), (x + size, y + size)], fill=color, width=1
                )
                draw.line(
                    [(x - size, y + size), (x + size, y - size)], fill=color, width=1
                )

                self.death_animations[agent_id] = (pos, frame + 1)
            else:
                deaths_to_remove.append(agent_id)

        for agent_id in deaths_to_remove:
            del self.death_animations[agent_id]

    def _draw_step_number(self, draw, params):
        """Draw the current step number on the visualization."""
        font_size = max(
            self.MIN_FONT_SIZE,
            int(min(params["width"], params["height"]) / self.FONT_SCALE_FACTOR),
        )
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.text(
            (self.PADDING, self.PADDING),
            f"Step: {self.current_step}",
            fill=self.BIRTH_MARK_COLOR,  # Using white color
            font=font,
        )

    def _update_charts(self):
        """Update the population and resource charts with historical data."""
        try:
            # Fetch historical data up to the current step
            history = self.db.get_historical_data()

            if not history["steps"]:
                return  # No data to plot

            # Convert to numpy arrays for better performance
            steps = np.array(history["steps"])
            system_agents = np.array(history["metrics"]["system_agents"])
            independent_agents = np.array(history["metrics"]["independent_agents"])
            control_agents = np.array(history["metrics"]["control_agents"])
            total_resources = np.array(history["metrics"]["total_resources"])

            # Update all lines with full data
            self.lines["system_agents"].set_data(steps, system_agents)
            self.lines["independent_agents"].set_data(steps, independent_agents)
            self.lines["control_agents"].set_data(steps, control_agents)
            self.lines["resources"].set_data(steps, total_resources)

            # Update current step line
            self.lines["current_step"].set_xdata([self.current_step, self.current_step])
            max_y = max(
                max(system_agents),
                max(independent_agents),
                max(control_agents),
                max(total_resources),
            )
            self.lines["current_step"].set_ydata([0, max_y])

            # Update axis limits with padding
            self.ax1.set_xlim(0, max(steps) + 10)
            self.ax1.set_ylim(
                0,
                max(max(system_agents), max(independent_agents), max(control_agents))
                * 1.1,
            )
            self.ax2.set_ylim(0, max(total_resources) * 1.1)

            # Redraw canvas
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to update charts: {str(e)}")

    def run(self):
        """Start the visualization."""
        # Make sure we're showing the first frame
        if self.current_step == 0:
            self._step_to(0)

    def close(self):
        """Clean up resources."""
        try:
            self.db.close()
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to close database connection: {str(e)}"
            )

    def _export_data(self):
        """Export simulation data to CSV."""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if filename:
                try:
                    self.db.export_data(filename)
                    messagebox.showinfo("Success", "Data exported successfully!")
                except Exception as e:
                    messagebox.showerror(
                        "Export Error", f"Failed to export data: {str(e)}"
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file dialog: {str(e)}")

    def _initialize_visualization(self):
        """Initialize the visualization with the first frame of data."""
        initial_data = self.db.get_simulation_data(0)
        if initial_data["agent_states"] or initial_data["resource_states"]:
            self._update_visualization(initial_data)
            # Force an update of the window
            self.root.update_idletasks()


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
