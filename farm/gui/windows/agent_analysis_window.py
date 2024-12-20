import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
import traceback
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sqlalchemy import func

from database.database import Agent, AgentAction, AgentState, SimulationDatabase
from gui.components.tooltips import ToolTip
from database.data_retrieval import DataRetriever


class AgentAnalysisWindow(ttk.Frame):
    """
    Frame for detailed analysis of individual agents.
    """

    def __init__(self, parent: tk.Widget, db_path: str):
        super().__init__(parent)
        self.db_path = db_path
        self.chart_canvas = None
        self.db = SimulationDatabase(db_path)
        self.retriever = DataRetriever(self.db)

        self.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._setup_ui()
        self._load_agents()

    def _setup_ui(self):
        """Setup the main UI components with a grid layout."""
        # Main container with padding
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=1)

        # Agent Selection Area
        selection_frame = ttk.Frame(main_container)
        selection_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 10))
        selection_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(selection_frame, text="Select Agent:").grid(
            row=0, column=0, padx=(0, 10)
        )

        self.agent_var = tk.StringVar()
        self.agent_combobox = ttk.Combobox(
            selection_frame,
            textvariable=self.agent_var,
            style="AgentAnalysis.TCombobox",
        )
        self.agent_combobox.grid(row=0, column=1, sticky="ew")
        self.agent_combobox.bind("<<ComboboxSelected>>", self._on_agent_selected)

        # Main Content Area (Bottom) - Using PanedWindow for resizable sections
        content_frame = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        content_frame.grid(row=1, column=0, sticky="nsew")

        # Left Side - Agent Information (reduced width)
        info_frame = ttk.LabelFrame(
            content_frame,
            text="Agent Information",
            padding=10,
            style="AgentAnalysis.TLabelframe",
            width=400,  # Set fixed initial width
        )

        # Right Side - Charts
        charts_frame = ttk.LabelFrame(
            content_frame,
            text="Agent Analytics",
            padding=10,
            style="AgentAnalysis.TLabelframe",
        )

        # Add frames to PanedWindow with appropriate weights
        content_frame.add(info_frame, weight=1)  # Reduced weight for info panel
        content_frame.add(charts_frame, weight=4)  # Increased weight for charts

        # Set minimum size for info frame
        info_frame.grid_propagate(
            False
        )  # Prevent frame from shrinking below specified width
        info_frame.pack_propagate(False)

        # Setup info panel with scrolling
        info_canvas = tk.Canvas(info_frame)
        scrollbar = ttk.Scrollbar(
            info_frame, orient="vertical", command=info_canvas.yview
        )
        self.scrollable_info = ttk.Frame(info_canvas)

        info_canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        info_canvas.pack(side="left", fill="both", expand=True)

        # Create window in canvas
        canvas_frame = info_canvas.create_window(
            (0, 0), window=self.scrollable_info, anchor="nw"
        )

        # Configure scrolling
        def configure_scroll_region(event):
            info_canvas.configure(scrollregion=info_canvas.bbox("all"))

        def configure_canvas_width(event):
            info_canvas.itemconfig(canvas_frame, width=event.width)

        self.scrollable_info.bind("<Configure>", configure_scroll_region)
        info_canvas.bind("<Configure>", configure_canvas_width)

        # Setup info sections in scrollable frame
        self._setup_info_panel(self.scrollable_info)

        # Setup charts in a notebook for tabbed view
        self.charts_notebook = ttk.Notebook(charts_frame)
        self.charts_notebook.pack(fill="both", expand=True)

        # Metrics tab
        self.metrics_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.metrics_frame, text="Metrics Over Time")

        # Actions tab
        self.actions_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.actions_frame, text="Action Analysis")

    def _setup_info_panel(self, parent):
        """Setup the left panel containing agent information."""
        # Create a more compact layout for info sections
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill="x", expand=True, padx=5)

        # Basic Info Section with two columns
        basic_frame = ttk.LabelFrame(
            info_frame, text="Basic Information", padding=(5, 2)
        )
        basic_frame.pack(fill="x", pady=(0, 5))

        # Create two columns for basic info
        left_basic = ttk.Frame(basic_frame)
        right_basic = ttk.Frame(basic_frame)
        left_basic.pack(side=tk.LEFT, expand=True, fill="x", padx=5)
        right_basic.pack(side=tk.LEFT, expand=True, fill="x", padx=5)

        # Split basic fields between columns
        basic_fields_left = [
            ("Type", "type"),
            ("Birth Time", "birth_time"),
            ("Death Time", "death_time"),
            ("Generation", "generation"),
        ]

        basic_fields_right = [
            ("Initial Resources", "initial_resources"),
            ("Starting Health", "starting_health"),
            ("Starvation Threshold", "starvation_threshold"),
            ("Genome ID", "genome_id"),
        ]

        self.info_labels = {}

        # Create left column labels
        for label, key in basic_fields_left:
            container = ttk.Frame(left_basic)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.info_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.info_labels[key].pack(side=tk.LEFT)

        # Create right column labels
        for label, key in basic_fields_right:
            container = ttk.Frame(right_basic)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.info_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.info_labels[key].pack(side=tk.LEFT)

        # Current Stats Section
        stats_frame = ttk.LabelFrame(info_frame, text="Current Status", padding=(5, 2))
        stats_frame.pack(fill="x", pady=5)

        # Create two columns for stats
        left_stats = ttk.Frame(stats_frame)
        right_stats = ttk.Frame(stats_frame)
        left_stats.pack(side=tk.LEFT, expand=True, fill="x", padx=5)
        right_stats.pack(side=tk.LEFT, expand=True, fill="x", padx=5)

        # Split current stats between columns
        stats_left = [
            ("Health", "health"),
            ("Resources", "resources"),
            ("Total Reward", "total_reward"),
        ]

        stats_right = [
            ("Age", "age"),
            ("Is Defending", "is_defending"),
            ("Position", "current_position"),
        ]

        self.stat_labels = {}

        # Create left column stats
        for label, key in stats_left:
            container = ttk.Frame(left_stats)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.stat_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.stat_labels[key].pack(side=tk.LEFT)

        # Create right column stats
        for label, key in stats_right:
            container = ttk.Frame(right_stats)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.stat_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.stat_labels[key].pack(side=tk.LEFT)

        # Performance Metrics Section
        metrics_frame = ttk.LabelFrame(info_frame, text="Performance", padding=(5, 2))
        metrics_frame.pack(fill="x", pady=(5, 0))

        # Create two columns for metrics
        left_metrics = ttk.Frame(metrics_frame)
        right_metrics = ttk.Frame(metrics_frame)
        left_metrics.pack(side=tk.LEFT, expand=True, fill="x", padx=5)
        right_metrics.pack(side=tk.LEFT, expand=True, fill="x", padx=5)

        # Split metrics between columns
        metrics_left = [
            ("Survival Time", "survival_time"),
            ("Peak Health", "peak_health"),
        ]

        metrics_right = [
            ("Peak Resources", "peak_resources"),
            ("Total Actions", "total_actions"),
        ]

        self.metric_labels = {}

        # Create left column metrics
        for label, key in metrics_left:
            container = ttk.Frame(left_metrics)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.metric_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.metric_labels[key].pack(side=tk.LEFT)

        # Create right column metrics
        for label, key in metrics_right:
            container = ttk.Frame(right_metrics)
            container.pack(fill="x", pady=1)
            ttk.Label(
                container, text=f"{label}:", width=15, style="InfoLabel.TLabel"
            ).pack(side=tk.LEFT)
            self.metric_labels[key] = ttk.Label(
                container, text="-", width=12, style="InfoValue.TLabel"
            )
            self.metric_labels[key].pack(side=tk.LEFT)

        # Add Children Table Section
        children_frame = ttk.LabelFrame(info_frame, text="Children", padding=(5, 2))
        children_frame.pack(fill="x", pady=(5, 0))

        # Create Treeview for children
        self.children_tree = ttk.Treeview(
            children_frame,
            columns=("child_id", "birth_time", "age"),
            show="headings",
            height=5,  # Show 5 rows at a time
        )

        # Configure columns
        self.children_tree.heading("child_id", text="Child ID")
        self.children_tree.heading("birth_time", text="Birth")
        self.children_tree.heading("age", text="Age")

        # Configure column widths
        self.children_tree.column("child_id", width=90)
        self.children_tree.column("birth_time", width=90)
        self.children_tree.column("age", width=90)

        # Add scrollbar
        children_scrollbar = ttk.Scrollbar(
            children_frame, orient="vertical", command=self.children_tree.yview
        )
        self.children_tree.configure(yscrollcommand=children_scrollbar.set)

        # Pack tree and scrollbar
        self.children_tree.pack(side="left", fill="x", expand=True)
        children_scrollbar.pack(side="right", fill="y")

        # Add tooltip
        ToolTip(children_frame, "List of agent's children")

    def _load_agents(self):
        """Load available agents from database."""
        try:
            db = SimulationDatabase(self.db_path)

            def _query(session):
                # Use SQLAlchemy ORM query
                agents = (
                    session.query(Agent.agent_id, Agent.agent_type, Agent.birth_time)
                    .order_by(Agent.birth_time.desc())
                    .all()
                )

                # Format combobox values
                values = [
                    f"Agent {agent.agent_id} ({agent.agent_type}) - Born: {agent.birth_time}"
                    for agent in agents
                ]
                return values

            values = db._execute_in_transaction(_query)

            self.agent_combobox["values"] = values

            # Auto-select first agent if available
            if values:
                self.agent_combobox.set(values[0])
                # Trigger the selection event
                self._on_agent_selected(None)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agents: {e}")

    def _on_agent_selected(self, event):
        """Handle agent selection."""
        if not self.agent_var.get():
            return

        # Extract agent_id from selection string
        agent_id = int(self.agent_var.get().split()[1])
        self._load_agent_data(agent_id)

    def _load_agent_data(self, agent_id: int):
        """Load and display all data for selected agent."""
        try:
            # Get comprehensive agent data using DataRetriever
            agent_data = self.retriever.get_agent_data(agent_id)
            agent_actions = self.retriever.get_agent_actions(agent_id)
            agent_decisions = self.retriever.get_agent_decisions(agent_id)

            # Update info labels with basic info
            self._update_info_labels(agent_data["basic_info"])

            # Update current state
            self._update_stat_labels(agent_data["current_state"])

            # Update performance metrics
            self._update_metric_labels(agent_data["historical_metrics"])

            # Update metrics chart with time series data
            self._update_metrics_chart(agent_data, agent_actions)

            # Update children table
            self._update_children_table(agent_id)

            # Update action analysis
            self._update_action_analysis(agent_actions, agent_decisions)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agent data: {e}")

    def _load_basic_info(self, db: SimulationDatabase, agent_id: int) -> Dict:
        """Load basic agent information from database."""

        def _query(session):
            agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()

            if agent:
                return {
                    "type": agent.agent_type,
                    "birth_time": agent.birth_time,
                    "death_time": agent.death_time,
                    "generation": agent.generation,
                    "initial_resources": agent.initial_resources,
                    "starting_health": agent.starting_health,
                    "starvation_threshold": agent.starvation_threshold,
                    "genome_id": agent.genome_id,
                }
            return {}

        return db._execute_in_transaction(_query)

    def _load_agent_stats(self, db: SimulationDatabase, agent_id: int) -> Dict:
        """Load current agent statistics from database."""

        def _query(session):
            # Get latest state using window function
            from sqlalchemy import func

            latest_state = (
                session.query(AgentState)
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number.desc())
                .first()
            )

            if latest_state:
                return {
                    "current_health": latest_state.current_health,
                    "resource_level": latest_state.resource_level,
                    "total_reward": latest_state.total_reward,
                    "age": latest_state.age,
                    "is_defending": latest_state.is_defending,
                    "current_position": f"{latest_state.position_x}, {latest_state.position_y}",
                }

            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": False,
                "current_position": "0, 0",
            }

        return db._execute_in_transaction(_query)

    def _load_performance_metrics(self, db: SimulationDatabase, agent_id: int) -> Dict:
        """Load agent performance metrics from database."""

        def _query(session):
            from sqlalchemy import func

            metrics = (
                session.query(
                    (
                        func.max(AgentState.step_number)
                        - func.min(AgentState.step_number)
                    ).label("survival_time"),
                    func.max(AgentState.current_health).label("peak_health"),
                    func.max(AgentState.resource_level).label("peak_resources"),
                    func.count(AgentAction.action_id).label("total_actions"),
                )
                .select_from(AgentState)
                .outerjoin(
                    AgentAction,
                    (AgentAction.agent_id == AgentState.agent_id)
                    & (AgentAction.step_number == AgentState.step_number),
                )
                .filter(AgentState.agent_id == agent_id)
                .group_by(AgentState.agent_id)
                .first()
            )

            if metrics:
                return {
                    "survival_time": metrics.survival_time or 0,
                    "peak_health": metrics.peak_health or 0,
                    "peak_resources": metrics.peak_resources or 0,
                    "total_actions": metrics.total_actions or 0,
                }

            return {
                "survival_time": 0,
                "peak_health": 0,
                "peak_resources": 0,
                "total_actions": 0,
            }

        return db._execute_in_transaction(_query)

    def _update_metrics_chart(self, db: SimulationDatabase, agent_id):
        """Update the metrics chart with agent's time series data."""

        def _query(session):
            states = (
                session.query(
                    AgentState.step_number,
                    AgentState.current_health,
                    AgentState.resource_level,
                    AgentState.total_reward,
                    AgentState.is_defending,
                )
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number)
                .all()
            )

            return pd.DataFrame(
                [
                    {
                        "step_number": state.step_number,
                        "current_health": state.current_health,
                        "resource_level": state.resource_level,
                        "total_reward": state.total_reward,
                        "is_defending": state.is_defending,
                    }
                    for state in states
                ]
            )

        df = db._execute_in_transaction(_query)

        # Clear previous chart
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        if not df.empty:
            self._create_metrics_plot(df)

    def _create_metrics_plot(self, df):
        """Create the metrics plot."""
        # Store DataFrame for event handlers
        self.df = df

        # Clear previous plot
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        # Create figure with explicit size and spacing
        fig = plt.figure(figsize=(10, 6))

        # Create gridspec with specific ratios and spacing
        gs = fig.add_gridspec(
            2,
            2,  # 2 rows, 2 columns
            width_ratios=[4, 1],  # Main plot takes 80% width, legend takes 20%
            height_ratios=[3, 1],  # Top plot takes 75% height, bottom plot takes 25%
            hspace=0.1,  # Minimal horizontal spacing
            wspace=0.3,  # Space for legend
        )

        # Metrics plot (top)
        ax1 = fig.add_subplot(gs[0, 0])  # Top-left position

        # Plot metrics with improved styling
        lines = []
        lines.append(
            ax1.plot(
                df["step_number"],
                df["current_health"],
                label="Health",
                color="#2ecc71",
                linewidth=2,
            )[0]
        )
        lines.append(
            ax1.plot(
                df["step_number"],
                df["resource_level"],
                label="Resources",
                color="#3498db",
                linewidth=2,
            )[0]
        )
        lines.append(
            ax1.plot(
                df["step_number"],
                df["total_reward"],
                label="Reward",
                color="#e74c3c",
                linewidth=2,
            )[0]
        )

        # Add action indicators
        y_min = (
            ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02
        )  # 2% below bottom

        # Get steps for each action type
        defense_steps = df[df["is_defending"] == 1]["step_number"]

        # Query attack and reproduction steps
        action_data = self._get_action_data(
            df["step_number"].min(), df["step_number"].max()
        )
        attack_steps = action_data[action_data["action_type"] == "attack"][
            "step_number"
        ]
        reproduce_steps = action_data[action_data["action_type"] == "reproduce"][
            "step_number"
        ]

        # Add markers for each action type
        if not defense_steps.empty:
            lines.append(
                ax1.scatter(
                    defense_steps,
                    [y_min] * len(defense_steps),
                    marker="^",
                    color="red",
                    label="Defending",
                    alpha=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )

        if not attack_steps.empty:
            lines.append(
                ax1.scatter(
                    attack_steps,
                    [y_min] * len(attack_steps),
                    marker="^",
                    color="orange",
                    label="Attack",
                    alpha=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )

        if not reproduce_steps.empty:
            lines.append(
                ax1.scatter(
                    reproduce_steps,
                    [y_min] * len(reproduce_steps),
                    marker="^",
                    color="purple",
                    label="Reproduce",
                    alpha=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )

        # Adjust bottom margin to make room for markers
        ax1.set_ylim(bottom=y_min - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.01)

        # Improve plot styling
        ax1.set_xlabel("")
        ax1.set_ylabel("Value", fontsize=10)
        ax1.set_title("Agent Metrics Over Time", fontsize=12, pad=15)
        ax1.grid(True, alpha=0.3)

        # Create legend axis
        legend_ax = fig.add_subplot(gs[0, 1])  # Top-right position
        legend_ax.axis("off")
        legend_ax.legend(
            lines, [l.get_label() for l in lines], loc="center", frameon=True
        )

        # Population and Resources plot (bottom)
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left position
        ax2.sharex(ax1)

        # Get and plot population/resource data
        pop_data = self._get_population_data(
            df["step_number"].min(), df["step_number"].max()
        )
        self._plot_population_timeline(ax2, pop_data)

        # Style the population timeline
        ax2.set_xlabel("Time Step", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Set x-axis limits with padding
        x_min = df["step_number"].min()
        x_max = df["step_number"].max()
        x_padding = (x_max - x_min) * 0.02  # 2% padding

        ax1.set_xlim(x_min, x_max + x_padding)  # Add padding only to right side
        ax1.set_xticks([])  # Remove x-ticks from top plot

        # Ensure bottom plot shares the same x-axis limits
        ax2.set_xlim(x_min, x_max + x_padding)

        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(
            left=0.1,
            right=0.85,
            bottom=0.1,
            top=0.9,
            hspace=0.1,
        )

        # Add vertical line for current step
        self.current_step_line = ax1.axvline(
            x=df["step_number"].iloc[0], color="gray", linestyle="--", alpha=0.5
        )

        # Create canvas with all event connections
        canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
        canvas.mpl_connect("button_press_event", lambda e: self._on_click(e))
        canvas.mpl_connect("button_release_event", lambda e: self._on_release(e))
        canvas.mpl_connect("motion_notify_event", lambda e: self._on_drag(e))
        canvas.mpl_connect("key_press_event", lambda e: self._on_key(e))

        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Store axes for event handlers
        self.ax1 = ax1
        self.ax2 = ax2

    def _get_population_data(self, start_step, end_step):
        """Get total population and resource data for the timeline."""
        try:
            db = SimulationDatabase(self.db_path)

            def _query(session):
                from database.database import SimulationStep

                steps = (
                    session.query(
                        SimulationStep.step_number,
                        SimulationStep.total_agents,
                        SimulationStep.total_resources,
                    )
                    .filter(
                        SimulationStep.step_number >= start_step,
                        SimulationStep.step_number <= end_step,
                    )
                    .order_by(SimulationStep.step_number)
                    .all()
                )

                return pd.DataFrame(
                    [
                        {
                            "step_number": step.step_number,
                            "total_agents": step.total_agents,
                            "total_resources": step.total_resources,
                        }
                        for step in steps
                    ]
                )

            return db._execute_in_transaction(_query)

        except Exception as e:
            print(f"Error getting population data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _plot_population_timeline(self, ax, df):
        """Plot total population and resource metrics."""
        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Create two y-axes
        ax2 = ax.twinx()

        # Plot total population data on left axis
        l1 = ax.plot(
            df["step_number"],
            df["total_agents"],
            color="blue",
            label="Total Agents",
            alpha=0.7,
            linewidth=2,
        )

        # Plot resources on right axis
        l2 = ax2.plot(
            df["step_number"],
            df["total_resources"],
            color="green",
            label="Resources",
            alpha=0.7,
            linewidth=2,
        )

        # Add legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.15, 1.0))

        # Set labels
        ax.set_ylabel("Population", color="black")
        ax2.set_ylabel("Resources", color="green", rotation=270, labelpad=15)

        # Add padding to y-axes (10% padding)
        y1_min, y1_max = ax.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        y1_padding = (y1_max - y1_min) * 0.1
        y2_padding = (y2_max - y2_min) * 0.1

        ax.set_ylim(max(0, y1_min - y1_padding), y1_max + y1_padding)
        ax2.set_ylim(max(0, y2_min - y2_padding), y2_max + y2_padding)

        # Set colors for tick labels
        ax.tick_params(axis="y", labelcolor="black")
        ax2.tick_params(axis="y", labelcolor="green")

        # Add grid
        ax.grid(True, alpha=0.2)

        # Add synchronized vertical line
        self.population_step_line = ax.axvline(
            x=self.df["step_number"].iloc[0],  # Start at first step
            color="gray",
            linestyle="--",
            alpha=0.5,
        )

    def _get_action_data(self, start_step, end_step):
        """Get action data for the timeline."""
        try:
            db = SimulationDatabase(self.db_path)
            agent_id = int(self.agent_var.get().split()[1])

            def _query(session):
                actions = (
                    session.query(
                        AgentAction.step_number,
                        AgentAction.action_type,
                        AgentAction.reward,
                        AgentAction.action_id,
                        AgentAction.action_target_id,
                        AgentAction.resources_before,
                        AgentAction.resources_after,
                        AgentAction.details,
                        AgentAction.agent_id,
                    )
                    .filter(
                        AgentAction.agent_id == agent_id,
                        AgentAction.step_number >= start_step,
                        AgentAction.step_number <= end_step,
                    )
                    .order_by(AgentAction.step_number)
                    .all()
                )

                return pd.DataFrame(
                    [
                        {
                            "step_number": action.step_number,
                            "action_type": (
                                action.action_type.lower()
                                if action.action_type
                                else None
                            ),
                            "reward": action.reward,
                            "action_id": action.action_id,
                            "action_target_id": action.action_target_id,
                            "resources_before": action.resources_before,
                            "resources_after": action.resources_after,
                            "details": action.details,
                            "agent_id": action.agent_id,
                        }
                        for action in actions
                    ]
                )

            return db._execute_in_transaction(_query)

        except Exception as e:
            print(f"Error getting action data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _plot_action_timeline(self, ax, df):
        """Plot actions as evenly distributed slices in a horizontal bar."""
        # Define action colors with clear semantic meaning
        action_colors = {
            "move": "#3498db",  # Blue
            "gather": "#2ecc71",  # Green
            "attack": "#e74c3c",  # Red
            "defend": "#f39c12",  # Orange
            "reproduce": "#9b59b6",  # Purple
            "share": "#1abc9c",  # Turquoise
            "rest": "#95a5a6",  # Gray
        }

        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No actions recorded",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Get the full time range from the axis limits
        start_step = int(ax.get_xlim()[0])
        end_step = int(ax.get_xlim()[1])

        # Clear previous plot content
        ax.clear()

        # For each action, create a rectangle patch
        for _, row in df.iterrows():
            action_type = row["action_type"].lower()
            color = action_colors.get(action_type, "#808080")  # Default to gray

            # Create rectangle for this action
            rect = plt.Rectangle(
                (row["step_number"], 0),  # (x, y)
                1,  # width (1 time step)
                1,  # height
                facecolor=color,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Create legend elements for all possible actions
        legend_elements = []
        for action, color in action_colors.items():
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=action.capitalize())
            )

        # Create a separate legend axis to avoid tight_layout issues
        legend_ax = ax.figure.add_axes([0.85, 0.1, 0.15, 0.8])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_elements,
            loc="center",
            title="Actions",
            frameon=True,
            fontsize=9,
        )

        # Configure main axis
        ax.set_yticks([])
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(start_step, end_step)

        # Add grid
        ax.grid(True, axis="x", alpha=0.2)

        # Adjust the main axis to make room for legend
        ax.set_position([0.1, 0.1, 0.7, 0.8])

    def _adjust_color_for_reward(self, base_color, reward):
        """Adjust color based on reward value."""
        # Convert reward to a scale factor (-1 to 1 range)
        scale = min(max(reward, -1), 1)

        if scale > 0:
            # Positive reward: blend with white
            return tuple(int(c + (255 - c) * scale * 0.5) for c in base_color)
        else:
            # Negative reward: blend with black
            return tuple(int(c * (1 + scale * 0.5)) for c in base_color)

    def _update_info_labels(self, info: Dict):
        """Update the basic information labels."""
        for key, label in self.info_labels.items():
            value = info.get(key.lower().replace(" ", "_"), "-")
            label.config(text=str(value))

    def _update_stat_labels(self, stats: Dict):
        """Update the current statistics labels."""
        for key, label in self.stat_labels.items():
            value = stats.get(key.lower().replace(" ", "_"), "-")
            label.config(
                text=f"{value:.2f}" if isinstance(value, float) else str(value)
            )

    def _update_metric_labels(self, metrics: Dict):
        """Update the performance metrics labels."""
        for key, label in self.metric_labels.items():
            value = metrics.get(key.lower().replace(" ", "_"), "-")
            label.config(
                text=f"{value:.2f}" if isinstance(value, float) else str(value)
            )

    def _update_actions_chart(self, conn, agent_id):
        """Update the actions distribution chart."""
        query = """
            SELECT 
                action_type,
                COUNT(*) as count,
                AVG(CASE WHEN reward IS NOT NULL THEN reward ELSE 0 END) as avg_reward
            FROM AgentActions
            WHERE agent_id = ?
            GROUP BY action_type
            ORDER BY count DESC
        """
        df = pd.read_sql_query(query, conn, params=(agent_id,))

        # Clear previous chart
        for widget in self.actions_frame.winfo_children():
            widget.destroy()

        if not df.empty:
            self._create_actions_plot(df)

    def _create_actions_plot(self, df):
        """Create the actions distribution and rewards plot."""
        # Create figure with more explicit size and spacing
        fig = plt.figure(figsize=(10, 6))

        # Add GridSpec to have more control over subplot layout
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

        # Create subplots using GridSpec
        ax1 = fig.add_subplot(gs[0])  # Left plot
        ax2 = fig.add_subplot(gs[1])  # Right plot

        # Action counts with improved styling
        bars = ax1.bar(df["action_type"], df["count"], color="#3498db", alpha=0.8)
        ax1.set_xlabel("Action Type", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.set_title("Action Distribution", fontsize=12, pad=15)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # Average rewards with improved styling
        bars = ax2.bar(df["action_type"], df["avg_reward"], color="#2ecc71", alpha=0.8)
        ax2.set_xlabel("Action Type", fontsize=10)
        ax2.set_ylabel("Average Reward", fontsize=10)
        ax2.set_title("Action Rewards", fontsize=12, pad=15)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # Instead of tight_layout, use figure.subplots_adjust
        fig.subplots_adjust(
            left=0.1,  # Left margin
            right=0.9,  # Right margin
            bottom=0.15,  # Bottom margin
            top=0.9,  # Top margin
            wspace=0.3,  # Width spacing between subplots
        )

        # Create canvas and pack
        canvas = FigureCanvasTkAgg(fig, master=self.actions_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _update_step_info(self, step: int):
        """Update the agent information for a specific step."""
        try:
            conn = sqlite3.connect(self.db_path)
            agent_id = int(self.agent_var.get().split()[1])

            # Get basic state information
            query = """
                SELECT 
                    s.current_health,
                    s.resource_level,
                    s.total_reward,
                    s.age,
                    s.is_defending,
                    s.position_x || ', ' || s.position_y as current_position,
                    s.agent_id
                FROM AgentStates s
                WHERE s.agent_id = ? AND s.step_number = ?
            """
            df = pd.read_sql_query(query, conn, params=(agent_id, step))

            if not df.empty:
                state = df.iloc[0]

                # Update stat labels with step-specific data
                self.stat_labels["health"].config(text=f"{state['current_health']:.2f}")
                self.stat_labels["resources"].config(
                    text=f"{state['resource_level']:.2f}"
                )
                self.stat_labels["total_reward"].config(
                    text=f"{state['total_reward']:.2f}"
                )
                self.stat_labels["age"].config(text=str(state["age"]))
                self.stat_labels["is_defending"].config(
                    text=str(bool(state["is_defending"]))
                )
                self.stat_labels["current_position"].config(
                    text=state["current_position"]
                )

                # Get action details for this step
                db = SimulationDatabase(self.db_path)
                action_details = db.get_step_actions(agent_id, step)

                # Update action details in the UI
                if action_details:
                    action_text = f"""Step {step} Action Details:
Action Type: {action_details['action_type']}
Target ID: {action_details['action_target_id'] or 'None'}
Resources Before: {action_details['resources_before']:.2f}
Resources After: {action_details['resources_after']:.2f}
Reward: {action_details['reward']:.2f}
"""
                    if action_details["details"]:
                        import json

                        details = json.loads(action_details["details"])
                        action_text += "\nAdditional Details:\n"
                        for key, value in details.items():
                            action_text += f"{key}: {value}\n"
                else:
                    action_text = f"No action recorded for step {step}"

                # Create or update action details label
                if not hasattr(self, "action_details_label"):
                    self.action_details_label = ttk.Label(
                        self.scrollable_info,
                        text=action_text,
                        style="InfoValue.TLabel",
                        justify=tk.LEFT,
                    )
                    self.action_details_label.pack(fill="x", pady=(10, 0), padx=5)
                else:
                    self.action_details_label.config(text=action_text)

            conn.close()

        except Exception as e:
            print(f"Error updating step info: {e}")
            traceback.print_exc()

    def _on_click(self, event):
        """Handle mouse click on the chart."""
        if event.inaxes in [self.ax1, self.ax2]:
            try:
                self.is_dragging = True
                step = int(round(event.xdata))
                # Constrain step to valid range
                step = max(
                    self.df["step_number"].min(),
                    min(step, self.df["step_number"].max()),
                )
                self._update_step_info(step)
                # Update both vertical lines
                self.current_step_line.set_xdata([step, step])
                self.population_step_line.set_xdata([step, step])
                event.canvas.draw()
            except Exception as e:
                print(f"Error in click handler: {e}")

    def _on_release(self, event):
        """Handle mouse release."""
        self.is_dragging = False

    def _on_drag(self, event):
        """Handle mouse drag on the chart."""
        if hasattr(self, "is_dragging") and self.is_dragging and event.inaxes:
            try:
                step = int(round(event.xdata))
                # Constrain step to valid range
                if hasattr(self, "df"):
                    step = max(
                        self.df["step_number"].min(),
                        min(step, self.df["step_number"].max()),
                    )
                    self._update_step_info(step)
                    # Update both vertical lines
                    self.current_step_line.set_xdata([step, step])
                    self.population_step_line.set_xdata([step, step])
                    event.canvas.draw()
            except Exception as e:
                print(f"Error in drag handler: {e}")

    def _on_key(self, event):
        """Handle keyboard navigation."""
        if event.inaxes:  # Only if mouse is over the plot
            try:
                current_x = self.current_step_line.get_xdata()[0]
                step = int(current_x)

                # Handle left/right arrow keys
                if event.key == "left":
                    step = max(self.df["step_number"].min(), step - 1)
                elif event.key == "right":
                    step = min(self.df["step_number"].max(), step + 1)
                else:
                    return

                # Update line positions and info
                self._update_step_info(step)
                self.current_step_line.set_xdata([step, step])
                self.population_step_line.set_xdata([step, step])
                event.canvas.draw()
            except Exception as e:
                print(f"Error in key handler: {e}")

    def _update_children_table(self, agent_id: int):
        """Update the children table for the given agent."""
        try:
            db = SimulationDatabase(self.db_path)

            def _query(session):
                from sqlalchemy import case

                children = (
                    session.query(
                        Agent.agent_id.label("child_id"),
                        Agent.birth_time,
                        case(
                            (Agent.death_time.is_(None), func.max(AgentState.age)),
                            else_=Agent.death_time - Agent.birth_time,
                        ).label("age"),
                    )
                    .outerjoin(AgentState)
                    .group_by(Agent.agent_id)
                    .order_by(Agent.birth_time.desc())
                    .all()
                )

                return children

            children = db._execute_in_transaction(_query)

            # Clear existing items
            for item in self.children_tree.get_children():
                self.children_tree.delete(item)

            # Add new data
            if children:
                for child in children:
                    self.children_tree.insert(
                        "",
                        "end",
                        values=(
                            child.child_id,
                            child.birth_time,
                            child.age or 0,  # Handle None case
                        ),
                    )
            else:
                self.children_tree.insert("", "end", values=("No children", "-", "-"))

        except Exception as e:
            print(f"Error updating children table: {e}")
            traceback.print_exc()

    def _update_agent_info(self, agent_id: int):
        """Update agent information display."""
        try:
            # Get agent data using DataRetriever
            data = self.retriever.get_simulation_data(self.current_step)
            agent_states = [s for s in data["agent_states"] if s[0] == agent_id]

            if agent_states:
                state = agent_states[0]
                self._update_info_labels(
                    {
                        "Agent ID": state[0],
                        "Type": state[1],
                        "Position": f"({state[2]:.1f}, {state[3]:.1f})",
                        "Resources": f"{state[4]:.1f}",
                        "Health": f"{state[5]:.1f}",
                    }
                )
        except Exception as e:
            self.show_error("Error", f"Failed to update agent info: {str(e)}")
