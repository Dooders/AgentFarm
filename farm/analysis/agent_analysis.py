import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sqlalchemy import func

from farm.database.database import Agent, AgentAction, AgentState


class AgentAnalysis:
    """Agent analysis window for visualizing individual agent data and metrics.

    This class provides a detailed view of a single agent's information, statistics,
    and performance metrics through an interactive GUI window. It includes:

    Attributes:
        window (tk.Toplevel): Main window for the analysis display
        canvas (FigureCanvasTkAgg): Canvas for displaying matplotlib charts
        info_frame (ttk.LabelFrame): Frame containing basic agent information
        stats_frame (ttk.LabelFrame): Frame containing current agent statistics
        metrics_frame (ttk.LabelFrame): Frame containing performance metrics

    Features:
        - Basic information display (type, birth/death time, generation, etc.)
        - Current statistics (health, resources, position, etc.)
        - Performance metrics (survival time, peak values, total actions)
        - Time series visualization of health, resources, and rewards
        - Action distribution and reward analysis charts

    The window automatically loads and displays all data when initialized
    with an agent ID.
    """

    def __init__(self, parent, db_path):
        self.parent = parent
        self.db_path = db_path
        self.chart_canvas = None

        # Create and configure window
        self.window = tk.Toplevel(self.parent)
        self.window.title("Agent Analysis")
        self.window.geometry("1200x800")

        # Configure grid weights for responsive layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)

        self._setup_ui()
        self._load_agents()

    def _setup_ui(self):
        """Setup the main UI components with a grid layout."""
        # Agent Selection Area (Top)
        selection_frame = ttk.LabelFrame(
            self.window, text="Agent Selection", padding=10
        )
        selection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Configure agent selection
        ttk.Label(selection_frame, text="Select Agent:").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.agent_var = tk.StringVar()
        self.agent_combobox = ttk.Combobox(
            selection_frame, textvariable=self.agent_var, width=60
        )
        self.agent_combobox.pack(side=tk.LEFT)
        self.agent_combobox.bind("<<ComboboxSelected>>", self._on_agent_selected)

        # Main Content Area (Bottom)
        content_frame = ttk.Frame(self.window)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Adjust column weights to make left panel wider
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(1, weight=3)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left Side - Agent Information
        self._setup_info_panel(content_frame)

        # Right Side - Charts
        self._setup_charts_panel(content_frame)

    def _setup_info_panel(self, parent):
        """Setup the left panel containing agent information."""
        info_frame = ttk.LabelFrame(parent, text="Agent Information", padding=10)
        info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Basic Info Section
        self.basic_info = ttk.LabelFrame(info_frame, text="Basic Details", padding=5)
        self.basic_info.pack(fill=tk.X, pady=(0, 10))

        # Create labels for basic info with increased width
        self.info_labels: Dict[str, ttk.Label] = {}
        basic_fields = [
            "Type",
            "Birth Time",
            "Death Time",
            "Generation",
            "Parent ID",
            "Initial Resources",
            "Max Health",
            "Starvation Threshold",
            "Genome ID",
        ]
        for field in basic_fields:
            container = ttk.Frame(self.basic_info)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{field}:", width=25).pack(side=tk.LEFT)
            self.info_labels[field] = ttk.Label(container, text="-", width=30)
            self.info_labels[field].pack(side=tk.LEFT)

        # Current Stats Section with increased widths
        stats_frame = ttk.LabelFrame(info_frame, text="Current Statistics", padding=5)
        stats_frame.pack(fill=tk.X)

        self.stat_labels: Dict[str, ttk.Label] = {}
        stats = [
            "Health",
            "Resources",
            "Total Reward",
            "Age",
            "Is Defending",
            "Current Position",
        ]
        for stat in stats:
            container = ttk.Frame(stats_frame)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{stat}:", width=25).pack(side=tk.LEFT)
            self.stat_labels[stat] = ttk.Label(container, text="-", width=30)
            self.stat_labels[stat].pack(side=tk.LEFT)

        # Performance Metrics Section with increased widths
        metrics_frame = ttk.LabelFrame(
            info_frame, text="Performance Metrics", padding=5
        )
        metrics_frame.pack(fill=tk.X, pady=(10, 0))

        self.metric_labels: Dict[str, ttk.Label] = {}
        for metric in [
            "Survival Time",
            "Peak Health",
            "Peak Resources",
            "Total Actions",
        ]:
            container = ttk.Frame(metrics_frame)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{metric}:", width=25).pack(side=tk.LEFT)
            self.metric_labels[metric] = ttk.Label(container, text="-", width=30)
            self.metric_labels[metric].pack(side=tk.LEFT)

    def _setup_charts_panel(self, parent):
        """Setup the right panel containing charts."""
        charts_frame = ttk.LabelFrame(parent, text="Agent Analytics", padding=10)
        charts_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid for charts
        charts_frame.grid_columnconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(1, weight=1)

        # Create frames for different charts
        self.metrics_chart_frame = ttk.Frame(charts_frame)
        self.metrics_chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.actions_chart_frame = ttk.Frame(charts_frame)
        self.actions_chart_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _load_agents(self):
        """Load available agents from database."""
        try:
            def _query(session):
                agents = (
                    session.query(Agent.agent_id, Agent.agent_type, Agent.birth_time)
                    .order_by(Agent.birth_time.desc())
                    .all()
                )
                return agents
            
            agents = self.db._execute_in_transaction(_query)

            # Format combobox values to show more info
            self.agent_combobox["values"] = [
                f"Agent {agent.agent_id} ({agent.agent_type}) - Born: {agent.birth_time}"
                for agent in agents
            ]
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
        """Load and display all data for selected agent.

        Args:
            agent_id (int): The ID of the agent to load data for.

        Loads basic info, statistics, performance metrics and updates all charts
        and labels with the agent's data from the database.
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Load basic agent info
            basic_info = self._load_basic_info(conn, agent_id)
            self._update_info_labels(basic_info)

            # Load agent statistics
            stats = self._load_agent_stats(conn, agent_id)
            self._update_stat_labels(stats)

            # Load performance metrics
            metrics = self._load_performance_metrics(conn, agent_id)
            self._update_metric_labels(metrics)

            # Update charts
            self._update_metrics_chart(conn, agent_id)
            self._update_actions_chart(conn, agent_id)

            conn.close()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agent data: {e}")

    def _load_basic_info(self, conn, agent_id) -> Dict:
        """Load basic agent information using SQLAlchemy."""
        def _query(session):
            agent = (
                session.query(Agent)
                .filter(Agent.agent_id == agent_id)
                .first()
            )
            
            if agent:
                return {
                    "agent_type": agent.agent_type,
                    "birth_time": agent.birth_time,
                    "death_time": agent.death_time,
                    "generation": agent.generation,
                    "initial_resources": agent.initial_resources,
                    "starting_health": agent.starting_health,
                    "starvation_threshold": agent.starvation_threshold,
                    "genome_id": agent.genome_id
                }
            return {}
            
        return self.db._execute_in_transaction(_query)

    def _load_agent_stats(self, conn, agent_id) -> Dict:
        """Load current agent statistics using SQLAlchemy."""
        def _query(session):
            # Get latest state for the agent
            latest_state = (
                session.query(
                    AgentState.current_health,
                    AgentState.resource_level,
                    AgentState.total_reward,
                    (AgentState.step_number - Agent.birth_time).label('age'),
                    AgentState.is_defending,
                    AgentState.position_x,
                    AgentState.position_y
                )
                .join(Agent)
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number.desc())
                .first()
            )
            
            if latest_state:
                return {
                    "current_health": latest_state[0],
                    "resource_level": latest_state[1],
                    "total_reward": latest_state[2],
                    "age": latest_state[3],
                    "is_defending": latest_state[4],
                    "current_position": f"{latest_state[5]}, {latest_state[6]}"
                }
            
            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": 0,
                "current_position": "0, 0"
            }
            
        return self.db._execute_in_transaction(_query)

    def _load_performance_metrics(self, conn, agent_id) -> Dict:
        """Load agent performance metrics using SQLAlchemy."""
        def _query(session):
            # Calculate survival time, peak health, peak resources
            metrics = (
                session.query(
                    func.max(AgentState.step_number) - func.min(AgentState.step_number),
                    func.max(AgentState.current_health),
                    func.max(AgentState.resource_level),
                    func.count(func.distinct(AgentAction.action_id))
                )
                .select_from(AgentState)
                .outerjoin(
                    AgentAction,
                    (AgentAction.agent_id == AgentState.agent_id) & 
                    (AgentAction.step_number == AgentState.step_number)
                )
                .filter(AgentState.agent_id == agent_id)
                .group_by(AgentState.agent_id)
                .first()
            )
            
            if metrics:
                return {
                    "survival_time": metrics[0] or 0,
                    "peak_health": metrics[1] or 0,
                    "peak_resources": metrics[2] or 0,
                    "total_actions": metrics[3] or 0
                }
            
            return {
                "survival_time": 0,
                "peak_health": 0,
                "peak_resources": 0,
                "total_actions": 0
            }
            
        return self.db._execute_in_transaction(_query)

    def _update_metrics_chart(self, conn, agent_id):
        """Update the metrics chart with agent's time series data."""
        def _query(session):
            states = (
                session.query(
                    AgentState.step_number,
                    AgentState.current_health,
                    AgentState.resource_level,
                    AgentState.total_reward,
                    AgentState.is_defending
                )
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number)
                .all()
            )
            return states
        
        states = self.db._execute_in_transaction(_query)

        # Clear previous chart
        for widget in self.metrics_chart_frame.winfo_children():
            widget.destroy()

        if states:
            df = pd.DataFrame(states, columns=[
                "step_number", "current_health", "resource_level", 
                "total_reward", "is_defending"
            ])
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                df["step_number"],
                df["current_health"],
                label="Health",
                color="#2ecc71",
                linewidth=2,
            )
            ax.plot(
                df["step_number"],
                df["resource_level"],
                label="Resources",
                color="#3498db",
                linewidth=2,
            )
            ax.plot(
                df["step_number"],
                df["total_reward"],
                label="Reward",
                color="#e74c3c",
                linewidth=2,
            )

            # Add defense indicators
            defense_steps = df[df["is_defending"] == True]["step_number"]
            if not defense_steps.empty:
                ax.scatter(
                    defense_steps,
                    [0] * len(defense_steps),
                    marker="^",
                    color="red",
                    label="Defending",
                    alpha=0.5,
                )

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.set_title("Agent Metrics Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

            canvas = FigureCanvasTkAgg(fig, master=self.metrics_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_actions_chart(self, conn, agent_id):
        """Update the actions distribution and rewards charts."""
        def _query(session):
            actions = (
                session.query(
                    AgentAction.action_type,
                    func.count().label('count'),
                    func.avg(AgentAction.reward).label('avg_reward')
                )
                .filter(AgentAction.agent_id == agent_id)
                .group_by(AgentAction.action_type)
                .order_by(func.count().desc())
                .all()
            )
            return actions
        
        actions = self.db._execute_in_transaction(_query)

        # Clear previous chart
        for widget in self.actions_chart_frame.winfo_children():
            widget.destroy()

        if actions:
            df = pd.DataFrame(actions, columns=["action_type", "count", "avg_reward"])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Action counts
            bars = ax1.bar(df["action_type"], df["count"])
            ax1.set_xlabel("Action Type")
            ax1.set_ylabel("Count")
            ax1.set_title("Action Distribution")
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

            # Average rewards
            bars = ax2.bar(df["action_type"], df["avg_reward"])
            ax2.set_xlabel("Action Type")
            ax2.set_ylabel("Average Reward")
            ax2.set_title("Action Rewards")
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

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.actions_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_info_labels(self, info: Dict):
        """Update the basic information labels in the UI.

        Args:
            info (Dict): Dictionary containing basic agent information
        """
        for key, label in self.info_labels.items():
            value = info.get(key.lower().replace(" ", "_"), "-")
            label.config(text=str(value))

    def _update_stat_labels(self, stats: Dict):
        """Update the current statistics labels in the UI.

        Args:
            stats (Dict): Dictionary containing current agent statistics
        """
        for key, label in self.stat_labels.items():
            value = stats.get(key.lower().replace(" ", "_"), "-")
            label.config(
                text=f"{value:.2f}" if isinstance(value, float) else str(value)
            )

    def _update_metric_labels(self, metrics: Dict):
        """Update the performance metrics labels in the UI.

        Args:
            metrics (Dict): Dictionary containing agent performance metrics
        """
        for key, label in self.metric_labels.items():
            value = metrics.get(key.lower().replace(" ", "_"), "-")
            label.config(
                text=f"{value:.2f}" if isinstance(value, float) else str(value)
            )
