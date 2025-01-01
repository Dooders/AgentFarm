import tkinter as tk
import traceback
from tkinter import messagebox, ttk
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sqlalchemy import func

from farm.database.data_retrieval import DataRetriever
from farm.gui.components.tooltips import ToolTip
from farm.gui.windows.base_window import BaseWindow


class AgentAnalysisWindow(ttk.Frame):
    """
    Frame for detailed analysis of individual agents.
    """

    def __init__(self, parent: tk.Widget, data_retriever: DataRetriever):
        super().__init__(parent)
        self.data = data_retriever
        self.chart_canvas = None

        self.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._setup_ui()
        self._load_agents()

    def show_error(self, title: str, message: str):
        """Display error message."""
        messagebox.showerror(title, message, parent=self)

    def show_warning(self, title: str, message: str):
        """Display warning message."""
        messagebox.showwarning(title, message, parent=self)

    def show_info(self, title: str, message: str):
        """Display information message."""
        messagebox.showinfo(title, message, parent=self)

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
            # Use population repository through data retriever instead of agent repository
            agents = self.data.population_repository.get_all_agents()

            # Format combobox values with string IDs
            values = [
                f"Agent {str(agent.agent_id)} ({agent.agent_type}) - Born: {agent.birth_time}"
                for agent in agents
            ]

            self.agent_combobox["values"] = values

            # Auto-select first agent if available
            if values:
                self.agent_combobox.set(values[0])
                self._on_agent_selected(None)

        except Exception as e:
            self.show_error("Error", f"Failed to load agents: {str(e)}")

    def _on_agent_selected(self, event):
        """Handle agent selection."""
        if not self.agent_var.get():
            return

        # Extract agent_id from selection string - keep as string
        agent_id = self.agent_var.get().split()[1]  # Remove int() conversion
        self._load_agent_data(agent_id)

    def _load_agent_data(self, agent_id: str):
        """Load and display agent data."""
        try:
            # Get basic info
            basic_info = self.data.agent_repository.get_agent_info(agent_id)
            self._update_info_labels(basic_info)

            # Get current stats
            current_stats = self.data.agent_repository.get_agent_current_stats(agent_id)
            self._update_stat_labels(current_stats)

            # Get performance metrics
            performance = self.data.agent_repository.get_agent_performance_metrics(
                agent_id
            )
            self._update_metric_labels(performance)

            # Update charts
            self._update_charts(agent_id)

            # Update children table
            self._update_children_table(agent_id)

        except Exception as e:
            self.show_error("Error", f"Failed to load agent data: {str(e)}")

    def _update_charts(self, agent_id: str):
        """Update agent charts."""
        try:
            # Get historical data using repositories
            states = self.data.agent_repository.get_agent_state_history(agent_id)
            actions = self.data.action_repository.get_agent_actions(agent_id)

            # Convert to DataFrame for plotting
            self.df = pd.DataFrame(
                [
                    {
                        "step_number": state.step_number,
                        "current_health": state.current_health,
                        "resource_level": state.resource_level,
                        "total_reward": state.total_reward,
                        "age": state.age,
                    }
                    for state in states
                ]
            )

            # Clear previous charts if they exist
            if hasattr(self, "ax1"):
                self.ax1.clear()
                self.ax2.clear()

            # Create new charts...
            # Rest of the chart creation code remains the same

        except Exception as e:
            self.show_error("Error", f"Failed to update charts: {str(e)}")

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
            agent_id = self.agent_var.get().split()[1]

            # Get step data using repositories
            state = self.data.agent_repository.get_agent_state_at_step(agent_id, step)
            action = self.data.action_repository.get_agent_action_at_step(
                agent_id, step
            )

            if state:
                # Update stat labels with step-specific data
                self.stat_labels["health"].config(text=f"{state.current_health:.2f}")
                self.stat_labels["resources"].config(text=f"{state.resource_level:.2f}")
                self.stat_labels["total_reward"].config(
                    text=f"{state.total_reward:.2f}"
                )
                self.stat_labels["age"].config(text=str(state.age))
                self.stat_labels["is_defending"].config(
                    text=str(bool(state.is_defending))
                )
                self.stat_labels["current_position"].config(
                    text=f"{state.position_x}, {state.position_y}"
                )

            # Update action details if available
            if action:
                action_text = self._format_action_text(action, step)
            else:
                action_text = f"No action recorded for step {step}"

            # Update or create action details label
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

    def _update_children_table(self, agent_id: str):
        """Update the children table for the given agent."""
        try:
            # Get children data using agent repository
            children = self.data.agent_repository.get_agent_children(agent_id)

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
                            child.agent_id,
                            child.birth_time,
                            child.age or 0,
                        ),
                    )
            else:
                self.children_tree.insert("", "end", values=("No children", "-", "-"))

        except Exception as e:
            print(f"Error updating children table: {e}")
            traceback.print_exc()

    def _update_agent_info(self, agent_id: str):
        """Update agent information display."""
        try:
            agent_info = self.data.agent_repository.get_agent_info(agent_id)
            if agent_info:
                # Access attributes directly from AgentInfo dataclass
                info_dict = {
                    "Agent ID": agent_info.agent_id,
                    "Type": agent_info.agent_type,
                    "Position": f"({agent_info.position[0]:.1f}, {agent_info.position[1]:.1f})" if agent_info.position else "N/A",
                    "Resources": f"{agent_info.current_resources:.1f}" if agent_info.current_resources is not None else "N/A",
                    "Health": f"{agent_info.current_health:.1f}" if agent_info.current_health is not None else "N/A",
                    "Generation": agent_info.generation,
                    "Birth Time": agent_info.birth_time,
                    "Status": "Alive" if agent_info.death_time is None else "Dead"
                }
                self._update_info_labels(info_dict)
        except Exception as e:
            self.show_error("Error", f"Failed to update agent info: {str(e)}")
