import tkinter as tk
from tkinter import ttk
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from farm.database.data_retrieval import DataRetriever
from farm.database.database import SimulationDatabase
from farm.gui.windows.base_window import BaseWindow


class StatisticsWindow(BaseWindow):
    """Window for displaying detailed simulation statistics."""

    def __init__(self, parent: tk.Tk, db_path: str):
        # Initialize database before calling super().__init__
        self.db_path = db_path
        self.db = SimulationDatabase(db_path)
        self.retriever = DataRetriever(self.db)

        # Initialize tooltips dictionary
        self.tooltips = {
            "Population Momentum": "Product of survival time and peak population",
            "Average Population": "Mean number of agents across all simulation steps",
            "Peak Population": "Maximum number of agents at any point",
            "Simulation Length": "Number of steps until last agent died",
            "Resource Utilization": "Proportion of available resources consumed",
            "Resources Consumed": "Total resources used by all agents",
            "Resources Available": "Total resources generated in simulation",
            "Utilization per Agent": "Average resources consumed per agent per step",
            "Population Variance": "Measure of population size fluctuation",
            "Coefficient of Variation": "Standardized measure of population stability",
        }

        # Now call parent's __init__
        super().__init__(parent, title="Simulation Statistics", size=(800, 600))

    def _setup_ui(self):
        """Setup the statistics window UI."""
        # Create notebook for different stat views
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        self._setup_summary_tab(summary_frame)

        # Population tab
        population_frame = ttk.Frame(self.notebook)
        self.notebook.add(population_frame, text="Population")
        self._setup_population_tab(population_frame)

        # Resources tab
        resources_frame = ttk.Frame(self.notebook)
        self.notebook.add(resources_frame, text="Resources")
        self._setup_resources_tab(resources_frame)

    def _setup_summary_tab(self, parent):
        """Setup the summary statistics tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get comprehensive statistics using DataRetriever
        population_stats = self.retriever.get_population_statistics()
        resource_stats = self.retriever.get_resource_statistics()
        learning_stats = self.retriever.get_learning_statistics()
        advanced_stats = self.retriever.get_advanced_statistics()

        # Population Dynamics section
        self._add_stat_section(scrollable_frame, "Population Dynamics")
        self._add_stat_row(
            "Average Population",
            f"{population_stats['basic_stats']['average_population']:,.2f}",
        )
        self._add_stat_row(
            "Peak Population",
            f"{population_stats['basic_stats']['peak_population']:,d}",
        )
        self._add_stat_row(
            "Population Stability",
            f"{advanced_stats['survival_metrics']['population_stability']:.2%}",
        )

        # Resource Metrics section
        self._add_stat_section(scrollable_frame, "Resource Metrics")
        self._add_stat_row(
            "Resource Utilization",
            f"{resource_stats['efficiency_metrics']['average_efficiency']:.2%}",
        )
        self._add_stat_row(
            "Resources per Agent",
            f"{population_stats['resource_metrics']['utilization_per_agent']:.2f}",
        )

        # Agent Distribution section
        self._add_stat_section(scrollable_frame, "Agent Distribution")
        for agent_type, ratio in advanced_stats["agent_distribution"].items():
            self._add_stat_row(
                f"{agent_type.replace('_', ' ').title()}", f"{ratio:.2%}"
            )

        # Update tooltips dictionary with new metrics
        self.tooltips.update(
            {
                "Peak-to-End Ratio": "Ratio of peak population to final population size",
                "Growth Rate": "Average change in population size per step",
                "Extinction Threshold": "Step when population fell below 10% of peak",
                "Average Health": "Mean health level across all agents over time",
                "Survivor Ratio": "Proportion of created agents that survived to the end",
                "Agent Diversity": "Shannon entropy of agent type distribution",
                "Interaction Rate": "Average interactions per agent per step",
                "Conflict/Cooperation Ratio": "Ratio of hostile to friendly interactions",
                "Scarcity Index": "Proportion of time resources were scarce",
            }
        )

        # Add tooltips for each metric
        self._add_tooltips(scrollable_frame)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

    def _add_tooltips(self, frame):
        """Add tooltips explaining each metric."""
        tooltips = {
            "Population Momentum": "Product of survival time and peak population",
            "Average Population": "Mean number of agents across all simulation steps",
            "Peak Population": "Maximum number of agents at any point",
            "Simulation Length": "Number of steps until last agent died",
            "Resource Utilization": "Proportion of available resources consumed",
            "Resources Consumed": "Total resources used by all agents",
            "Resources Available": "Total resources generated in simulation",
            "Utilization per Agent": "Average resources consumed per agent per step",
            "Population Variance": "Measure of population size fluctuation",
            "Coefficient of Variation": "Standardized measure of population stability",
            "Peak-to-End Ratio": "Ratio of peak population to final population size",
            "Growth Rate": "Average change in population size per step",
            "Extinction Threshold": "Step when population fell below 10% of peak",
            "Average Health": "Mean health level across all agents over time",
            "Survivor Ratio": "Proportion of created agents that survived to the end",
            "Agent Diversity": "Shannon entropy of agent type distribution",
            "Interaction Rate": "Average interactions per agent per step",
            "Conflict/Cooperation Ratio": "Ratio of hostile to friendly interactions",
            "Scarcity Index": "Proportion of time resources were scarce",
        }

        def _find_labels(widget):
            """Recursively find all Label widgets."""
            labels = []
            if isinstance(widget, tk.Label) or isinstance(widget, ttk.Label):
                labels.append(widget)
            for child in widget.winfo_children():
                labels.extend(_find_labels(child))
            return labels

        # Find all labels recursively
        for label in _find_labels(frame):
            label_text = label.cget("text").replace(":", "")
            if label_text in tooltips:
                from gui.components.tooltips import ToolTip

                ToolTip(label, tooltips[label_text])

    def _add_stat_section(self, parent, title: str):
        """Add a new section header."""
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill="x", padx=5, pady=5)
        self.current_section = section_frame

    def _add_stat_row(self, label: str, value: str):
        """Add a row of statistics."""
        row = ttk.Frame(self.current_section)
        row.pack(fill="x", pady=2)

        ttk.Label(row, text=label + ":", width=30, anchor="w").pack(side="left")
        ttk.Label(row, text=value).pack(side="left", padx=10)

    def _setup_population_tab(self, parent):
        """Setup the population statistics tab."""
        # Create figure and canvas
        fig = plt.figure(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=parent)

        # Get data using DataRetriever
        population_stats = self.retriever.get_population_statistics()
        data = population_stats["population_over_time"]

        # Plot population trends
        ax = fig.add_subplot(111)
        ax.plot(data["steps"], data["total"], label="Total")
        ax.plot(data["steps"], data["system"], label="System")
        ax.plot(data["steps"], data["independent"], label="Independent")

        ax.set_xlabel("Step")
        ax.set_ylabel("Population")
        ax.legend()

        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_resources_tab(self, parent):
        """Setup the resource statistics tab."""
        # Create figure and canvas
        fig = plt.figure(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=parent)

        def _get_data(session):
            # Use SQLAlchemy to get resource data
            return self.db.get_historical_data()

        data = self.db._execute_in_transaction(_get_data)

        # Plot resource trends
        ax = fig.add_subplot(111)
        ax.plot(
            data["steps"], data["metrics"]["total_resources"], label="Total Resources"
        )
        ax.plot(
            data["steps"],
            data["metrics"]["average_agent_resources"],
            label="Avg Resources/Agent",
        )

        ax.set_xlabel("Step")
        ax.set_ylabel("Resources")
        ax.legend()

        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
