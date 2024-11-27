import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import replace
import json
import os.path

from gui.components.charts import SimulationChart
from gui.components.controls import ControlPanel
from gui.components.environment import EnvironmentView
from gui.components.stats import StatsPanel
from gui.components.tooltips import ToolTip
from gui.utils.styles import configure_ttk_styles
from gui.windows.agent_analysis_window import AgentAnalysisWindow
from gui.components.notes import NotesPanel
from gui.components.chat_assistant import ChatAssistant

from config import SimulationConfig
from database import SimulationDatabase
from simulation import run_simulation

import logging


class SimulationGUI:
    """Main GUI application for running and visualizing agent-based simulations."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Agent-Based Simulation")
        
        # Maximize the window
        self.root.state('zoomed')  # For Windows
        # Alternative for Linux/Mac:
        # self.root.attributes('-zoomed', True)  # For Linux
        # self.root.attributes('-fullscreen', True)  # For Mac
        
        # Initialize variables
        self.current_db_path = None
        self.current_step = 0
        self.components = {}
        self.playback_timer = None
        self.last_config_path = "simulations/last_config.json"
        
        # Configure styles
        self._configure_styles()
        configure_ttk_styles()

        # Setup main components
        self._setup_menu()
        self._setup_main_frame()
        self._show_welcome_screen()

    def _configure_styles(self):
        """Configure custom styles for the application."""
        style = ttk.Style()
        
        # Configure Notebook (Tab) styles
        style.configure(
            "Custom.TNotebook", 
            background="#f0f0f0",    # Light gray background
            borderwidth=0,           # Remove border
            padding=5                # Add some padding
        )
        
        # Configure tab styles
        style.configure(
            "Custom.TNotebook.Tab",
            padding=(15, 8),         # Wider tabs with more vertical padding
            font=("Arial", 10, "bold")
        )
        
        # Map colors for different tab states
        style.map("Custom.TNotebook.Tab",
            background=[
                ("selected", "#c2e6f7"),     # Selected simulation tab
                ("!selected", "#e8f4f9")     # Unselected simulation tab
            ],
            foreground=[
                ("selected", "#1a5276"),     # Selected text color
                ("!selected", "#2c3e50")     # Unselected text color
            ],
            expand=[("selected", (0, 0, 0, 2))]
        )
        
        # Configure frame styles for tab content
        style.configure(
            "TabContent.TFrame",
            background="#ffffff",    # White background
            relief="flat"           # No border
        )

    def _setup_main_frame(self) -> None:
        """Setup the main container frame."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=2)  # Left pane
        self.main_frame.grid_columnconfigure(1, weight=3)  # Right pane
        self.main_frame.grid_rowconfigure(0, weight=1)

    def _load_last_config(self) -> dict:
        """Load the last used configuration if available."""
        try:
            if os.path.exists(self.last_config_path):
                with open(self.last_config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load last config: {e}")
        return {}

    def _save_last_config(self) -> None:
        """Save the current configuration."""
        try:
            os.makedirs(os.path.dirname(self.last_config_path), exist_ok=True)
            config = {key: var.get() for key, var in self.config_vars.items()}
            with open(self.last_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save config: {e}")

    def _show_welcome_screen(self):
        """Show the welcome screen with configuration options."""
        # Clear existing components
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create welcome frame
        welcome_frame = ttk.Frame(self.main_frame)
        welcome_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        welcome_frame.grid_columnconfigure(0, weight=1)

        # Quick action buttons at top left
        button_frame = ttk.Frame(welcome_frame)
        button_frame.grid(row=0, column=0, sticky="w", padx=20, pady=20)

        # Create a custom style for welcome screen buttons
        style = ttk.Style()
        style.configure(
            "Welcome.TButton",
            padding=(20, 10),  # Wider horizontal padding
            font=("Arial", 11),  # Slightly larger font
        )

        new_sim_btn = ttk.Button(
            button_frame,
            text="New Simulation",
            command=self._new_simulation,
            style="Welcome.TButton"
        )
        new_sim_btn.pack(side=tk.LEFT, padx=10)  # Increased spacing between buttons
        ToolTip(new_sim_btn, "Start a new simulation with custom parameters")

        open_sim_btn = ttk.Button(
            button_frame,
            text="Open Simulation",
            command=self._open_simulation,
            style="Welcome.TButton"
        )
        open_sim_btn.pack(side=tk.LEFT, padx=10)
        ToolTip(open_sim_btn, "Load and analyze an existing simulation")

        # Load default configuration
        try:
            config = SimulationConfig.from_yaml("config.yaml")
            # Load last used config
            last_config = self._load_last_config()
        except Exception as e:
            self.show_error("Configuration Error", f"Failed to load configuration: {str(e)}")
            config = SimulationConfig()
            last_config = {}

        # Configuration section
        config_frame = ttk.LabelFrame(
            welcome_frame,
            text="Simulation Configuration",
            padding=15,
            style="Config.TLabelframe"
        )
        config_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        # Create sections for different configuration categories
        sections = {
            "Environment Settings": [
                ("Environment Width", "width", config.width, "Width of the simulation environment"),
                ("Environment Height", "height", config.height, "Height of the simulation environment"),
                ("Initial Resources", "initial_resources", config.initial_resources, "Starting amount of resources"),
                ("Resource Regen Rate", "resource_regen_rate", config.resource_regen_rate, "Rate at which resources regenerate"),
                ("Max Resource Amount", "max_resource_amount", config.max_resource_amount, "Maximum resources per cell"),
            ],
            "Agent Population": [
                ("System Agents", "system_agents", config.system_agents, "Number of system-controlled agents"),
                ("Independent Agents", "independent_agents", config.independent_agents, "Number of independently-controlled agents"),
                ("Control Agents", "control_agents", config.control_agents, "Number of control group agents"),
                ("Max Population", "max_population", config.max_population, "Maximum total agent population"),
            ],
            "Simulation Parameters": [
                ("Simulation Steps", "simulation_steps", config.simulation_steps, "Number of steps to run the simulation"),
                ("Base Consumption Rate", "base_consumption_rate", config.base_consumption_rate, "Rate at which agents consume resources"),
                ("Max Movement", "max_movement", config.max_movement, "Maximum distance agents can move per step"),
                ("Gathering Range", "gathering_range", config.gathering_range, "Range at which agents can gather resources"),
            ]
        }

        # Create three columns for different sections
        column_frames = []
        for i in range(3):
            frame = ttk.Frame(config_frame)
            frame.grid(row=0, column=i, sticky="nsew", padx=10)
            frame.grid_columnconfigure(0, weight=1)
            column_frames.append(frame)

        # Initialize config_vars dictionary
        self.config_vars = {}

        # Distribute sections across columns
        for i, (section_name, fields) in enumerate(sections.items()):
            section_frame = ttk.LabelFrame(
                column_frames[i],
                text=section_name,
                padding=10,
                style="ConfigSection.TLabelframe"
            )
            section_frame.pack(fill="x", expand=True)

            # Add fields to section
            for label, key, default, tooltip in fields:
                container = ttk.Frame(section_frame)
                container.pack(fill="x", pady=4)
                
                # Label
                ttk.Label(
                    container,
                    text=f"{label}:",
                    style="ConfigLabel.TLabel"
                ).pack(side=tk.LEFT, padx=(0, 5))
                
                # Use last config value if available, otherwise use default
                value = last_config.get(key, str(default))
                var = tk.StringVar(value=str(value))
                entry = ttk.Entry(
                    container,
                    textvariable=var,
                    width=12,
                    style="Config.TEntry"
                )
                entry.pack(side=tk.RIGHT)
                self.config_vars[key] = var
                
                # Add tooltip with specific description
                ToolTip(entry, tooltip)

        # Welcome message below configuration
        welcome_text = (
            "\nWelcome to Agent-Based Simulation\n\n"
            "Configure simulation parameters above and use the buttons to:\n"
            "• Start a new simulation\n"
            "• Open an existing simulation"
        )
        
        welcome_label = ttk.Label(
            welcome_frame,
            text=welcome_text,
            justify=tk.CENTER,
            font=("Arial", 12)
        )
        welcome_label.grid(row=2, column=0, pady=20)

    def _setup_simulation_view(self):
        """Setup the simulation visualization components."""
        # Clear existing components
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create notebook for tabs with custom style
        self.notebook = ttk.Notebook(
            self.main_frame, 
            style="Custom.TNotebook"
        )
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create main simulation tab with styled frame
        sim_tab = ttk.Frame(
            self.notebook, 
            style="TabContent.TFrame",
            padding=10
        )
        self.notebook.add(
            sim_tab, 
            text="Simulation View"
        )

        # Create agent analysis tab with styled frame
        agent_tab = ttk.Frame(
            self.notebook, 
            style="TabContent.TFrame",
            padding=10
        )
        self.notebook.add(
            agent_tab, 
            text="Agent Analysis"
        )

        # Create notes tab
        notes_tab = ttk.Frame(
            self.notebook,
            style="TabContent.TFrame",
            padding=10
        )
        self.notebook.add(
            notes_tab,
            text="Notes & Observations"
        )
        
        # Add notes panel
        self.components["notes"] = NotesPanel(notes_tab)
        self.components["notes"].pack(fill="both", expand=True)

        # Create chat assistant tab
        chat_tab = ttk.Frame(
            self.notebook,
            style="TabContent.TFrame",
            padding=10
        )
        self.notebook.add(
            chat_tab,
            text="AI Assistant"
        )
        
        # Add chat assistant
        self.components["chat"] = ChatAssistant(chat_tab)
        self.components["chat"].pack(fill="both", expand=True)

        # After adding tabs, configure their colors using tag_configure
        self.notebook.configure(style="Custom.TNotebook")
        
        # Setup simulation components in sim_tab
        # Create left and right panes
        left_pane = ttk.Frame(sim_tab, style="SimPane.TFrame")
        right_pane = ttk.Frame(sim_tab, style="SimPane.TFrame")
        
        left_pane.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_pane.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Configure weights for resizing
        sim_tab.grid_columnconfigure(0, weight=2)  # Left pane
        sim_tab.grid_columnconfigure(1, weight=3)  # Right pane
        sim_tab.grid_rowconfigure(0, weight=1)

        # Left pane components
        self.components["stats"] = StatsPanel(left_pane)
        self.components["stats"].pack(fill="both", expand=True, padx=5, pady=5)
        
        self.components["environment"] = EnvironmentView(left_pane)
        self.components["environment"].pack(fill="both", expand=True, padx=5, pady=5)

        # Right pane - Chart
        self.components["chart"] = SimulationChart(right_pane)
        self.components["chart"].pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom controls - spans both panes
        controls_frame = ttk.Frame(sim_tab, style="Controls.TFrame")
        controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        self.components["controls"] = ControlPanel(
            controls_frame,
            play_callback=self._toggle_playback,
            step_callback=self._step_to,
            export_callback=self._export_data
        )
        self.components["controls"].pack(fill="x", expand=True)

        # Setup agent analysis in agent_tab
        self.components["agent_analysis"] = AgentAnalysisWindow(agent_tab, self.current_db_path)
        self.components["agent_analysis"].pack(fill="both", expand=True)

        # Configure component frames
        for component in self.components.values():
            if isinstance(component, (StatsPanel, EnvironmentView, SimulationChart)):
                component.configure(relief="solid", borderwidth=1)

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Connect environment view to agent analysis
        def on_agent_selected(agent_id):
            if agent_id is not None:
                # Switch to agent analysis tab
                self.notebook.select(1)  # Select agent analysis tab
                
                # Find the agent in the combobox
                for i, value in enumerate(self.components["agent_analysis"].agent_combobox["values"]):
                    if f"Agent {agent_id}" in value:
                        self.components["agent_analysis"].agent_combobox.current(i)
                        self.components["agent_analysis"]._on_agent_selected(None)
                        break

        self.components["environment"].set_agent_selected_callback(on_agent_selected)

    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        if tab_text == "Agent Analysis":
            # Pause simulation if it's playing
            if self.components["controls"].playing:
                self.components["controls"].set_playing(False)

    def _new_simulation(self) -> None:
        """Start a new simulation with current configuration."""
        try:
            # Close any existing database connections
            if hasattr(self, 'db'):
                try:
                    self.db.close()
                    delattr(self, 'db')
                except Exception:
                    pass
            
            # Close database connections in components
            for component in self.components.values():
                if hasattr(component, 'db'):
                    try:
                        component.db.close()
                    except Exception:
                        pass

            # Load base config to get default values
            base_config = SimulationConfig.from_yaml("config.yaml")
            
            # Create a dictionary of the updated values
            config_updates = {}
            for key, var in self.config_vars.items():
                try:
                    # Convert string values to appropriate types
                    value = var.get().strip()
                    if isinstance(getattr(base_config, key), float):
                        config_updates[key] = float(value)
                    else:
                        config_updates[key] = int(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {var.get()}")

            # Create new config by updating base config with new values
            config = replace(base_config, **config_updates)
            
            # Save the current configuration
            self._save_last_config()

            # Create new database
            self.current_db_path = "simulations/simulation.db"
            os.makedirs("simulations", exist_ok=True)
            
            # Remove existing database file if it exists
            if os.path.exists(self.current_db_path):
                try:
                    os.remove(self.current_db_path)
                except PermissionError:
                    raise Exception("Cannot overwrite existing simulation - database file is in use. Please restart the application.")

            # Show progress screen
            self._show_progress_screen("Running simulation...")

            # Run simulation in separate thread
            import threading
            sim_thread = threading.Thread(
                target=self._run_simulation,
                args=(config,)
            )
            sim_thread.start()

        except ValueError as e:
            self.show_error("Configuration Error", str(e))
        except Exception as e:
            self.show_error("Error", f"Failed to start simulation: {str(e)}")

    def _clear_progress_screen(self):
        """Clear the progress screen and all its components."""
        # First stop the progress bar if it exists
        if hasattr(self, 'progress_bar'):
            try:
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
            except Exception:
                pass
            delattr(self, 'progress_bar')
        
        # Remove the progress frame
        if hasattr(self, 'progress_frame'):
            try:
                self.progress_frame.grid_remove()
                self.progress_frame.destroy()
            except Exception:
                pass
            delattr(self, 'progress_frame')
        
        # Clear all widgets from main frame
        for widget in self.main_frame.winfo_children():
            try:
                widget.grid_remove()
                widget.destroy()
            except Exception:
                pass
        
        # Update the display
        self.main_frame.update()

    def _show_progress_screen(self, message: str):
        """Show progress screen while simulation is running."""
        # Clear existing screen
        self._clear_progress_screen()

        # Create progress frame
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_frame.grid_rowconfigure(0, weight=1)

        # Progress message
        ttk.Label(
            self.progress_frame,
            text=message,
            font=("Arial", 12)
        ).grid(row=0, column=0, pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode="indeterminate",
            length=300
        )
        self.progress_bar.grid(row=1, column=0, pady=10)
        self.progress_bar.start()

    def _setup_menu(self) -> None:
        """Create the application menu bar."""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(
            label="New Simulation",
            command=self._show_welcome_screen,
            accelerator="Ctrl+N"
        )
        file_menu.add_command(
            label="Open Simulation",
            command=self._open_simulation,
            accelerator="Ctrl+O"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Data",
            command=self._export_data,
            accelerator="Ctrl+E"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # Simulation Menu
        sim_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Run Batch", command=self._run_batch)
        sim_menu.add_command(label="Configure", command=self._configure_simulation)

        # Analysis Menu
        analysis_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)
        analysis_menu.add_command(label="View Statistics", command=self._view_statistics)
        analysis_menu.add_command(
            label="Agent Analysis",
            command=self._open_agent_analysis_window
        )

        # Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self._show_welcome_screen())
        self.root.bind("<Control-o>", lambda e: self._open_simulation())
        self.root.bind("<Control-e>", lambda e: self._export_data())

    def _run_simulation(self, config: SimulationConfig) -> None:
        """Run simulation in background thread."""
        try:
            run_simulation(
                num_steps=config.simulation_steps,
                config=config,
                db_path=self.current_db_path
            )
            self.root.after(0, self._simulation_complete)
        except Exception as e:
            self.root.after(0, self._simulation_error, str(e))

    def _start_visualization(self) -> None:
        """Initialize visualization components with simulation data."""
        if not self.current_db_path:
            return

        try:
            # Initialize database connection
            db = SimulationDatabase(self.current_db_path)
            
            # Get historical data for the chart
            historical_data = db.get_historical_data()
            
            # Get configuration from database
            config = db.get_configuration()
            
            # Store the full data in the chart but don't display it yet
            if historical_data and "metrics" in historical_data:
                logging.debug("Setting full data in chart")
                self.components["chart"].set_full_data({
                    "steps": historical_data["steps"],
                    "metrics": {
                        "system_agents": historical_data["metrics"]["system_agents"],
                        "independent_agents": historical_data["metrics"]["independent_agents"],
                        "control_agents": historical_data["metrics"]["control_agents"],
                        "total_resources": historical_data["metrics"]["total_resources"],
                    }
                })
            
            # Reset to initial state (step 0)
            initial_data = db.get_simulation_data(0)
            self.current_step = 0
            
            # Set up timeline interaction callbacks
            logging.debug("Setting up timeline callbacks")
            self.components["chart"].set_timeline_callback(self._step_to)
            self.components["chart"].set_playback_callback(
                lambda: self.components["controls"].set_playing(True)
            )
            # Add toggle callback for double-click behavior
            self.components["chart"].set_playback_toggle_callback(
                lambda: self.components["controls"].set_playing(
                    not self.components["controls"].playing
                )
            )
            
            # Update visualization components
            updatable_components = ["stats", "environment", "chart"]
            for name in updatable_components:
                if name in self.components and hasattr(self.components[name], "update"):
                    self.components[name].update(initial_data)

            # Update chat assistant with simulation data
            if "chat" in self.components:
                simulation_data = {
                    "config": config if config else {},
                    "metrics": historical_data.get("metrics", {})
                }
                self.components["chat"].set_simulation_data(simulation_data)

        except Exception as e:
            logging.error(f"Error starting visualization: {str(e)}", exc_info=True)
            self.show_error("Visualization Error", f"Failed to initialize visualization: {str(e)}")

    def _open_simulation(self) -> None:
        """Open existing simulation database."""
        filepath = filedialog.askopenfilename(
            title="Open Simulation",
            initialdir="simulations",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")]
        )
        if filepath:
            self.current_db_path = filepath
            self._setup_simulation_view()
            self._start_visualization()

    def _run_batch(self) -> None:
        """Run batch simulation."""
        messagebox.showinfo("Not Implemented", "Batch simulation not yet implemented.")

    def _configure_simulation(self) -> None:
        """Open configuration dialog."""
        messagebox.showinfo("Not Implemented", "Configuration dialog not yet implemented.")

    def _generate_report(self) -> None:
        """Generate analysis report."""
        messagebox.showinfo("Not Implemented", "Report generation not yet implemented.")

    def _view_statistics(self) -> None:
        """Show statistics window."""
        if not self.current_db_path:
            messagebox.showwarning("No Data", "Please open or run a simulation first.")
            return
        
        from gui.windows.statistics_window import StatisticsWindow
        stats_window = StatisticsWindow(self.root, self.current_db_path)
        stats_window.show()

    def _open_agent_analysis_window(self) -> None:
        """Switch to agent analysis tab."""
        if not self.current_db_path:
            messagebox.showwarning("No Data", "Please open or run a simulation first.")
            return
        
        # Switch to agent analysis tab
        self.notebook.select(1)  # Select second tab

    def _return_to_simulation(self) -> None:
        """Return to simulation view from agent analysis."""
        self._setup_simulation_view()
        self._start_visualization()

    def _show_documentation(self) -> None:
        """Show documentation window."""
        messagebox.showinfo("Not Implemented", "Documentation not yet implemented.")

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo("About", 
            "Agent-Based Simulation\n\n"
            "A tool for running and analyzing agent-based simulations."
        )

    def _on_exit(self) -> None:
        """Handle application exit."""
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.quit()

    def _export_data(self) -> None:
        """Export simulation data."""
        if not self.current_db_path:
            messagebox.showwarning("No Data", "Please run or open a simulation first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                db = SimulationDatabase(self.current_db_path)
                db.export_data(filepath)
                messagebox.showinfo("Success", "Data exported successfully!")
            except Exception as e:
                self.show_error("Export Error", f"Failed to export data: {str(e)}")

    def _toggle_playback(self, playing: bool) -> None:
        """Handle playback state change."""
        if playing:
            # Start playback
            self._play_simulation()
        else:
            # Stop playback
            self._stop_simulation()

    def _play_simulation(self):
        """Start simulation playback."""
        if not self.current_db_path:
            return
        
        try:
            # Cancel any existing timer
            if self.playback_timer:
                self.root.after_cancel(self.playback_timer)
                self.playback_timer = None

            db = SimulationDatabase(self.current_db_path)
            data = db.get_simulation_data(self.current_step + 1)
            
            # Check if we've reached the end of the data
            if not data or not data.get("metrics"):
                # Stop playback
                self.components["controls"].set_playing(False)
                return
            
            self.current_step += 1
            
            # Update each component with the data, except controls
            for name, component in self.components.items():
                if name != "controls" and hasattr(component, "update"):
                    try:
                        # Skip agent_analysis updates during playback
                        if name == "agent_analysis":
                            continue
                            
                        # Ensure data is passed as a dictionary
                        if not isinstance(data, dict):
                            logging.warning(f"Invalid data format for {name}: {type(data)}")
                            continue
                        component.update(data)
                    except Exception as comp_error:
                        logging.error(f"Error updating {name}: {str(comp_error)}")
                    
            # Schedule next update if still playing
            if self.components["controls"].playing:
                delay = self.components["controls"].get_delay()
                self.playback_timer = self.root.after(delay, self._play_simulation)
                
        except Exception as e:
            self.show_error("Playback Error", f"Failed to update simulation: {str(e)}")
            self.components["controls"].set_playing(False)

    def _stop_simulation(self):
        """Stop simulation playback."""
        if self.playback_timer:
            self.root.after_cancel(self.playback_timer)
            self.playback_timer = None

    def _step_to(self, step: int) -> None:
        """Move to specific simulation step."""
        if not self.current_db_path:
            return
            
        try:
            db = SimulationDatabase(self.current_db_path)
            
            # Ensure step is within valid range
            if step < 0:
                step = 0
            
            # Get max step from chart's full data
            max_step = len(self.components["chart"].full_data["steps"]) - 1
            if step > max_step:
                step = max_step
            
            data = db.get_simulation_data(step)
            
            if not isinstance(data, dict):
                raise ValueError(f"Invalid data format: expected dict, got {type(data)}")
            
            # Update current step
            self.current_step = step
            
            # Reset chart history to current step
            self.components["chart"].reset_history_to_step(step)
            
            # Update only visualization components that have an update method
            updatable_components = ["stats", "environment", "chart"]
            for name in updatable_components:
                if name in self.components and hasattr(self.components[name], "update"):
                    self.components[name].update(data)
                    
        except Exception as e:
            self.show_error("Navigation Error", f"Failed to move to step {step}: {str(e)}")

    def _simulation_complete(self) -> None:
        """Handle simulation completion."""
        try:
            # Close any existing database connections
            if hasattr(self, 'db') and self.db:
                self.db.close()
            
            # Clear the progress screen first
            self._clear_progress_screen()
            
            # Setup and start visualization
            self._setup_simulation_view()
            
            # Set simulation ID for notes after view is setup
            if "notes" in self.components:
                self.components["notes"].set_simulation(
                    os.path.basename(self.current_db_path)
                )
            
            self._start_visualization()
            
        except Exception as e:
            logging.error(f"Error during simulation completion: {str(e)}")
            self.show_error("Error", "Failed to complete simulation setup")

    def _simulation_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self.progress_bar.stop()
        self._show_welcome_screen()
        self.show_error("Simulation Error", error_msg)

    def show_error(self, title: str, message: str):
        """Display error message."""
        messagebox.showerror(title, message, parent=self.root)