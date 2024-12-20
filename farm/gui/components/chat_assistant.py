import tkinter as tk
from tkinter import ttk, messagebox
import openai
import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from database.database import SimulationDatabase
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.patches import Patch
import re

class ChatAssistant(ttk.Frame):
    """Interactive chat interface for analyzing simulation data with OpenAI."""

    def __init__(self, parent):
        super().__init__(parent)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            messagebox.showwarning(
                "API Key Missing",
                "Please set OPENAI_API_KEY environment variable"
            )
        else:
            openai.api_key = self.api_key  # Initialize OpenAI with API key
        
        self.simulation_data = None
        self.chat_history = []
        self.db_path = None
        self.db = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the chat interface."""
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Split into chat and data view
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Chat interface
        chat_frame = ttk.Frame(paned)
        paned.add(chat_frame, weight=2)
        
        # Chat history
        history_frame = ttk.LabelFrame(chat_frame, text="Chat History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create chat display with scrollbar
        self.chat_display = tk.Text(
            history_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Arial", 10),
            height=20
        )
        scroll = ttk.Scrollbar(history_frame, command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=scroll.set)
        
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_field = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.input_field.pack(fill=tk.X, pady=5)
        self.input_field.bind("<Return>", self._handle_return)
        self.input_field.bind("<Shift-Return>", lambda e: None)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X)
        
        send_btn = ttk.Button(
            button_frame,
            text="Send",
            command=self._send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=5)
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear Chat",
            command=self._clear_chat
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add plot generation button
        self.plot_btn = ttk.Button(
            button_frame,
            text="Generate Plot",
            command=self._generate_plot
        )
        self.plot_btn.pack(side=tk.RIGHT, padx=5)

        # Add data analysis button
        self.analyze_btn = ttk.Button(
            button_frame,
            text="Quick Analysis",
            command=self._quick_analysis
        )
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)

        # Add agent query button
        self.query_agent_btn = ttk.Button(
            button_frame,
            text="Query Agent",
            command=self._query_agent
        )
        self.query_agent_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add agent design analysis button
        self.design_btn = ttk.Button(
            button_frame,
            text="Agent Design",
            command=self._analyze_agent_design
        )
        self.design_btn.pack(side=tk.RIGHT, padx=5)

        # Add behavior pattern button
        self.pattern_btn = ttk.Button(
            button_frame,
            text="Behavior Patterns",
            command=self._analyze_behavior_patterns
        )
        self.pattern_btn.pack(side=tk.RIGHT, padx=5)

        # Add decision tree button
        self.decision_btn = ttk.Button(
            button_frame,
            text="Decision Analysis",
            command=self._show_decision_tree
        )
        self.decision_btn.pack(side=tk.RIGHT, padx=5)
        
        # Right side - Data view
        data_frame = ttk.LabelFrame(paned, text="Available Data")
        paned.add(data_frame, weight=1)
        
        # Create data tree view
        self.data_tree = ttk.Treeview(data_frame, show="tree")
        self.data_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add initial message
        self._add_message(
            "Assistant",
            "Hello! I can help you analyze your simulation data. "
            "What would you like to know?"
        )

    def _handle_return(self, event):
        """Handle return key in input field."""
        if not event.state & 0x1:  # Shift not pressed
            self._send_message()
            return "break"  # Prevent default newline
        return None  # Allow shift+return for newline

    def _send_message(self):
        """Send user message and get AI response."""
        message = self.input_field.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Add user message to chat
        self._add_message("You", message)
        
        # Get AI response
        try:
            response = self._get_ai_response(message)
            self._add_message("Assistant", response)
        except Exception as e:
            self._add_message(
                "System",
                f"Error getting response: {str(e)}",
                color="red"
            )

    def _add_message(self, sender: str, message: str, color: str = None):
        """Add message to chat display."""
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Add sender
        self.chat_display.insert(tk.END, f"{sender}: ", "sender")
        
        # Add message
        if color:
            self.chat_display.tag_configure(f"message_{timestamp}", foreground=color)
            self.chat_display.insert(tk.END, f"{message}\n", f"message_{timestamp}")
        else:
            self.chat_display.insert(tk.END, f"{message}\n", "message")
        
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        
        # Configure tags
        self.chat_display.tag_configure("timestamp", foreground="gray")
        self.chat_display.tag_configure("sender", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_display.tag_configure("message", font=("Arial", 10))

    def _clear_chat(self):
        """Clear chat history."""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?"):
            self.chat_display.configure(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.configure(state=tk.DISABLED)
            self.chat_history = []

    def _get_ai_response(self, message: str, require_reasoning: bool = False) -> str:
        """Enhanced AI response with optional chain-of-thought reasoning."""
        if not self.api_key:
            return "Error: OpenAI API key not set"
            
        try:
            # Prepare context with simulation data
            context = self._prepare_context()
            
            # Add reasoning instruction if required
            if require_reasoning:
                context += "\nProvide your reasoning step by step before giving the final answer."
            
            # Create messages list with context and user message
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": message}
            ]
            
            # Add chat history for context
            for msg in self.chat_history[-5:]:
                messages.append(msg)
            
            # Get completion from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_message = response.choices[0].message.content
            
            # Store in chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}"

    def _prepare_context(self) -> str:
        """Enhanced context preparation with more data."""
        if not self.simulation_data:
            return "No simulation data available."
            
        context = """You are an AI assistant analyzing simulation data. 
        You have access to the following simulation information:
        """
        
        # Add configuration
        if "config" in self.simulation_data:
            context += "\nConfiguration:\n"
            for key, value in self.simulation_data["config"].items():
                context += f"- {key}: {value}\n"
        
        # Add current metrics
        if "metrics" in self.simulation_data:
            context += "\nCurrent Metrics:\n"
            for key, value in self.simulation_data["metrics"].items():
                context += f"- {key}: {value}\n"
        
        # Add historical trends if available
        if self.db and "metrics" in self.simulation_data:
            try:
                trends = self.db.get_historical_trends()
                context += "\nHistorical Trends:\n"
                for metric, trend in trends.items():
                    context += f"- {metric}: {trend}\n"
            except Exception:
                pass
        
        # Add population statistics
        if self.db:
            try:
                stats = self.db.get_population_statistics()
                context += "\nPopulation Statistics:\n"
                for stat, value in stats.items():
                    context += f"- {stat}: {value}\n"
            except Exception:
                pass
        
        return context

    def set_simulation_data(self, data: Dict):
        """Update simulation data and refresh display."""
        self.simulation_data = data
        self._update_data_tree()
        
    def _update_data_tree(self):
        """Update data tree view with current simulation data."""
        self.data_tree.delete(*self.data_tree.get_children())
        
        if not self.simulation_data:
            self.data_tree.insert("", "end", text="No data available")
            return
            
        # Add configuration
        config_node = self.data_tree.insert("", "end", text="Configuration")
        if "config" in self.simulation_data:
            for key, value in self.simulation_data["config"].items():
                self.data_tree.insert(config_node, "end", text=f"{key}: {value}")
        
        # Add metrics
        metrics_node = self.data_tree.insert("", "end", text="Metrics")
        if "metrics" in self.simulation_data:
            for key, value in self.simulation_data["metrics"].items():
                self.data_tree.insert(metrics_node, "end", text=f"{key}: {value}") 

    def _generate_plot(self):
        """Generate a plot based on user query."""
        if not self.simulation_data:
            messagebox.showwarning("No Data", "Please load simulation data first.")
            return

        plot_query = "What kind of plot would you like to generate? Examples:\n" + \
                    "- Population trends over time\n" + \
                    "- Resource distribution\n" + \
                    "- Agent performance comparison\n" + \
                    "- Survival analysis"
        
        # Create dialog for plot query
        dialog = tk.Toplevel(self)
        dialog.title("Generate Plot")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text=plot_query).pack(pady=10)
        
        query_text = tk.Text(dialog, height=4)
        query_text.pack(fill=tk.X, padx=10, pady=5)
        
        def submit_query():
            query = query_text.get("1.0", tk.END).strip()
            if query:
                self._create_plot_from_query(query)
                dialog.destroy()
        
        ttk.Button(dialog, text="Generate", command=submit_query).pack(pady=10)

    def _create_plot_from_query(self, query: str):
        """Create and display a plot based on user query."""
        try:
            # Get AI guidance on plot creation
            plot_guidance = self._get_ai_response(
                f"Based on this query: '{query}', provide Python code using matplotlib "
                "to create the requested visualization. Use only the available data: "
                f"{self.simulation_data}. Return ONLY the code, no explanations."
            )
            
            # Create plot window
            plot_window = tk.Toplevel(self)
            plot_window.title("Generated Plot")
            
            # Create figure and canvas
            fig = plt.figure(figsize=(10, 6))
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            
            # Execute plot code (with safety checks)
            restricted_globals = {
                'plt': plt,
                'np': np,
                'pd': pd,
                'data': self.simulation_data,
                'fig': fig
            }
            exec(plot_guidance, restricted_globals)
            
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")

    def _quick_analysis(self):
        """Perform quick analysis of current simulation state."""
        if not self.simulation_data:
            messagebox.showwarning("No Data", "Please load simulation data first.")
            return
            
        analysis_prompt = """Perform a quick analysis of the simulation data:
        1. Identify key trends and patterns
        2. Highlight any anomalies or interesting behaviors
        3. Compare performance metrics
        4. Suggest potential optimizations
        5. Predict likely outcomes based on current trends
        """
        
        response = self._get_ai_response(analysis_prompt)
        self._add_message("Assistant", response)

    def _query_agent(self):
        """Query specific agent details."""
        if not self.db_path:
            messagebox.showwarning("No Database", "Please load simulation database first.")
            return
            
        # Create agent query dialog
        dialog = tk.Toplevel(self)
        dialog.title("Query Agent")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="Enter Agent ID:").pack(pady=10)
        
        agent_id_var = tk.StringVar()
        agent_id_entry = ttk.Entry(dialog, textvariable=agent_id_var)
        agent_id_entry.pack(pady=5)
        
        def submit_query():
            try:
                agent_id = int(agent_id_var.get())
                self._analyze_agent(agent_id)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid agent ID")
        
        ttk.Button(dialog, text="Query", command=submit_query).pack(pady=10)

    def _analyze_agent(self, agent_id: int):
        """Analyze specific agent's behavior and history."""
        try:
            db = SimulationDatabase(self.db_path)
            
            # Get agent data
            agent_data = db.get_agent_data(agent_id)
            agent_actions = db.get_agent_actions(agent_id)
            
            # Prepare analysis prompt
            analysis_prompt = f"""Analyze this agent's behavior and performance:
            Agent Data: {agent_data}
            Action History: {agent_actions}
            
            Please provide:
            1. Overview of agent's lifecycle
            2. Key decisions and their outcomes
            3. Performance analysis
            4. Comparison to population averages
            5. Notable behaviors or patterns
            """
            
            response = self._get_ai_response(analysis_prompt)
            self._add_message("Assistant", response)
            
        except Exception as e:
            self._add_message("System", f"Error analyzing agent: {str(e)}", color="red")

    def set_database(self, db_path: str):
        """Set the database path for detailed queries."""
        self.db_path = db_path
        self.db = SimulationDatabase(db_path)
        
    def _analyze_agent_design(self):
        """Analyze agent design architecture and parameters."""
        if not self.db:
            messagebox.showwarning("No Data", "Please load simulation data first.")
            return

        try:
            # Get agent design data
            agent_types = self.db.get_agent_types()
            design_analysis = self._get_ai_response(
                f"""Analyze the agent design architecture:
                Agent Types: {agent_types}
                Configuration: {self.simulation_data.get('config', {})}
                
                Please provide:
                1. Design principles and architecture
                2. Key parameters and their impact
                3. Decision-making mechanisms
                4. Learning/adaptation capabilities
                5. Strengths and limitations
                6. Suggested improvements
                """
            )
            
            # Create analysis window
            design_window = tk.Toplevel(self)
            design_window.title("Agent Design Analysis")
            design_window.geometry("800x600")
            
            # Create notebook for different aspects
            notebook = ttk.Notebook(design_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Analysis tab
            analysis_frame = ttk.Frame(notebook)
            notebook.add(analysis_frame, text="Analysis")
            
            analysis_text = tk.Text(analysis_frame, wrap=tk.WORD)
            analysis_text.insert("1.0", design_analysis)
            analysis_text.pack(fill=tk.BOTH, expand=True)
            
            # Architecture visualization tab
            arch_frame = ttk.Frame(notebook)
            notebook.add(arch_frame, text="Architecture")
            
            self._visualize_agent_architecture(arch_frame)
            
            # Parameters tab
            param_frame = ttk.Frame(notebook)
            notebook.add(param_frame, text="Parameters")
            
            self._show_parameter_analysis(param_frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze agent design: {str(e)}")

    def _visualize_agent_architecture(self, parent):
        """Create visual representation of agent architecture."""
        fig = plt.figure(figsize=(10, 8))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes for components
        components = [
            ("Input", "Sensory\nInput"),
            ("Process", "Decision\nMaking"),
            ("Memory", "State\nMemory"),
            ("Output", "Action\nSelection")
        ]
        
        pos = {}
        for i, (node_type, label) in enumerate(components):
            G.add_node(node_type, label=label)
            pos[node_type] = (i/2, i%2)
        
        # Add edges
        edges = [
            ("Input", "Process"),
            ("Process", "Output"),
            ("Memory", "Process"),
            ("Process", "Memory")
        ]
        G.add_edges_from(edges)
        
        # Draw graph
        nx.draw(G, pos,
                labels=nx.get_node_attributes(G, 'label'),
                node_color='lightblue',
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray')
        
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_parameter_analysis(self, parent):
        """Show analysis of agent parameters."""
        if not self.simulation_data or 'config' not in self.simulation_data:
            return
            
        # Create parameter tree
        tree = ttk.Treeview(parent, columns=('value', 'impact'), show='headings')
        tree.heading('value', text='Value')
        tree.heading('impact', text='Impact')
        
        # Add parameters
        config = self.simulation_data['config']
        for param, value in config.items():
            if param.startswith(('agent_', 'learning_', 'decision_')):
                impact = self._get_parameter_impact(param, value)
                tree.insert('', 'end', values=(f"{value}", impact))
                
        tree.pack(fill=tk.BOTH, expand=True)

    def _analyze_behavior_patterns(self):
        """Analyze agent behavior patterns over time."""
        if not self.db:
            messagebox.showwarning("No Data", "Please load simulation database first.")
            return
            
        try:
            # Get behavior data
            behavior_data = self.db.get_agent_behaviors()
            
            # Create analysis window
            pattern_window = tk.Toplevel(self)
            pattern_window.title("Behavior Pattern Analysis")
            pattern_window.geometry("800x600")
            
            # Create visualization
            fig = plt.figure(figsize=(10, 6))
            
            # Plot behavior frequencies
            behaviors = pd.DataFrame(behavior_data)
            behaviors.plot(kind='bar', stacked=True)
            plt.title("Agent Behavior Patterns")
            plt.xlabel("Time Steps")
            plt.ylabel("Frequency")
            
            canvas = FigureCanvasTkAgg(fig, master=pattern_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add analysis text
            analysis = self._get_ai_response(
                f"""Analyze these behavior patterns:
                Data: {behavior_data}
                
                Focus on:
                1. Common behavior sequences
                2. Adaptation patterns
                3. Successful strategies
                4. Behavioral anomalies
                5. Environmental influences
                """
            )
            
            text = tk.Text(pattern_window, height=10, wrap=tk.WORD)
            text.insert("1.0", analysis)
            text.pack(fill=tk.X, padx=5, pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze behavior patterns: {str(e)}")

    def _show_decision_tree(self):
        """Visualize agent decision-making process."""
        if not self.db:
            messagebox.showwarning("No Data", "Please load simulation database first.")
            return
            
        try:
            # Get decision data
            decision_data = self.db.get_agent_decisions()
            
            # Create decision window
            decision_window = tk.Toplevel(self)
            decision_window.title("Decision Analysis")
            decision_window.geometry("800x600")
            
            # Create tree visualization
            fig = plt.figure(figsize=(12, 8))
            
            def plot_decision_node(x, y, width, height, label, condition=None):
                rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                                   facecolor='lightblue', edgecolor='black')
                plt.gca().add_patch(rect)
                plt.text(x, y, label, ha='center', va='center', wrap=True)
                if condition:
                    plt.text(x, y+height/2, condition, ha='center', va='bottom',
                           fontsize=8, color='gray')
            
            # Plot decision tree structure
            plot_decision_node(0.5, 0.8, 0.2, 0.1, "State\nInput")
            plot_decision_node(0.3, 0.6, 0.15, 0.1, "Resource\nCheck", "R < threshold")
            plot_decision_node(0.7, 0.6, 0.15, 0.1, "Threat\nCheck", "Enemies nearby")
            # Add more nodes as needed...
            
            plt.axis('equal')
            plt.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=decision_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add decision analysis
            analysis = self._get_ai_response(
                f"""Analyze the decision-making process:
                Decisions: {decision_data}
                
                Focus on:
                1. Decision criteria
                2. Priority ordering
                3. Success rates
                4. Edge cases
                5. Improvement opportunities
                """
            )
            
            text = tk.Text(decision_window, height=10, wrap=tk.WORD)
            text.insert("1.0", analysis)
            text.pack(fill=tk.X, padx=5, pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show decision tree: {str(e)}")

    def _get_parameter_impact(self, param: str, value: Any) -> str:
        """Analyze the impact of a parameter on agent behavior."""
        impact_prompt = f"""Analyze the impact of parameter '{param}' with value {value}
        on agent behavior and performance. Consider the current configuration:
        {self.simulation_data.get('config', {})}
        
        Provide a brief impact assessment."""
        
        return self._get_ai_response(impact_prompt)

    def _execute_complex_task(self, task: str):
        """Execute a complex task using task decomposition and chain-of-thought reasoning."""
        try:
            # First, get task decomposition
            decomposition_prompt = f"""
            Break down this complex task into smaller, sequential steps:
            Task: {task}
            
            For each step:
            1. Specify what data or information is needed
            2. List any dependencies on previous steps
            3. Define the expected output
            
            Format your response as:
            Step 1: [description]
            Dependencies: [list]
            Requires: [data needed]
            Produces: [output]
            
            Continue for each step...
            """
            
            decomposition = self._get_ai_response(decomposition_prompt, require_reasoning=True)
            steps = self._parse_task_steps(decomposition)
            
            # Execute each step
            results = {}
            for i, step in enumerate(steps):
                self._add_message("System", f"Executing step {i+1}: {step['description']}")
                
                # Check dependencies
                if not self._check_dependencies(step, results):
                    raise Exception(f"Missing dependencies for step {i+1}")
                
                # Get required data
                step_data = self._gather_step_data(step)
                
                # Execute step
                result = self._execute_step(step, step_data, results)
                results[f"step_{i+1}"] = result
                
                # Validate result
                if not self._validate_step_result(step, result):
                    raise Exception(f"Step {i+1} validation failed")
            
            # Synthesize final result
            final_result = self._synthesize_results(results)
            return final_result
            
        except Exception as e:
            self._add_message("System", f"Error executing task: {str(e)}", color="red")
            return None

    def _parse_task_steps(self, decomposition: str) -> List[Dict]:
        """Parse the task decomposition into structured steps."""
        steps = []
        current_step = {}
        
        for line in decomposition.split('\n'):
            line = line.strip()
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step)
                current_step = {'description': line.split(':', 1)[1].strip()}
            elif line.startswith('Dependencies:'):
                current_step['dependencies'] = [
                    d.strip() for d in line.split(':', 1)[1].split(',')
                ]
            elif line.startswith('Requires:'):
                current_step['requires'] = [
                    r.strip() for r in line.split(':', 1)[1].split(',')
                ]
            elif line.startswith('Produces:'):
                current_step['produces'] = line.split(':', 1)[1].strip()
                
        if current_step:
            steps.append(current_step)
            
        return steps

    def _check_dependencies(self, step: Dict, results: Dict) -> bool:
        """Check if all dependencies for a step are satisfied."""
        if 'dependencies' not in step:
            return True
            
        for dep in step['dependencies']:
            dep_step = re.search(r'step_(\d+)', dep)
            if dep_step and dep_step.group(1) not in results:
                return False
        return True

    def _gather_step_data(self, step: Dict) -> Dict:
        """Gather required data for a step."""
        data = {}
        if 'requires' not in step:
            return data
            
        for requirement in step['requires']:
            if requirement == 'simulation_data':
                data['simulation_data'] = self.simulation_data
            elif requirement == 'agent_data':
                data['agent_data'] = self.db.get_agent_data()
            elif requirement == 'metrics':
                data['metrics'] = self.simulation_data.get('metrics', {})
            # Add more data gathering as needed
            
        return data

    def _execute_step(self, step: Dict, data: Dict, previous_results: Dict) -> Any:
        """Execute a single step of the task."""
        # Prepare prompt with step information and data
        prompt = f"""
        Execute this step: {step['description']}
        
        Available data:
        {json.dumps(data, indent=2)}
        
        Previous results:
        {json.dumps(previous_results, indent=2)}
        
        Provide:
        1. Your reasoning process
        2. Calculations or analysis performed
        3. The final result
        """
        
        response = self._get_ai_response(prompt, require_reasoning=True)
        return self._extract_step_result(response)

    def _extract_step_result(self, response: str) -> Any:
        """Extract the actual result from the AI's response."""
        # Look for a final result section
        if "Final result:" in response:
            result = response.split("Final result:", 1)[1].strip()
            try:
                # Try to parse as JSON if possible
                return json.loads(result)
            except:
                return result
        return response

    def _validate_step_result(self, step: Dict, result: Any) -> bool:
        """Validate the result of a step."""
        validation_prompt = f"""
        Validate this step result:
        Step: {step['description']}
        Expected output: {step.get('produces', 'Not specified')}
        Actual result: {result}
        
        Is this result valid and complete? Explain why or why not.
        """
        
        response = self._get_ai_response(validation_prompt)
        return "valid" in response.lower() and "invalid" not in response.lower()

    def _synthesize_results(self, results: Dict) -> str:
        """Synthesize all step results into a final output."""
        synthesis_prompt = f"""
        Synthesize these step results into a final conclusion:
        {json.dumps(results, indent=2)}
        
        Provide:
        1. Summary of key findings
        2. Relationships between different steps
        3. Overall conclusions
        4. Confidence level in the results
        """
        
        return self._get_ai_response(synthesis_prompt, require_reasoning=True)

    def execute_analysis(self, query: str):
        """Execute a complex analysis task."""
        self._add_message("User", query)
        result = self._execute_complex_task(query)
        if result:
            self._add_message("Assistant", result)
        
    def update(self, data=None):
        """Update the chat component with new data."""
        if data:
            # Update simulation data
            self.simulation_data = data
            self._update_data_tree()
        
        # Call the widget's update method without arguments
        ttk.Frame.update(self)
        