import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import markdown
from tkhtmlview import HTMLLabel
import json
import os
from datetime import datetime

class NotesPanel(ttk.Frame):
    """Panel for writing and viewing simulation notes in markdown."""

    def __init__(self, parent):
        super().__init__(parent)
        self.notes_file = "simulations/notes.json"
        self._setup_ui()
        self._load_notes()

    def _setup_ui(self):
        """Setup the notes interface with editor and preview."""
        # Create paned window for split view
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side - Editor
        editor_frame = ttk.LabelFrame(self.paned, text="Markdown Editor", padding=5)
        self.editor = tk.Text(editor_frame, wrap=tk.WORD, width=50)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.editor.bind('<KeyRelease>', self._update_preview)
        
        # Add editor toolbar
        toolbar = ttk.Frame(editor_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Save button
        self.save_btn = ttk.Button(
            toolbar, 
            text="Save Notes", 
            command=self._save_notes
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        self.clear_btn = ttk.Button(
            toolbar, 
            text="Clear", 
            command=self._clear_editor
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Add to paned window
        self.paned.add(editor_frame)

        # Add search frame above editor:
        search_frame = ttk.Frame(editor_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._highlight_search)
        
        search_entry = ttk.Entry(
            search_frame,
            textvariable=self.search_var,
            width=30
        )
        search_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)

        # Right side - Preview
        preview_frame = ttk.LabelFrame(self.paned, text="Preview", padding=5)
        
        # Create preview with scrollbar
        preview_scroll = ttk.Scrollbar(preview_frame)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.preview = HTMLLabel(
            preview_frame,
            html="<h1>Preview</h1>",
            width=50
        )
        self.preview.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        preview_scroll.config(command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        
        # Add to paned window
        self.paned.add(preview_frame)

        # Template dropdown
        templates = {
            "Basic Observation": """# Simulation Observation

## Setup
- Configuration:
- Initial conditions:

## Observations
- Key findings:
- Unexpected behaviors:
- Agent interactions:

## Analysis
- Patterns noticed:
- Potential improvements:
""",
            "Agent Behavior Analysis": """# Agent Behavior Analysis

## Population Dynamics
- Population growth rate:
- Survival patterns:
- Resource distribution:

## Agent Interactions
- Cooperation observed:
- Competition patterns:
- Resource sharing:

## Learning & Adaptation
- Strategy changes:
- Successful behaviors:
- Failed approaches:
""",
            "Performance Analysis": """# Performance Analysis

## Metrics
- Peak population:
- Resource efficiency:
- Survival rates:

## System Stability
- Equilibrium points:
- Instability factors:
- Recovery patterns:

## Optimization Opportunities
- Bottlenecks:
- Improvement areas:
- Suggested changes:
"""
        }
        
        template_frame = ttk.Frame(toolbar)
        template_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(template_frame, text="Template:").pack(side=tk.LEFT, padx=2)
        
        self.template_var = tk.StringVar()
        template_combo = ttk.Combobox(
            template_frame,
            textvariable=self.template_var,
            values=list(templates.keys()),
            width=20,
            state="readonly"
        )
        template_combo.pack(side=tk.LEFT, padx=2)
        
        def apply_template(*args):
            selected = self.template_var.get()
            if selected and selected in templates:
                if self.editor.get("1.0", tk.END).strip():
                    if messagebox.askyesno("Apply Template", 
                        "This will replace current content. Continue?"):
                        self.editor.delete("1.0", tk.END)
                        self.editor.insert("1.0", templates[selected])
                else:
                    self.editor.insert("1.0", templates[selected])
                self._update_preview()
        
        template_combo.bind("<<ComboboxSelected>>", apply_template)

        # Add to toolbar:
        self.export_btn = ttk.Button(
            toolbar,
            text="Export MD",
            command=self._export_markdown
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Add history button to toolbar:
        self.history_btn = ttk.Button(
            toolbar,
            text="History",
            command=self._show_history
        )
        self.history_btn.pack(side=tk.LEFT, padx=5)

    def _update_preview(self, event=None):
        """Convert markdown to HTML and update preview."""
        try:
            md_text = self.editor.get("1.0", tk.END)
            html = markdown.markdown(md_text)
            self.preview.set_html(html)
        except Exception as e:
            self.preview.set_html(f"<p style='color: red;'>Preview error: {str(e)}</p>")

    def _save_notes(self):
        """Save notes to JSON file."""
        try:
            os.makedirs("simulations", exist_ok=True)
            
            # Load existing notes
            notes = []
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r') as f:
                    notes = json.load(f)

            # Add new note
            notes.append({
                'timestamp': datetime.now().isoformat(),
                'content': self.editor.get("1.0", tk.END.strip()),
                'simulation_id': getattr(self, 'current_simulation_id', None)
            })

            # Save to file
            with open(self.notes_file, 'w') as f:
                json.dump(notes, f, indent=2)

            messagebox.showinfo("Success", "Notes saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save notes: {str(e)}")

    def _load_notes(self):
        """Load existing notes for current simulation."""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r') as f:
                    notes = json.load(f)
                
                # Find notes for current simulation
                if hasattr(self, 'current_simulation_id'):
                    sim_notes = [
                        note for note in notes 
                        if note['simulation_id'] == self.current_simulation_id
                    ]
                    if sim_notes:
                        self.editor.delete("1.0", tk.END)
                        self.editor.insert("1.0", sim_notes[-1]['content'])
                        self._update_preview()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load notes: {str(e)}")

    def _clear_editor(self):
        """Clear the editor content."""
        if messagebox.askyesno("Clear Notes", "Are you sure you want to clear the editor?"):
            self.editor.delete("1.0", tk.END)
            self._update_preview()

    def set_simulation(self, simulation_id):
        """Set current simulation and load its notes."""
        self.current_simulation_id = simulation_id
        self._load_notes()

    def get_markdown(self) -> str:
        """Get current markdown content."""
        return self.editor.get("1.0", tk.END.strip())

    def set_markdown(self, content: str):
        """Set markdown content."""
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", content)
        self._update_preview() 

    def _export_markdown(self):
        """Export notes as markdown file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
                initialfile=f"notes_{self.current_simulation_id}.md"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.get_markdown())
                messagebox.showinfo("Success", "Notes exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export notes: {str(e)}") 

    def _highlight_search(self, *args):
        """Highlight search terms in editor."""
        # Remove existing tags
        self.editor.tag_remove('search', '1.0', tk.END)
        
        search_term = self.search_var.get()
        if not search_term:
            return
        
        # Add new tags
        start_pos = '1.0'
        while True:
            start_pos = self.editor.search(
                search_term, start_pos, tk.END, nocase=True
            )
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(search_term)}c"
            self.editor.tag_add('search', start_pos, end_pos)
            start_pos = end_pos
        
        # Configure tag appearance
        self.editor.tag_config('search', background='yellow')

    def _show_history(self):
        """Show version history of notes."""
        history_window = tk.Toplevel(self)
        history_window.title("Notes History")
        history_window.geometry("600x400")
        
        # Create treeview for history
        tree = ttk.Treeview(
            history_window,
            columns=("timestamp", "preview"),
            show="headings"
        )
        tree.heading("timestamp", text="Timestamp")
        tree.heading("preview", text="Preview")
        
        # Load history
        try:
            with open(self.notes_file, 'r') as f:
                notes = json.load(f)
            
            # Filter notes for current simulation
            sim_notes = [
                note for note in notes 
                if note['simulation_id'] == self.current_simulation_id
            ]
            
            # Add to treeview
            for note in sim_notes:
                preview = note['content'][:50] + "..." if len(note['content']) > 50 else note['content']
                tree.insert("", 0, values=(note['timestamp'], preview))
                
            tree.pack(fill=tk.BOTH, expand=True)
            
            # Add restore button
            def restore_version():
                selected = tree.selection()
                if selected:
                    idx = tree.index(selected[0])
                    note = sim_notes[-(idx+1)]  # Reverse index since we inserted at 0
                    self.set_markdown(note['content'])
                    history_window.destroy()
            
            ttk.Button(
                history_window,
                text="Restore Selected Version",
                command=restore_version
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load history: {str(e)}")