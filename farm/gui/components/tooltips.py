import tkinter as tk
from tkinter import ttk


class ToolTip:
    """
    Create a tooltip for a given widget.
    
    Displays a small popup with explanatory text when hovering over a widget.
    
    Attributes:
        widget: The widget to attach the tooltip to
        text (str): The text to display in the tooltip
        tooltip_window: The popup window containing the tooltip
        id: ID of the current after callback
        
    Example:
        button = ttk.Button(parent, text="Help")
        ToolTip(button, "Click for help")
    """

    def __init__(self, widget, text: str, delay: int = 500, wrap_length: int = 180):
        """
        Initialize tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Text to display
            delay: Delay in ms before showing tooltip
            wrap_length: Maximum line length before wrapping
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap_length = wrap_length
        self.tooltip_window = None
        self.id = None

        # Bind mouse events
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)
        self.widget.bind("<ButtonPress>", self._on_leave)

    def _on_enter(self, event=None):
        """Handle mouse entering widget."""
        self._schedule()

    def _on_leave(self, event=None):
        """Handle mouse leaving widget."""
        self._unschedule()
        self._hide()

    def _schedule(self):
        """Schedule tooltip to appear after delay."""
        self._unschedule()
        self.id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        """Cancel scheduled tooltip."""
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def _show(self):
        """Display the tooltip."""
        # Get display coordinates
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # Creates a toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)
        
        # Remove window decorations
        self.tooltip_window.wm_overrideredirect(True)

        # Position tooltip
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create tooltip label
        label = ttk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=self.wrap_length
        )
        label.pack(padx=2, pady=2)

        # Make sure tooltip stays on top
        self.tooltip_window.lift()
        
        # Add bindings to handle tooltip behavior
        self.tooltip_window.bind("<Enter>", self._on_tooltip_enter)
        self.tooltip_window.bind("<Leave>", self._on_tooltip_leave)

    def _hide(self):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def _on_tooltip_enter(self, event):
        """Handle mouse entering tooltip."""
        # Cancel any pending hide operations
        self._unschedule()

    def _on_tooltip_leave(self, event):
        """Handle mouse leaving tooltip."""
        self._hide()

    def update_text(self, new_text: str):
        """Update tooltip text.
        
        Args:
            new_text: New text to display
        """
        self.text = new_text
        # Update tooltip if it's currently shown
        if self.tooltip_window:
            self._hide()
            self._show() 