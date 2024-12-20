import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple


class BaseWindow:
    """
    Base class for application windows.
    
    Provides common functionality for:
    - Window creation and configuration
    - Error handling
    - Message display
    - Layout management
    - Window state management
    
    Attributes:
        parent (tk.Tk): Parent window
        window (tk.Toplevel): Window instance
        title (str): Window title
        size (Tuple[int, int]): Window dimensions (width, height)
        resizable (Tuple[bool, bool]): Window resizability (width, height)
    """

    def __init__(
        self,
        parent: tk.Tk,
        title: str = "Window",
        size: Tuple[int, int] = (800, 600),
        resizable: Tuple[bool, bool] = (True, True)
    ):
        """
        Initialize base window.

        Args:
            parent: Parent window
            title: Window title
            size: Window dimensions (width, height)
            resizable: Window resizability (width, height)
        """
        self.parent = parent
        self.title = title
        self.size = size
        
        # Create and configure window
        self.window = tk.Toplevel(self.parent)
        self._configure_window(resizable)
        
        # Initialize UI
        self._setup_ui()
        
        # Center window
        self.center_window()

    def _configure_window(self, resizable: Tuple[bool, bool]):
        """Configure window properties."""
        self.window.title(self.title)
        self.window.geometry(f"{self.size[0]}x{self.size[1]}")
        self.window.resizable(resizable[0], resizable[1])
        
        # Make window modal
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_ui(self):
        """
        Setup the user interface.
        
        Override this method in subclasses to create the window's UI.
        """
        pass

    def center_window(self):
        """Center window on screen."""
        self.window.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - self.size[0]) // 2
        y = (screen_height - self.size[1]) // 2
        
        # Position window
        self.window.geometry(f"+{x}+{y}")

    def show_error(self, title: str, message: str):
        """Display error message."""
        messagebox.showerror(title, message, parent=self.window)

    def show_warning(self, title: str, message: str):
        """Display warning message."""
        messagebox.showwarning(title, message, parent=self.window)

    def show_info(self, title: str, message: str):
        """Display information message."""
        messagebox.showinfo(title, message, parent=self.window)

    def ask_yes_no(self, title: str, message: str) -> bool:
        """
        Ask yes/no question.
        
        Returns:
            bool: True if user clicked Yes, False otherwise
        """
        return messagebox.askyesno(title, message, parent=self.window)

    def create_scrolled_frame(
        self,
        parent: tk.Widget,
        **kwargs
    ) -> Tuple[ttk.Frame, tk.Canvas]:
        """
        Create a scrollable frame.
        
        Args:
            parent: Parent widget
            **kwargs: Additional frame configuration
            
        Returns:
            Tuple containing the frame and canvas
        """
        # Create canvas and scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=canvas.yview
        )
        
        # Create frame inside canvas
        frame = ttk.Frame(canvas, **kwargs)
        
        # Configure scrolling
        frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        return frame, canvas

    def create_labeled_entry(
        self,
        parent: tk.Widget,
        label_text: str,
        default_value: str = "",
        width: int = 20
    ) -> Tuple[ttk.Label, ttk.Entry]:
        """
        Create a labeled entry field.
        
        Args:
            parent: Parent widget
            label_text: Label text
            default_value: Default entry value
            width: Entry width
            
        Returns:
            Tuple containing the label and entry widgets
        """
        container = ttk.Frame(parent)
        container.pack(fill="x", padx=5, pady=2)
        
        label = ttk.Label(container, text=label_text)
        label.pack(side="left")
        
        entry = ttk.Entry(container, width=width)
        entry.insert(0, default_value)
        entry.pack(side="right", fill="x", expand=True)
        
        return label, entry

    def create_button_group(
        self,
        parent: tk.Widget,
        buttons: list,
        side: str = "right"
    ) -> dict:
        """
        Create a group of buttons.
        
        Args:
            parent: Parent widget
            buttons: List of (text, command) tuples
            side: Pack side for buttons
            
        Returns:
            Dictionary of created buttons
        """
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=5)
        
        created_buttons = {}
        for text, command in buttons:
            btn = ttk.Button(frame, text=text, command=command)
            btn.pack(side=side, padx=2)
            created_buttons[text] = btn
            
        return created_buttons

    def close(self):
        """Close the window."""
        self.window.grab_release()
        self.window.destroy()

    def show(self):
        """Show the window and wait for it to close."""
        self.window.wait_window()

    def enable(self):
        """Enable window interaction."""
        self.window.grab_set()

    def disable(self):
        """Disable window interaction."""
        self.window.grab_release() 