import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from gui.components.tooltips import ToolTip


class ControlPanel(ttk.Frame):
    """
    Control panel for simulation playback.
    
    Provides:
    - Play/pause button
    - Step controls (forward/backward)
    - Speed adjustment slider
    - Export button
    
    Attributes:
        play_button (ttk.Button): Toggle playback button
        speed_scale (ttk.Scale): Playback speed control
        playing (bool): Current playback state
    """

    def __init__(
        self,
        parent,
        play_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        export_callback: Optional[Callable] = None
    ):
        super().__init__(parent, padding=10)
        
        self.play_callback = play_callback
        self.step_callback = step_callback
        self.export_callback = export_callback
        
        self.playing = False
        
        self._setup_controls()

    def _setup_controls(self):
        """Setup the playback control buttons and slider."""
        # Control buttons frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(side="left", fill="x", expand=True)

        # Playback controls
        self.play_button = ttk.Button(
            buttons_frame,
            text="▶ Play/Pause",
            command=self._toggle_playback,
            style="Control.TButton"
        )
        self.play_button.pack(side="left", padx=5)
        ToolTip(self.play_button, "Start or pause the simulation playback")

        # Step controls with consistent styling and tooltips
        step_controls = [
            ("⏪", -10, "Go back 10 steps"),
            ("◀", -1, "Previous step"),
            ("▶", 1, "Next step"),
            ("⏩", 10, "Skip forward 10 steps"),
        ]

        for text, step_size, tooltip in step_controls:
            btn = ttk.Button(
                buttons_frame,
                text=text,
                command=lambda s=step_size: self._step(s),
                style="Control.TButton"
            )
            btn.pack(side="left", padx=2)
            ToolTip(btn, tooltip)

        # Speed control frame
        speed_frame = ttk.Frame(self)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        speed_label = ttk.Label(speed_frame, text="Playback Speed:")
        speed_label.pack(side="left", padx=5)
        ToolTip(speed_label, "Adjust the simulation playback speed")

        # Speed scale with tooltip
        self.speed_scale = ttk.Scale(
            speed_frame,
            from_=1,
            to=50,
            orient="horizontal"
        )
        self.speed_scale.set(10)  # Default speed
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)
        ToolTip(self.speed_scale, "1 = Slowest, 50 = Fastest")

        # Export button with tooltip
        if self.export_callback:
            export_btn = ttk.Button(
                self,
                text="Export Data",
                command=self.export_callback,
                style="Control.TButton"
            )
            export_btn.pack(side="right", padx=5)
            ToolTip(export_btn, "Export simulation data to CSV file")

    def _toggle_playback(self):
        """Toggle simulation playback state."""
        self.playing = not self.playing
        
        # Update button text
        self.play_button.config(
            text="⏸ Pause" if self.playing else "▶ Play"
        )
        
        # Notify parent
        if self.play_callback:
            self.play_callback(self.playing)

    def _step(self, step_size: int):
        """Step simulation forward or backward."""
        if self.step_callback:
            self.step_callback(step_size)

    def get_speed(self) -> float:
        """Get current playback speed setting."""
        return self.speed_scale.get()

    def set_playing(self, state: bool):
        """Set playback state without triggering callback."""
        if self.playing != state:
            self.playing = state
            self.play_button.config(
                text="⏸ Pause" if self.playing else "▶ Play"
            )

    def enable(self):
        """Enable all controls."""
        self.play_button.config(state="normal")
        self.speed_scale.config(state="normal")
        for child in self.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state="normal")

    def disable(self):
        """Disable all controls."""
        self.play_button.config(state="disabled")
        self.speed_scale.config(state="disabled")
        for child in self.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state="disabled")

    def set_speed(self, speed: float):
        """Set playback speed externally."""
        self.speed_scale.set(speed)

    def get_delay(self) -> int:
        """Calculate delay in milliseconds based on speed setting."""
        return int(1000 / self.get_speed())

    def update_tooltips(self):
        """Update tooltips based on current state."""
        play_tooltip = "Pause simulation" if self.playing else "Start simulation"
        ToolTip(self.play_button, play_tooltip) 