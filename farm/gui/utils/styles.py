"""Shared styles and constants for GUI components"""

CARD_COLORS = {
    "total_agents": "#4a90e2",  # Blue
    "system_agents": "#50c878",  # Emerald green
    "independent_agents": "#e74c3c",  # Red
    "control_agents": "#DAA520",  # Goldenrod
    "total_resources": "#f39c12",  # Orange
    "average_agent_resources": "#9b59b6",  # Purple
}

AGENT_COLORS = {
    "SystemAgent": "blue",
    "IndependentAgent": "red",
    "ControlAgent": "#DAA520"  # Goldenrod
}

VISUALIZATION_CONSTANTS = {
    "DEFAULT_CANVAS_SIZE": (400, 400),
    "PADDING": 20,
    "MAX_ANIMATION_FRAMES": 5,
    "ANIMATION_MIN_DELAY": 50,
    "MAX_RESOURCE_AMOUNT": 30,
    "RESOURCE_GLOW": {
        "RED": 50,
        "GREEN": 255,
        "BLUE": 50
    },
    "AGENT_RADIUS_SCALE": 2,
    "BIRTH_RADIUS_SCALE": 4,
    "DEATH_MARK_SCALE": 1.5,
    "MIN_FONT_SIZE": 10,
    "FONT_SCALE_FACTOR": 40
}

def configure_ttk_styles():
    """Configure common ttk styles used across the application"""
    from tkinter import ttk
    
    style = ttk.Style()
    
    # Control button style
    style.configure("Control.TButton", padding=5)
    
    # Card styles
    style.configure("Card.TFrame", background="white", relief="solid")
    style.configure("CardLabel.TLabel", background="white", font=("Arial", 10))
    style.configure("CardValue.TLabel", background="white", font=("Arial", 14, "bold"))
    
    # Scale style
    style.layout("Horizontal.TScale", [
        ("Horizontal.Scale.trough", {
            "sticky": "nswe",
            "children": [
                ("Horizontal.Scale.slider", {"side": "left", "sticky": ""})
            ]
        })
    ])
    style.configure("Horizontal.TScale", background="white")
    
    # Welcome screen button style
    style.configure(
        "Welcome.TButton",
        padding=(20, 10),
        font=("Arial", 11),
        background="#f0f0f0",  # Light gray background
        relief="raised"
    )
    
    # Configure hover effect for welcome buttons
    style.map(
        "Welcome.TButton",
        background=[("active", "#e0e0e0")],  # Slightly darker when hovered
        relief=[("pressed", "sunken")]  # Pressed effect
    )
    
    # Configuration section styles
    style.configure(
        "Config.TLabelframe",
        background="#f5f5f5",  # Light gray background
        relief="solid",
    )
    style.configure(
        "Config.TLabelframe.Label",
        font=("Arial", 12, "bold"),
        padding=(0, 5),
        background="#f5f5f5"  # Match parent background
    )
    
    style.configure(
        "ConfigSection.TLabelframe",
        background="#ffffff",  # White background for sections
        relief="solid",
    )
    style.configure(
        "ConfigSection.TLabelframe.Label",
        font=("Arial", 10, "bold"),
        padding=(0, 3),
        background="#ffffff"  # Match section background
    )
    
    style.configure(
        "ConfigLabel.TLabel",
        font=("Arial", 9),
        background="#ffffff"  # Match section background
    )
    
    style.configure(
        "Config.TEntry",
        padding=5,
        relief="solid",
        fieldbackground="white",  # Ensure white background for text entry
    )
    
    # Configure frames to match their parents' backgrounds
    style.configure("TFrame", background="#f5f5f5")
    
    # Make sure entry widgets have white backgrounds
    style.map("TEntry",
        fieldbackground=[("readonly", "white"), ("disabled", "#f0f0f0")],
        background=[("readonly", "white"), ("disabled", "#f0f0f0")]
    )
    
    # Simulation view styles
    style.configure(
        "SimPane.TFrame",
        background="#f8f9fa",  # Light background
        relief="flat",
    )
    
    style.configure(
        "Controls.TFrame",
        background="#ffffff",
        relief="solid",
        borderwidth=1,
        padding=5
    )
    
    # Update existing Card styles
    style.configure(
        "Card.TFrame",
        background="#ffffff",
        relief="solid",
        borderwidth=1,
        padding=8
    )
    
    # Agent Analysis Window styles
    style.configure(
        "AgentAnalysis.TLabelframe",
        background="#ffffff",
        relief="solid",
        borderwidth=1,
        padding=10
    )
    
    style.configure(
        "AgentAnalysis.TLabelframe.Label",
        font=("Arial", 10, "bold"),
        background="#ffffff"
    )
    
    style.configure(
        "AgentAnalysis.TCombobox",
        padding=5,
        relief="solid"
    )
    
    # Info panel styles
    style.configure(
        "InfoLabel.TLabel",
        font=("Arial", 9),
        background="#ffffff",
        padding=(2, 0)
    )
    
    style.configure(
        "InfoValue.TLabel",
        font=("Arial", 9, "bold"),
        background="#ffffff",
        padding=(2, 0)
    )
    
    # Other existing styles... 