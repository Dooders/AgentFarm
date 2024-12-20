import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageTk

from farm.gui.utils.styles import AGENT_COLORS, VISUALIZATION_CONSTANTS as VC


class EnvironmentView(ttk.Frame):
    """
    Component for visualizing the simulation environment.
    
    Displays:
    - Agent positions and states
    - Resource distributions
    - Birth/death animations
    - Step information
    
    Attributes:
        canvas (tk.Canvas): Main drawing canvas
        canvas_size (Tuple[int, int]): Current canvas dimensions
        previous_agent_ids (set): Set of agent IDs from previous frame
        previous_agent_states (List): Agent states from previous frame
        birth_animations (Dict): Active birth animation states
        death_animations (Dict): Active death animation states
        logger (Logger): Data logger for this component
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas_size = VC["DEFAULT_CANVAS_SIZE"]
        
        # Initialize animation tracking
        self.previous_agent_ids = set()
        self.previous_agent_states = []
        self.birth_animations = {}
        self.death_animations = {}
        self.logger = None  # Initialize logger as None
        
        # Add selection tracking
        self.selected_agent_id = None
        self.on_agent_selected = None  # Callback for agent selection
        
        self._setup_canvas()

    def _setup_canvas(self):
        """Initialize the drawing canvas."""
        style = ttk.Style()
        style.configure("Environment.TFrame", background="black")
        
        # Create canvas frame
        canvas_frame = ttk.Frame(self, style="Environment.TFrame")
        canvas_frame.pack(fill="both", expand=True)

        # Create canvas with black background
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Bind resize event and click event
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def set_agent_selected_callback(self, callback):
        """Set callback for agent selection."""
        self.on_agent_selected = callback

    def select_agent(self, agent_id):
        """Select an agent by ID."""
        self.selected_agent_id = agent_id
        self.update()  # Redraw with selection

    def _on_canvas_click(self, event):
        """Handle canvas click events."""
        if not hasattr(self, 'last_agent_positions'):
            return

        # Convert click coordinates to simulation coordinates
        click_x = event.x
        click_y = event.y
        
        # Calculate click radius for selection (in pixels)
        click_radius = max(5, int(VC["AGENT_RADIUS_SCALE"] * 2))
        
        # Check each agent position
        for agent_id, (x, y) in self.last_agent_positions.items():
            # Convert simulation coordinates to screen coordinates
            screen_x, screen_y = self._transform_coords(x, y, self.last_transform_params)
            
            # Calculate distance
            distance = ((screen_x - click_x) ** 2 + (screen_y - click_y) ** 2) ** 0.5
            
            # If click is within radius of agent
            if distance <= click_radius:
                self.selected_agent_id = agent_id
                if self.on_agent_selected:
                    self.on_agent_selected(agent_id)
                self.update()
                return

        # If no agent was clicked, deselect
        self.selected_agent_id = None
        if self.on_agent_selected:
            self.on_agent_selected(None)
        self.update()

    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        self.canvas_size = (event.width, event.height)
        self.update()  # Redraw with new dimensions

    def update(self, data: Dict = None):
        """Update visualization with new simulation data."""
        if not data:
            return
            
        agent_states = data.get("agent_states", [])
        resource_states = data.get("resource_states", [])
        
        self._draw_environment(agent_states, resource_states)

    def clear(self):
        """Clear the canvas."""
        self.canvas.delete("all")

    def show_message(self, message: str):
        """Display a message on the canvas."""
        self.clear()
        self.canvas.create_text(
            self.canvas_size[0] / 2,
            self.canvas_size[1] / 2,
            text=message,
            fill="white",
            justify=tk.CENTER,
            font=("Arial", 12)
        )

    def _draw_environment(self, agent_states: List, resource_states: List):
        """Draw the current state of the environment."""
        # Get canvas dimensions and create image
        width, height = self.canvas_size
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        # Calculate transformation parameters
        transform_params = self._calculate_transform_params(
            resource_states, width, height
        )

        # Track current agents for birth/death detection
        self._update_animation_states(agent_states)

        # Draw environment elements
        self._draw_resources(draw, resource_states, transform_params)
        self._draw_agents(draw, agent_states, transform_params)
        self._draw_birth_animations(draw, transform_params)
        self._draw_death_animations(draw, transform_params)
        self._draw_step_number(draw, transform_params)

        # Update canvas
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.image = photo  # Keep reference

    def _calculate_transform_params(self, resource_states: List, width: int, height: int) -> Dict:
        """Calculate scaling and offset parameters for coordinate transformation."""
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])

        scale_x = (width - 2 * VC["PADDING"]) / env_width
        scale_y = (height - 2 * VC["PADDING"]) / env_height
        scale = min(scale_x, scale_y)

        offset_x = max(VC["PADDING"], (width - (env_width * scale)) / 2)
        offset_y = max(VC["PADDING"], (height - (env_height * scale)) / 2)

        return {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "padding": VC["PADDING"],
            "width": width,
            "height": height,
        }

    def _transform_coords(self, x: float, y: float, params: Dict) -> Tuple[float, float]:
        """Transform environment coordinates to screen coordinates."""
        return (
            params["offset_x"] + x * params["scale"],
            params["offset_y"] + y * params["scale"]
        )

    def _update_animation_states(self, agent_states: List):
        """Update birth and death animation states."""
        current_agent_ids = {agent[0] for agent in agent_states}

        # Check for new births
        new_births = current_agent_ids - self.previous_agent_ids
        for agent_id in new_births:
            agent_data = next(a for a in agent_states if a[0] == agent_id)
            pos = (agent_data[2], agent_data[3])
            self.birth_animations[agent_id] = (pos, 0)

        # Check for deaths
        deaths = self.previous_agent_ids - current_agent_ids
        for agent_id in deaths:
            if agent_data := next(
                (a for a in self.previous_agent_states if a[0] == agent_id), None
            ):
                pos = (agent_data[2], agent_data[3])
                self.death_animations[agent_id] = (pos, 0)

        self.previous_agent_ids = current_agent_ids
        self.previous_agent_states = agent_states

    def _draw_resources(self, draw: ImageDraw, resource_states: List, params: Dict):
        """Draw resources as glowing squares."""
        for resource in resource_states:
            amount = resource[1]
            if amount > 0:
                x, y = self._transform_coords(resource[2], resource[3], params)

                # Calculate color intensity and size
                intensity = amount / VC["MAX_RESOURCE_AMOUNT"]
                resource_color = (
                    int(VC["RESOURCE_GLOW"]["RED"] * intensity),
                    int(VC["RESOURCE_GLOW"]["GREEN"] * intensity),
                    int(VC["RESOURCE_GLOW"]["BLUE"] * intensity)
                )

                size = max(1, int(VC["AGENT_RADIUS_SCALE"] * params["scale"]))
                radius = int(size * 0.2)

                self._draw_rounded_rectangle(draw, x, y, size, radius, resource_color)

    def _draw_rounded_rectangle(self, draw: ImageDraw, x: float, y: float, 
                              size: int, radius: int, color: Tuple[int, int, int]):
        """Draw a rectangle with rounded corners."""
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size

        # Main rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color)

        # Corner circles
        corners = [
            (x1, y1),  # Top-left
            (x2 - radius * 2, y1),  # Top-right
            (x1, y2 - radius * 2),  # Bottom-left
            (x2 - radius * 2, y2 - radius * 2),  # Bottom-right
        ]

        for corner_x, corner_y in corners:
            draw.ellipse(
                [corner_x, corner_y, corner_x + radius * 2, corner_y + radius * 2],
                fill=color
            )

    def _draw_agents(self, draw: ImageDraw, agent_states: List, params: Dict):
        """Draw agents as colored circles."""
        # Store agent positions for click detection
        self.last_agent_positions = {}
        self.last_transform_params = params

        for agent in agent_states:
            agent_id = agent[0]
            x, y = self._transform_coords(agent[2], agent[3], params)
            color = AGENT_COLORS.get(agent[1], "white")
            
            # Store position for click detection
            self.last_agent_positions[agent_id] = (agent[2], agent[3])

            radius = max(1, int(VC["AGENT_RADIUS_SCALE"] * params["scale"]))
            
            # Draw selection glow if this agent is selected
            if agent_id == self.selected_agent_id:
                # Draw outer glow
                glow_radius = radius + 3
                glow_color = (255, 255, 255, 128)  # Semi-transparent white
                draw.ellipse(
                    [(x - glow_radius, y - glow_radius),
                     (x + glow_radius, y + glow_radius)],
                    fill=None,
                    outline=glow_color,
                    width=2
                )

            # Draw agent
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color
            )

    def _draw_birth_animations(self, draw: ImageDraw, params: Dict):
        """Draw expanding circle animations for new agents."""
        births_to_remove = []
        for agent_id, (pos, frame) in self.birth_animations.items():
            if frame < VC["MAX_ANIMATION_FRAMES"]:
                x, y = self._transform_coords(pos[0], pos[1], params)
                radius = (
                    max(2, int(VC["BIRTH_RADIUS_SCALE"] * params["scale"]))
                    * (frame + 1)
                    / VC["MAX_ANIMATION_FRAMES"]
                )
                opacity = int(255 * (1 - frame / VC["MAX_ANIMATION_FRAMES"]))
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=(255, 255, 255, opacity)
                )
                self.birth_animations[agent_id] = (pos, frame + 1)
            else:
                births_to_remove.append(agent_id)

        for agent_id in births_to_remove:
            del self.birth_animations[agent_id]

    def _draw_death_animations(self, draw: ImageDraw, params: Dict):
        """Draw fading X mark animations for dying agents."""
        deaths_to_remove = []
        for agent_id, (pos, frame) in self.death_animations.items():
            if frame < VC["MAX_ANIMATION_FRAMES"]:
                x, y = self._transform_coords(pos[0], pos[1], params)
                size = max(1, int(VC["DEATH_MARK_SCALE"] * params["scale"]))
                opacity = int(128 * (1 - frame / VC["MAX_ANIMATION_FRAMES"]))
                color = (255, 0, 0, opacity)  # Red with fading opacity

                draw.line(
                    [(x - size, y - size), (x + size, y + size)],
                    fill=color,
                    width=1
                )
                draw.line(
                    [(x - size, y + size), (x + size, y - size)],
                    fill=color,
                    width=1
                )

                self.death_animations[agent_id] = (pos, frame + 1)
            else:
                deaths_to_remove.append(agent_id)

        for agent_id in deaths_to_remove:
            del self.death_animations[agent_id]

    def _draw_step_number(self, draw: ImageDraw, params: Dict):
        """Draw the current step number on the visualization."""
        font_size = max(
            VC["MIN_FONT_SIZE"],
            int(min(params["width"], params["height"]) / VC["FONT_SCALE_FACTOR"])
        )
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.text(
            (VC["PADDING"], VC["PADDING"]),
            f"Step: {getattr(self, 'current_step', 0)}",
            fill=(255, 255, 255),  # White
            font=font
        ) 

    def set_logger(self, logger):
        """Set the data logger for this component."""
        self.logger = logger

    def collect_action(self, **action_data):
        """Collect an action for batch processing."""
        if self.logger is not None:
            self.logger.log_agent_action(**action_data) 