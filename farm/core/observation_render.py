"""
Observation Rendering System for AgentFarm

This module provides comprehensive rendering utilities for visualizing
multichannel agent observation data. It supports both static image generation
and interactive HTML viewers for exploring observation tensors.

The rendering system is designed to handle the complex multichannel observation
data produced by AgentObservation, providing both programmatic access through
static images and interactive exploration through web-based viewers.

Key Components:
    - ChannelStyle: Configuration class for channel rendering styles
    - ObservationRenderer: Main static class providing rendering methods
    - Utility functions: Helper functions for image processing and HTML generation
    - Interactive viewer: Self-contained HTML/JavaScript viewer for observation exploration

Rendering Modes:
    - Overlay: Alpha-blended composite of all channels for compact visualization
    - Gallery: Grid layout showing each channel as separate tiles
    - Interactive HTML: Web-based viewer with zoom, channel selection, and tooltips

Features:
    - Support for custom color palettes and colormaps
    - Grid lines and center crosshairs for spatial reference
    - Multiple output formats (PIL Image, numpy array, PNG bytes)
    - Interactive controls for channel exploration
    - Responsive design for various screen sizes
    - Keyboard shortcuts for navigation

Usage:
    # Static rendering
    from farm.core.observation_render import ObservationRenderer
    image = ObservationRenderer.render(
        observation_tensor,
        channel_names=["SELF_HP", "ALLIES_HP", "ENEMIES_HP"],
        mode="overlay"
    )

    # Interactive HTML viewer
    html = ObservationRenderer.render_interactive_html(
        observation_tensor,
        channel_names=["SELF_HP", "ALLIES_HP", "ENEMIES_HP"],
        title="Agent Observation"
    )
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # Colormap support for continuous channels
    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mpl_colors
except Exception:  # pragma: no cover - optional dependency at runtime
    mpl_cm = None
    mpl_colors = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None


RgbTuple = Tuple[int, int, int]


def _hex_to_rgb(color_hex: str) -> RgbTuple:
    """Convert a hex color string to an RGB tuple.

    Args:
        color_hex: Hex color string with or without leading '#'. Supports both
            3-digit and 6-digit formats.

    Returns:
        Tuple of (r, g, b) integers in range [0, 255].

    Examples:
        >>> _hex_to_rgb("#ff0000")
        (255, 0, 0)
        >>> _hex_to_rgb("00f")
        (0, 0, 255)
    """
    color_hex = color_hex.lstrip("#")
    if len(color_hex) == 3:
        color_hex = "".join([c * 2 for c in color_hex])
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return (r, g, b)


def _ensure_numpy01(array_like: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:  # type: ignore
    """Ensure input array is a numpy array normalized to [0,1] range.

    Handles both numpy arrays and PyTorch tensors, converting them to float32
    numpy arrays with values clamped to [0, 1]. NaN values are replaced with 0.

    Args:
        array_like: Input array-like object (numpy array or torch tensor).

    Returns:
        Numpy array with dtype float32 and values in range [0, 1].

    Note:
        Uses torch if available, otherwise falls back to numpy-only conversion.
    """
    if torch is not None and isinstance(array_like, torch.Tensor):
        if array_like.is_cuda:
            array_like = array_like.detach().cpu()
        array_like = array_like.detach().to(dtype=torch.float32)
        npy = array_like.numpy()
    else:
        npy = np.asarray(array_like, dtype=np.float32)
    # Normalize/clamp to [0,1]
    npy = np.nan_to_num(npy, nan=0.0, posinf=1.0, neginf=0.0)
    npy = np.clip(npy, 0.0, 1.0)
    return npy


@dataclass
class ChannelStyle:
    """
    Styling configuration for rendering a single observation channel.

    This class defines the visual appearance of an observation channel when rendered
    as an image. It supports both solid color rendering and matplotlib colormaps for
    continuous value visualization.

    For solid color mode (when color_hex is provided):
    - The channel values modulate both the opacity and color intensity
    - Higher channel values result in more opaque and brighter colors
    - Useful for binary or discrete channels (e.g., obstacles, visibility)

    For colormap mode (when colormap is provided):
    - Channel values are mapped to colors using matplotlib colormaps
    - The colormap determines the color mapping, alpha controls overall opacity
    - Useful for continuous channels (e.g., terrain cost, health gradients)

    Attributes:
        color_hex: Optional hex color string (e.g., "#ff0000", "#00ff00") for solid color mode.
                  If None, colormap mode is used. Supports both 3-digit and 6-digit formats.
        alpha: Opacity multiplier in range [0.0, 1.0]. Final opacity = channel_value * alpha.
              Values outside this range raise ValueError.
        colormap: Optional matplotlib colormap name for continuous channels.
                 Common options: "viridis", "plasma", "magma", "inferno", "cividis".

    Raises:
        ValueError: If alpha is not in the range [0.0, 1.0].

    Examples:
        >>> # Solid color for discrete channels
        >>> style = ChannelStyle("#ff0000", 0.8)
        >>> style.color_hex
        '#ff0000'
        >>> style.alpha
        0.8

        >>> # Colormap for continuous channels
        >>> terrain_style = ChannelStyle(None, 0.9, "magma")
        >>> terrain_style.colormap
        'magma'

        >>> # Short hex format also works
        >>> style = ChannelStyle("#f00", 0.6)  # equivalent to "#ff0000"
    """

    color_hex: Optional[str]
    alpha: float
    colormap: Optional[str] = None

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"ChannelStyle.alpha must be in [0, 1], got {self.alpha}")


def get_default_palette(channel_names: List[str]) -> Dict[str, ChannelStyle]:
    """
    Generate a colorblind-friendly default palette for observation channels.

    This function creates a comprehensive color palette optimized for visualization
    of multichannel observation data. It provides predefined styles for standard
    AgentFarm observation channels and generates fallback colors for custom channels.

    The palette is designed with accessibility in mind, using colors that are
    distinguishable for viewers with various forms of color vision deficiency.

    Predefined Channels:
        - SELF_HP: Cyan (#00e5ff) - Agent's own health
        - ALLIES_HP: Green (#00c853) - Ally health values
        - ENEMIES_HP: Red (#ff1744) - Enemy health values
        - RESOURCES: Light green (#2ecc71) - Resource availability
        - OBSTACLES: Gray (#9e9e9e) - Obstacle/passability
        - TERRAIN_COST: Magma colormap - Movement cost (continuous)
        - VISIBILITY: White (#ffffff) - Field-of-view mask
        - KNOWN_EMPTY: Light blue-gray (#90a4ae) - Previously observed empty cells
        - DAMAGE_HEAT: Orange (#ff9100) - Recent damage events
        - TRAILS: Light blue (#00b8d4) - Movement trails
        - ALLY_SIGNAL: Yellow (#ffd600) - Communication signals
        - GOAL: Purple (#d500f9) - Goal/waypoint positions
        - LANDMARKS: Purple-blue (#7e57c2) - Permanent landmarks

    Fallback Colors:
        For unknown/custom channels, colors are assigned from a curated palette:
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
         "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    Args:
        channel_names: List of channel names to generate styles for.
                     Can include both predefined and custom channel names.

    Returns:
        Dictionary mapping each channel name to its corresponding ChannelStyle object.
        All channels in the input list will have an entry in the returned dictionary.

    Example:
        >>> palette = get_default_palette(["SELF_HP", "ENEMIES_HP", "CUSTOM_CHANNEL"])
        >>> palette["SELF_HP"].color_hex
        '#00e5ff'
        >>> palette["CUSTOM_CHANNEL"].alpha
        0.6
    """
    # Colorblind-friendly inspired palette
    default: Dict[str, ChannelStyle] = {
        "SELF_HP": ChannelStyle("#00e5ff", 0.95, None),
        "ALLIES_HP": ChannelStyle("#00c853", 0.85, None),
        "ENEMIES_HP": ChannelStyle("#ff1744", 0.85, None),
        "RESOURCES": ChannelStyle("#2ecc71", 0.60, None),
        "OBSTACLES": ChannelStyle("#9e9e9e", 0.65, None),
        "TERRAIN_COST": ChannelStyle(None, 0.90, "magma"),
        "VISIBILITY": ChannelStyle("#ffffff", 0.30, None),
        "KNOWN_EMPTY": ChannelStyle("#90a4ae", 0.15, None),
        "DAMAGE_HEAT": ChannelStyle("#ff9100", 0.75, None),
        "TRAILS": ChannelStyle("#00b8d4", 0.55, None),
        "ALLY_SIGNAL": ChannelStyle("#ffd600", 0.45, None),
        "GOAL": ChannelStyle("#d500f9", 0.95, None),
        "LANDMARKS": ChannelStyle("#7e57c2", 0.80, None),
    }
    # Add fallback for any unknown/custom channels
    fallback_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    idx = 0
    for name in channel_names:
        if name not in default:
            color = fallback_colors[idx % len(fallback_colors)]
            default[name] = ChannelStyle(color, 0.6, None)
            idx += 1
    return default


def _render_channels_overlay(
    gridC01: np.ndarray,  # (C,H,W) in [0,1]
    channel_names: List[str],
    palette: Dict[str, ChannelStyle],
    background: str,
) -> np.ndarray:
    """Render multiple channels as an alpha-blended overlay.

    Composites all channels on top of each other using alpha blending.
    Channels are rendered in order, with later channels appearing on top.

    Args:
        gridC01: Channel data with shape (C, H, W), values in [0, 1].
        channel_names: Names of channels corresponding to first dimension.
        palette: Dictionary mapping channel names to rendering styles.
        background: Background color as hex string.

    Returns:
        RGB image array with shape (H, W, 3) and values in [0, 1].

    Note:
        For colormap channels, uses matplotlib colormaps if available.
        For regular color channels, modulates color intensity by channel values.
        Alpha blending formula: out = src * a + dst * (1-a)
    """
    C, H, W = gridC01.shape
    base = np.zeros((H, W, 3), dtype=np.float32)
    bg_rgb = np.array(_hex_to_rgb(background), dtype=np.float32) / 255.0
    base[:] = bg_rgb

    for idx, name in enumerate(channel_names):
        layer = gridC01[idx]  # (H,W)
        style = palette.get(name, ChannelStyle("#ffffff", 0.5, None))

        if style.colormap and mpl_cm is not None:
            cmap = mpl_cm.get_cmap(style.colormap)
            rgba = cmap(layer)  # (H,W,4), already 0..1
            alpha = np.clip(rgba[..., 3] * style.alpha, 0.0, 1.0)
            color_rgb = rgba[..., :3]
        else:
            color_rgb = (
                np.array(_hex_to_rgb(style.color_hex or "#ffffff"), dtype=np.float32)
                / 255.0
            )
            # Use value as intensity for alpha scaling to emphasize higher values
            alpha = np.clip(layer * style.alpha, 0.0, 1.0)
            # Also modulate color intensity by value to keep minimalist look
            color_rgb = color_rgb[None, None, :] * layer[..., None]

        # Alpha blend: out = src * a + dst * (1-a)
        base = color_rgb * alpha[..., None] + base * (1.0 - alpha[..., None])

    return np.clip(base, 0.0, 1.0)


def _nearest_scale(img_rgb01: np.ndarray, size: int) -> Image.Image:
    """Scale an RGB image using nearest neighbor interpolation.

    Scales the image so that the longest dimension becomes approximately
    the target size, while maintaining aspect ratio.

    Args:
        img_rgb01: RGB image array with values in [0, 1] range.
        size: Target size for the longest dimension in pixels.

    Returns:
        PIL Image object scaled using nearest neighbor interpolation.
    """
    H, W = img_rgb01.shape[:2]
    # Decide target keeping square, use longest side = size
    scale = max(1, int(math.ceil(size / max(H, W))))
    target = (W * scale, H * scale)
    pil = Image.fromarray((img_rgb01 * 255.0).astype(np.uint8), mode="RGB")
    return pil.resize(target, resample=Image.NEAREST)


def _draw_grid_and_center(
    image: Image.Image, draw_grid: bool, draw_center: bool, cell_size: int
) -> None:
    """Draw grid lines and center crosshairs on an image.

    Args:
        image: PIL Image to draw on (modified in-place).
        draw_grid: Whether to draw grid lines.
        draw_center: Whether to draw center crosshairs.
        cell_size: Size of each cell in pixels for grid alignment.

    Note:
        Grid lines are only drawn when cell_size >= 8 to avoid clutter.
        Grid lines use semi-transparent white, center lines use more opaque white.
    """
    if not (draw_grid or draw_center):
        return
    draw = ImageDraw.Draw(image)
    W, H = image.size
    if draw_grid and cell_size >= 8:
        # subtle grid lines
        grid_color = (255, 255, 255, 32)
        for x in range(0, W, cell_size):
            draw.line([(x, 0), (x, H)], fill=grid_color)
        for y in range(0, H, cell_size):
            draw.line([(0, y), (W, y)], fill=grid_color)
    if draw_center:
        cx = (W // (2 * cell_size)) * cell_size + cell_size // 2
        cy = (H // (2 * cell_size)) * cell_size + cell_size // 2
        cross_color = (255, 255, 255, 128)
        draw.line([(0, cy), (W, cy)], fill=cross_color)
        draw.line([(cx, 0), (cx, H)], fill=cross_color)


def _render_gallery(
    gridC01: np.ndarray,
    channel_names: List[str],
    palette: Dict[str, ChannelStyle],
    background: str,
    size: int,
) -> Image.Image:
    """Render channels as a grid gallery with individual channel tiles.

    Creates a grid layout where each channel is rendered as a separate tile
    with its name labeled. Tiles are arranged to approximate a square grid.

    Args:
        gridC01: Channel data with shape (C, H, W), values in [0, 1].
        channel_names: Names of channels for labeling tiles.
        palette: Dictionary mapping channel names to rendering styles.
        background: Background color as hex string.
        size: Target size for the entire gallery (longest dimension).

    Returns:
        PIL Image containing the grid of channel tiles.

    Note:
        Each tile shows only one channel overlaid on the background.
        Channel names are drawn as labels in the top-left corner of each tile.
        Grid layout automatically adjusts columns/rows for best fit.
    """
    C, H, W = gridC01.shape
    cols = int(math.ceil(math.sqrt(C)))
    rows = int(math.ceil(C / cols))
    # Render each as overlay of a single channel for consistency
    tiles: List[Image.Image] = []
    for idx, name in enumerate(channel_names):
        single = gridC01[idx : idx + 1, :, :]
        img01 = _render_channels_overlay(single, [name], palette, background)
        tile = _nearest_scale(img01, size // max(cols, rows))
        # label
        draw = ImageDraw.Draw(tile)
        label_bg = (0, 0, 0, 128)
        text = name
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([(2, 2), (2 + 6 * len(text), 16)], fill=label_bg)
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        tiles.append(tile)

    tile_w, tile_h = tiles[0].size
    grid_img = Image.new(
        "RGB", (tile_w * cols, tile_h * rows), color=_hex_to_rgb(background)
    )
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        grid_img.paste(tile, (c * tile_w, r * tile_h))
    return grid_img


class ObservationRenderer:
    """Static utility class for rendering multichannel observation data.

    Provides methods to render AgentObservation data as static images or
    interactive HTML viewers. Supports both overlay and gallery rendering modes.
    """

    @staticmethod
    def render(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        mode: str = "overlay",
        size: int = 256,
        palette: Optional[Dict[str, ChannelStyle]] = None,
        grid: bool = False,
        legend: bool = False,  # reserved for future use
        background: str = "#111213",
        draw_center: bool = True,
        return_type: str = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """Render observation data as an image.

        Args:
            observation: Channel data with shape (C, H, W). Can be numpy array or torch tensor.
            channel_names: Names of channels for styling and labeling.
            mode: Rendering mode - "overlay" for alpha-blended composite, "gallery" for grid.
            size: Target size for output image (longest dimension).
            palette: Optional custom color palette. Uses defaults if None.
            grid: Whether to draw grid lines on the output.
            legend: Reserved for future use (currently unused).
            background: Background color as hex string.
            draw_center: Whether to draw center crosshairs.
            return_type: Output format - "pil" for PIL Image, "numpy" for array, "bytes" for PNG bytes.

        Returns:
            Rendered image in requested format.

        Raises:
            ValueError: If observation shape is invalid or channel count doesn't match names.

        Examples:
            >>> obs = np.random.rand(3, 10, 10)
            >>> img = ObservationRenderer.render(obs, ["hp", "enemies", "terrain"])
            >>> img.show()  # PIL Image
        """
        gridC01 = _ensure_numpy01(observation)
        if gridC01.ndim != 3:
            raise ValueError("observation must have shape (C,H,W)")
        C = gridC01.shape[0]
        if len(channel_names) != C:
            raise ValueError("channel_names length must match observation channels")

        style_palette = palette or get_default_palette(channel_names)

        if mode == "overlay":
            img01 = _render_channels_overlay(
                gridC01, channel_names, style_palette, background
            )
            img = _nearest_scale(img01, size)
            cell_size = max(1, img.size[0] // gridC01.shape[2])
            _draw_grid_and_center(img, grid, draw_center, cell_size)
        elif mode == "gallery":
            img = _render_gallery(
                gridC01, channel_names, style_palette, background, size
            )
        else:
            raise ValueError("mode must be 'overlay' or 'gallery'")

        if return_type == "pil":
            return img
        if return_type == "numpy":
            return np.asarray(img)
        if return_type == "bytes":
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        raise ValueError("return_type must be 'pil', 'numpy', or 'bytes'")

    @staticmethod
    def to_interactive_json(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        meta: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> Dict:
        """Convert observation data to JSON format for interactive viewing.

        Serializes observation data into a format suitable for the interactive
        HTML viewer, including channel names, grid data, and optional metadata.

        Args:
            observation: Channel data with shape (C, H, W). Can be numpy array or torch tensor.
            channel_names: Names of channels for labeling.
            meta: Optional metadata dictionary to include in JSON output.

        Returns:
            Dictionary containing observation data in JSON-serializable format.

        Note:
            The returned dictionary includes:
            - "shape": [C, H, W] dimensions
            - "channels": list of channel names
            - "grid": 3D array of observation values
            - "meta": metadata dictionary or empty dict
        """
        gridC01 = _ensure_numpy01(observation)
        C, H, W = gridC01.shape
        payload = {
            "shape": [int(C), int(H), int(W)],
            "channels": list(channel_names),
            "grid": gridC01.tolist(),
            "meta": meta or {},
        }
        return payload

    @staticmethod
    def render_interactive_html(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        outfile: Optional[str] = None,
        title: str = "Observation Viewer",
        background: str = "#0b0c0f",
        palette: Optional[Dict[str, ChannelStyle]] = None,
        initial_scale: int = 16,
    ) -> str:
        """Generate interactive HTML viewer for observation data.

        Creates a complete HTML page with embedded JavaScript that provides
        an interactive viewer for multichannel observation data. Features include:
        - Channel selection and overlay toggles
        - Zoom controls with scale slider
        - Grid and center crosshair toggles
        - Hover tooltips showing channel values
        - Keyboard shortcuts ([/] for prev/next channel)

        Args:
            observation: Channel data with shape (C, H, W). Can be numpy array or torch tensor.
            channel_names: Names of channels for labeling and controls.
            outfile: Optional file path to save HTML output.
            title: Title for the HTML page.
            background: Background color for the viewer interface.
            palette: Optional custom color palette. Uses defaults if None.
            initial_scale: Initial pixel scale factor for rendering.

        Returns:
            Complete HTML string containing the interactive viewer.

        Note:
            If outfile is provided, the HTML is also saved to that file.
            The viewer uses a dark theme with shadcn-inspired styling.
        """
        data = ObservationRenderer.to_interactive_json(observation, channel_names)
        palette_dict = {
            name: {"color": cs.color_hex, "alpha": cs.alpha, "cmap": cs.colormap}
            for name, cs in (palette or get_default_palette(channel_names)).items()
        }
        html = _build_interactive_html(
            data, title, background, palette_dict, initial_scale
        )
        if outfile:
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def _build_interactive_html(
    data: Dict,
    title: str,
    background: str,
    palette: Dict[str, Dict],
    initial_scale: int,
) -> str:
    """Build the complete HTML string for the interactive observation viewer.

    Constructs a self-contained HTML page with embedded CSS and JavaScript
    that provides an interactive interface for exploring multichannel observation data.

    Args:
        data: JSON-serializable observation data from to_interactive_json().
        title: Page title and heading.
        background: Hex color for the viewer's background theme.
        palette: Channel styling information for the JavaScript renderer.
        initial_scale: Starting pixel scale factor for the canvas.

    Returns:
        Complete HTML document as a string.

    Note:
        The generated HTML includes:
        - Responsive dark theme with CSS custom properties
        - Canvas-based rendering with zoom and pan controls
        - Interactive controls for channel selection and visualization options
        - Hover tooltips showing precise channel values
        - Keyboard shortcuts for navigation
    """
    # Minimal, shadcn-inspired styling with a clean dark theme.
    json_data = json.dumps(data)
    json_palette = json.dumps(palette)
    title_placeholder = title
    background_placeholder = background
    initial_scale_placeholder = str(int(initial_scale))
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: __BACKGROUND__;
      --fg: #e5e7eb;
      --muted: #a1a1aa;
      --card: #111217;
      --accent: #3b82f6;
      --ring: #60a5fa;
      --border: #1f2937;
    }}
    html, body {{ height: 100%; margin: 0; background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, Apple Color Emoji, Segoe UI Emoji; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }}
    .row {{ display: flex; gap: 12px; align-items: stretch; }}
    .col {{ flex: 1; min-width: 0; }}
    .panel {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
    .select, .slider {{ appearance: none; background: var(--bg); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }}
    .btn {{ background: #0f172a; color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; cursor: pointer; }}
    .btn:focus {{ outline: 2px solid var(--ring); outline-offset: 2px; }}
    .pill {{ padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; cursor: pointer; font-size: 12px; }}
    .pill.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    .legend-item {{ display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--muted); }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--border); }}
    #canvas-wrap {{ position: relative; }}
    #tooltip {{ position: absolute; pointer-events: none; background: rgba(17,18,23,0.95); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; font-size: 12px; display: none; min-width: 160px; }}
    .kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace; background: #0b1220; border: 1px solid var(--border); border-radius: 6px; padding: 1px 4px; font-size: 11px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"row\" style=\"gap: 16px\">
      <div class=\"col\">
        <div class=\"card\">
          <div class=\"panel\">
            <select id=\"channelSelect\" class=\"select\"></select>
            <button id=\"prevBtn\" class=\"btn\">Prev (<span class=\"kbd\">[</span>)</button>
            <button id=\"nextBtn\" class=\"btn\">Next (<span class=\"kbd\">]</span>)</button>
            <label class=\"pill\"><input type=\"checkbox\" id=\"overlayToggle\" checked /> Overlay</label>
            <label class=\"pill\"><input type=\"checkbox\" id=\"gridToggle\" /> Grid</label>
            <label class=\"pill\"><input type=\"checkbox\" id=\"centerToggle\" checked /> Center</label>
            <label style=\"margin-left:8px\">Scale <input type=\"range\" id=\"scale\" class=\"slider\" min=\"4\" max=\"48\" step=\"1\" value=\"__INITIAL_SCALE__\" /></label>
          </div>
          <div id=\"canvas-wrap\" class=\"card\" style=\"padding:8px\">
            <canvas id=\"canvas\"></canvas>
            <div id=\"tooltip\"></div>
          </div>
          <div class=\"legend\" id=\"legend\"></div>
        </div>
      </div>
    </div>
  </div>
  <script id=\"obs-data\" type=\"application/json\">__DATA__</script>
  <script id=\"palette\" type=\"application/json\">__PALETTE__</script>
  <script>
    const data = JSON.parse(document.getElementById('obs-data').textContent);
    const palette = JSON.parse(document.getElementById('palette').textContent);
    const C = data.shape[0], H = data.shape[1], W = data.shape[2];
    const channels = data.channels;
    const grid = data.grid; // C x H x W
    let scale = parseInt(document.getElementById('scale').value, 10) || 16;
    let overlay = true;
    let showGrid = false;
    let showCenter = true;
    let currentIndex = 0;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');

    function setCanvasSize() {
      canvas.width = W * scale;
      canvas.height = H * scale;
    }

    function hexToRgb(hex) {
      hex = hex.replace('#','');
      if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
      const num = parseInt(hex, 16);
      return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
    }

    function drawGrid() {
      if (!showGrid || scale < 8) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= W; x++) {
        ctx.beginPath(); ctx.moveTo(x*scale + 0.5, 0); ctx.lineTo(x*scale + 0.5, H*scale); ctx.stroke();
      }
      for (let y = 0; y <= H; y++) {
        ctx.beginPath(); ctx.moveTo(0, y*scale + 0.5); ctx.lineTo(W*scale, y*scale + 0.5); ctx.stroke();
      }
      ctx.restore();
    }

    function drawCenter() {
      if (!showCenter) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      const cx = Math.floor(W/2)*scale + Math.floor(scale/2) + 0.5;
      const cy = Math.floor(H/2)*scale + Math.floor(scale/2) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W*scale, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H*scale); ctx.stroke();
      ctx.restore();
    }

    function lerp(a,b,t){ return a + (b-a)*t; }

    function drawSingle(idx) {
      const imgData = ctx.createImageData(W*scale, H*scale);
      const channelName = channels[idx];
      const pal = palette[channelName] || { color: '#ffffff', alpha: 0.6 };
      const base = hexToRgb('__BACKGROUND__');
      // Pre-fill background
      for (let i=0;i<imgData.data.length;i+=4){
        imgData.data[i] = base.r; imgData.data[i+1] = base.g; imgData.data[i+2] = base.b; imgData.data[i+3] = 255;
      }
      const layer = grid[idx];
      const color = hexToRgb(pal.color || '#ffffff');
      for (let y=0;y<H;y++){
        for (let x=0;x<W;x++){
          const v = Math.max(0, Math.min(1, layer[y][x]));
          const a = Math.max(0, Math.min(1, v * (pal.alpha ?? 0.6)));
          const r = Math.round(lerp(base.r, color.r * v, a));
          const g = Math.round(lerp(base.g, color.g * v, a));
          const b = Math.round(lerp(base.b, color.b * v, a));
          for (let sy=0; sy<scale; sy++){
            for (let sx=0; sx<scale; sx++){
              const px = (y*scale + sy) * (W*scale) + (x*scale + sx);
              const off = px*4;
              imgData.data[off] = r; imgData.data[off+1] = g; imgData.data[off+2] = b; imgData.data[off+3] = 255;
            }
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      drawGrid();
      drawCenter();
    }

    function drawOverlay() {
      // Start with background
      ctx.save();
      const base = '__BACKGROUND__';
      ctx.fillStyle = base; ctx.fillRect(0,0,W*scale,H*scale);
      ctx.restore();
      for (let idx=0; idx<C; idx++){
        const channelName = channels[idx];
        const pal = palette[channelName] || { color: '#ffffff', alpha: 0.6 };
        const color = hexToRgb(pal.color || '#ffffff');
        const layer = grid[idx];
        ctx.save();
        for (let y=0;y<H;y++){
          for (let x=0;x<W;x++){
            const v = Math.max(0, Math.min(1, layer[y][x]));
            const a = v * (pal.alpha ?? 0.6);
            if (a <= 0) continue;
            ctx.globalAlpha = Math.max(0, Math.min(1, a));
            ctx.fillStyle = `rgb(${Math.round(color.r*v)},${Math.round(color.g*v)},${Math.round(color.b*v)})`;
            ctx.fillRect(x*scale, y*scale, scale, scale);
          }
        }
        ctx.restore();
      }
      drawGrid();
      drawCenter();
    }

    function render() {
      setCanvasSize();
      if (overlay) drawOverlay(); else drawSingle(currentIndex);
    }

    function updateLegend(){
      const legend = document.getElementById('legend');
      legend.innerHTML='';
      channels.forEach((name, i) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        const sw = document.createElement('div');
        sw.className = 'swatch';
        const pal = palette[name] || { color: '#ffffff' };
        sw.style.background = pal.color || '#ffffff';
        const label = document.createElement('span');
        label.textContent = name;
        if (i === currentIndex && !overlay) {
          item.classList.add('active');
          item.style.color = 'var(--fg)';
        }
        item.appendChild(sw); item.appendChild(label);
        legend.appendChild(item);
      });
    }

    // UI wiring
    const select = document.getElementById('channelSelect');
    channels.forEach((name, i) => {
      const opt = document.createElement('option'); opt.value = i; opt.text = name; select.appendChild(opt);
    });
    select.addEventListener('change', (e) => { currentIndex = parseInt(e.target.value,10) || 0; if(!overlay) render(); updateLegend(); });
    document.getElementById('prevBtn').addEventListener('click', () => { currentIndex = (currentIndex - 1 + C) % C; select.value = currentIndex; if(!overlay) render(); updateLegend(); });
    document.getElementById('nextBtn').addEventListener('click', () => { currentIndex = (currentIndex + 1) % C; select.value = currentIndex; if(!overlay) render(); updateLegend(); });
    document.getElementById('overlayToggle').addEventListener('change', (e) => { overlay = !!e.target.checked; render(); updateLegend(); });
    document.getElementById('gridToggle').addEventListener('change', (e) => { showGrid = !!e.target.checked; render(); });
    document.getElementById('centerToggle').addEventListener('change', (e) => { showCenter = !!e.target.checked; render(); });
    document.getElementById('scale').addEventListener('input', (e) => { scale = parseInt(e.target.value,10)||16; render(); });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === '[') { document.getElementById('prevBtn').click(); }
      if (e.key === ']') { document.getElementById('nextBtn').click(); }
    });

    // Hover tooltip
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / scale);
      const y = Math.floor((e.clientY - rect.top) / scale);
      if (x < 0 || y < 0 || x >= W || y >= H){ tooltip.style.display='none'; return; }
      const values = channels.map((name, i) => ({ name, v: grid[i][y][x] }));
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
      tooltip.style.top = (e.clientY - rect.top + 12) + 'px';
      const centerY = Math.floor(H/2), centerX = Math.floor(W/2);
      const dy = y - centerY, dx = x - centerX;
      const dist = Math.sqrt(dx*dx + dy*dy).toFixed(2);
      let html = `<div style="font-weight:600">(y=${y}, x=${x}) d=${dist}</div>`;
      values.forEach(it => { html += `<div><span style="color:#94a3b8">${it.name}</span>: ${it.v.toFixed(3)}</div>`; });
      tooltip.innerHTML = html;
    });
    canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });

    // Initial
    render(); updateLegend();
  </script>
</body>
</html>"""
    html = html.replace("__TITLE__", title_placeholder)
    html = html.replace("__BACKGROUND__", background_placeholder)
    html = html.replace("__INITIAL_SCALE__", initial_scale_placeholder)
    html = html.replace("__DATA__", json_data)
    html = html.replace("__PALETTE__", json_palette)
    return html
