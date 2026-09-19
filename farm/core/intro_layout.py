"""Frame geometry for the intro explainer.

Pure arithmetic in Manim units so the layout can be tested without a render.
The frame is the 16:9 Manim default: 14.222 wide by 8.0 tall, origin centered.
"""

from __future__ import annotations

FRAME_WIDTH = 14.222222
FRAME_HEIGHT = 8.0

MARGIN_TOP = 0.44
MARGIN_BOTTOM = 0.40
MARGIN_SIDE = 0.60

# Space between the top banner and the stage, and between the stage and a footer.
GAP_BANNER = 0.52
GAP_FOOTER = 0.46

# Space between a section heading and its caption.
GAP_HEADING = 0.30

MAX_CELL = 0.56


def frame_top() -> float:
    return FRAME_HEIGHT / 2


def frame_bottom() -> float:
    return -FRAME_HEIGHT / 2


def safe_width() -> float:
    return FRAME_WIDTH - 2 * MARGIN_SIDE


def banner_top() -> float:
    """Y coordinate where the top banner starts."""
    return frame_top() - MARGIN_TOP


def stage_bounds(banner_height: float, footer_height: float = 0.0) -> tuple[float, float]:
    """Top and bottom Y of the free area between the banner and any footer."""
    top = banner_top() - banner_height - GAP_BANNER
    bottom = frame_bottom() + MARGIN_BOTTOM + GAP_FOOTER
    if footer_height > 0:
        bottom += footer_height
    if bottom >= top:
        raise ValueError(f"No stage room: banner={banner_height}, footer={footer_height}")
    return top, bottom


def stage_center(top: float, bottom: float) -> float:
    """Y coordinate of the middle of a stage span."""
    return (top + bottom) / 2


def fit_cell(top: float, bottom: float, cells: int, max_cell: float = MAX_CELL) -> float:
    """Largest square cell that lets ``cells`` rows fit inside a stage span."""
    if cells <= 0:
        raise ValueError(f"cells must be positive, got {cells}")
    span = top - bottom
    if span <= 0:
        raise ValueError(f"Stage span must be positive, got {span}")
    return min(max_cell, span / cells)


def fits_frame(top: float, bottom: float) -> bool:
    """True if a block stays inside the vertical safe area."""
    return top <= frame_top() - MARGIN_TOP + 1e-6 and bottom >= frame_bottom() + MARGIN_BOTTOM - 1e-6
