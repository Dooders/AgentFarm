import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Create directory for saving visualizations
os.makedirs("output", exist_ok=True)


def generate_memory_agent_overview():
    """Generate a high-level visualization of the Memory Agent architecture with improved spacing."""
    # -- Figure and axis setup --
    fig, ax = plt.subplots(figsize=(14, 9))  # Increase figure size
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.92, bottom=0.08
    )  # Add a bit of margin

    # -- Color palette --
    colors = {
        "background": "#f5f5f5",
        "agent": "#e1f5fe",
        "stm": "#ffcdd2",
        "im": "#ffe0b2",
        "ltm": "#c8e6c9",
        "encoder": "#d1c4e9",
        "retrieval": "#bbdefb",
        "decision": "#b2dfdb",
        "arrows": "#78909c",
        "text": "#37474f",
    }

    # -- Background --
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["background"])

    # -- Agent boundary (slightly smaller & more transparent for clarity) --
    agent_rect = patches.Rectangle(
        (0.7, 0.7),
        11,
        7,
        linewidth=2,
        edgecolor="#455a64",
        facecolor=colors["agent"],
        alpha=0.2,
        zorder=0,
    )
    ax.add_patch(agent_rect)

    # -- Memory components: shift rectangles up/down slightly to avoid collisions --
    stm_rect = patches.Rectangle(
        (1, 5.3),
        3,
        1.4,
        linewidth=1.5,
        edgecolor="#b71c1c",
        facecolor=colors["stm"],
        alpha=0.7,
        zorder=2,
    )
    im_rect = patches.Rectangle(
        (4.7, 5.3),
        3,
        1.4,
        linewidth=1.5,
        edgecolor="#e65100",
        facecolor=colors["im"],
        alpha=0.7,
        zorder=2,
    )
    ltm_rect = patches.Rectangle(
        (8.4, 5.3),
        3,
        1.4,
        linewidth=1.5,
        edgecolor="#2e7d32",
        facecolor=colors["ltm"],
        alpha=0.7,
        zorder=2,
    )

    # -- Encoders (shift slightly) --
    stm_encoder = patches.Rectangle(
        (3.2, 3.7),
        1.3,
        0.8,
        linewidth=1,
        edgecolor="#4527a0",
        facecolor=colors["encoder"],
        alpha=0.7,
        zorder=2,
    )
    im_encoder = patches.Rectangle(
        (6.9, 3.7),
        1.3,
        0.8,
        linewidth=1,
        edgecolor="#4527a0",
        facecolor=colors["encoder"],
        alpha=0.7,
        zorder=2,
    )

    # -- Retrieval and decision components --
    retrieval = patches.Rectangle(
        (3.0, 2.0),
        3,
        1,
        linewidth=1.5,
        edgecolor="#01579b",
        facecolor=colors["retrieval"],
        alpha=0.7,
        zorder=2,
    )
    decision = patches.Rectangle(
        (6.7, 2.0),
        3,
        1,
        linewidth=1.5,
        edgecolor="#00695c",
        facecolor=colors["decision"],
        alpha=0.7,
        zorder=2,
    )

    # -- Environment input/output (shift them slightly outward) --
    env_input = patches.Rectangle(
        (0.2, 1.5),
        1.5,
        2,
        linewidth=1.5,
        edgecolor="#263238",
        facecolor="#cfd8dc",
        alpha=0.7,
        zorder=2,
    )
    env_output = patches.Rectangle(
        (11.2, 1.5),
        1.5,
        2,
        linewidth=1.5,
        edgecolor="#263238",
        facecolor="#cfd8dc",
        alpha=0.7,
        zorder=2,
    )

    # -- Add all patches --
    for patch in [
        stm_rect,
        im_rect,
        ltm_rect,
        stm_encoder,
        im_encoder,
        retrieval,
        decision,
        env_input,
        env_output,
    ]:
        ax.add_patch(patch)

    # -- Arrows: narrower heads to reduce overlap --
    arrow_params = dict(
        color=colors["arrows"], width=0.01, head_width=0.06, head_length=0.09
    )

    # STM → IM via encoder
    draw_arrow(ax, 4, 6, 4.7, 6, **arrow_params)
    draw_arrow(ax, 2.5, 5.3, 3.2, 4.5, **arrow_params)

    # IM → LTM via encoder
    draw_arrow(ax, 7.7, 6, 8.4, 6, **arrow_params)
    draw_arrow(ax, 5.8, 5.3, 6.9, 4.5, **arrow_params)

    # Memory retrieval flows (use dotted/dashed styles)
    draw_arrow(ax, 2.5, 5.3, 4.5, 2.5, linestyle=":", **arrow_params)
    draw_arrow(ax, 6, 5.3, 4.5, 2.5, linestyle=":", **arrow_params)
    draw_arrow(ax, 9.5, 5.3, 4.5, 2.5, linestyle="--", **arrow_params)

    # Decision flow
    draw_arrow(ax, 6, 2.5, 6.7, 2.5, **arrow_params)

    # Environment interactions
    draw_arrow(ax, 1.7, 2.5, 3, 2.5, **arrow_params)  # Env → Agent
    draw_arrow(ax, 9.7, 2.5, 11.2, 2.5, **arrow_params)  # Agent → Env

    # -- Labels (slightly smaller font to reduce collisions) --
    label_fontsize = 9
    add_label(
        ax, 2.5, 6.0, "Short-Term Memory", fontsize=label_fontsize, color=colors["text"]
    )
    add_label(
        ax,
        6.2,
        6.0,
        "Intermediate Memory",
        fontsize=label_fontsize,
        color=colors["text"],
    )
    add_label(
        ax, 9.9, 6.0, "Long-Term Memory", fontsize=label_fontsize, color=colors["text"]
    )
    add_label(ax, 3.9, 4.1, "STM→IM\nEncoder", fontsize=8, color=colors["text"])
    add_label(ax, 7.55, 4.1, "IM→LTM\nEncoder", fontsize=8, color=colors["text"])
    add_label(
        ax, 4.5, 2.5, "Memory\nRetrieval", fontsize=label_fontsize, color=colors["text"]
    )
    add_label(
        ax, 8.2, 2.5, "Decision\nMaking", fontsize=label_fontsize, color=colors["text"]
    )
    add_label(ax, 1, 2.5, "Input\nPerception", fontsize=8, color=colors["text"])
    add_label(ax, 12, 2.5, "Action\nOutput", fontsize=8, color=colors["text"])

    # -- Title (move slightly up) --
    ax.text(
        6.5,
        8.3,
        "Memory Agent Architecture",
        fontsize=14,
        ha="center",
        fontweight="bold",
        color="#263238",
    )

    # -- Memory tier details (shift them slightly) --
    add_memory_details(
        ax,
        1.1,
        5.0,
        "STM",
        "• Full fidelity\n• Recent experiences\n• High dimensionality",
    )
    add_memory_details(
        ax,
        4.8,
        5.0,
        "IM",
        "• Medium compression\n• Moderate age\n• Semantic preservation",
    )
    add_memory_details(
        ax,
        8.6,
        5.0,
        "LTM",
        "• High compression\n• Distant experiences\n• Core patterns only",
    )

    # -- Legend: place in top-left corner to avoid crowding near env input --
    legend_x, legend_y = 1.0, 4.2
    draw_arrow(
        ax, legend_x, legend_y, legend_x + 0.6, legend_y, colors["arrows"], width=0.01
    )
    add_label(
        ax,
        legend_x + 1.3,
        legend_y,
        "Data flow",
        fontsize=8,
        color=colors["text"],
        ha="left",
    )

    legend_y -= 0.3
    draw_arrow(
        ax,
        legend_x,
        legend_y,
        legend_x + 0.6,
        legend_y,
        colors["arrows"],
        width=0.01,
        linestyle=":",
    )
    add_label(
        ax,
        legend_x + 1.3,
        legend_y,
        "STM/IM retrieval",
        fontsize=8,
        color=colors["text"],
        ha="left",
    )

    legend_y -= 0.3
    draw_arrow(
        ax,
        legend_x,
        legend_y,
        legend_x + 0.6,
        legend_y,
        colors["arrows"],
        width=0.01,
        linestyle="--",
    )
    add_label(
        ax,
        legend_x + 1.3,
        legend_y,
        "LTM retrieval",
        fontsize=8,
        color=colors["text"],
        ha="left",
    )

    # -- Ax limits and final save (no tight_layout, we use subplots_adjust instead) --
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    plt.savefig("output/memory_agent_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "docs/experiments/memory_agent/visualizations/memory_agent_architecture.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated Memory Agent architecture visualization (refined).")


def generate_memory_entry_diagram():
    """Generate a visualization of a MemoryEntry structure with improved spacing."""
    fig, ax = plt.subplots(figsize=(11, 7))  # Slightly bigger than original
    fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1)

    # -- Color palette --
    colors = {
        "background": "#f5f5f5",
        "metadata": "#e1f5fe",
        "content": "#fff9c4",
        "embedding": "#d1c4e9",
        "compression": "#dcedc8",
        "border": "#455a64",
        "text": "#37474f",
        "highlight": "#f44336",
    }

    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["background"])

    # -- Outline --
    memory_rect = patches.Rectangle(
        (0.8, 0.8),
        9,
        5,
        linewidth=2,
        edgecolor=colors["border"],
        facecolor="white",
        alpha=0.8,
        zorder=0,
    )
    ax.add_patch(memory_rect)

    # -- Components: shift slightly for more consistent margins --
    metadata = patches.Rectangle(
        (1.2, 4.3),
        8,
        1,
        linewidth=1,
        edgecolor=colors["border"],
        facecolor=colors["metadata"],
        alpha=0.7,
        zorder=2,
    )
    content = patches.Rectangle(
        (1.2, 2.9),
        3.8,
        1,
        linewidth=1,
        edgecolor=colors["border"],
        facecolor=colors["content"],
        alpha=0.7,
        zorder=2,
    )
    embedding = patches.Rectangle(
        (5.3, 2.9),
        3.8,
        1,
        linewidth=1,
        edgecolor=colors["border"],
        facecolor=colors["embedding"],
        alpha=0.7,
        zorder=2,
    )
    compression = patches.Rectangle(
        (1.2, 1.6),
        8,
        1,
        linewidth=1,
        edgecolor=colors["border"],
        facecolor=colors["compression"],
        alpha=0.7,
        zorder=2,
    )

    for patch in [metadata, content, embedding, compression]:
        ax.add_patch(patch)

    add_label(ax, 5.2, 4.8, "Memory Metadata", fontsize=11, color=colors["text"])
    add_label(ax, 3.1, 3.4, "Original Content", fontsize=11, color=colors["text"])
    add_label(ax, 7.2, 3.4, "Vector Embedding", fontsize=11, color=colors["text"])
    add_label(ax, 5.2, 2.1, "Compression Data", fontsize=11, color=colors["text"])

    # -- Title --
    ax.text(
        5.2,
        6.2,
        "Memory Entry Structure",
        fontsize=16,
        ha="center",
        fontweight="bold",
        color="#263238",
    )

    # -- Metadata fields --
    y_pos = 4.1
    field_x = 1.4
    fields_meta = [
        "memory_id: str",
        "creation_time: int",
        "last_access_time: int",
        "importance: float",
        "retrieval_count: int",
    ]
    for field in fields_meta:
        add_label(
            ax, field_x, y_pos, field, fontsize=9, color=colors["text"], ha="left"
        )
        y_pos -= 0.25

    # -- Content fields --
    y_pos = 2.7
    field_x = 1.4
    fields_content = [
        "agent_state: AgentState",
        "perception: PerceptionData",
        "action_taken: Action",
        "reward: float",
    ]
    for field in fields_content:
        add_label(
            ax, field_x, y_pos, field, fontsize=9, color=colors["text"], ha="left"
        )
        y_pos -= 0.25

    # -- Embedding fields --
    y_pos = 2.7
    field_x = 5.5
    fields_emb = [
        "embedding: ndarray",
        "dimensions: int",
        "created_from: str",
        "similarity_index: float",
    ]
    for field in fields_emb:
        add_label(
            ax, field_x, y_pos, field, fontsize=9, color=colors["text"], ha="left"
        )
        y_pos -= 0.25

    # -- Compression fields --
    y_pos = 1.4
    field_x = 1.4
    fields_comp = [
        "compression_level: int",
        "compressed_data: Optional[ndarray]",
        "reconstruction_loss: float",
        "decompression_fn: Callable",
    ]
    for field in fields_comp:
        add_label(
            ax, field_x, y_pos, field, fontsize=9, color=colors["text"], ha="left"
        )
        y_pos -= 0.25

    # -- Memory tier indicators (shift them a bit) --
    add_memory_tier_indicator(ax, 8.7, 4.6, "STM", "#ef5350")
    add_memory_tier_indicator(ax, 8.7, 4.3, "IM", "#ff9800")
    add_memory_tier_indicator(ax, 8.7, 4.0, "LTM", "#66bb6a")

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    plt.savefig("output/memory_entry_structure.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "docs/experiments/memory_agent/visualizations/memory_entry_structure.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated MemoryEntry structure visualization (refined).")


def generate_memory_transition_diagram():
    """Generate a visualization of memory transitions between tiers with improved layout."""
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.1)

    # -- Color palette --
    colors = {
        "background": "#f5f5f5",
        "stm": "#ffcdd2",
        "im": "#ffe0b2",
        "ltm": "#c8e6c9",
        "text": "#37474f",
        "arrows": "#78909c",
        "highlight": "#f44336",
    }

    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["background"])

    # -- Cylinders for memory tiers: slightly shift them for clarity --
    stm_ellipse_top = patches.Ellipse(
        (2, 3.3),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#b71c1c",
        facecolor=colors["stm"],
        alpha=0.8,
        zorder=2,
    )
    stm_rect = patches.Rectangle(
        (0.75, 1.8),
        2.5,
        1.5,
        linewidth=1.5,
        edgecolor="#b71c1c",
        facecolor=colors["stm"],
        alpha=0.8,
        zorder=1,
    )
    stm_ellipse_bottom = patches.Ellipse(
        (2, 1.8),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#b71c1c",
        facecolor=colors["stm"],
        alpha=0.8,
        zorder=0,
    )

    im_ellipse_top = patches.Ellipse(
        (6, 3.3),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#e65100",
        facecolor=colors["im"],
        alpha=0.4,
        zorder=2,
    )
    im_rect = patches.Rectangle(
        (4.75, 1.8),
        2.5,
        1.5,
        linewidth=1.5,
        edgecolor="#e65100",
        facecolor=colors["im"],
        alpha=0.4,
        zorder=1,
    )
    im_ellipse_bottom = patches.Ellipse(
        (6, 1.8),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#e65100",
        facecolor=colors["im"],
        alpha=0.4,
        zorder=0,
    )

    ltm_ellipse_top = patches.Ellipse(
        (10, 3.3),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#2e7d32",
        facecolor=colors["ltm"],
        alpha=0.2,
        zorder=2,
    )
    ltm_rect = patches.Rectangle(
        (8.75, 1.8),
        2.5,
        1.5,
        linewidth=1.5,
        edgecolor="#2e7d32",
        facecolor=colors["ltm"],
        alpha=0.2,
        zorder=1,
    )
    ltm_ellipse_bottom = patches.Ellipse(
        (10, 1.8),
        2.5,
        0.6,
        linewidth=1.5,
        edgecolor="#2e7d32",
        facecolor=colors["ltm"],
        alpha=0.2,
        zorder=0,
    )

    for patch in [
        stm_ellipse_top,
        stm_rect,
        stm_ellipse_bottom,
        im_ellipse_top,
        im_rect,
        im_ellipse_bottom,
        ltm_ellipse_top,
        ltm_rect,
        ltm_ellipse_bottom,
    ]:
        ax.add_patch(patch)

    # -- Memory dots (reduced number for clarity) --
    add_memory_dots(ax, 2, 2.6, 10, "#b71c1c", 0.1, high_dim=True)
    add_memory_dots(ax, 6, 2.6, 6, "#e65100", 0.08, high_dim=False)
    add_memory_dots(ax, 10, 2.6, 4, "#2e7d32", 0.06, high_dim=False)

    # -- Transition arrows + labels (shift text a bit) --
    draw_transition_arrow(ax, 3.3, 2.6, 4.7, 2.6, "Compression\n(~50%)")
    draw_transition_arrow(ax, 7.3, 2.6, 8.7, 2.6, "Compression\n(~75%)")

    # -- Transition criteria --
    add_label(
        ax,
        6.5,
        6.5,
        "Memory Transition Criteria",
        fontsize=12,
        color=colors["text"],
        weight="bold",
    )

    # Left-align bullet points
    bullet_x = 4.0  # Starting x position for left alignment
    add_label(
        ax,
        bullet_x,
        6.1,
        "• Age-based: Oldest memories move first",
        fontsize=10,
        color=colors["text"],
        ha="left",
    )
    add_label(
        ax,
        bullet_x,
        5.8,
        "• Importance-based: Low importance moves first",
        fontsize=10,
        color=colors["text"],
        ha="left",
    )
    add_label(
        ax,
        bullet_x,
        5.5,
        "• Hybrid approach combines both factors",
        fontsize=10,
        color=colors["text"],
        ha="left",
    )

    # -- Tier labels --
    add_label(
        ax, 2, 4.8, "Short-Term Memory", fontsize=11, color="#b71c1c", weight="bold"
    )
    add_label(
        ax, 6, 4.8, "Intermediate Memory", fontsize=11, color="#e65100", weight="bold"
    )
    add_label(
        ax, 10, 4.8, "Long-Term Memory", fontsize=11, color="#2e7d32", weight="bold"
    )

    # -- Example memory dimension rectangles (repositioned with equal vertical spacing) --
    entry_height = 0.4
    entry_width = 0.8

    # Calculate position for better vertical alignment
    title_y = 4.8
    cylinder_top_y = 3.3
    spacing = (title_y - cylinder_top_y) / 2
    dimension_y = title_y - spacing  # Halfway between title and cylinder top

    # Position under the titles and above the cylinders
    stm_entry = patches.Rectangle(
        (1.6, dimension_y - entry_height / 2),
        entry_width,
        entry_height,
        linewidth=1,
        edgecolor="#b71c1c",
        facecolor=colors["stm"],
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(stm_entry)
    add_label(ax, 2.0, dimension_y, "500D", fontsize=9, color="#b71c1c")

    im_entry = patches.Rectangle(
        (5.7, dimension_y - entry_height / 2),
        entry_width * 0.8,
        entry_height,
        linewidth=1,
        edgecolor="#e65100",
        facecolor=colors["im"],
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(im_entry)
    add_label(ax, 6.0, dimension_y, "100D", fontsize=9, color="#e65100")

    ltm_entry = patches.Rectangle(
        (9.8, dimension_y - entry_height / 2),
        entry_width * 0.5,
        entry_height,
        linewidth=1,
        edgecolor="#2e7d32",
        facecolor=colors["ltm"],
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(ltm_entry)
    add_label(ax, 10.0, dimension_y, "20D", fontsize=9, color="#2e7d32")

    # -- Arrows for dimension rectangles --
    arrow_params = dict(
        color=colors["arrows"], width=0.01, head_width=0.05, head_length=0.07
    )
    draw_arrow(ax, 2.4, dimension_y, 5.7, dimension_y, **arrow_params)
    draw_arrow(ax, 6.34, dimension_y, 9.8, dimension_y, **arrow_params)

    # -- Memory descriptors (under cylinders) --
    add_label(
        ax, 2, 0.9, "High-Dimensional\nFull Resolution", fontsize=8, color="#b71c1c"
    )
    add_label(
        ax,
        6,
        0.9,
        "Medium Compression\nKey Features Preserved",
        fontsize=8,
        color="#e65100",
    )
    add_label(
        ax, 10, 0.9, "High Compression\nCore Patterns Only", fontsize=8, color="#2e7d32"
    )

    # -- Capacity indicators: position them slightly below each cylinder --
    add_capacity_indicator(ax, 2, 0.5, 100, 1)
    add_capacity_indicator(ax, 6, 0.5, 500, 0.7)
    add_capacity_indicator(ax, 10, 0.5, 1000, 0.6)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    plt.savefig("output/memory_transitions.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "docs/experiments/memory_agent/visualizations/memory_transitions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated memory transitions visualization (refined).")


# ----------------
# Helper functions
# ----------------


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color,
    width=0.02,
    head_width=0.1,
    head_length=0.1,
    linestyle="-",
):
    """Draw an arrow from (x1, y1) to (x2, y2)."""
    ax.arrow(
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        head_width=head_width,
        head_length=head_length,
        fc=color,
        ec=color,
        linewidth=1,
        length_includes_head=True,
        linestyle=linestyle,
        alpha=0.8,
    )


def add_label(
    ax,
    x,
    y,
    text,
    fontsize=10,
    color="black",
    ha="center",
    va="center",
    weight="normal",
):
    """Add a text label at position (x, y)."""
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        fontweight=weight,
        zorder=10,
    )


def add_memory_details(ax, x, y, title, text):
    """Add memory tier details with a bold title and bullet text."""
    ax.text(
        x, y, title + ":", fontsize=9, fontweight="bold", color="#263238", ha="left"
    )
    ax.text(x, y - 0.1, text, fontsize=8, color="#37474f", va="top", ha="left")


def add_memory_tier_indicator(ax, x, y, text, color):
    """Add a colored box + label to indicate a memory tier."""
    box = patches.Rectangle(
        (x, y - 0.08),
        0.3,
        0.16,
        linewidth=1,
        edgecolor=color,
        facecolor=color,
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(box)
    add_label(ax, x + 0.15, y, text, fontsize=8, color="white")


def add_memory_dots(ax, center_x, center_y, num_dots, color, radius, high_dim=False):
    """Scatter small circles to represent memory entries."""
    np.random.seed(42)  # For reproducibility

    # Generate random positions within the cylinder
    x_offset = np.random.uniform(-1.0, 1.0, num_dots)
    y_offset = np.random.uniform(-0.6, 0.6, num_dots)

    # Draw the dots
    for i in range(num_dots):
        dot_x = center_x + x_offset[i]
        dot_y = center_y + y_offset[i]

        # Draw connections for high-dimensional entries
        if high_dim and i < num_dots - 1:
            next_x = center_x + x_offset[i + 1]
            next_y = center_y + y_offset[i + 1]
            ax.plot(
                [dot_x, next_x], [dot_y, next_y], color=color, alpha=0.3, linewidth=0.5
            )

        # Draw the dot
        circle = patches.Circle(
            (dot_x, dot_y),
            radius=radius,
            linewidth=0,
            facecolor=color,
            alpha=0.8,
            zorder=5,
        )
        ax.add_patch(circle)


def add_capacity_indicator(ax, x, y, capacity, fill_ratio):
    """Add a capacity bar indicator showing memory utilization."""
    width = 2.0
    height = 0.2

    # Background bar (total capacity)
    background = patches.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=1,
        edgecolor="#455a64",
        facecolor="#e0e0e0",
        alpha=0.7,
        zorder=3,
    )
    ax.add_patch(background)

    # Fill bar (used capacity)
    fill = patches.Rectangle(
        (x - width / 2, y - height / 2),
        width * fill_ratio,
        height,
        linewidth=0,
        facecolor="#455a64",
        alpha=0.8,
        zorder=4,
    )
    ax.add_patch(fill)

    # Capacity label
    add_label(ax, x, y - 0.25, f"Capacity: {capacity}", fontsize=8, color="#37474f")


def draw_transition_arrow(ax, x1, y1, x2, y2, label_text):
    """Draw a labeled transition arrow between memory tiers."""
    arrow_params = dict(color="#546e7a", width=0.015, head_width=0.1, head_length=0.15)
    draw_arrow(ax, x1, y1, x2, y2, **arrow_params)

    # Position the label above the arrow
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2 + 0.3
    add_label(ax, mid_x, mid_y, label_text, fontsize=8, color="#37474f")


# Execute all visualization functions if this script is run directly
if __name__ == "__main__":
    print("Generating memory agent visualizations...")
    generate_memory_agent_overview()
    generate_memory_entry_diagram()
    generate_memory_transition_diagram()
    print("All visualizations completed successfully.")
