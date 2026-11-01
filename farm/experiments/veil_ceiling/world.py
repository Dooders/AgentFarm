"""Resource field for the veil-ceiling grid world.

Nodes sit on distinct cells. Stock regenerates logistically toward
``node_max_amount`` while it is at or above ``regen_threshold``; below the
threshold regeneration is multiplied by ``suppressed_regen_factor``. Ordinary
gathering never draws a node below the threshold, so the only way to suppress
a node is the over-harvest affordance.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.veil_ceiling.config import WorldConfig


class ResourceField:
    """Vectorised store of resource nodes plus per-cell neighbourhood lookups."""

    def __init__(self, cfg: WorldConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        cells = rng.choice(cfg.n_cells, size=cfg.n_nodes, replace=False)
        cells.sort()
        self.xs = (cells % cfg.width).astype(np.int64)
        self.ys = (cells // cfg.width).astype(np.int64)
        self.amount = np.full(cfg.n_nodes, float(cfg.node_initial_amount))
        self.cell_to_node = np.full(cfg.n_cells, -1, dtype=np.int64)
        self.cell_to_node[cells] = np.arange(cfg.n_nodes)
        self._neighbours = self._build_neighbour_lists(cfg.harvest_range)

    def _build_neighbour_lists(self, radius: int) -> list[np.ndarray]:
        cfg = self.cfg
        out: list[np.ndarray] = []
        for cell in range(cfg.n_cells):
            cx, cy = cell % cfg.width, cell // cfg.width
            near = np.flatnonzero((np.abs(self.xs - cx) <= radius) & (np.abs(self.ys - cy) <= radius))
            out.append(near)
        return out

    def cell_index(self, x: int, y: int) -> int:
        return y * self.cfg.width + x

    def nodes_in_range(self, x: int, y: int) -> np.ndarray:
        return self._neighbours[self.cell_index(x, y)]

    def best_node_in_range(self, x: int, y: int) -> int:
        """Index of the richest node within harvest range, or -1."""
        near = self.nodes_in_range(x, y)
        if near.size == 0:
            return -1
        return int(near[int(np.argmax(self.amount[near]))])

    def nearest_healthy_node(self, x: int, y: int) -> tuple[int, float]:
        """Nearest node whose stock is above the regeneration threshold.

        Returns ``(node_index, chebyshev_distance)``; ``(-1, inf)`` when no
        node is currently healthy.
        """
        healthy = np.flatnonzero(self.amount > self.cfg.regen_threshold)
        if healthy.size == 0:
            return -1, float("inf")
        dist = np.maximum(np.abs(self.xs[healthy] - x), np.abs(self.ys[healthy] - y))
        k = int(np.argmin(dist))
        return int(healthy[k]), float(dist[k])

    def regenerate(self) -> None:
        cfg = self.cfg
        stock = np.maximum(self.amount, cfg.regen_floor)
        growth = cfg.regen_rate * stock * (1.0 - self.amount / cfg.node_max_amount)
        suppressed = self.amount < cfg.regen_threshold
        growth = np.where(suppressed, growth * cfg.suppressed_regen_factor, growth)
        self.amount = np.minimum(self.amount + np.maximum(growth, 0.0), cfg.node_max_amount)

    def gather(self, node: int) -> float:
        """Sustainable draw: never takes the node below the regeneration threshold."""
        available = self.amount[node] - self.cfg.regen_threshold
        taken = float(min(self.cfg.gather_amount, max(available, 0.0)))
        self.amount[node] -= taken
        return taken

    def over_harvest(self, node: int) -> float:
        """Defection: draws below the threshold, suppressing future regeneration."""
        taken = float(min(self.cfg.gather_amount, max(self.amount[node], 0.0)))
        self.amount[node] -= taken
        return taken

    def gather_yield(self, node: int) -> float:
        return float(min(self.cfg.gather_amount, max(self.amount[node] - self.cfg.regen_threshold, 0.0)))

    def over_harvest_yield(self, node: int) -> float:
        return float(min(self.cfg.gather_amount, max(self.amount[node], 0.0)))

    def is_defection_opportunity(self, node: int) -> bool:
        """True when over-harvesting ``node`` pays strictly more, net of effort, than gathering.

        Restricting opportunities to individually profitable defections keeps the
        denominator of the defection rate to decisions where the affordance
        actually tempts (design section 5.2).
        """
        if node < 0:
            return False
        if self.amount[node] < self.cfg.regen_threshold:
            return False
        return self.over_harvest_yield(node) - self.cfg.over_harvest_effort > self.gather_yield(node)

    def total_stock(self) -> float:
        return float(self.amount.sum())

    def suppressed_fraction(self) -> float:
        return float(np.mean(self.amount < self.cfg.regen_threshold))

    def node_position(self, node: int) -> tuple[int, int] | None:
        if node < 0:
            return None
        return int(self.xs[node]), int(self.ys[node])
