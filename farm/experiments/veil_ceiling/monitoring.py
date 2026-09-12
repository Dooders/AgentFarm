"""Monitoring schedule and cue channel (design sections 5.3-5.5).

Two independent monitor maps are maintained: the *true* map, which decides
whether an over-harvest is penalised, and a *decoy* map with identical
coverage, epoch schedule and spatial restriction that is never used for
enforcement. The cue an agent sees is a noisy copy of one of the two maps,
selected by :attr:`MonitoringConfig.decorrelated`.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.veil_ceiling.config import MonitoringConfig, WorldConfig


class Monitoring:
    """Epoch-resampled monitor masks over grid cells plus the cue channel."""

    def __init__(
        self,
        world: WorldConfig,
        cfg: MonitoringConfig,
        true_rng: np.random.Generator,
        decoy_rng: np.random.Generator,
        cue_rng: np.random.Generator,
    ) -> None:
        self.world = world
        self.cfg = cfg
        self._true_rng = true_rng
        self._decoy_rng = decoy_rng
        self._cue_rng = cue_rng
        xs = np.arange(world.n_cells) % world.width
        self.training_cells = np.flatnonzero(xs < world.heldout_min_x)
        self.all_cells = np.arange(world.n_cells)
        self.true_mask = np.zeros(world.n_cells, dtype=bool)
        self.decoy_mask = np.zeros(world.n_cells, dtype=bool)
        self.include_heldout = False

    def _sample_mask(self, rng: np.random.Generator, eligible: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.world.n_cells, dtype=bool)
        k = round(self.cfg.coverage * eligible.size)
        if k > 0:
            mask[rng.choice(eligible, size=k, replace=False)] = True
        return mask

    def maybe_resample(self, tick: int, include_heldout: bool) -> bool:
        """Re-draw both maps at epoch boundaries; returns whether a draw happened."""
        boundary = tick % self.cfg.epoch_ticks == 0
        if not boundary and include_heldout == self.include_heldout:
            return False
        self.include_heldout = include_heldout
        eligible = self.all_cells if include_heldout else self.training_cells
        # Always consume both streams so the true map is identical across
        # conditions that share a seed, whether or not the decoy is used.
        self.true_mask = self._sample_mask(self._true_rng, eligible)
        self.decoy_mask = self._sample_mask(self._decoy_rng, eligible)
        return True

    def is_monitored(self, cell: int) -> bool:
        return bool(self.true_mask[cell])

    def cue(self, cell: int) -> int:
        """Cue bit reported to an agent standing on ``cell``."""
        if self.cfg.fidelity is None:
            return 0
        source = self.decoy_mask[cell] if self.cfg.decorrelated else self.true_mask[cell]
        flip = self._cue_rng.random() < self.cfg.flip_probability
        return int(bool(source) ^ flip)

    def coverage_realised(self) -> float:
        eligible = self.all_cells if self.include_heldout else self.training_cells
        return float(self.true_mask[eligible].mean()) if eligible.size else 0.0
