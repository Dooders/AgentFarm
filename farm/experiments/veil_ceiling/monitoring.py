"""Monitoring schedule and cue channel (design sections 5.3-5.5).

Two independent monitor maps are maintained: the *true* map, which decides
whether an over-harvest is penalised, and a *decoy* map with identical
coverage, epoch schedule and spatial restriction that is never used for
enforcement. The cue an agent sees is a noisy copy of one of the two maps,
selected by :attr:`MonitoringConfig.decorrelated`.

Adaptive policies keep the same coverage budget and re-weight the next
epoch's true map from last epoch's observed residual. The decoy map is
always drawn uniformly so a leaked cue of an adaptive true map cannot be
confounded with a leaked cue of an adaptive decoy.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.veil_ceiling.config import (
    MONITOR_POLICY_ADAPTIVE_BLIND,
    MONITOR_POLICY_ADAPTIVE_CELLS,
    MONITOR_POLICY_ADAPTIVE_MOVE,
    MONITOR_POLICY_STATIC,
    MonitoringConfig,
    WorldConfig,
)

RESIDUAL_DEFECTIONS = "defections"
RESIDUAL_MOVES = "moves"


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
        self.epoch_index = -1
        self.last_overlap = float("nan")
        self.last_kl = float("nan")
        self.last_weight_gini = float("nan")
        self.n_adaptive_draws = 0
        self._residual = np.zeros(world.n_cells)
        self._eligible = self.training_cells

    def record_residual(self, cell: int, kind: str) -> None:
        """Accumulate an observed residual on ``cell`` during the current epoch."""
        if (
            kind == RESIDUAL_DEFECTIONS
            and self.cfg.policy
            in (
                MONITOR_POLICY_ADAPTIVE_CELLS,
                MONITOR_POLICY_ADAPTIVE_BLIND,
            )
            or kind == RESIDUAL_MOVES
            and self.cfg.policy == MONITOR_POLICY_ADAPTIVE_MOVE
        ):
            self._residual[cell] += 1.0

    def _sample_uniform(self, rng: np.random.Generator, eligible: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.world.n_cells, dtype=bool)
        k = round(self.cfg.coverage * eligible.size)
        if k > 0:
            mask[rng.choice(eligible, size=k, replace=False)] = True
        return mask

    def _weights(self, eligible: np.ndarray) -> np.ndarray:
        floor = self.cfg.smoothing if self.cfg.smoothing > 0.0 else 1e-12
        residual = self._residual[eligible] + floor
        total = residual.sum()
        if total <= 0.0:
            return np.full(eligible.size, 1.0 / eligible.size)
        return residual / total

    @staticmethod
    def _gini(weights: np.ndarray) -> float:
        n = weights.size
        if n <= 1:
            return 0.0
        ordered = np.sort(weights)
        ranks = np.arange(1, n + 1)
        return float((2.0 * np.dot(ranks, ordered) / (n * ordered.sum())) - (n + 1) / n)

    @staticmethod
    def _kl(weights: np.ndarray) -> float:
        uniform = 1.0 / weights.size
        clipped = np.clip(weights, 1e-12, 1.0)
        return float(np.sum(clipped * np.log(clipped / uniform)))

    def _sample_weighted(self, rng: np.random.Generator, eligible: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.world.n_cells, dtype=bool)
        k = round(self.cfg.coverage * eligible.size)
        if k <= 0:
            self.last_overlap = float("nan")
            self.last_kl = 0.0
            self.last_weight_gini = 0.0
            return mask
        weights = self._weights(eligible)
        chosen = rng.choice(eligible, size=k, replace=False, p=weights)
        mask[chosen] = True
        prev = self.true_mask[eligible]
        next_ = mask[eligible]
        denom = max(int(prev.sum()), 1)
        self.last_overlap = float((prev & next_).sum() / denom)
        self.last_kl = self._kl(weights)
        self.last_weight_gini = self._gini(weights)
        self.n_adaptive_draws += 1
        return mask

    def maybe_resample(self, tick: int, include_heldout: bool) -> bool:
        """Re-draw both maps at epoch boundaries; returns whether a draw happened."""
        boundary = tick % self.cfg.epoch_ticks == 0
        if not boundary and include_heldout == self.include_heldout:
            return False
        self.include_heldout = include_heldout
        eligible = self.all_cells if include_heldout else self.training_cells
        self._eligible = eligible
        # Always consume the decoy stream so seed-matched static cells keep
        # the same true-map stream as the original design. Adaptive true maps
        # consume extra RNG only from the true stream, after the first epoch.
        previous = self.true_mask.copy()
        use_adaptive = (
            self.cfg.policy != MONITOR_POLICY_STATIC
            and self.epoch_index >= 0
            and not include_heldout
            and self._residual[eligible].sum() > 0.0
        )
        if use_adaptive:
            self.true_mask = self._sample_weighted(self._true_rng, eligible)
        else:
            self.true_mask = self._sample_uniform(self._true_rng, eligible)
            if self.epoch_index >= 0 and previous.any():
                prev = previous[eligible]
                next_ = self.true_mask[eligible]
                denom = max(int(prev.sum()), 1)
                self.last_overlap = float((prev & next_).sum() / denom)
            else:
                self.last_overlap = float("nan")
            self.last_kl = 0.0
            self.last_weight_gini = 0.0
        self.decoy_mask = self._sample_uniform(self._decoy_rng, eligible)
        self._residual[:] = 0.0
        self.epoch_index += 1
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
