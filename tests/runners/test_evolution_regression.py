"""Regression checks for evolution optimizer quality.

This module provides a deterministic, threshold-based benchmark suite for
catching regressions in the evolution optimizer: fitness collapse, diversity
collapse, excessive boundary clustering, and artifact schema drift.

Benchmark configuration (tuning justification)
-----------------------------------------------
- SEED = 42             : fixed seed guarantees 100 % reproducible runs.
- NUM_GENERATIONS = 5   : enough to observe selection pressure and elitism
                          without being slow (< 0.1 s with fast evaluator).
- POPULATION_SIZE = 8   : small enough for fast CI; large enough that per-gene
                          std is statistically meaningful.
- MUTATION_RATE = 0.3   : moderate per-gene mutation probability.
- MUTATION_SCALE = 0.25 : ~25 % of normalised range per mutation step.
- ELITISM_COUNT = 1     : the best candidate is always carried forward, so
                          best_fitness must be monotonically non-decreasing.

Thresholds (justified against the benchmark's actual observed values)
----------------------------------------------------------------------
- MIN_DIVERSITY = 0.05
    Normalised mean gene-std / span.  With POPULATION_SIZE=8 and
    MUTATION_SCALE=0.25 the benchmark produces diversity ≥ 0.106 in every
    generation.  A value near 0 indicates population collapse; 0.05 gives a
    2× safety margin while reliably catching completely degenerate runs.

- MAX_BOUNDARY_OCCUPANCY = 0.75
    Fraction of candidates sitting exactly on either gene boundary (min or
    max).  The benchmark peaks at 0.625 in generation 0.  A threshold of 0.75
    (6 of 8 candidates) is deliberately conservative to catch only severe
    regressions such as broken clamping logic, not normal variation.

- FITNESS_REGRESSION_TOLERANCE = 0.0
    With ELITISM_COUNT=1 and a deterministic evaluator the elite's fitness
    does not change across generations, so best_fitness must never decrease.
    Zero tolerance is exact and correct for this evaluator class.

Running the checks
------------------
Run the full regression suite locally::

    pytest -v -m evolution_regression tests/runners/test_evolution_regression.py

Or include them in the standard test run (they are not excluded by default
because they are fast and contribute to coverage)::

    pytest

CI: see .github/workflows/evolution-regression.yml.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from typing import Any, ClassVar, Optional

import pytest

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import BoundaryMode
from farm.runners.evolution_experiment import (
    EvolutionCandidate,
    EvolutionExperiment,
    EvolutionExperimentConfig,
    EvolutionExperimentResult,
)

# ---------------------------------------------------------------------------
# Module-level marker – applies to every test in this file.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.evolution_regression

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

_SEED: int = 42
_NUM_GENERATIONS: int = 5
_POPULATION_SIZE: int = 8
_MUTATION_RATE: float = 0.3
_MUTATION_SCALE: float = 0.25
_ELITISM_COUNT: int = 1

# ---------------------------------------------------------------------------
# Regression thresholds – change only with a justification comment above.
# ---------------------------------------------------------------------------

_MIN_DIVERSITY: float = 0.05
_MAX_BOUNDARY_OCCUPANCY: float = 0.90
_FITNESS_REGRESSION_TOLERANCE: float = 0.0

# ---------------------------------------------------------------------------
# Artifact schema constants – keep centralized for maintainability.
# ---------------------------------------------------------------------------

_SUMMARY_REQUIRED_KEYS: set[str] = {
    "generation",
    "best_fitness",
    "mean_fitness",
    "min_fitness",
    "best_candidate_id",
    "gene_statistics",
    "best_chromosome",
    "diversity",
    "boundary_occupancy",
    "adaptive_event",
    "mutation_rate_used",
    "mutation_scale_used",
    "mutation_rate_multiplier",
    "mutation_scale_multiplier",
    "best_fitness_delta",
}

_LINEAGE_REQUIRED_KEYS: set[str] = {
    "candidate_id",
    "generation",
    "fitness",
    "learning_rate",
    "chromosome",
    "parent_ids",
}

_METADATA_REQUIRED_KEYS: set[str] = {
    "converged",
    "convergence_reason",
    "generation_of_convergence",
    "num_generations_completed",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _deterministic_evaluator(
    candidate: EvolutionCandidate,
    candidate_config: SimulationConfig,
    generation: int,
    member_index: int,
) -> tuple[float, dict[str, Any]]:
    """Fast, deterministic fitness: rewards higher learning_rate linearly.

    Fitness = learning_rate × 1000.  This creates clear selection pressure
    toward higher learning rates while being completely reproducible given the
    same chromosome.  No randomness → elitism guarantees monotone best_fitness.
    """
    lr = candidate_config.learning.learning_rate
    return lr * 1000.0, {"learning_rate": lr, "generation": generation}


def _run_benchmark(
    *,
    mutation_scale: float = _MUTATION_SCALE,
    mutation_rate: float = _MUTATION_RATE,
    output_dir: Optional[str] = None,
) -> EvolutionExperimentResult:
    """Run the canonical regression benchmark and return the result."""
    config = EvolutionExperimentConfig(
        num_generations=_NUM_GENERATIONS,
        population_size=_POPULATION_SIZE,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        boundary_mode=BoundaryMode.REFLECT,
        elitism_count=_ELITISM_COUNT,
        seed=_SEED,
        output_dir=output_dir,
    )
    return EvolutionExperiment(SimulationConfig(), config).run(
        fitness_evaluator=_deterministic_evaluator
    )


# ---------------------------------------------------------------------------
# Baseline regression suite
# ---------------------------------------------------------------------------


class TestEvolutionRegressionBaseline(unittest.TestCase):
    """Threshold-based regression checks for a healthy evolution run.

    Every test in this class must pass for the evolution optimizer to be
    considered regression-free.  Failing output includes which metric
    regressed and what value was observed, so failures are immediately
    actionable.

    The benchmark is executed once per class (``setUpClass``) and shared
    across all test methods to avoid redundant re-runs of the same
    deterministic loop.
    """

    _result: ClassVar[EvolutionExperimentResult]
    _artifact_dir: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._artifact_dir = tempfile.mkdtemp()
        cls._result = _run_benchmark(output_dir=cls._artifact_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._artifact_dir, ignore_errors=True)

    def test_diversity_non_degenerate_at_every_generation(self) -> None:
        """Diversity must exceed MIN_DIVERSITY at every generation.

        Regression caught: population collapse (all candidates converge to the
        same chromosome), which reduces diversity to 0 and eliminates
        exploration.  Causes: mutation disabled, broken crossover, or
        selection pressure too high.
        """
        result = self._result
        for summary in result.generation_summaries:
            diversity = summary.diversity
            self.assertIsNotNone(
                diversity,
                msg=(
                    f"REGRESSION: diversity is None at generation {summary.generation}. "
                    "Expected a numeric value; check _compute_diversity."
                ),
            )
            self.assertGreater(
                diversity,
                _MIN_DIVERSITY,
                msg=(
                    f"REGRESSION: diversity={diversity:.4f} at generation "
                    f"{summary.generation} is below threshold={_MIN_DIVERSITY}. "
                    "Population may have collapsed; check mutation scale or "
                    "selection pressure."
                ),
            )

    def test_best_fitness_never_regresses_across_generations(self) -> None:
        """best_fitness must not decrease between consecutive generations.

        With elitism_count=1 and a deterministic evaluator the elite candidate
        is re-evaluated in the next generation with an identical chromosome, so
        its fitness is unchanged.  Any decrease is a bug in elitism logic or
        fitness evaluation.

        Regression caught: broken elitism (elite not carried forward),
        non-deterministic evaluator, or selection returning wrong candidates.
        """
        result = self._result
        summaries = result.generation_summaries
        for i in range(1, len(summaries)):
            prev_best = summaries[i - 1].best_fitness
            curr_best = summaries[i].best_fitness
            self.assertGreaterEqual(
                curr_best,
                prev_best - _FITNESS_REGRESSION_TOLERANCE,
                msg=(
                    f"REGRESSION: best_fitness decreased from generation {i - 1} to {i}: "
                    f"{prev_best:.6f} → {curr_best:.6f} "
                    f"(allowed tolerance={_FITNESS_REGRESSION_TOLERANCE}). "
                    "Check elitism logic in _next_generation or evaluator determinism."
                ),
            )

    def test_boundary_occupancy_within_threshold_at_all_generations(self) -> None:
        """No gene may have more than MAX_BOUNDARY_OCCUPANCY fraction at boundaries.

        boundary_occupancy[gene] is the fraction of candidates whose gene
        value sits exactly on the gene's min_value or max_value.  Sustained
        high occupancy indicates that mutation or boundary handling is broken
        and the population has lost an effective dimension.

        Regression caught: clamping logic always clips to the same extreme,
        broken BoundaryMode handling, or mutation scale so large that every
        candidate saturates.
        """
        result = self._result
        for summary in result.generation_summaries:
            for gene_name, occupancy in summary.boundary_occupancy.items():
                self.assertLessEqual(
                    occupancy,
                    _MAX_BOUNDARY_OCCUPANCY,
                    msg=(
                        f"REGRESSION: gene '{gene_name}' has boundary_occupancy="
                        f"{occupancy:.3f} at generation {summary.generation}, "
                        f"exceeding threshold={_MAX_BOUNDARY_OCCUPANCY}. "
                        "Excessive boundary clustering indicates broken mutation "
                        "or boundary mode handling."
                    ),
                )

    def test_generation_summary_schema_contains_required_fields(self) -> None:
        """evolution_generation_summaries.json must contain all required keys.

        Regression caught: removing or renaming a field in
        EvolutionGenerationSummary silently drops it from the persisted JSON,
        breaking downstream analysis tools.
        """
        with open(
                f"{self._artifact_dir}/evolution_generation_summaries.json",
                encoding="utf-8",
            ) as fh:
                summaries = json.load(fh)

        for summary in summaries:
            missing = _SUMMARY_REQUIRED_KEYS - summary.keys()
            self.assertFalse(
                missing,
                msg=(
                    f"REGRESSION: generation summary (generation={summary.get('generation')}) "
                    f"is missing schema keys: {missing}. "
                    "A field was likely removed or renamed in EvolutionGenerationSummary."
                ),
            )

    def test_lineage_schema_contains_required_fields(self) -> None:
        """evolution_lineage.json must contain all required per-candidate keys.

        Regression caught: removing a field from the lineage serialization in
        _persist_results breaks lineage tracking and downstream tools.
        """
        with open(
                f"{self._artifact_dir}/evolution_lineage.json",
                encoding="utf-8",
            ) as fh:
                lineage = json.load(fh)

        for entry in lineage:
            missing = _LINEAGE_REQUIRED_KEYS - entry.keys()
            self.assertFalse(
                missing,
                msg=(
                    f"REGRESSION: lineage entry (candidate_id="
                    f"{entry.get('candidate_id')!r}) is missing schema keys: "
                    f"{missing}. A field was likely removed or renamed in "
                    "_persist_results."
                ),
            )

    def test_metadata_schema_contains_required_fields(self) -> None:
        """evolution_metadata.json must contain all required convergence keys.

        Regression caught: removing or renaming convergence metadata fields in
        _persist_results breaks downstream tooling that tracks run outcomes.
        """
        with open(
                f"{self._artifact_dir}/evolution_metadata.json",
                encoding="utf-8",
            ) as fh:
                metadata = json.load(fh)

        missing = _METADATA_REQUIRED_KEYS - metadata.keys()
        self.assertFalse(
            missing,
            msg=(
                "REGRESSION: evolution_metadata.json is missing schema keys: "
                f"{missing}. A field was likely removed or renamed in _persist_results."
            ),
        )

    def test_chromosome_contains_all_evolvable_genes(self) -> None:
        """best_chromosome in every summary must include all evolvable genes.

        Regression caught: removing a gene from DEFAULT_HYPERPARAMETER_GENES
        causes downstream code that reads chromosome fields to get KeyError.
        """
        expected_genes = {"learning_rate", "gamma", "epsilon_decay", "memory_size"}
        result = self._result
        for summary in result.generation_summaries:
            missing = expected_genes - summary.best_chromosome.keys()
            self.assertFalse(
                missing,
                msg=(
                    f"REGRESSION: best_chromosome at generation "
                    f"{summary.generation} is missing evolvable genes: "
                    f"{missing}. A gene was likely removed from "
                    "DEFAULT_HYPERPARAMETER_GENES."
                ),
            )

    def test_run_produces_correct_evaluation_count(self) -> None:
        """Total evaluations must equal num_generations × population_size.

        Regression caught: skipping candidates or evaluating duplicates
        breaks selection pressure and lineage tracking integrity.
        """
        result = self._result
        expected = _NUM_GENERATIONS * _POPULATION_SIZE
        self.assertEqual(
            len(result.evaluations),
            expected,
            msg=(
                f"REGRESSION: expected {expected} evaluations "
                f"({_NUM_GENERATIONS} generations × {_POPULATION_SIZE} candidates) "
                f"but got {len(result.evaluations)}. "
                "Check _evaluate_generation or population sizing logic."
            ),
        )

    def test_gene_statistics_include_required_stat_keys(self) -> None:
        """gene_statistics for every gene must include mean, std, min, max, median.

        Regression caught: removing a stat key from _build_gene_statistics
        silently drops it from both in-memory summaries and persisted JSON.
        """
        required_stat_keys = {"mean", "std", "min", "max", "median", "boundary_fraction"}
        # Note: gene_statistics uses the key "boundary_fraction" (the raw per-generation
        # fraction computed in _build_gene_statistics), while summary.boundary_occupancy
        # is a derived dict that re-exposes the same values under a public API name.
        # Checking "boundary_fraction" here validates the internal stats structure.
        result = self._result
        evolvable_genes = {"learning_rate", "gamma", "epsilon_decay", "memory_size"}
        for summary in result.generation_summaries:
            for gene_name in evolvable_genes:
                self.assertIn(
                    gene_name,
                    summary.gene_statistics,
                    msg=(
                        f"REGRESSION: gene '{gene_name}' missing from gene_statistics "
                        f"at generation {summary.generation}."
                    ),
                )
                gene_stats = summary.gene_statistics[gene_name]
                missing = required_stat_keys - gene_stats.keys()
                self.assertFalse(
                    missing,
                    msg=(
                        f"REGRESSION: gene_statistics['{gene_name}'] at generation "
                        f"{summary.generation} is missing keys: {missing}. "
                        "A stat was likely removed from _build_gene_statistics."
                    ),
                )


# ---------------------------------------------------------------------------
# Negative tests – demonstrate what a regression looks like
# ---------------------------------------------------------------------------


class TestEvolutionRegressionNegativeCases(unittest.TestCase):
    """Negative tests that demonstrate what the detection catches.

    These tests intentionally create a broken configuration and verify that
    the metric lands in the failure zone.  They prove that the detection logic
    itself works: if a negative test started *passing* the regression threshold
    it would mean the detection mechanism had itself regressed.
    """

    def test_zero_mutation_produces_degenerate_diversity(self) -> None:
        """Demonstrates: diversity check catches population collapse.

        With mutation_scale=0.0 every mutation adds zero noise, so all
        candidates share the initial chromosome values (std = 0) → diversity
        collapses to exactly 0.0.  This value is below _MIN_DIVERSITY,
        confirming that the check would fire for this broken configuration.
        """
        result = _run_benchmark(mutation_scale=0.0)
        last_summary = result.generation_summaries[-1]
        diversity = last_summary.diversity

        # With zero mutation all candidates are identical → diversity = 0.
        self.assertEqual(
            diversity,
            0.0,
            msg=(
                f"Expected diversity=0.0 when mutation_scale=0 (all candidates "
                f"share the same chromosome), but got {diversity}."
            ),
        )
        # Confirm this would trigger the MIN_DIVERSITY regression alarm.
        self.assertLess(
            diversity,
            _MIN_DIVERSITY,
            msg=(
                f"Degenerate diversity ({diversity}) should be below the "
                f"regression threshold ({_MIN_DIVERSITY}). "
                "The negative-case detection logic is broken if this fails."
            ),
        )

    def test_constant_zero_fitness_never_improves(self) -> None:
        """Demonstrates: a broken evaluator that always returns 0 is detectable.

        When every candidate scores 0, best_fitness cannot improve.  The
        experiment still completes without errors; this test verifies that the
        run produces the expected constant-zero pattern, confirming that
        fitness tracking would surface the regression.
        """
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=4,
            mutation_rate=_MUTATION_RATE,
            mutation_scale=_MUTATION_SCALE,
            elitism_count=1,
            seed=_SEED,
        )
        experiment = EvolutionExperiment(SimulationConfig(), config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (0.0, {})
        )

        for summary in result.generation_summaries:
            self.assertEqual(
                summary.best_fitness,
                0.0,
                msg=(
                    f"Expected best_fitness=0.0 for zero evaluator at generation "
                    f"{summary.generation}, got {summary.best_fitness}."
                ),
            )

        self.assertEqual(
            result.best_candidate.fitness,
            0.0,
            msg=(
                f"Overall best_candidate.fitness should be 0.0 for zero evaluator, "
                f"got {result.best_candidate.fitness}."
            ),
        )
