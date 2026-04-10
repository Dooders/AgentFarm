"""Systematic crossover + fine-tune search orchestration.

This module implements the search layer that generates many child Q-networks
by sweeping over crossover strategies × fine-tune hyperparameter recipes and
ranks outcomes so the project can recommend a default strategy.

The entry point is :func:`run_crossover_search`.  It produces a
:class:`ManifestEntry` per child and writes the following to *run_dir*:

``<run_dir>/<child_id>/child.pt``
    Best-checkpoint child state dict (float32).
``<run_dir>/<child_id>/eval_report.json``
    Full :class:`~farm.core.decision.training.recombination_eval.RecombinationReport`
    as JSON.
``<run_dir>/<child_id>/run_config.json``
    Config JSON capturing parents, crossover params, fine-tune params, seeds,
    and PyTorch version for exact reproducibility.
``<run_dir>/manifest.json``
    One entry per child; includes all hyperparameters and summary metrics.
``<run_dir>/leaderboard.csv``
    Manifest sorted by *primary_metric* (``min(agree_a, agree_b)``) descending;
    parent baselines appended at the bottom.
``<run_dir>/leaderboard.json``
    Same data as the CSV in JSON format.
``<run_dir>/recommendation.txt``
    Human-readable strategy recommendation.

Public API
----------
- :class:`CrossoverRecipe`     – one crossover configuration (mode + alpha + seed)
- :class:`FineTuneRegime`      – one fine-tune hyperparameter set
- :class:`SearchConfig`        – full search space (recipes × regimes)
- :class:`ManifestEntry`       – one leaderboard row (paths + metrics)
- :func:`run_crossover_search` – main entry point (restores parent ``train`` /
  ``requires_grad`` state after the search; optional batched evaluation;
  optional ``num_workers`` > 1 for process-parallel children when parents are
  :class:`~farm.core.decision.base_dqn.BaseQNetwork`)
- :func:`build_leaderboard`    – sort and annotate entries
- :func:`generate_recommendation` – text summary from manifest

Typical usage
-------------
::

    import numpy as np
    from farm.core.decision.base_dqn import BaseQNetwork
    from farm.core.decision.training.crossover_search import (
        SearchConfig, run_crossover_search, build_leaderboard,
        generate_recommendation,
    )

    parent_a = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    parent_b = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    states = np.random.randn(500, 8).astype("float32")

    cfg = SearchConfig.default()
    manifest, leaderboard = run_crossover_search(
        parent_a, parent_b, states, cfg, run_dir="/tmp/crossover_search"
    )
    print(generate_recommendation(manifest))

See also
--------
- :mod:`farm.core.decision.training.crossover` – crossover operators
- :mod:`farm.core.decision.training.finetune`  – fine-tuning pipeline
- :mod:`farm.core.decision.training.recombination_eval` – evaluation harness
- ``scripts/run_crossover_search.py``          – single-generation CLI wrapper
- ``scripts/run_multi_gen_search.py``          – multi-generation evolutionary CLI
- ``docs/design/crossover_search_space.md``    – search space specification
"""

from __future__ import annotations

import copy
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.utils.logging import get_logger

from .crossover import CROSSOVER_MODES, MutationConfig, crossover_quantized_state_dict, mutate_state_dict
from .finetune import FineTuner, FineTuningConfig, _sanitize_for_json
from .recombination_eval import (
    RecombinationEvaluator,
    RecombinationReport,
    RecombinationThresholds,
)

logger = get_logger(__name__)


def _infer_base_qnetwork_arch(module: nn.Module) -> Tuple[int, int, int]:
    """Return ``(input_dim, hidden_size, output_dim)`` from a :class:`BaseQNetwork`."""
    if not isinstance(module, BaseQNetwork):
        raise TypeError(
            "Parallel crossover search (num_workers > 1) requires parent_a and parent_b "
            f"to be BaseQNetwork instances; got {type(module).__name__}. "
            "Use num_workers=1 for other module types."
        )
    linear_layers = [m for m in module.network.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 3:
        raise ValueError("Could not infer BaseQNetwork architecture from Linear layers.")
    input_dim = int(linear_layers[0].in_features)
    hidden_size = int(linear_layers[0].out_features)
    output_dim = int(linear_layers[-1].out_features)
    return input_dim, hidden_size, output_dim


def _snapshot_train_state(module: nn.Module) -> Tuple[bool, List[Tuple[nn.Parameter, bool]]]:
    """Capture ``training`` flag and per-parameter ``requires_grad`` for restore."""
    return module.training, [(p, p.requires_grad) for p in module.parameters()]


def _restore_train_state(
    module: nn.Module,
    snapshot: Tuple[bool, List[Tuple[nn.Parameter, bool]]],
) -> None:
    """Restore module mode and ``requires_grad`` flags (undo search-runner side effects)."""
    training, pairs = snapshot
    module.train(mode=training)
    for param, requires_grad in pairs:
        param.requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# Search-space dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CrossoverRecipe:
    """One crossover configuration (mode + hyperparameters).

    Attributes
    ----------
    mode:
        Crossover strategy: ``"random"``, ``"layer"``, or ``"weighted"``.
        Must be one of :data:`~farm.core.decision.training.crossover.CROSSOVER_MODES`.
    alpha:
        Blend coefficient for parent A.  In ``"random"`` mode: probability of
        selecting from parent A per tensor.  In ``"weighted"`` mode: linear
        blend weight.  Ignored for ``"layer"``.  Must be in ``[0, 1]``.
    seed:
        Integer RNG seed for the ``"random"`` mode.  ``None`` means a
        non-reproducible RNG.  Ignored for deterministic modes (``"layer"``,
        ``"weighted"``).
    """

    mode: str
    alpha: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in CROSSOVER_MODES:
            raise ValueError(
                f"CrossoverRecipe.mode must be one of {CROSSOVER_MODES!r}; "
                f"got {self.mode!r}"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"CrossoverRecipe.alpha must be in [0, 1]; got {self.alpha!r}"
            )


@dataclass
class FineTuneRegime:
    """One fine-tune hyperparameter set.

    Attributes
    ----------
    name:
        Short human-readable label used in ``child_id`` strings and reports
        (e.g. ``"short"``, ``"long"``, ``"lr_high"``).
    epochs:
        Number of full passes over the state buffer.
    lr:
        Learning rate (passed to :attr:`~farm.core.decision.training.finetune.FineTuningConfig.learning_rate`).
    optimizer:
        Built-in optimiser name; see :data:`~farm.core.decision.training.finetune.FINETUNE_OPTIMIZERS`.
    optimizer_kwargs:
        Extra optimiser keyword arguments.
    early_stopping_patience:
        Early stopping patience (0 = disabled); requires ``val_fraction > 0``.
    quantization_applied:
        Passed to :class:`~farm.core.decision.training.finetune.FineTuningConfig`.
        Use ``"none"`` (default) for full-precision float fine-tuning.
    loss_fn:
        ``"kl"`` (default) or ``"mse"`` soft distillation loss.
    batch_size:
        Mini-batch size.
    val_fraction:
        Fraction of states held out for validation (``0`` = no validation).
    seed:
        Fine-tuning RNG seed for reproducibility.
    """

    name: str
    epochs: int
    lr: float
    quantization_applied: str = "none"
    loss_fn: str = "kl"
    batch_size: int = 32
    val_fraction: float = 0.1
    seed: Optional[int] = None
    optimizer: str = "adam"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 0


@dataclass
class SearchConfig:
    """Full search space for the crossover + fine-tune experiment.

    A *search config* is a Cartesian product of :attr:`crossover_recipes` ×
    :attr:`finetune_regimes`.  Each combination yields one child.

    Attributes
    ----------
    crossover_recipes:
        List of :class:`CrossoverRecipe` objects defining which crossover
        configurations to explore.
    finetune_regimes:
        List of :class:`FineTuneRegime` objects defining which fine-tune
        hyperparameter sets to test.
    max_runs:
        Optional cap on the total number of children generated.  When set,
        the Cartesian product is truncated to the first *max_runs* pairs in
        the order ``(recipe_0, regime_0), (recipe_0, regime_1), …``.
        ``None`` means no limit.
    degenerate_threshold:
        Primary metric (``min(agree_a, agree_b)``) below which a child is
        flagged as **degenerate**.  ``0.0`` disables the flag.
    """

    crossover_recipes: List[CrossoverRecipe] = field(default_factory=list)
    finetune_regimes: List[FineTuneRegime] = field(default_factory=list)
    max_runs: Optional[int] = None
    degenerate_threshold: float = 0.0

    @classmethod
    def default(cls) -> "SearchConfig":
        """Return the standard search space (≥ 9 children) used for the paper.

        Crossover recipes (7):
        - ``random`` at 3 seeds (0, 1, 2), alpha=0.5 — explores diversity
        - ``layer`` — structural coherence
        - ``weighted`` at alpha ∈ {0.3, 0.5, 0.7} — smooth interpolation

        Fine-tune regimes (2):
        - ``short``  — 5 epochs, lr=1e-3 (quick recovery)
        - ``long``   — 20 epochs, lr=1e-4 (deeper adaptation)

        Total children: 7 × 2 = **14**.
        """
        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=0),
            CrossoverRecipe("random", alpha=0.5, seed=1),
            CrossoverRecipe("random", alpha=0.5, seed=2),
            CrossoverRecipe("layer"),
            CrossoverRecipe("weighted", alpha=0.3),
            CrossoverRecipe("weighted", alpha=0.5),
            CrossoverRecipe("weighted", alpha=0.7),
        ]
        regimes = [
            FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
        ]
        return cls(crossover_recipes=recipes, finetune_regimes=regimes)

    @classmethod
    def minimal(cls) -> "SearchConfig":
        """Return the minimal 3 × 3 grid (exactly 9 children).

        One recipe per crossover mode × three fine-tune regimes.  Useful as a
        fast smoke-check to verify runner wiring and produce a first leaderboard.

        Crossover recipes (3):
        - ``random`` seed=0
        - ``layer``
        - ``weighted`` alpha=0.5

        Fine-tune regimes (3):
        - ``short``   — 5 epochs, lr=1e-3
        - ``medium``  — 10 epochs, lr=5e-4
        - ``long``    — 20 epochs, lr=1e-4

        Total children: 3 × 3 = **9**.
        """
        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=0),
            CrossoverRecipe("layer"),
            CrossoverRecipe("weighted", alpha=0.5),
        ]
        regimes = [
            FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            FineTuneRegime("medium", epochs=10, lr=5e-4, seed=42),
            FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
        ]
        return cls(crossover_recipes=recipes, finetune_regimes=regimes)

    @classmethod
    def default_with_qat(cls) -> "SearchConfig":
        """Same crossover recipes as :meth:`default` with **three** fine-tune regimes.

        Adds **short_qat** (``quantization_applied="ptq_dynamic"``) beside ``short``
        and ``long`` float regimes so each recipe is trained both in full float
        and under fake-quant weight noise (7 × 3 = **21** children).
        """
        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=0),
            CrossoverRecipe("random", alpha=0.5, seed=1),
            CrossoverRecipe("random", alpha=0.5, seed=2),
            CrossoverRecipe("layer"),
            CrossoverRecipe("weighted", alpha=0.3),
            CrossoverRecipe("weighted", alpha=0.5),
            CrossoverRecipe("weighted", alpha=0.7),
        ]
        regimes = [
            FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
            FineTuneRegime(
                "short_qat",
                epochs=5,
                lr=1e-4,
                seed=42,
                quantization_applied="ptq_dynamic",
                batch_size=16,
                val_fraction=0.1,
            ),
        ]
        return cls(crossover_recipes=recipes, finetune_regimes=regimes)

    @classmethod
    def minimal_with_qat(cls) -> "SearchConfig":
        """Minimal grid with one **QAT** column: 3 recipes × 3 regimes = **9** children.

        Regimes: ``short`` (float), ``short_qat`` (``ptq_dynamic``), ``long`` (float).
        """
        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=0),
            CrossoverRecipe("layer"),
            CrossoverRecipe("weighted", alpha=0.5),
        ]
        regimes = [
            FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            FineTuneRegime(
                "short_qat",
                epochs=5,
                lr=1e-4,
                seed=42,
                quantization_applied="ptq_dynamic",
                batch_size=16,
                val_fraction=0.1,
            ),
            FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
        ]
        return cls(crossover_recipes=recipes, finetune_regimes=regimes)

    def pairs(self) -> List[Tuple[CrossoverRecipe, FineTuneRegime]]:
        """Return the ordered list of (recipe, regime) pairs to evaluate.

        The Cartesian product is iterated in recipe-major order: all regimes
        are tested for each recipe before moving to the next recipe.  The
        result is truncated to :attr:`max_runs` when set.
        """
        all_pairs: List[Tuple[CrossoverRecipe, FineTuneRegime]] = []
        for recipe in self.crossover_recipes:
            for regime in self.finetune_regimes:
                all_pairs.append((recipe, regime))
        if self.max_runs is not None:
            return all_pairs[: self.max_runs]
        return all_pairs


def _search_config_with_generation_seed_bump(
    search_config: SearchConfig,
    bump: int,
) -> SearchConfig:
    """Return a shallow copy of *search_config* with integer seeds shifted by *bump*.

    Each :class:`CrossoverRecipe` / :class:`FineTuneRegime` ``seed`` that is not
    ``None`` becomes ``seed + bump``.  ``None`` seeds stay ``None``.  When
    *bump* is zero, returns *search_config* unchanged (same object).
    """
    if bump == 0:
        return search_config
    new_recipes = [
        replace(r, seed=None if r.seed is None else r.seed + bump)
        for r in search_config.crossover_recipes
    ]
    new_regimes = [
        replace(r, seed=None if r.seed is None else r.seed + bump)
        for r in search_config.finetune_regimes
    ]
    return replace(
        search_config,
        crossover_recipes=new_recipes,
        finetune_regimes=new_regimes,
    )


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One row in the run manifest — one evaluated child.

    Attributes
    ----------
    child_id:
        Unique string identifier for this child run
        (e.g. ``"000_random_a0p50_s0_short"``).
    crossover_mode, crossover_alpha, crossover_seed:
        Crossover recipe hyperparameters.
    finetune_regime, finetune_epochs, finetune_lr, finetune_quantization_applied:
        Fine-tune regime hyperparameters.
    child_pt_path:
        Absolute path to the saved child checkpoint (``.pt``).
    eval_report_path:
        Absolute path to the JSON evaluation report.
    run_config_path:
        Absolute path to the JSON run config (for re-run reproducibility).
    primary_metric:
        ``min(child_vs_parent_a_agreement, child_vs_parent_b_agreement)`` —
        higher is better; used for leaderboard ordering.
    child_vs_parent_a_agreement, child_vs_parent_b_agreement:
        Top-1 action-agreement rates from the evaluation.
    oracle_agreement:
        Fraction of states where the child matches *at least one* parent.
    kl_divergence_a, kl_divergence_b:
        Mean KL divergence vs each parent.
    mse_a, mse_b:
        Mean squared error vs each parent's raw Q-logits.
    cosine_a, cosine_b:
        Mean cosine similarity vs each parent's Q-value vectors.
    degenerate:
        ``True`` when *primary_metric* falls below
        :attr:`SearchConfig.degenerate_threshold`.
    """

    child_id: str
    crossover_mode: str
    crossover_alpha: float
    crossover_seed: Optional[int]
    finetune_regime: str
    finetune_epochs: int
    finetune_lr: float
    finetune_quantization_applied: str
    child_pt_path: str
    eval_report_path: str
    run_config_path: str
    primary_metric: float
    child_vs_parent_a_agreement: float
    child_vs_parent_b_agreement: float
    oracle_agreement: Optional[float]
    kl_divergence_a: float
    kl_divergence_b: float
    mse_a: float
    mse_b: float
    cosine_a: float
    cosine_b: float
    degenerate: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "child_id": self.child_id,
            "crossover_mode": self.crossover_mode,
            "crossover_alpha": self.crossover_alpha,
            "crossover_seed": self.crossover_seed,
            "finetune_regime": self.finetune_regime,
            "finetune_epochs": self.finetune_epochs,
            "finetune_lr": self.finetune_lr,
            "finetune_quantization_applied": self.finetune_quantization_applied,
            "child_pt_path": self.child_pt_path,
            "eval_report_path": self.eval_report_path,
            "run_config_path": self.run_config_path,
            "primary_metric": self.primary_metric,
            "child_vs_parent_a_agreement": self.child_vs_parent_a_agreement,
            "child_vs_parent_b_agreement": self.child_vs_parent_b_agreement,
            "oracle_agreement": self.oracle_agreement,
            "kl_divergence_a": self.kl_divergence_a,
            "kl_divergence_b": self.kl_divergence_b,
            "mse_a": self.mse_a,
            "mse_b": self.mse_b,
            "cosine_a": self.cosine_a,
            "cosine_b": self.cosine_b,
            "degenerate": self.degenerate,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_child_id(recipe: CrossoverRecipe, regime: FineTuneRegime, run_idx: int) -> str:
    """Return a deterministic, human-readable child identifier string."""
    seed_str = f"s{recipe.seed}" if recipe.seed is not None else "sNone"
    alpha_str = f"a{recipe.alpha:.2f}".replace(".", "p")
    return f"{run_idx:03d}_{recipe.mode}_{alpha_str}_{seed_str}_{regime.name}"


def _run_one(
    parent_a: nn.Module,
    parent_b: nn.Module,
    states: np.ndarray,
    recipe: CrossoverRecipe,
    regime: FineTuneRegime,
    run_dir: str,
    run_idx: int,
    thresholds: RecombinationThresholds,
    include_parent_baseline: bool,
    degenerate_threshold: float,
    *,
    eval_batch_size: Optional[int] = None,
) -> Tuple[ManifestEntry, Dict[str, Any]]:
    """Run one (recipe × regime) pair: crossover → fine-tune → evaluate.

    Parameters
    ----------
    parent_a, parent_b:
        Frozen parent networks in eval mode.
    states:
        NumPy array of evaluation states, shape ``(N, input_dim)``.
    recipe:
        Crossover configuration for this run.
    regime:
        Fine-tune regime for this run.
    run_dir:
        Root directory; per-child outputs are written to ``run_dir/<child_id>/``.
    run_idx:
        Sequential run index used to generate ``child_id``.
    thresholds:
        Pass/fail thresholds forwarded to :class:`RecombinationEvaluator`.
    include_parent_baseline:
        Whether to compute the parent A vs parent B comparison as a baseline.
    degenerate_threshold:
        Primary-metric floor below which the child is flagged as degenerate.

    Returns
    -------
    Tuple[ManifestEntry, Dict[str, Any]]
        The manifest entry and the raw eval report dict.
    """
    child_id = _make_child_id(recipe, regime, run_idx)
    child_dir = os.path.join(run_dir, child_id)
    os.makedirs(child_dir, exist_ok=True)

    logger.info(
        "crossover_search_run_start",
        run_idx=run_idx,
        child_id=child_id,
        crossover_mode=recipe.mode,
        alpha=recipe.alpha,
        seed=recipe.seed,
        finetune_regime=regime.name,
        epochs=regime.epochs,
        lr=regime.lr,
    )

    # ------------------------------------------------------------------
    # 1. Crossover: produce child state dict
    # ------------------------------------------------------------------
    child_sd = crossover_quantized_state_dict(
        parent_a.state_dict(),
        parent_b.state_dict(),
        mode=recipe.mode,
        alpha=recipe.alpha,
        seed=recipe.seed,
    )
    # Instantiate child with parent_a's architecture (deep-copy ensures
    # independent parameters even when parent_a is modified later).
    child = copy.deepcopy(parent_a)
    child.load_state_dict(child_sd)
    # Re-enable gradients: parent_a may have been frozen in run_crossover_search;
    # the child must remain trainable for FineTuner.
    for p in child.parameters():
        p.requires_grad_(True)
    child.train()

    # ------------------------------------------------------------------
    # 2. Fine-tune child against parent_a as reference teacher
    # ------------------------------------------------------------------
    ft_cfg = FineTuningConfig(
        learning_rate=regime.lr,
        epochs=regime.epochs,
        batch_size=regime.batch_size,
        val_fraction=regime.val_fraction,
        loss_fn=regime.loss_fn,
        seed=regime.seed,
        quantization_applied=regime.quantization_applied,
        optimizer=regime.optimizer,
        optimizer_kwargs=dict(regime.optimizer_kwargs),
        early_stopping_patience=regime.early_stopping_patience,
    )
    child_pt_path = os.path.join(child_dir, "child.pt")
    tuner = FineTuner(reference=parent_a, child=child, config=ft_cfg)
    ft_metrics = tuner.finetune(states, checkpoint_path=child_pt_path)

    # Reload best checkpoint into a clean float child for evaluation.
    # This is necessary in QAT mode, where _active_child is a QAT copy;
    # loading the saved state dict into the original float model works
    # because WeightOnlyFakeQuantLinear shares the same weight/bias keys
    # as nn.Linear.
    best_sd = torch.load(child_pt_path, map_location="cpu", weights_only=True)
    eval_child = copy.deepcopy(parent_a)
    eval_child.load_state_dict(best_sd)
    eval_child.eval()

    # ------------------------------------------------------------------
    # 3. Evaluate child vs both parents
    # ------------------------------------------------------------------
    evaluator = RecombinationEvaluator(
        parent_a, parent_b, eval_child, thresholds=thresholds
    )
    report: RecombinationReport = evaluator.evaluate(
        states,
        include_parent_baseline=include_parent_baseline,
        k_values=[1, 2, 3],
        n_latency_warmup=3,
        n_latency_repeats=20,
        states_source="crossover_search",
        eval_batch_size=eval_batch_size,
        model_paths={
            "parent_a": "(in-memory)",
            "parent_b": "(in-memory)",
            "child": child_pt_path,
        },
    )
    report_dict = report.to_dict()

    # ------------------------------------------------------------------
    # 4. Persist per-child artifacts
    # ------------------------------------------------------------------
    eval_report_path = os.path.join(child_dir, "eval_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, allow_nan=False)

    run_config: Dict[str, Any] = {
        "child_id": child_id,
        "crossover": {
            "mode": recipe.mode,
            "alpha": recipe.alpha,
            "seed": recipe.seed,
        },
        "finetune": {
            "regime": regime.name,
            "epochs": regime.epochs,
            "lr": regime.lr,
            "batch_size": regime.batch_size,
            "val_fraction": regime.val_fraction,
            "loss_fn": regime.loss_fn,
            "seed": regime.seed,
            "quantization_applied": regime.quantization_applied,
            "optimizer": regime.optimizer,
            "optimizer_kwargs": regime.optimizer_kwargs,
            "early_stopping_patience": regime.early_stopping_patience,
        },
        "finetune_metrics": _sanitize_for_json(ft_metrics.to_dict()),
        "torch_version": torch.__version__,
    }
    run_config_path = os.path.join(child_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2, allow_nan=False)

    # ------------------------------------------------------------------
    # 5. Assemble manifest entry
    # ------------------------------------------------------------------
    cmp_a = report_dict["comparisons"]["child_vs_parent_a"]
    cmp_b = report_dict["comparisons"]["child_vs_parent_b"]
    agree_a = float(cmp_a["action_agreement"])
    agree_b = float(cmp_b["action_agreement"])
    primary = min(agree_a, agree_b)
    oracle = report_dict["summary"].get("oracle_agreement")

    entry = ManifestEntry(
        child_id=child_id,
        crossover_mode=recipe.mode,
        crossover_alpha=recipe.alpha,
        crossover_seed=recipe.seed,
        finetune_regime=regime.name,
        finetune_epochs=regime.epochs,
        finetune_lr=regime.lr,
        finetune_quantization_applied=regime.quantization_applied,
        child_pt_path=child_pt_path,
        eval_report_path=eval_report_path,
        run_config_path=run_config_path,
        primary_metric=primary,
        child_vs_parent_a_agreement=agree_a,
        child_vs_parent_b_agreement=agree_b,
        oracle_agreement=float(oracle) if oracle is not None else None,
        kl_divergence_a=float(cmp_a["kl_divergence"]),
        kl_divergence_b=float(cmp_b["kl_divergence"]),
        mse_a=float(cmp_a["mse"]),
        mse_b=float(cmp_b["mse"]),
        cosine_a=float(cmp_a["mean_cosine_similarity"]),
        cosine_b=float(cmp_b["mean_cosine_similarity"]),
        degenerate=primary < degenerate_threshold if degenerate_threshold > 0 else False,
    )

    logger.info(
        "crossover_search_run_complete",
        child_id=child_id,
        primary_metric=round(primary, 4),
        agree_a=round(agree_a, 4),
        agree_b=round(agree_b, 4),
        oracle=round(float(oracle), 4) if oracle is not None else None,
        degenerate=entry.degenerate,
    )
    return entry, report_dict


def _manifest_entry_from_dict(d: Dict[str, Any]) -> ManifestEntry:
    """Rebuild :class:`ManifestEntry` from :meth:`ManifestEntry.to_dict` output."""
    return ManifestEntry(**d)


def _crossover_parallel_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one search pair in a child process (parents loaded from disk)."""
    recipe = CrossoverRecipe(**job["recipe"])
    regime = FineTuneRegime(**job["regime"])
    thresholds = RecombinationThresholds(**job["thresholds"])
    parent_a = BaseQNetwork(
        input_dim=job["input_dim"],
        output_dim=job["output_dim"],
        hidden_size=job["hidden_size"],
    )
    parent_a.load_state_dict(
        torch.load(job["parent_a_path"], map_location="cpu", weights_only=True)
    )
    parent_b = BaseQNetwork(
        input_dim=job["input_dim"],
        output_dim=job["output_dim"],
        hidden_size=job["hidden_size"],
    )
    parent_b.load_state_dict(
        torch.load(job["parent_b_path"], map_location="cpu", weights_only=True)
    )
    states = np.load(job["states_path"])
    for m in (parent_a, parent_b):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    entry, _ = _run_one(
        parent_a,
        parent_b,
        states,
        recipe,
        regime,
        job["run_dir"],
        job["run_idx"],
        thresholds,
        job["include_parent_baseline"],
        job["degenerate_threshold"],
        eval_batch_size=job.get("eval_batch_size"),
    )
    return {"run_idx": job["run_idx"], "entry_dict": entry.to_dict()}


# ---------------------------------------------------------------------------
# Public API: run_crossover_search
# ---------------------------------------------------------------------------


def run_crossover_search(
    parent_a: nn.Module,
    parent_b: nn.Module,
    states: np.ndarray,
    search_config: SearchConfig,
    run_dir: str,
    *,
    thresholds: Optional[RecombinationThresholds] = None,
    include_parent_baseline: bool = True,
    eval_batch_size: Optional[int] = 2048,
    num_workers: int = 1,
) -> Tuple[List[ManifestEntry], List[Dict[str, Any]]]:
    """Generate and evaluate many children over the configured search space.

    For each *(recipe × regime)* pair in *search_config*:

    1. Construct a child by crossover of *parent_a* and *parent_b*.
    2. Fine-tune the child against *parent_a* as a soft reference teacher.
    3. Evaluate child vs both parents using :class:`RecombinationEvaluator`.
    4. Record config JSON, eval JSON, and checkpoint under
       ``<run_dir>/<child_id>/``.

    After all runs, writes ``manifest.json``, ``leaderboard.csv``,
    ``leaderboard.json``, and ``recommendation.txt`` to *run_dir*.

    Parameters
    ----------
    parent_a:
        First parent network.  Used both as the reference teacher during
        fine-tuning and as one half of the child-vs-parent evaluation.
    parent_b:
        Second parent network.  Used only in the child-vs-parent evaluation.
    states:
        NumPy array of shape ``(N, input_dim)`` with evaluation states.
        The same array is shared across all children to ensure fair
        comparison (fixed evaluation harness).
    search_config:
        :class:`SearchConfig` defining the search space.
    run_dir:
        Root output directory.  Created if it does not exist.
    thresholds:
        Optional :class:`RecombinationThresholds` for pass/fail gates in
        evaluation reports.  Defaults to lenient report-only thresholds.
    include_parent_baseline:
        When ``True``, each eval report includes a parent A vs parent B
        comparison for baseline context.
    eval_batch_size:
        Optional max batch size for :class:`RecombinationEvaluator` forward
        passes.  ``None`` evaluates all states in one batch (highest memory).
        Default ``2048`` caps peak activations on large state buffers.
    num_workers:
        When ``> 1``, run each child in a separate process via
        :class:`~concurrent.futures.ProcessPoolExecutor`.  Parents must be
        :class:`BaseQNetwork` instances (state dicts are written under
        ``<run_dir>/.crossover_parallel_cache/``).  Quantized int8 parents are
        not supported on this path—use ``num_workers=1``.  ``1`` (default) is
        sequential and works for any parent type.

    Returns
    -------
    manifest : List[ManifestEntry]
        One entry per child, in run order (not sorted).
    leaderboard : List[Dict[str, Any]]
        Sorted leaderboard dicts (primary_metric descending), including
        parent baseline rows.
    """
    os.makedirs(run_dir, exist_ok=True)

    if thresholds is None:
        thresholds = RecombinationThresholds(report_only=True)

    snap_a = _snapshot_train_state(parent_a)
    snap_b = _snapshot_train_state(parent_b)
    try:
        # Search uses parents as frozen references; restore train/RNG state after.
        for m in (parent_a, parent_b):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        pairs = search_config.pairs()

        logger.info(
            "crossover_search_start",
            n_pairs=len(pairs),
            n_states=len(states),
            run_dir=run_dir,
        )

        manifest: List[ManifestEntry] = []

        if num_workers > 1 and len(pairs) > 0:
            in_d, hid, out_d = _infer_base_qnetwork_arch(parent_a)
            in_b, hid_b, out_b = _infer_base_qnetwork_arch(parent_b)
            if (in_d, hid, out_d) != (in_b, hid_b, out_b):
                raise ValueError(
                    "parent_a and parent_b must share the same architecture for parallel search."
                )
            cache_dir = os.path.join(run_dir, ".crossover_parallel_cache")
            os.makedirs(cache_dir, exist_ok=True)
            pa_path = os.path.join(cache_dir, "parent_a.pt")
            pb_path = os.path.join(cache_dir, "parent_b.pt")
            st_path = os.path.join(cache_dir, "states.npy")
            torch.save(parent_a.state_dict(), pa_path)
            torch.save(parent_b.state_dict(), pb_path)
            np.save(st_path, states)
            thr_dict = asdict(thresholds)
            jobs: List[Dict[str, Any]] = []
            for run_idx, (recipe, regime) in enumerate(pairs):
                job: Dict[str, Any] = {
                    "run_idx": run_idx,
                    "recipe": asdict(recipe),
                    "regime": asdict(regime),
                    "thresholds": thr_dict,
                    "parent_a_path": pa_path,
                    "parent_b_path": pb_path,
                    "states_path": st_path,
                    "run_dir": run_dir,
                    "input_dim": in_d,
                    "output_dim": out_d,
                    "hidden_size": hid,
                    "include_parent_baseline": include_parent_baseline,
                    "degenerate_threshold": search_config.degenerate_threshold,
                }
                if eval_batch_size is not None:
                    job["eval_batch_size"] = eval_batch_size
                jobs.append(job)
            logger.info(
                "crossover_search_parallel",
                num_workers=num_workers,
                n_jobs=len(jobs),
                cache_dir=cache_dir,
            )
            raw_results: List[Dict[str, Any]] = []
            # Use spawn: fork-after-Torch (and pytest threads) commonly deadlocks workers.
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
                futures = [pool.submit(_crossover_parallel_worker, j) for j in jobs]
                for fut in as_completed(futures):
                    raw_results.append(fut.result())
            raw_results.sort(key=lambda r: r["run_idx"])
            manifest = [_manifest_entry_from_dict(r["entry_dict"]) for r in raw_results]
        else:
            if num_workers > 1 and len(pairs) == 0:
                logger.info("crossover_search_parallel_skipped", reason="empty_pairs")
            for run_idx, (recipe, regime) in enumerate(pairs):
                entry, _ = _run_one(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    states=states,
                    recipe=recipe,
                    regime=regime,
                    run_dir=run_dir,
                    run_idx=run_idx,
                    thresholds=thresholds,
                    include_parent_baseline=include_parent_baseline,
                    degenerate_threshold=search_config.degenerate_threshold,
                    eval_batch_size=eval_batch_size,
                )
                manifest.append(entry)

        # ------------------------------------------------------------------
        # Compute parent-vs-parent baseline for the leaderboard
        # ------------------------------------------------------------------
        parent_baseline_agreement: Optional[float] = None
        if include_parent_baseline and manifest:
            # Re-use the last eval report (all have the baseline if requested).
            last_report_path = manifest[-1].eval_report_path
            try:
                with open(last_report_path, encoding="utf-8") as fh:
                    last_report = json.load(fh)
                pvp = last_report.get("comparisons", {}).get("parent_a_vs_parent_b", {})
                parent_baseline_agreement = pvp.get("action_agreement")
            except (OSError, json.JSONDecodeError, KeyError):
                pass

        # ------------------------------------------------------------------
        # Persist manifest + leaderboard
        # ------------------------------------------------------------------
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in manifest], fh, indent=2, allow_nan=False)

        leaderboard = build_leaderboard(manifest, parent_baseline_agreement)

        _write_leaderboard(leaderboard, run_dir)

        recommendation = generate_recommendation(manifest, parent_baseline_agreement)
        rec_path = os.path.join(run_dir, "recommendation.txt")
        with open(rec_path, "w", encoding="utf-8") as fh:
            fh.write(recommendation)

        logger.info(
            "crossover_search_complete",
            n_children=len(manifest),
            manifest_path=manifest_path,
            best_child_id=leaderboard[0].get("child_id") if leaderboard else None,
            best_primary_metric=leaderboard[0].get("primary_metric") if leaderboard else None,
        )

        return manifest, leaderboard
    finally:
        _restore_train_state(parent_a, snap_a)
        _restore_train_state(parent_b, snap_b)


# ---------------------------------------------------------------------------
# Public API: build_leaderboard
# ---------------------------------------------------------------------------

#: Column names used for the leaderboard CSV header.
LEADERBOARD_COLUMNS = [
    "rank",
    "child_id",
    "crossover_mode",
    "crossover_alpha",
    "crossover_seed",
    "finetune_regime",
    "finetune_epochs",
    "finetune_lr",
    "finetune_quantization_applied",
    "primary_metric",
    "child_vs_parent_a_agreement",
    "child_vs_parent_b_agreement",
    "oracle_agreement",
    "kl_divergence_a",
    "kl_divergence_b",
    "mse_a",
    "mse_b",
    "cosine_a",
    "cosine_b",
    "degenerate",
    "child_pt_path",
    "eval_report_path",
    "run_config_path",
]


def build_leaderboard(
    manifest: List[ManifestEntry],
    parent_baseline_agreement: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return a sorted leaderboard list with optional parent baseline rows.

    Children are ranked by *primary_metric* (``min(agree_a, agree_b)``)
    descending (higher = better blend of both parents).  Two parent baseline
    rows are appended after all children.

    Parameters
    ----------
    manifest:
        List of :class:`ManifestEntry` from :func:`run_crossover_search`.
    parent_baseline_agreement:
        The parent A vs parent B top-1 action agreement, computed separately
        (e.g. from the ``parent_a_vs_parent_b`` comparison in an eval report).
        When provided, two baseline rows are appended: one for parent A and
        one for parent B.  ``None`` omits the baselines.

    Returns
    -------
    List[Dict[str, Any]]
        Leaderboard rows (JSON-serialisable dicts) in rank order.  Each dict
        has all :data:`LEADERBOARD_COLUMNS` keys plus ``"is_parent_baseline"``
        (``bool``).
    """
    rows: List[Dict[str, Any]] = []
    for entry in sorted(manifest, key=lambda e: e.primary_metric, reverse=True):
        d = entry.to_dict()
        d["is_parent_baseline"] = False
        rows.append(d)

    # Parent baseline rows: each parent's "primary metric" is its agreement
    # with the other parent (how alike the parents are, providing context).
    if parent_baseline_agreement is not None:
        for parent_label in ("parent_a", "parent_b"):
            baseline_row: Dict[str, Any] = {
                "child_id": parent_label,
                "crossover_mode": "—",
                "crossover_alpha": None,
                "crossover_seed": None,
                "finetune_regime": "—",
                "finetune_epochs": None,
                "finetune_lr": None,
                "finetune_quantization_applied": "—",
                "child_pt_path": "—",
                "eval_report_path": "—",
                "run_config_path": "—",
                "primary_metric": parent_baseline_agreement,
                "child_vs_parent_a_agreement": (
                    1.0 if parent_label == "parent_a" else parent_baseline_agreement
                ),
                "child_vs_parent_b_agreement": (
                    parent_baseline_agreement if parent_label == "parent_a" else 1.0
                ),
                "oracle_agreement": None,
                "kl_divergence_a": 0.0 if parent_label == "parent_a" else None,
                "kl_divergence_b": None if parent_label == "parent_a" else 0.0,
                "mse_a": 0.0 if parent_label == "parent_a" else None,
                "mse_b": None if parent_label == "parent_a" else 0.0,
                "cosine_a": 1.0 if parent_label == "parent_a" else None,
                "cosine_b": None if parent_label == "parent_a" else 1.0,
                "degenerate": False,
                "is_parent_baseline": True,
            }
            rows.append(baseline_row)

    # Assign ranks (baselines share rank = len(children)+1)
    n_children = len(manifest)
    for i, row in enumerate(rows):
        row["rank"] = i + 1 if not row.get("is_parent_baseline") else n_children + 1

    return rows


# ---------------------------------------------------------------------------
# Public API: generate_recommendation
# ---------------------------------------------------------------------------


def generate_recommendation(
    manifest: List[ManifestEntry],
    parent_baseline_agreement: Optional[float] = None,
) -> str:
    """Produce a human-readable strategy recommendation from the manifest.

    Analyses the manifest to identify the best-performing crossover mode and
    fine-tune regime by average primary metric, lists the top-3 children,
    and flags the degenerate count.  The output is suitable for writing to
    ``recommendation.txt`` and for inclusion in PR descriptions or docs.

    Parameters
    ----------
    manifest:
        List of :class:`ManifestEntry` from :func:`run_crossover_search`.
    parent_baseline_agreement:
        Optional parent A vs parent B agreement for contextual comparison.

    Returns
    -------
    str
        Multi-line text recommendation.
    """
    if not manifest:
        return "No runs to analyse — manifest is empty."

    sorted_entries = sorted(manifest, key=lambda e: e.primary_metric, reverse=True)
    best = sorted_entries[0]

    # Average primary metric per crossover mode
    mode_scores: Dict[str, List[float]] = {}
    for e in manifest:
        mode_scores.setdefault(e.crossover_mode, []).append(e.primary_metric)
    mode_avg = {m: sum(sc) / len(sc) for m, sc in mode_scores.items()}
    best_mode = max(mode_avg, key=lambda m: mode_avg[m])

    # Average primary metric per fine-tune regime
    regime_scores: Dict[str, List[float]] = {}
    for e in manifest:
        regime_scores.setdefault(e.finetune_regime, []).append(e.primary_metric)
    regime_avg = {r: sum(sc) / len(sc) for r, sc in regime_scores.items()}
    best_regime = max(regime_avg, key=lambda r: regime_avg[r])

    degenerate_count = sum(1 for e in manifest if e.degenerate)

    lines: List[str] = [
        "=" * 72,
        "Crossover Search — Strategy Recommendation",
        "=" * 72,
        "",
        f"Total children evaluated : {len(manifest)}",
        f"Degenerate children      : {degenerate_count}",
        "",
        "Top-3 children (by primary metric = min(agree_a, agree_b)):",
    ]
    for rank, entry in enumerate(sorted_entries[:3], start=1):
        lines.append(
            f"  #{rank:>2}  {entry.child_id:<40}  "
            f"primary={entry.primary_metric:.4f}  "
            f"agree_a={entry.child_vs_parent_a_agreement:.4f}  "
            f"agree_b={entry.child_vs_parent_b_agreement:.4f}"
        )

    lines += [
        "",
        "Best crossover mode (by avg primary metric across all regimes):",
    ]
    for mode, avg in sorted(mode_avg.items(), key=lambda kv: kv[1], reverse=True):
        marker = " ← recommended" if mode == best_mode else ""
        lines.append(f"  {mode:<12}  avg={avg:.4f}{marker}")

    lines += [
        "",
        "Best fine-tune regime (by avg primary metric across all modes):",
    ]
    for regime, avg in sorted(regime_avg.items(), key=lambda kv: kv[1], reverse=True):
        marker = " ← recommended" if regime == best_regime else ""
        lines.append(f"  {regime:<12}  avg={avg:.4f}{marker}")

    # Hyperparams for the recommended regime: use the best primary_metric child in that
    # regime (not the first manifest row with the same name).
    in_best_regime = [e for e in manifest if e.finetune_regime == best_regime]
    best_in_regime = max(in_best_regime, key=lambda e: e.primary_metric)
    best_lr = best_in_regime.finetune_lr
    best_epochs = best_in_regime.finetune_epochs
    lines += [
        "",
        "Recommended default strategy:",
        f"  Crossover : {best_mode}",
        f"  Fine-tune : {best_regime}  (LR={best_lr:.0e}  epochs={best_epochs})",
    ]

    if parent_baseline_agreement is not None:
        lines += [
            "",
            f"Parent A vs Parent B agreement (inter-parent diversity baseline) : "
            f"{parent_baseline_agreement:.4f}",
        ]

    lines += [
        "",
        "Best individual child:",
        f"  child_id           : {best.child_id}",
        f"  crossover_mode     : {best.crossover_mode}",
        f"  crossover_alpha    : {best.crossover_alpha}",
        f"  crossover_seed     : {best.crossover_seed}",
        f"  finetune_regime    : {best.finetune_regime}",
        f"  finetune_epochs    : {best.finetune_epochs}",
        f"  finetune_lr        : {best.finetune_lr:.0e}",
        f"  primary_metric     : {best.primary_metric:.4f}",
        f"  agree_a / agree_b  : {best.child_vs_parent_a_agreement:.4f} / "
        f"{best.child_vs_parent_b_agreement:.4f}",
        (
            f"  oracle_agreement   : {best.oracle_agreement:.4f}"
            if best.oracle_agreement is not None
            else "  oracle_agreement   : N/A"
        ),
        "",
        "=" * 72,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------


def _write_leaderboard(leaderboard: List[Dict[str, Any]], run_dir: str) -> None:
    """Write leaderboard.csv and leaderboard.json to *run_dir*."""
    csv_path = os.path.join(run_dir, "leaderboard.csv")
    json_path = os.path.join(run_dir, "leaderboard.json")

    # CSV — include all LEADERBOARD_COLUMNS + is_parent_baseline
    all_cols = LEADERBOARD_COLUMNS + ["is_parent_baseline"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(leaderboard)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(leaderboard, fh, indent=2, allow_nan=False, default=str)


# ---------------------------------------------------------------------------
# Multi-generation pipeline
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Configuration for a multi-generation evolutionary search.

    Each generation consists of a full :func:`run_crossover_search` run.
    After each generation the top-*k* children (ranked by *primary_metric*)
    are promoted as the parent pair for the next generation.

    Attributes
    ----------
    num_generations:
        Total number of generations to run.  Must be >= 1.
    search_config:
        :class:`SearchConfig` used for the crossover + fine-tune search in
        every generation (shared / frozen across all generations).
    keep_top_k:
        When selecting the **second** parent for the next generation under
        ``selection_strategy="best"``, only the first *keep_top_k* entries of
        the leaderboard (by *primary_metric*) are considered: parent B is the
        rank-2 child **within that prefix**.  ``keep_top_k=1`` therefore has
        no second distinct child and parent B falls back to the original
        parent B (same as a single-child generation).  ``None`` uses the full
        sorted leaderboard for that selection.  **Lineage JSON always records
        every child** in the manifest; this knob does not trim stored lineage.
    mutation_config:
        Optional :class:`~farm.core.decision.training.crossover.MutationConfig`
        applied to the **promoted parent** state dicts before each new
        generation begins.  ``None`` disables mutation (pure crossover).
    selection_strategy:
        How to pick the two parents for the next generation from the
        leaderboard:

        ``"best"``
            Select the top-1 child as parent A and the top-2 child (or
            parent B if only one child was produced) as parent B.

        ``"best_vs_original"``
            Parent A is always the best child; parent B is the original
            parent B supplied to :func:`run_multi_generation_search`.

    seed:
        Optional integer combined with the zero-based generation index *g* to
        diversify stochastic work **per generation**.  When not ``None``,
        every non-``None`` crossover and fine-tune seed in
        :attr:`search_config` is offset by ``seed + g`` for generation *g*
        (so generation 0 matches the template when ``seed`` is ``0``).  When
        :attr:`mutation_config` is set, its NumPy RNG seed is further combined
        with ``seed`` when mutating promoted parents (see
        :func:`run_multi_generation_search`).  ``None`` leaves recipe/regime
        seeds exactly as given in :attr:`search_config` and does not inject a
        mutation RNG seed when ``mutation_config.seed`` is also ``None``.
    """

    num_generations: int
    search_config: SearchConfig
    keep_top_k: Optional[int] = None
    mutation_config: Optional[MutationConfig] = None
    selection_strategy: str = "best"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_generations < 1:
            raise ValueError(
                f"GenerationConfig.num_generations must be >= 1; "
                f"got {self.num_generations!r}"
            )
        if self.selection_strategy not in ("best", "best_vs_original"):
            raise ValueError(
                f"GenerationConfig.selection_strategy must be 'best' or "
                f"'best_vs_original'; got {self.selection_strategy!r}"
            )


@dataclass
class LineageRecord:
    """Provenance record for a single child across generations.

    Attributes
    ----------
    generation:
        Zero-based generation index.
    child_id:
        Child identifier (same as :attr:`ManifestEntry.child_id`).
    parent_a_id:
        Identifier of the parent A used in this generation.  For generation 0
        this is ``"original_parent_a"``; for later generations it is the
        child_id of the promoted parent.
    parent_b_id:
        Identifier of parent B, analogous to *parent_a_id*.
    primary_metric:
        Evaluation primary metric for this child.
    mutation_applied:
        ``True`` if the parents' weights were perturbed by mutation noise
        before this generation's crossover.
    """

    generation: int
    child_id: str
    parent_a_id: str
    parent_b_id: str
    primary_metric: float
    mutation_applied: bool


@dataclass
class GenerationSummary:
    """Aggregated metrics for a single completed generation.

    Attributes
    ----------
    generation:
        Zero-based generation index.
    n_children:
        Number of children evaluated in this generation.
    best_primary_metric:
        Highest primary metric among all children in this generation.
        ``None`` when ``n_children == 0``.
    mean_primary_metric:
        Mean primary metric across all children.
        ``None`` when ``n_children == 0``.
    best_child_id:
        ``child_id`` of the top-ranked child.
    promoted_parent_a_id:
        Child (or original parent) selected as parent A for the *next*
        generation.  ``None`` for the final generation.
    promoted_parent_b_id:
        Child (or original parent) selected as parent B for the *next*
        generation.  ``None`` for the final generation.
    run_dir:
        Filesystem path of this generation's output directory.
    """

    generation: int
    n_children: int
    best_primary_metric: Optional[float]
    mean_primary_metric: Optional[float]
    best_child_id: str
    promoted_parent_a_id: Optional[str]
    promoted_parent_b_id: Optional[str]
    run_dir: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "generation": self.generation,
            "n_children": self.n_children,
            "best_primary_metric": self.best_primary_metric,
            "mean_primary_metric": self.mean_primary_metric,
            "best_child_id": self.best_child_id,
            "promoted_parent_a_id": self.promoted_parent_a_id,
            "promoted_parent_b_id": self.promoted_parent_b_id,
            "run_dir": self.run_dir,
        }


def _load_child_as_module(
    parent_template: nn.Module,
    child_pt_path: str,
) -> nn.Module:
    """Load a saved child checkpoint into a deep-copy of *parent_template*."""
    child = copy.deepcopy(parent_template)
    sd = torch.load(child_pt_path, map_location="cpu", weights_only=True)
    child.load_state_dict(sd)
    return child


def run_multi_generation_search(
    parent_a: nn.Module,
    parent_b: nn.Module,
    states: np.ndarray,
    generation_config: GenerationConfig,
    run_dir: str,
    *,
    thresholds: Optional[RecombinationThresholds] = None,
    include_parent_baseline: bool = True,
    eval_batch_size: Optional[int] = 2048,
) -> Tuple[List[GenerationSummary], List[LineageRecord]]:
    """Run an iterated multi-generation evolutionary crossover search.

    Each generation produces a set of child Q-networks via
    :func:`run_crossover_search`.  The best child(ren) from that generation
    are then promoted as parents for the next generation, enabling iterated
    evolutionary improvement.

    The pipeline per generation *g*:

    1. Run :func:`run_crossover_search` with the current parents and the
       shared :attr:`GenerationConfig.search_config`.
    2. Apply optional :attr:`GenerationConfig.mutation_config` Gaussian noise
       to the promoted parent state dicts.
    3. Select the next parent pair from the leaderboard according to
       :attr:`GenerationConfig.selection_strategy`.
    4. Repeat from step 1 for *g+1*.

    Output is written under ``<run_dir>/gen_<N>/`` for each generation *N*,
    with a summary file ``<run_dir>/multi_gen_summary.json`` and
    ``<run_dir>/lineage.json`` containing all :class:`LineageRecord` entries.

    Parameters
    ----------
    parent_a:
        First parent network (generation 0).
    parent_b:
        Second parent network (generation 0).
    states:
        Shared evaluation state buffer — all generations use the same array.
    generation_config:
        :class:`GenerationConfig` controlling the multi-generation search.
    run_dir:
        Root output directory; per-generation outputs under ``gen_0/``, etc.
    thresholds:
        Optional pass/fail thresholds for :class:`RecombinationEvaluator`.
    include_parent_baseline:
        Forwarded to :func:`run_crossover_search`.
    eval_batch_size:
        Forwarded to :func:`run_crossover_search`.

    Returns
    -------
    summaries : List[GenerationSummary]
        One :class:`GenerationSummary` per generation, in order.
    lineage : List[LineageRecord]
        Full lineage registry: one :class:`LineageRecord` per child across
        all generations.
    """
    os.makedirs(run_dir, exist_ok=True)

    summaries: List[GenerationSummary] = []
    lineage: List[LineageRecord] = []

    current_parent_a = parent_a
    current_parent_b = parent_b
    current_parent_a_id = "original_parent_a"
    current_parent_b_id = "original_parent_b"

    # Keep a reference to the original parent B for the "best_vs_original" strategy.
    original_parent_b = parent_b
    original_parent_b_id = "original_parent_b"

    logger.info(
        "multi_gen_search_start",
        num_generations=generation_config.num_generations,
        selection_strategy=generation_config.selection_strategy,
        mutation_enabled=generation_config.mutation_config is not None,
        run_dir=run_dir,
    )

    for gen_idx in range(generation_config.num_generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen_idx}")

        seed_bump = 0
        if generation_config.seed is not None:
            seed_bump = int(generation_config.seed) + gen_idx
        search_cfg = _search_config_with_generation_seed_bump(
            generation_config.search_config,
            seed_bump,
        )

        logger.info(
            "multi_gen_search_generation_start",
            generation=gen_idx,
            parent_a_id=current_parent_a_id,
            parent_b_id=current_parent_b_id,
            gen_dir=gen_dir,
            generation_seed_bump=seed_bump,
        )

        # ------------------------------------------------------------------
        # Run one generation of crossover search
        # ------------------------------------------------------------------
        manifest, _ = run_crossover_search(
            current_parent_a,
            current_parent_b,
            states,
            search_cfg,
            gen_dir,
            thresholds=thresholds,
            include_parent_baseline=include_parent_baseline,
            eval_batch_size=eval_batch_size,
        )

        if not manifest:
            logger.warning(
                "multi_gen_search_empty_manifest",
                generation=gen_idx,
            )
            summaries.append(
                GenerationSummary(
                    generation=gen_idx,
                    n_children=0,
                    best_primary_metric=None,
                    mean_primary_metric=None,
                    best_child_id="",
                    promoted_parent_a_id=None,
                    promoted_parent_b_id=None,
                    run_dir=gen_dir,
                )
            )
            break

        # ------------------------------------------------------------------
        # Lineage tracking
        # ------------------------------------------------------------------
        for entry in manifest:
            lineage.append(
                LineageRecord(
                    generation=gen_idx,
                    child_id=entry.child_id,
                    parent_a_id=current_parent_a_id,
                    parent_b_id=current_parent_b_id,
                    primary_metric=entry.primary_metric,
                    mutation_applied=generation_config.mutation_config is not None and gen_idx > 0,
                )
            )

        # ------------------------------------------------------------------
        # Generation summary
        # ------------------------------------------------------------------
        sorted_manifest = sorted(manifest, key=lambda e: e.primary_metric, reverse=True)
        keep_k = generation_config.keep_top_k
        top_entries = sorted_manifest[:keep_k] if keep_k is not None else sorted_manifest
        best = top_entries[0]
        metrics = [e.primary_metric for e in manifest]
        mean_metric = sum(metrics) / len(metrics)

        is_last = gen_idx == generation_config.num_generations - 1

        # ------------------------------------------------------------------
        # Select next-generation parents (skip on the final generation)
        # ------------------------------------------------------------------
        next_parent_a_id: Optional[str] = None
        next_parent_b_id: Optional[str] = None
        next_parent_a: Optional[nn.Module] = None
        next_parent_b: Optional[nn.Module] = None

        if not is_last:
            # Load best child as next parent A
            next_parent_a = _load_child_as_module(current_parent_a, best.child_pt_path)
            next_parent_a_id = best.child_id

            if generation_config.selection_strategy == "best" and len(top_entries) >= 2:
                next_parent_b = _load_child_as_module(current_parent_a, top_entries[1].child_pt_path)
                next_parent_b_id = top_entries[1].child_id
            else:
                # "best_vs_original" or only one child available
                next_parent_b = original_parent_b
                next_parent_b_id = original_parent_b_id

            # Apply optional mutation to parent state dicts before promoting
            if generation_config.mutation_config is not None:
                mut_cfg = generation_config.mutation_config
                gen_base = 0 if generation_config.seed is None else int(generation_config.seed)
                if mut_cfg.seed is not None:
                    mut_cfg = replace(
                        mut_cfg,
                        seed=int(mut_cfg.seed) + gen_base + gen_idx + 1,
                    )
                elif gen_base != 0:
                    mut_cfg = replace(mut_cfg, seed=gen_base + gen_idx + 1)

                mutated_a_sd = mutate_state_dict(next_parent_a.state_dict(), mut_cfg)
                next_parent_a.load_state_dict(mutated_a_sd)

                if next_parent_b is not original_parent_b:
                    mut_cfg_b = mut_cfg
                    if mut_cfg.seed is not None:
                        mut_cfg_b = replace(mut_cfg, seed=int(mut_cfg.seed) + 1)
                    mutated_b_sd = mutate_state_dict(next_parent_b.state_dict(), mut_cfg_b)
                    next_parent_b.load_state_dict(mutated_b_sd)

        summaries.append(
            GenerationSummary(
                generation=gen_idx,
                n_children=len(manifest),
                best_primary_metric=best.primary_metric,
                mean_primary_metric=mean_metric,
                best_child_id=best.child_id,
                promoted_parent_a_id=next_parent_a_id,
                promoted_parent_b_id=next_parent_b_id,
                run_dir=gen_dir,
            )
        )

        logger.info(
            "multi_gen_search_generation_complete",
            generation=gen_idx,
            n_children=len(manifest),
            best_primary_metric=round(best.primary_metric, 4),
            mean_primary_metric=round(mean_metric, 4),
            best_child_id=best.child_id,
            promoted_parent_a_id=next_parent_a_id,
            promoted_parent_b_id=next_parent_b_id,
        )

        # Advance parents for the next generation
        if not is_last and next_parent_a is not None:
            current_parent_a = next_parent_a
            current_parent_a_id = next_parent_a_id  # type: ignore[assignment]
            current_parent_b = next_parent_b  # type: ignore[assignment]
            current_parent_b_id = next_parent_b_id  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Persist multi-generation summary and lineage
    # ------------------------------------------------------------------
    summary_path = os.path.join(run_dir, "multi_gen_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in summaries], fh, indent=2, allow_nan=False)

    lineage_path = os.path.join(run_dir, "lineage.json")
    lineage_dicts = [
        {
            "generation": lr.generation,
            "child_id": lr.child_id,
            "parent_a_id": lr.parent_a_id,
            "parent_b_id": lr.parent_b_id,
            "primary_metric": lr.primary_metric,
            "mutation_applied": lr.mutation_applied,
        }
        for lr in lineage
    ]
    with open(lineage_path, "w", encoding="utf-8") as fh:
        json.dump(lineage_dicts, fh, indent=2, allow_nan=False)

    logger.info(
        "multi_gen_search_complete",
        num_generations=len(summaries),
        summary_path=summary_path,
        lineage_path=lineage_path,
        total_children=len(lineage),
    )

    return summaries, lineage

