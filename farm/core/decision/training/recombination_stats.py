"""Statistics helpers for aggregating repeated-run recombination experiments.

This module provides a thin, numpy-first analysis layer on top of the JSON
artefacts produced by :func:`~farm.core.decision.training.crossover_search.run_crossover_search`.
It covers three main use-cases:

1. **Loading** – read one or many ``manifest.json`` files (one per seed run)
   into a flat list of metric dictionaries.
2. **Summary stats** – compute per-condition *mean ± std* and *count* for any
   numeric metric column (uses only numpy; SciPy is optional).
3. **Significance tests** – compare two conditions with a paired t-test, a
   Welch t-test, or a bootstrap confidence interval.  All three helpers use
   SciPy when available and fall back to a pure-numpy implementation
   otherwise; assumptions are clearly documented in each docstring.

Typical workflow
----------------
::

    from farm.core.decision.training.recombination_stats import (
        load_manifest_entries,
        aggregate_conditions,
        welch_ttest,
        bootstrap_ci,
    )

    # Load two seed-run directories into a combined table.
    rows = load_manifest_entries(["run_seed0/manifest.json",
                                  "run_seed1/manifest.json"])

    # Summarise by (crossover_mode, finetune_regime).
    summaries = aggregate_conditions(
        rows, group_by=("crossover_mode", "finetune_regime")
    )
    for key, s in summaries.items():
        print(key, s.mean_primary_metric, "±", s.std_primary_metric)

    # Compare crossover vs distill-only on primary_metric.
    crossover_vals = [r["primary_metric"] for r in rows
                      if r["crossover_mode"] == "weighted"]
    distill_vals   = [r["primary_metric"] for r in rows
                      if r["crossover_mode"] == "random"]
    result = welch_ttest(crossover_vals, distill_vals)
    print(result)

Public API
----------
- :class:`ConditionSummary`    – named summary statistics for one condition
- :func:`load_manifest_entries` – parse manifest.json file(s) → list of dicts
- :func:`compute_condition_summary` – mean/std/min/max/count for a metric list
- :func:`aggregate_conditions` – group rows and return per-key summaries
- :func:`paired_ttest`         – paired (dependent-samples) t-test
- :func:`welch_ttest`          – Welch (independent, unequal-variance) t-test
- :func:`bootstrap_ci`         – bootstrap confidence interval for the mean
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# SciPy is an optional runtime dep; we fall back to numpy implementations.
try:
    from scipy import stats as _scipy_stats  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Numeric metric keys present in every ManifestEntry / manifest row.
# ---------------------------------------------------------------------------

NUMERIC_METRIC_KEYS: Tuple[str, ...] = (
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
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ConditionSummary:
    """Descriptive statistics for one experimental condition across N runs.

    Attributes
    ----------
    condition_key:
        A tuple of ``(group_field, value)`` pairs that identify the condition,
        e.g. ``(("crossover_mode", "weighted"), ("finetune_regime", "long"))``.
    n_runs:
        Number of rows (individual children/seeds) aggregated.
    mean_primary_metric, std_primary_metric:
        Mean and sample standard deviation of ``primary_metric`` across runs.
        Returns ``float("nan")`` when ``n_runs < 2`` for std.
    mean_agree_a, std_agree_a:
        Statistics for ``child_vs_parent_a_agreement``.
    mean_agree_b, std_agree_b:
        Statistics for ``child_vs_parent_b_agreement``.
    mean_oracle_agreement, std_oracle_agreement:
        Statistics for ``oracle_agreement`` (``None`` values are skipped).
    n_degenerate:
        Count of rows marked ``degenerate=True``.
    extra:
        Any additional per-metric stats computed via
        :func:`compute_condition_summary`.
    """

    condition_key: Tuple[Tuple[str, Any], ...]
    n_runs: int
    mean_primary_metric: float
    std_primary_metric: float
    mean_agree_a: float
    std_agree_a: float
    mean_agree_b: float
    std_agree_b: float
    mean_oracle_agreement: float
    std_oracle_agreement: float
    n_degenerate: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "condition_key": list(self.condition_key),
            "n_runs": self.n_runs,
            "mean_primary_metric": self.mean_primary_metric,
            "std_primary_metric": self.std_primary_metric,
            "mean_agree_a": self.mean_agree_a,
            "std_agree_a": self.std_agree_a,
            "mean_agree_b": self.mean_agree_b,
            "std_agree_b": self.std_agree_b,
            "mean_oracle_agreement": self.mean_oracle_agreement,
            "std_oracle_agreement": self.std_oracle_agreement,
            "n_degenerate": self.n_degenerate,
            **self.extra,
        }


@dataclass
class TTestResult:
    """Result of a two-sample t-test.

    Attributes
    ----------
    statistic:
        The t-statistic.
    pvalue:
        Two-sided p-value.
    dof:
        Degrees of freedom used.
    mean_a, mean_b:
        Sample means.
    mean_diff:
        ``mean_a - mean_b``.
    method:
        Either ``"paired"`` or ``"welch"``.
    scipy_used:
        ``True`` when SciPy provided the computation.
    """

    statistic: float
    pvalue: float
    dof: float
    mean_a: float
    mean_b: float
    mean_diff: float
    method: str
    scipy_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "dof": self.dof,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "method": self.method,
            "scipy_used": self.scipy_used,
        }


@dataclass
class BootstrapCIResult:
    """Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    mean:
        Observed sample mean.
    ci_low, ci_high:
        Lower and upper confidence-interval bounds.
    confidence_level:
        Requested coverage (e.g. 0.95).
    n_bootstrap:
        Number of bootstrap resamples used.
    """

    mean: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_bootstrap: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_manifest_entries(
    sources: Union[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    """Load manifest rows from one or more ``manifest.json`` paths.

    Each ``manifest.json`` (written by :func:`run_crossover_search`) is a JSON
    array of :class:`~farm.core.decision.training.crossover_search.ManifestEntry`
    dictionaries.  This function concatenates all rows from all files into a
    single flat list so that callers can aggregate across seeds / reruns.

    Parameters
    ----------
    sources:
        A single file path string **or** an iterable of file path strings.
        Each file must contain a JSON array (the manifest).

    Returns
    -------
    List[Dict[str, Any]]
        Flat list of manifest-row dictionaries, one per evaluated child.
        A ``"_source_file"`` key is injected into every row so that the
        origin of each row can be traced.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    ValueError
        If a file does not parse as a JSON array.
    """
    if isinstance(sources, str):
        sources = [sources]

    rows: List[Dict[str, Any]] = []
    for path in sources:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"{path!r} does not contain a JSON array; got {type(data).__name__}."
            )
        for row in data:
            row = dict(row)
            row.setdefault("_source_file", path)
            rows.append(row)
    return rows


def load_eval_reports(
    paths: Sequence[str],
) -> List[Dict[str, Any]]:
    """Load individual ``eval_report.json`` files into a flat list.

    Use this when you have a collection of ``eval_report.json`` files rather
    than a ``manifest.json``.  The returned dicts contain the full report
    including ``"comparisons"`` and ``"summary"`` sections; metric extraction
    is left to the caller.

    Parameters
    ----------
    paths:
        Iterable of absolute or relative file paths to ``eval_report.json``
        files.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per file, with an added ``"_source_file"`` key.
    """
    reports: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = dict(data)
        data.setdefault("_source_file", path)
        reports.append(data)
    return reports


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_condition_summary(
    values: Sequence[float],
) -> Dict[str, float]:
    """Compute descriptive statistics for a sequence of scalar metric values.

    All statistics use **sample** standard deviation (ddof=1) so they are
    unbiased estimators when ``len(values) >= 2``.

    Parameters
    ----------
    values:
        Numeric observations (e.g. primary_metric values across N seeds).
        ``None`` and ``float("nan")`` values are dropped before computation.

    Returns
    -------
    Dict[str, float]
        Keys: ``mean``, ``std``, ``min``, ``max``, ``count``.
        ``std`` is ``float("nan")`` when fewer than two finite values remain.
    """
    arr = np.array([v for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": n,
    }


def aggregate_conditions(
    rows: Sequence[Dict[str, Any]],
    group_by: Sequence[str] = ("crossover_mode", "finetune_regime"),
    extra_metrics: Sequence[str] = (),
) -> Dict[Tuple[Tuple[str, Any], ...], ConditionSummary]:
    """Group manifest rows by condition and return per-condition summaries.

    Parameters
    ----------
    rows:
        Flat list of manifest-row dicts as returned by
        :func:`load_manifest_entries`.
    group_by:
        Column names to group on.  Defaults to
        ``("crossover_mode", "finetune_regime")``.
    extra_metrics:
        Additional numeric keys from each row to include in
        :attr:`ConditionSummary.extra` as ``{"<key>": compute_condition_summary(...)}``.

    Returns
    -------
    Dict[Tuple[Tuple[str, Any], ...], ConditionSummary]
        Mapping from condition key tuple → :class:`ConditionSummary`.  The key
        is a tuple of ``(field_name, value)`` pairs in ``group_by`` order, e.g.
        ``(("crossover_mode", "weighted"), ("finetune_regime", "long"))``.
    """
    # Group rows.
    groups: Dict[Tuple[Tuple[str, Any], ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple((col, row.get(col)) for col in group_by)
        groups.setdefault(key, []).append(row)

    summaries: Dict[Tuple[Tuple[str, Any], ...], ConditionSummary] = {}
    for key, group_rows in groups.items():
        pm = [r.get("primary_metric") for r in group_rows]
        aa = [r.get("child_vs_parent_a_agreement") for r in group_rows]
        ab = [r.get("child_vs_parent_b_agreement") for r in group_rows]
        oa = [r.get("oracle_agreement") for r in group_rows]
        n_degen = sum(1 for r in group_rows if r.get("degenerate", False))

        pm_s = compute_condition_summary(pm)
        aa_s = compute_condition_summary(aa)
        ab_s = compute_condition_summary(ab)
        oa_s = compute_condition_summary(oa)

        extra: Dict[str, Any] = {}
        for metric in extra_metrics:
            vals = [r.get(metric) for r in group_rows]
            extra[metric] = compute_condition_summary(vals)

        summaries[key] = ConditionSummary(
            condition_key=key,
            n_runs=len(group_rows),
            mean_primary_metric=pm_s["mean"],
            std_primary_metric=pm_s["std"],
            mean_agree_a=aa_s["mean"],
            std_agree_a=aa_s["std"],
            mean_agree_b=ab_s["mean"],
            std_agree_b=ab_s["std"],
            mean_oracle_agreement=oa_s["mean"],
            std_oracle_agreement=oa_s["std"],
            n_degenerate=n_degen,
            extra=extra,
        )

    return summaries


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def paired_ttest(
    a: Sequence[float],
    b: Sequence[float],
) -> TTestResult:
    """Paired (dependent-samples) t-test comparing two equal-length conditions.

    **Assumptions**

    * Each element ``a[i]`` and ``b[i]`` is a paired observation from the
      *same* experimental unit (e.g. the same random seed run under two
      different crossover strategies).
    * Differences ``a[i] - b[i]`` are approximately normally distributed.
      Violated by very small samples (N < 5) or heavily skewed differences.
    * Observations are independent across pairs.

    When SciPy is available, delegates to
    :func:`scipy.stats.ttest_rel`; otherwise uses the equivalent manual
    computation: ``t = mean(d) / (std(d, ddof=1) / sqrt(N))``,
    ``dof = N - 1``, two-sided p-value from the t-distribution CDF.

    Parameters
    ----------
    a, b:
        Equal-length sequences of numeric values (floats).

    Returns
    -------
    TTestResult

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or fewer than 2 pairs.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired_ttest requires equal-length inputs; got {len(a_arr)} vs {len(b_arr)}."
        )
    n = len(a_arr)
    if n < 2:
        raise ValueError("paired_ttest requires at least 2 paired observations.")

    if _SCIPY_AVAILABLE:
        res = _scipy_stats.ttest_rel(a_arr, b_arr)
        t_stat = float(res.statistic)
        pval = float(res.pvalue)
        dof = float(n - 1)
        scipy_used = True
    else:
        diff = a_arr - b_arr
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else float("nan")
        dof = float(n - 1)
        # Two-sided p-value via normal approximation for large N, else nan.
        pval = _two_sided_pvalue_from_t(t_stat, dof)
        scipy_used = False

    return TTestResult(
        statistic=t_stat,
        pvalue=pval,
        dof=dof,
        mean_a=float(np.mean(a_arr)),
        mean_b=float(np.mean(b_arr)),
        mean_diff=float(np.mean(a_arr - b_arr)),
        method="paired",
        scipy_used=scipy_used,
    )


def welch_ttest(
    a: Sequence[float],
    b: Sequence[float],
) -> TTestResult:
    """Welch's t-test for two independent samples with potentially unequal variance.

    **Assumptions**

    * Samples ``a`` and ``b`` are drawn **independently** (not paired).
    * Each sample is approximately normally distributed.  Robust to moderate
      non-normality for N ≥ 10 per group by the Central Limit Theorem.
    * Unlike Student's t-test, Welch's test does *not* assume equal variances
      (heteroscedastic-safe).

    When SciPy is available, delegates to
    :func:`scipy.stats.ttest_ind` with ``equal_var=False`` (Welch); otherwise
    uses Welch–Satterthwaite degrees-of-freedom and the t-statistic formula.

    Parameters
    ----------
    a, b:
        Independent samples of numeric values.  May have different lengths.

    Returns
    -------
    TTestResult

    Raises
    ------
    ValueError
        If either sample has fewer than 2 observations.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 2:
        raise ValueError("welch_ttest: sample 'a' must have at least 2 observations.")
    if len(b_arr) < 2:
        raise ValueError("welch_ttest: sample 'b' must have at least 2 observations.")

    if _SCIPY_AVAILABLE:
        res = _scipy_stats.ttest_ind(a_arr, b_arr, equal_var=False)
        t_stat = float(res.statistic)
        pval = float(res.pvalue)
        # Welch–Satterthwaite dof from scipy result
        dof = float(getattr(res, "df", len(a_arr) + len(b_arr) - 2))
        scipy_used = True
    else:
        n_a, n_b = len(a_arr), len(b_arr)
        var_a = float(np.var(a_arr, ddof=1))
        var_b = float(np.var(b_arr, ddof=1))
        mean_a = float(np.mean(a_arr))
        mean_b = float(np.mean(b_arr))
        se = math.sqrt(var_a / n_a + var_b / n_b)
        t_stat = (mean_a - mean_b) / se if se > 0 else float("nan")
        # Welch–Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        dof = num / den if den > 0 else float("nan")
        pval = _two_sided_pvalue_from_t(t_stat, dof)
        scipy_used = False

    return TTestResult(
        statistic=t_stat,
        pvalue=pval,
        dof=dof,
        mean_a=float(np.mean(a_arr)),
        mean_b=float(np.mean(b_arr)),
        mean_diff=float(np.mean(a_arr)) - float(np.mean(b_arr)),
        method="welch",
        scipy_used=scipy_used,
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    rng: Optional[Union[int, np.random.Generator]] = None,
    statistic: str = "mean",
) -> BootstrapCIResult:
    """Non-parametric bootstrap confidence interval for a sample statistic.

    **Assumptions**

    * Observations are i.i.d. (independent and identically distributed).
    * The bootstrap distribution is a good approximation of the true sampling
      distribution.  Accuracy improves with larger ``n_bootstrap`` and larger
      sample sizes.  Use at least 1 000 resamples for publication-quality CIs.
    * The percentile method is used (``lower = alpha/2``, ``upper = 1 - alpha/2``
      quantiles of the bootstrap distribution).  For strongly skewed
      distributions, consider bias-corrected and accelerated (BCa) CIs
      (not implemented here; SciPy's :func:`scipy.stats.bootstrap` provides BCa).

    Parameters
    ----------
    values:
        Sample observations.  Must have at least 2 elements.
    confidence_level:
        Target coverage, e.g. ``0.95`` for a 95 % CI.  Must be in (0, 1).
    n_bootstrap:
        Number of bootstrap resamples.
    rng:
        An integer seed or a :class:`numpy.random.Generator` for
        reproducibility.  If ``None``, uses the default global RNG.
    statistic:
        ``"mean"`` (default) or ``"median"``.

    Returns
    -------
    BootstrapCIResult

    Raises
    ------
    ValueError
        If ``values`` has fewer than 2 elements, or ``confidence_level`` is
        outside (0, 1), or ``statistic`` is not recognised.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        raise ValueError("bootstrap_ci requires at least 2 observations.")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}.")
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median'; got {statistic!r}.")

    if rng is None:
        gen = np.random.default_rng()
    elif isinstance(rng, int):
        gen = np.random.default_rng(rng)
    else:
        gen = rng

    stat_fn = np.mean if statistic == "mean" else np.median
    observed = float(stat_fn(arr))

    # Draw bootstrap resamples.
    indices = gen.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_stats = np.array([stat_fn(arr[idx]) for idx in indices])

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        mean=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _two_sided_pvalue_from_t(t_stat: float, dof: float) -> float:
    """Approximate two-sided p-value using the incomplete beta function.

    Falls back to a normal approximation when ``dof`` is large (≥ 30).
    This is used only when SciPy is unavailable.
    """
    if math.isnan(t_stat) or math.isnan(dof) or dof <= 0:
        return float("nan")

    if dof >= 30:
        # Normal approximation for large dof.
        abs_t = abs(t_stat)
        # Two-sided normal p-value via complementary error function.
        pval = math.erfc(abs_t / math.sqrt(2))
        return float(pval)

    # Regularised incomplete beta function B(x; a, b) where
    # x = dof / (dof + t^2), a = dof/2, b = 0.5.
    try:
        x = dof / (dof + t_stat ** 2)
        pval = _betainc(dof / 2.0, 0.5, x)
        return float(pval)
    except Exception:
        return float("nan")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction.

    Adapted from Numerical Recipes §6.4 (Lentz's method).  Accurate for the
    range of parameters typical in t-distribution p-value computation.
    """
    if x < 0.0 or x > 1.0:
        return float("nan")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use symmetry relation when x > (a+1)/(a+b+2) for better convergence.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    # Lentz's continued fraction.
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, 201):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break

    return front * h
