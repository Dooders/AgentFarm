"""Supervised classification metrics for policies with discrete action labels.

This module computes **accuracy**, **macro-averaged F1**, and a **confusion
matrix** when ground-truth action labels are available alongside evaluation
states.  These metrics complement the Q-value fidelity metrics (KL divergence,
MSE, cosine similarity) and the action-agreement metrics already provided by
:class:`~farm.core.decision.training.trainer_distill.StudentValidator`.

When to use which metric
------------------------
* **Action agreement** (``StudentValidator``): measures how often the *student*
  and *parent* networks choose the *same* action, regardless of which action is
  "correct" in a ground-truth sense.  This is the primary metric for
  distillation fidelity – you care that the student mimics the parent.
* **Supervised accuracy / macro-F1** (this module): measures how often the
  *student* (or parent) network selects the *ground-truth* correct action as
  defined by an external label column.  Use these when you have an offline
  supervised evaluation set (e.g. expert demonstrations or labelled replay
  buffers) and want to report standard classification metrics alongside Q-value
  fidelity.

The two families are intentionally kept separate: a student can achieve
near-perfect action agreement with a parent while both networks make the same
systematic mistakes relative to ground-truth labels (or vice-versa).

All computation is done with pure **NumPy** – no sklearn dependency is
required.

Example
-------
::

    import numpy as np
    from farm.core.decision.training.label_metrics import compute_label_metrics

    # Ground-truth labels and model predictions (integer action indices)
    labels      = np.array([0, 1, 2, 1, 0])
    predictions = np.array([0, 1, 1, 1, 0])

    metrics = compute_label_metrics(labels, predictions)
    print(metrics.accuracy)    # 0.8
    print(metrics.macro_f1)    # macro-averaged F1
    print(metrics.confusion_matrix)  # 3x3 numpy array
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LabelMetrics:
    """Classification metrics computed against ground-truth action labels.

    Attributes
    ----------
    accuracy:
        Fraction of samples where the predicted action equals the ground-truth
        label.  In ``[0, 1]``.
    macro_f1:
        Macro-averaged F1 score: unweighted mean of per-class F1 scores.
        Classes absent from the ground-truth set are excluded from the average.
        In ``[0, 1]``.
    confusion_matrix:
        ``n_classes × n_classes`` confusion matrix as a nested list of ints.
        ``confusion_matrix[i][j]`` is the number of samples with true label
        ``i`` predicted as label ``j``.
    n_classes:
        Number of unique action classes inferred (or supplied).
    support:
        Per-class sample counts (number of ground-truth samples per class) as a
        list of ints with length ``n_classes``.
    per_class_f1:
        Per-class F1 scores as a list of floats with length ``n_classes``.
        Classes with zero true positives *and* zero false positives receive an
        F1 of 0.0.
    """

    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    n_classes: int
    support: List[int]
    per_class_f1: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the metrics."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "n_classes": self.n_classes,
            "support": self.support,
            "per_class_f1": self.per_class_f1,
        }


def compute_label_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    n_classes: Optional[int] = None,
) -> LabelMetrics:
    """Compute accuracy, macro-F1, and confusion matrix using pure NumPy.

    Parameters
    ----------
    labels:
        1-D integer array of ground-truth class indices with shape ``(N,)``.
    predictions:
        1-D integer array of predicted class indices with shape ``(N,)``.
        Must have the same length as *labels*.
    n_classes:
        Number of classes.  When ``None`` (default) it is inferred as
        ``max(labels.max(), predictions.max()) + 1``.  Supply this explicitly
        when not all classes appear in the evaluation batch.

    Returns
    -------
    LabelMetrics

    Raises
    ------
    ValueError
        If *labels* or *predictions* is not 1-D, if they have different
        lengths, or if *n_classes* is less than 2.

    Examples
    --------
    Perfect predictions::

        labels      = np.array([0, 1, 2])
        predictions = np.array([0, 1, 2])
        m = compute_label_metrics(labels, predictions)
        assert m.accuracy == 1.0 and m.macro_f1 == 1.0

    With errors::

        labels      = np.array([0, 1, 2, 1, 0])
        predictions = np.array([0, 1, 1, 1, 0])
        m = compute_label_metrics(labels, predictions)
        assert m.accuracy == pytest.approx(0.8)
    """
    labels_in = np.asarray(labels)
    predictions_in = np.asarray(predictions)

    if labels_in.ndim != 1:
        raise ValueError(
            f"labels must be a 1-D array; got shape {labels_in.shape!r}"
        )
    if predictions_in.ndim != 1:
        raise ValueError(
            f"predictions must be a 1-D array; got shape {predictions_in.shape!r}"
        )

    labels = labels_in
    predictions = predictions_in

    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must have the same length; "
            f"got {len(labels)} and {len(predictions)}"
        )
    if len(labels) == 0:
        raise ValueError("labels and predictions must be non-empty.")

    if n_classes is None:
        n_classes = int(max(labels.max(), predictions.max())) + 1
    if n_classes < 2:
        raise ValueError(
            f"n_classes must be at least 2; got {n_classes}"
        )

    # --- Accuracy ---
    accuracy = float(np.mean(labels == predictions))

    # --- Confusion matrix: cm[true][pred] ---
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, predictions):
        cm[int(t), int(p)] += 1

    # --- Per-class F1 ---
    per_class_f1: List[float] = []
    support: List[int] = []
    classes_in_labels = set(int(x) for x in labels)

    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        sup = tp + fn
        support.append(sup)

        if c not in classes_in_labels:
            # Skip classes absent from ground-truth; they don't count toward macro-F1
            per_class_f1.append(0.0)
            continue

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        per_class_f1.append(float(f1))

    # Macro-F1: mean over classes that appear in the ground-truth labels
    included_f1 = [per_class_f1[c] for c in range(n_classes) if c in classes_in_labels]
    macro_f1 = float(np.mean(included_f1)) if included_f1 else 0.0

    return LabelMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=cm.tolist(),
        n_classes=n_classes,
        support=support,
        per_class_f1=per_class_f1,
    )
