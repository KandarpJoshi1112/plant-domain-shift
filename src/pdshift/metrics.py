"""Evaluation metrics: classification quality, calibration and statistical tests."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

logger = logging.getLogger(__name__)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    if probs.ndim != 2 or len(probs) != len(labels):
        raise ValueError("probs must be (n, k) and match labels")
    if len(labels) == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + eps)))


def bootstrap_ci(
    labels: np.ndarray, preds: np.ndarray, metric: str, n_samples: int = 2000, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for accuracy or macro-F1."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, n)
        if metric == "accuracy":
            scores[i] = accuracy_score(labels[idx], preds[idx])
        elif metric == "macro_f1":
            scores[i] = f1_score(labels[idx], preds[idx], average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_test(labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test comparing two models on the same test images.

    Uses only discordant pairs: b = A right/B wrong, c = A wrong/B right.
    """
    if not (len(labels) == len(preds_a) == len(preds_b)):
        raise ValueError("All arrays must have the same length")
    a_ok, b_ok = preds_a == labels, preds_b == labels
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def classification_report(
    probs: np.ndarray, labels: np.ndarray, class_names: list[str], ece_bins: int, n_boot: int,
) -> dict[str, Any]:
    """Compute the full metric set for one evaluation domain."""
    preds = probs.argmax(axis=1)
    k = len(class_names)
    present = np.unique(labels)
    report: dict[str, Any] = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=present, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, labels, ece_bins),
        "nll": negative_log_likelihood(probs, labels),
        "accuracy_ci95": bootstrap_ci(labels, preds, "accuracy", n_boot),
        "macro_f1_ci95": bootstrap_ci(labels, preds, "macro_f1", n_boot),
        "per_class_recall": {
            class_names[c]: float(r) for c, r in zip(
                present, recall_score(labels, preds, labels=present, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(k))).tolist(),
    }
    return report
