import numpy as np
import pytest

from pdshift.metrics import bootstrap_ci, expected_calibration_error, mcnemar_test


def test_ece_zero_for_perfect_confident_predictions():
    labels = np.array([0, 1, 2, 1])
    probs = np.eye(3)[labels]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0]] * 4)  # always predicts class 0 with full confidence, 50% right
    assert expected_calibration_error(probs, labels) == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 300)
    preds = np.where(rng.random(300) < 0.8, labels, (labels + 1) % 3)
    lo, hi = bootstrap_ci(labels, preds, "accuracy", n_samples=500)
    acc = (labels == preds).mean()
    assert lo <= acc <= hi


def test_mcnemar_identical_models():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    res = mcnemar_test(labels, preds, preds)
    assert res["p_value"] == 1.0 and res["a_only_correct"] == 0
