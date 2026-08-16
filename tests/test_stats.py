"""Tests for the statistical evaluation module (F-test, confusion matrix, chi-square)."""

from __future__ import annotations

import numpy as np
import pytest

from brainframe.classification.stats import (
    StatsReport,
    calibration_curve,
    chi_square_goodness_of_fit,
    chi_square_independence,
    confusion_matrix,
    evaluate_classifier,
    evaluate_trained_mlp,
    f_test,
    multiclass_brier_score,
    multiclass_log_loss,
    pr_curves,
    roc_curves,
    top_k_accuracy,
)


def test_confusion_matrix_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(y, y, class_names=["a", "b", "c"])
    assert cm.matrix.shape == (3, 3)
    assert cm.accuracy == 1.0
    # Diagonal = support; off-diagonal = 0.
    for i in range(3):
        assert cm.matrix[i, i] == 2
        assert cm.precision[i] == 1.0
        assert cm.recall[i] == 1.0
        assert cm.f1[i] == 1.0
        assert cm.support[i] == 2


def test_confusion_matrix_with_misclassification():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])  # class 0 -> 1 once, class 2 -> 0 once
    cm = confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])
    assert cm.matrix[0, 1] == 1
    assert cm.matrix[2, 0] == 1
    assert cm.accuracy == pytest.approx(4 / 6)
    # class 1 precision: 2 correct / (2+1 predicted) = 2/3
    assert cm.precision[1] == pytest.approx(2 / 3)
    # class 0 recall: 1 correct / 2 true = 0.5
    assert cm.recall[0] == pytest.approx(0.5)
    d = cm.to_dict()
    assert d["matrix"][0][1] == 1
    assert len(d["class_names"]) == 3


def test_confusion_matrix_handles_unseen_classes():
    # Predicted class 3 never appears in true labels -> matrix still 4x4.
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 3])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.matrix.shape == (4, 4)
    assert cm.matrix[2, 3] == 1


def test_f_test_discriminative_feature_has_high_f():
    # Feature 0 differs strongly by class; feature 1 is noise.
    rng = np.random.default_rng(0)
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    X = np.zeros((n, 2))
    X[:, 0] = y * 10.0 + rng.normal(0, 0.1, n)  # strongly class-dependent
    X[:, 1] = rng.normal(0, 1, n)  # noise
    ft = f_test(X, y, feature_names=["informative", "noise"])
    assert ft.f_statistic[0] > ft.f_statistic[1] * 50
    assert ft.significant[0] is True
    assert ft.p_value[0] < 1e-10
    # Noise feature is unlikely to be significant.
    assert ft.significant[1] is False or ft.p_value[1] > 0.01
    assert len(ft.feature_names) == 2


def test_f_test_constant_feature_returns_zero_f():
    y = np.array([0, 0, 1, 1, 2, 2])
    X = np.ones((6, 1))  # constant -> no variance -> F=0
    ft = f_test(X, y, feature_names=["constant"])
    assert ft.f_statistic[0] == 0.0
    assert ft.significant[0] is False


def test_chi_square_independence_significant_when_dependent():
    # Prediction perfectly determined by the feature -> dependent.
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    feature = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    res = chi_square_independence(labels, feature, "region")
    assert res.chi2 > 0
    assert res.p_value < 0.05
    assert res.significant is True
    assert res.dof == 4  # (3-1)*(3-1)
    assert len(res.contingency) == 3


def test_chi_square_independence_not_significant_when_independent():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=300)
    feature = rng.choice(["a", "b", "c"], size=300)
    res = chi_square_independence(labels, feature, "random")
    # With random data, p should usually be > 0.05.
    assert res.p_value > 0.01
    assert res.dof == 4


def test_chi_square_independence_degenerate_returns_trivial():
    # Single category -> can't test independence.
    res = chi_square_independence(np.array([0, 1, 2]), np.array(["x", "x", "x"]), "single")
    assert res.chi2 == 0.0
    assert res.significant is False


def test_chi_square_goodness_of_fit_uniform_not_significant():
    # Counts ~ uniform -> goodness-of-fit should NOT reject.
    res = chi_square_goodness_of_fit(np.array([33, 33, 34]), n_classes=3)
    assert res.chi2 < 1.0
    assert res.significant is False


def test_chi_square_goodness_of_fit_skewed_significant():
    res = chi_square_goodness_of_fit(np.array([90, 5, 5]), n_classes=3)
    assert res.chi2 > 20
    assert res.p_value < 0.001
    assert res.significant is True


def test_evaluate_classifier_returns_full_report():
    rng = np.random.default_rng(2)
    n_classes = 5
    n = 150
    y = np.repeat(np.arange(n_classes), n // n_classes)
    X = np.zeros((n, 3))
    X[:, 0] = y * 5 + rng.normal(0, 0.2, n)
    X[:, 1] = rng.normal(0, 1, n)
    X[:, 2] = (y == 0).astype(float)
    y_pred = y.copy()
    y_pred[::7] = (y_pred[::7] + 1) % n_classes  # introduce some errors
    rep = evaluate_classifier(
        X, y, y_pred,
        class_names=[f"d{i}" for i in range(n_classes)],
        feature_names=["vol", "noise", "is_class0"],
        alpha=0.05,
    )
    assert isinstance(rep, StatsReport)
    assert rep.n_samples == n
    assert rep.n_classes == n_classes
    assert rep.confusion.matrix.shape == (n_classes, n_classes)
    assert 0 < rep.confusion.accuracy <= 1.0
    assert len(rep.f_test.f_statistic) == 3
    assert rep.f_test.f_statistic[0] > rep.f_test.f_statistic[1]
    # At least the chi-square goodness-of-fit + independence tests present.
    assert len(rep.chi_square) >= 1
    d = rep.to_dict()
    assert "confusion" in d and "f_test" in d and "chi_square" in d


def test_evaluate_trained_mlp_runs_and_has_sensible_metrics():
    rep = evaluate_trained_mlp(n_per_class=60, seed=42, alpha=0.05)
    assert rep.n_classes == 36
    assert rep.n_samples > 0
    # A trained MLP on signature data should beat random chance (>1/36≈0.028).
    assert rep.confusion.accuracy > 0.15
    assert len(rep.f_test.f_statistic) > 0
    # The most discriminative feature should be significant.
    assert any(rep.f_test.significant)
    # Chi-square results include the goodness-of-fit + categorical tests.
    names = [c.name for c in rep.chi_square]
    assert any("distribution" in n for n in names)
    # The probabilistic metrics should be populated (evaluate_trained_mlp passes y_proba).
    assert rep.roc is not None
    assert rep.pr is not None
    assert rep.calibration is not None
    assert rep.top_k is not None
    assert rep.log_loss is not None
    assert rep.brier_score is not None
    # AUC should be well above chance (0.5) for a trained model.
    assert rep.roc.macro_auc > 0.8
    # Top-3 accuracy must be >= top-1 accuracy (monotonic).
    accs = rep.top_k.accuracies
    assert accs[2] >= accs[0]
    assert rep._y_true is not None
    assert rep._y_proba is not None


# ---------------------------------------------------------------------------
# Tests for the new aggregate / probabilistic metrics.
# ---------------------------------------------------------------------------


def test_confusion_matrix_aggregate_metrics_perfect():
    """Perfect predictions -> kappa, MCC, balanced accuracy all = 1.0."""
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = confusion_matrix(y, y, class_names=["a", "b", "c", "d"])
    assert cm.macro_f1 == pytest.approx(1.0)
    assert cm.macro_precision == pytest.approx(1.0)
    assert cm.macro_recall == pytest.approx(1.0)
    assert cm.balanced_accuracy == pytest.approx(1.0)
    assert cm.cohen_kappa == pytest.approx(1.0)
    assert cm.mcc == pytest.approx(1.0)
    # Macro specificity: with perfect predictions, TN/(TN+FP) = 1 for every class.
    assert cm.macro_specificity == pytest.approx(1.0)


def test_confusion_matrix_kappa_for_random_predictions_is_low():
    """Predictions no better than chance -> kappa near 0."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=2000)
    y_pred = rng.integers(0, 5, size=2000)
    cm = confusion_matrix(y_true, y_pred)
    # Kappa should be near 0 (within a tolerance for sampling noise).
    assert -0.05 < cm.cohen_kappa < 0.05
    # MCC should also be near 0.
    assert -0.05 < cm.mcc < 0.05


def test_confusion_matrix_weighted_vs_macro():
    """Weighted metrics should differ from macro when classes are imbalanced."""
    # Class 0 has 10x the support of the others.
    y_true = np.array([0] * 100 + [1] * 10 + [2] * 10)
    y_pred = y_true.copy()  # perfect
    cm = confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])
    assert cm.macro_f1 == pytest.approx(1.0)
    assert cm.weighted_f1 == pytest.approx(1.0)
    # Now misclassify only class-0 (the majority) heavily.
    y_pred = np.where(y_true == 0, 1, y_true)
    cm2 = confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])
    # Weighted F1 (dominated by class 0) should be much lower than macro F1.
    assert cm2.weighted_f1 < cm2.macro_f1


def test_confusion_matrix_specificity_computed():
    """Specificity (true-negative rate) per class is correct."""
    # 3x3 matrix with a known FP for class 0.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 2, 1, 1, 2, 2])  # one class-0 sample predicted as 2
    cm = confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])
    # Class 0: TN=4 (all non-class-0 true samples not predicted as 0), FP=0.
    # Actually FP for class 0 = predicted as 0 but true != 0 = 0.
    # TN for class 0 = total - TP(1) - FP(0) - FN(1) = 6-1-0-1 = 4.
    # specificity = TN / (TN + FP) = 4 / 4 = 1.0
    assert cm.specificity[0] == pytest.approx(1.0)
    # All specificity values in [0, 1].
    for s in cm.specificity:
        assert 0.0 <= s <= 1.0


def test_roc_curves_perfect_classifier():
    """Perfectly separable probs -> AUC = 1.0 for every class."""
    n = 30
    y = np.repeat([0, 1, 2], n // 3)
    proba = np.zeros((n, 3))
    proba[np.arange(n), y] = 0.95
    # Spread remaining mass so rows sum to 1.
    proba = proba + 0.025
    proba = proba / proba.sum(axis=1, keepdims=True)
    roc = roc_curves(y, proba, class_names=["a", "b", "c"])
    assert all(a == pytest.approx(1.0) for a in roc.auc)
    assert roc.macro_auc == pytest.approx(1.0)
    assert len(roc.fpr) == 3
    assert len(roc.tpr) == 3


def test_pr_curves_returns_average_precision():
    rng = np.random.default_rng(3)
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    proba = rng.random((n, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)
    pr = pr_curves(y, proba, class_names=["a", "b", "c"])
    assert len(pr.average_precision) == 3
    for ap in pr.average_precision:
        assert 0.0 <= ap <= 1.0
    assert 0.0 <= pr.macro_ap <= 1.0


def test_calibration_curve_well_calibrated():
    """When confidence == accuracy in every bin, ECE ≈ 0."""
    n = 2000
    rng = np.random.default_rng(7)
    y = rng.integers(0, 4, size=n)
    # Perfect predictions with confidence 1.0 -> accuracy 1.0 in the top bin.
    proba = np.zeros((n, 4))
    proba[np.arange(n), y] = 1.0
    cal = calibration_curve(y, proba, n_bins=10)
    assert cal.ece == pytest.approx(0.0, abs=0.01)


def test_calibration_curve_overconfident():
    """Confidence 1.0 but 50% accuracy -> large ECE."""
    rng = np.random.default_rng(11)
    n = 1000
    y = rng.integers(0, 2, size=n)
    proba = np.full((n, 2), 0.5)
    proba[np.arange(n), y] = 1.0  # always confident in a (correct) class
    # But flip half the predictions to be wrong yet confident.
    flip = rng.random(n) < 0.5
    proba[flip] = 0.0
    proba[flip, 1 - y[flip]] = 1.0
    cal = calibration_curve(y, proba, n_bins=10)
    # Confidence ~1.0 but accuracy ~0.5 -> ECE should be large.
    assert cal.ece > 0.3


def test_top_k_accuracy_monotonic_and_correct():
    """top-K accuracy is non-decreasing in K; top-1 == accuracy."""
    rng = np.random.default_rng(5)
    n = 100
    y = rng.integers(0, 5, size=n)
    proba = rng.random((n, 5))
    proba = proba / proba.sum(axis=1, keepdims=True)
    res = top_k_accuracy(y, proba, k_values=(1, 3, 5))
    assert res.k_values == [1, 3, 5]
    assert res.accuracies[0] <= res.accuracies[1] <= res.accuracies[2]
    # top-5 (all classes) must be 1.0 when K >= n_classes.
    assert res.accuracies[2] == pytest.approx(1.0)


def test_multiclass_log_loss_perfect_is_near_zero():
    """Near-perfect confident predictions -> log loss near 0."""
    n = 30
    y = np.repeat([0, 1, 2], n // 3)
    proba = np.full((n, 3), 1e-6)
    proba[np.arange(n), y] = 1 - 2e-6
    ll = multiclass_log_loss(y, proba)
    assert ll < 0.001


def test_multiclass_log_loss_uniform_is_high():
    """Uniform predictions on 3 classes -> log loss ≈ ln(3) ≈ 1.099."""
    n = 90
    y = np.repeat([0, 1, 2], n // 3)
    proba = np.full((n, 3), 1 / 3)
    ll = multiclass_log_loss(y, proba)
    assert ll == pytest.approx(np.log(3), abs=0.01)


def test_multiclass_brier_score_perfect_is_zero():
    n = 30
    y = np.repeat([0, 1, 2], n // 3)
    proba = np.zeros((n, 3))
    proba[np.arange(n), y] = 1.0
    assert multiclass_brier_score(y, proba, 3) == pytest.approx(0.0)


def test_multiclass_brier_score_uniform():
    """Uniform probs on K=3 -> per-sample Brier = 2/3 (one (p-1)^2 + two (p-0)^2)."""
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    proba = np.full((n, 3), 1 / 3)
    # Per sample: (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3.
    expected = 2 / 3
    assert multiclass_brier_score(y, proba, 3) == pytest.approx(expected)


def test_evaluate_classifier_with_y_proba_populates_probabilistic_metrics():
    rng = np.random.default_rng(2)
    n_classes = 5
    n = 150
    y = np.repeat(np.arange(n_classes), n // n_classes)
    X = np.zeros((n, 3))
    X[:, 0] = y * 5 + rng.normal(0, 0.2, n)
    X[:, 1] = rng.normal(0, 1, n)
    X[:, 2] = (y == 0).astype(float)
    # Build a probability matrix that's roughly correct.
    proba = np.full((n, n_classes), 0.05)
    proba[np.arange(n), y] = 0.80
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = proba.argmax(1)
    rep = evaluate_classifier(
        X, y, y_pred, y_proba=proba,
        class_names=[f"d{i}" for i in range(n_classes)],
        feature_names=["vol", "noise", "is_class0"],
        alpha=0.05,
    )
    assert rep.roc is not None
    assert rep.pr is not None
    assert rep.calibration is not None
    assert rep.top_k is not None
    assert rep.log_loss is not None
    assert rep.brier_score is not None
    assert isinstance(rep, StatsReport)
    d = rep.to_dict()
    assert d["roc"] is not None
    assert d["pr"] is not None
    assert d["calibration"] is not None
    assert d["top_k"] is not None


def test_evaluate_classifier_without_y_proba_leaves_probabilistic_none():
    """When y_proba is None, the probabilistic fields stay unset."""
    rng = np.random.default_rng(2)
    n_classes = 5
    n = 150
    y = np.repeat(np.arange(n_classes), n // n_classes)
    X = np.zeros((n, 3))
    X[:, 0] = y * 5 + rng.normal(0, 0.2, n)
    y_pred = y.copy()
    rep = evaluate_classifier(
        X, y, y_pred, class_names=[f"d{i}" for i in range(n_classes)],
        feature_names=["vol", "noise", "is_class0"],
    )
    assert rep.roc is None
    assert rep.pr is None
    assert rep.calibration is None
    assert rep.top_k is None
    assert rep.log_loss is None
    assert rep.brier_score is None
