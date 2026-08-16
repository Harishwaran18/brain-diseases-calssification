"""Statistical evaluation of the disease classifier.

Three complementary, classical statistical techniques are provided to rigorously
characterise the trained classifier beyond point-estimate accuracy:

1. **Confusion matrix** -- the N x N contingency of true vs predicted class,
   with per-class precision / recall / F1 / support. The standard way to see
   *which* diseases get confused with which.

2. **F-test (one-way ANOVA)** -- for every input feature, tests whether the
   feature's mean differs significantly across the disease classes. A large
   F-statistic with a small p-value means the feature discriminates between
   diseases (is informative for classification). This is the textbook
   feature-selection F-test (scikit-learn ``f_classif``).

3. **Chi-square test of independence** -- tests whether the predicted disease
   label is statistically independent of a categorical lesion feature (region,
   laterality, pattern). A small p-value rejects independence, i.e. the
   prediction depends on that feature -- exactly what a well-behaved
   evidence-based classifier should show. Also reports a goodness-of-fit
   chi-square of the predicted-class distribution against a uniform prior.

All three are computed on the *same* held-out signature-derived dataset used to
train the MLP, so the numbers are reproducible and require no external data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ConfusionMatrix:
    """A multi-class confusion matrix with per-class and aggregate metrics."""

    matrix: np.ndarray  # shape (n_classes, n_classes); rows=true, cols=pred
    class_names: list[str]
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    support: list[int] = field(default_factory=list)
    specificity: list[float] = field(default_factory=list)
    accuracy: float = 0.0
    # Aggregate (single-number) summaries.
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    macro_specificity: float = 0.0
    balanced_accuracy: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "class_names": self.class_names,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "macro_specificity": self.macro_specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "cohen_kappa": self.cohen_kappa,
            "mcc": self.mcc,
        }


@dataclass
class FTestResult:
    """Per-feature one-way ANOVA F-test results."""

    feature_names: list[str]
    f_statistic: list[float]
    p_value: list[float]
    significant: list[bool]  # True where p < alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "f_statistic": self.f_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
        }


@dataclass
class ChiSquareResult:
    """A chi-square test of independence (or goodness-of-fit) result."""

    name: str
    chi2: float
    dof: int
    p_value: float
    significant: bool
    contingency: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chi2": self.chi2,
            "dof": self.dof,
            "p_value": self.p_value,
            "significant": self.significant,
            "contingency": self.contingency,
        }


@dataclass
class ROCResult:
    """Per-class one-vs-rest ROC curves + macro/weighted AUC."""

    class_names: list[str]
    fpr: list[list[float]]  # per-class false-positive-rate arrays
    tpr: list[list[float]]  # per-class true-positive-rate arrays
    thresholds: list[list[float]]
    auc: list[float]  # per-class AUC
    macro_auc: float
    weighted_auc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "thresholds": self.thresholds,
            "auc": self.auc,
            "macro_auc": self.macro_auc,
            "weighted_auc": self.weighted_auc,
        }


@dataclass
class PRResult:
    """Per-class one-vs-rest precision-recall curves + average precision."""

    class_names: list[str]
    precision: list[list[float]]
    recall: list[list[float]]
    thresholds: list[list[float]]
    average_precision: list[float]
    macro_ap: float
    weighted_ap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "precision": self.precision,
            "recall": self.recall,
            "thresholds": self.thresholds,
            "average_precision": self.average_precision,
            "macro_ap": self.macro_ap,
            "weighted_ap": self.weighted_ap,
        }


@dataclass
class CalibrationResult:
    """Reliability-diagram data + Expected Calibration Error."""

    bin_centers: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    ece: float  # expected calibration error
    max_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_centers": self.bin_centers,
            "bin_accuracies": self.bin_accuracies,
            "bin_confidences": self.bin_confidences,
            "bin_counts": self.bin_counts,
            "ece": self.ece,
            "max_calibration_error": self.max_calibration_error,
        }


@dataclass
class TopKResult:
    """Top-K accuracy for multiple K values (differential-diagnosis quality)."""

    k_values: list[int]
    accuracies: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_values": self.k_values,
            "accuracies": self.accuracies,
        }


@dataclass
class StatsReport:
    """Bundle of all statistical evaluations."""

    confusion: ConfusionMatrix
    f_test: FTestResult
    chi_square: list[ChiSquareResult]
    n_samples: int
    n_classes: int
    # Probabilistic / threshold-free metrics (require y_proba).
    roc: ROCResult | None = None
    pr: PRResult | None = None
    calibration: CalibrationResult | None = None
    top_k: TopKResult | None = None
    log_loss: float | None = None
    brier_score: float | None = None
    # Raw arrays retained (not serialised) for interactive charts like the
    # confidence histogram that need per-sample probabilities.
    _y_true: np.ndarray | None = field(default=None, repr=False, compare=False)
    _y_proba: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion": self.confusion.to_dict(),
            "f_test": self.f_test.to_dict(),
            "chi_square": [c.to_dict() for c in self.chi_square],
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "roc": self.roc.to_dict() if self.roc else None,
            "pr": self.pr.to_dict() if self.pr else None,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "top_k": self.top_k.to_dict() if self.top_k else None,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
        }


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> ConfusionMatrix:
    """Build a multi-class confusion matrix with per-class and aggregate metrics.

    Parameters
    ----------
    y_true, y_pred
        1-D integer arrays of true and predicted class ids.
    class_names
        Display names ordered by class id. Defaults to ``"class_{i}"``.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    n = int(max(y_true.max(), y_pred.max())) + 1 if len(y_true) else 1
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred, strict=False):
        mat[t, p] += 1
    names = class_names or [f"class_{i}" for i in range(n)]
    names = names[:n] if len(names) >= n else names + [f"class_{i}" for i in range(len(names), n)]
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    specificity: list[float] = []
    support: list[int] = []
    for i in range(n):
        tp = int(mat[i, i])
        fp = int(mat[:, i].sum() - tp)
        fn = int(mat[i, :].sum() - tp)
        tn = int(mat.sum() - tp - fp - fn)
        sup = int(mat[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision.append(float(prec))
        recall.append(float(rec))
        f1.append(float(f))
        specificity.append(float(spec))
        support.append(sup)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    total = int(mat.sum()) or 1
    weights = np.array(support, dtype=float) / total
    macro_prec = float(np.mean(precision)) if precision else 0.0
    macro_rec = float(np.mean(recall)) if recall else 0.0
    macro_f1 = float(np.mean(f1)) if f1 else 0.0
    macro_spec = float(np.mean(specificity)) if specificity else 0.0
    w_prec = float(np.dot(precision, weights)) if precision else 0.0
    w_rec = float(np.dot(recall, weights)) if recall else 0.0
    w_f1 = float(np.dot(f1, weights)) if f1 else 0.0
    # Balanced accuracy = mean recall (sensitivity).
    balanced = macro_rec
    # Cohen's kappa: pe = sum(p_i * q_i) over classes (marginal-probability product).
    po = acc
    row = mat.sum(axis=1)  # true marginal counts (1-D, length n)
    col = mat.sum(axis=0)  # predicted marginal counts (1-D, length n)
    pe = float(np.dot(row, col)) / (total * total)
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-12 else 0.0
    # Matthews correlation coefficient (multiclass formula).
    mcc = _multiclass_mcc(mat)
    return ConfusionMatrix(
        matrix=mat, class_names=names, precision=precision, recall=recall,
        f1=f1, support=support, specificity=specificity, accuracy=acc,
        macro_precision=macro_prec, macro_recall=macro_rec, macro_f1=macro_f1,
        weighted_precision=w_prec, weighted_recall=w_rec, weighted_f1=w_f1,
        macro_specificity=macro_spec, balanced_accuracy=balanced,
        cohen_kappa=float(kappa), mcc=float(mcc),
    )


def _multiclass_mcc(mat: np.ndarray) -> float:
    """Matthews correlation coefficient for a multiclass confusion matrix.

    Uses the Gorodkin generalisation (R-style): a stable covariance formula
    over the multi-class contingency table.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return 0.0
    t_k = mat.sum(axis=1)  # true counts per class
    p_k = mat.sum(axis=0)  # predicted counts per class
    c = mat.trace()        # correctly classified
    s = mat.sum()
    sum_pk = float((p_k * p_k).sum())
    sum_tk = float((t_k * t_k).sum())
    sum_pk_tk = float((p_k * t_k).sum())
    cov_ytyp = float(c) * s - sum_pk_tk
    cov_ypyp = s * s - sum_pk
    cov_ytyt = s * s - sum_tk
    denom = np.sqrt(cov_ypyp * cov_ytyt) if cov_ypyp > 0 and cov_ytyt > 0 else 0.0
    if denom < 1e-12:
        return 0.0
    return cov_ytyp / denom


def f_test(
    X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None, alpha: float = 0.05
) -> FTestResult:
    """One-way ANOVA F-test per feature (scikit-learn ``f_classif`` equivalent).

    For each feature column, tests H0: "the feature has the same mean across all
    disease classes". A large F / small p means the feature discriminates between
    classes (informative). Implemented directly with ``scipy.stats.f_oneway`` so it
    has no sklearn dependency at runtime.

    Parameters
    ----------
    X
        2-D array of shape (n_samples, n_features).
    y
        1-D integer class labels of shape (n_samples,).
    feature_names
        Display names for each feature column.
    alpha
        Significance threshold; ``significant[i] = p_value[i] < alpha``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    n_feat = X.shape[1] if X.ndim == 2 else 1
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = feature_names or [f"feature_{i}" for i in range(n_feat)]
    f_stats: list[float] = []
    p_vals: list[float] = []
    classes = np.unique(y)
    for j in range(n_feat):
        groups = [X[y == c, j] for c in classes]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            # Constant / zero-variance features yield NaN -> treat as non-discriminative.
            f = float(f) if not np.isnan(f) else 0.0
            p = float(p) if not np.isnan(p) else 1.0
            f_stats.append(f)
            p_vals.append(p)
        else:
            f_stats.append(0.0)
            p_vals.append(1.0)
    significant = [p < alpha for p in p_vals]
    return FTestResult(
        feature_names=names, f_statistic=f_stats, p_value=p_vals, significant=significant,
    )


def chi_square_independence(
    labels: np.ndarray, feature: np.ndarray, name: str, alpha: float = 0.05
) -> ChiSquareResult:
    """Chi-square test of independence between predicted labels and a feature.

    Builds a contingency table (label x feature-category) and tests H0:
    "the prediction is independent of this feature". A small p-value rejects
    independence -- the prediction genuinely depends on the feature.

    Parameters
    ----------
    labels
        1-D array of predicted (or true) class ids.
    feature
        1-D array of categorical feature values (strings or ints).
    name
        Display name for the feature under test.
    alpha
        Significance threshold.
    """
    labels = np.asarray(labels).ravel()
    feature = np.asarray(feature).ravel()
    label_cats = np.unique(labels)
    feat_cats = np.unique(feature)
    if len(label_cats) < 2 or len(feat_cats) < 2:
        return ChiSquareResult(name=name, chi2=0.0, dof=0, p_value=1.0, significant=False)
    table = np.zeros((len(label_cats), len(feat_cats)), dtype=float)
    l_idx = {c: i for i, c in enumerate(label_cats)}
    f_idx = {c: i for i, c in enumerate(feat_cats)}
    for lv, fv in zip(labels, feature, strict=False):
        table[l_idx[lv], f_idx[fv]] += 1
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return ChiSquareResult(
        name=name, chi2=float(chi2), dof=int(dof), p_value=float(p),
        significant=bool(p < alpha), contingency=table.tolist(),
    )


def chi_square_goodness_of_fit(
    observed: np.ndarray, n_classes: int, name: str = "predicted-class distribution", alpha: float = 0.05
) -> ChiSquareResult:
    """Chi-square goodness-of-fit of observed class counts vs a uniform prior.

    Tests whether the predicted-class distribution is uniform (H0). A small
    p-value means the classifier predicts some diseases more often than others
    (i.e. it is not just guessing uniformly).
    """
    observed = np.asarray(observed, dtype=float).ravel()
    expected = np.full(n_classes, observed.sum() / n_classes)
    # Keep only classes that exist in the expected vector.
    k = min(len(observed), n_classes)
    chi2, p = stats.chisquare(observed[:k], expected[:k])
    dof = k - 1
    return ChiSquareResult(
        name=name, chi2=float(chi2), dof=int(dof), p_value=float(p),
        significant=bool(p < alpha),
    )


# ---------------------------------------------------------------------------
# Probabilistic / threshold-free metrics (ROC, PR, calibration, top-K, log loss).
# ---------------------------------------------------------------------------


def roc_curves(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str] | None = None
) -> ROCResult:
    """One-vs-rest ROC curves + AUC for every class.

    Uses scikit-learn's numerically stable implementation. Returns per-class
    FPR/TPR/threshold arrays plus macro- and weighted-average AUC.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1 - 1e-12)
    n_classes = y_proba.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]
    fpr_list: list[list[float]] = []
    tpr_list: list[list[float]] = []
    thr_list: list[list[float]] = []
    auc_list: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, y_proba[:, c])
        fpr_list.append(fpr.tolist())
        tpr_list.append(tpr.tolist())
        thr_list.append(thr.tolist())
        try:
            auc_list.append(float(roc_auc_score(y_bin, y_proba[:, c])))
        except ValueError:
            auc_list.append(0.5)
    # Macro / weighted AUC over all classes via softmax probs.
    try:
        macro_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float(np.mean(auc_list)) if auc_list else 0.5
    try:
        weighted_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except ValueError:
        weighted_auc = macro_auc
    return ROCResult(
        class_names=names, fpr=fpr_list, tpr=tpr_list, thresholds=thr_list,
        auc=auc_list, macro_auc=macro_auc, weighted_auc=weighted_auc,
    )


def pr_curves(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str] | None = None
) -> PRResult:
    """One-vs-rest precision-recall curves + average precision per class."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1 - 1e-12)
    n_classes = y_proba.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]
    prec_list: list[list[float]] = []
    rec_list: list[list[float]] = []
    thr_list: list[list[float]] = []
    ap_list: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        prec, rec, thr = precision_recall_curve(y_bin, y_proba[:, c])
        prec_list.append(prec.tolist())
        rec_list.append(rec.tolist())
        thr_list.append(thr.tolist())
        try:
            ap_list.append(float(average_precision_score(y_bin, y_proba[:, c])))
        except ValueError:
            ap_list.append(0.0)
    macro_ap = float(np.mean(ap_list)) if ap_list else 0.0
    try:
        weighted_ap = float(average_precision_score(y_true, y_proba, average="weighted"))
    except ValueError:
        weighted_ap = macro_ap
    return PRResult(
        class_names=names, precision=prec_list, recall=rec_list, thresholds=thr_list,
        average_precision=ap_list, macro_ap=macro_ap, weighted_ap=weighted_ap,
    )


def calibration_curve(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> CalibrationResult:
    """Reliability diagram + Expected Calibration Error (ECE).

    Bins predictions by max-softmax confidence and compares the mean
    confidence in each bin to the empirical accuracy in that bin. ECE is the
    bin-weighted absolute gap — a single-number calibration score.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1.0)
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    bin_acc: list[float] = []
    bin_conf: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    max_gap = 0.0
    n = len(confidences) or 1
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        gap = abs(acc - conf)
        ece += (cnt / n) * gap
        max_gap = max(max_gap, gap)
        bin_centers.append(float((lo + hi) / 2))
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_counts.append(cnt)
    return CalibrationResult(
        bin_centers=bin_centers, bin_accuracies=bin_acc, bin_confidences=bin_conf,
        bin_counts=bin_counts, ece=float(ece), max_calibration_error=float(max_gap),
    )


def top_k_accuracy(
    y_true: np.ndarray, y_proba: np.ndarray, k_values: tuple[int, ...] = (1, 2, 3, 5)
) -> TopKResult:
    """Top-K accuracy for each K — how often the true class is in the top-K predictions.

    Vital for differential diagnosis: with 21 diseases, top-3 accuracy is often
    the clinically meaningful metric (the shortlist a radiologist would consider).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1.0)
    k_sorted = sorted(k_values)
    max_k = k_sorted[-1]
    # Indices of top-max_k predictions per sample (descending probability).
    topk_idx = np.argsort(-y_proba, axis=1)[:, :max_k]
    accs: list[float] = []
    for k in k_sorted:
        correct = np.any(topk_idx[:, :k] == y_true.reshape(-1, 1), axis=1)
        accs.append(float(correct.mean()) if len(y_true) else 0.0)
    return TopKResult(k_values=list(k_sorted), accuracies=accs)


def multiclass_log_loss(
    y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15
) -> float:
    """Cross-entropy / log loss for a multi-class probabilistic prediction."""
    from sklearn.metrics import log_loss

    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    try:
        return float(log_loss(y_true, y_proba, normalize=True))
    except ValueError:
        return float("inf")


def multiclass_brier_score(
    y_true: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> float:
    """Multiclass Brier score (lower = better calibrated predictions).

    Mean squared difference between the predicted probability vector and the
    one-hot true label, summed over classes and averaged over samples.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 0.0, 1.0)
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


# Human-readable names for the MLP feature vector (matches _encode_features).
# Built from the live taxonomy vocabularies so it stays in sync with
# ``_FEATURE_DIM = 2 + len(PATTERNS) + len(LATERALITIES) + len(REGIONS)``.
def _build_feature_names() -> list[str]:
    from brainframe.classification.diseases import LATERALITIES, PATTERNS, REGIONS

    return [
        "log(volume)",
        "n_regions",
        *(f"pattern:{p}" for p in PATTERNS),
        *(f"laterality:{lat}" for lat in LATERALITIES),
        *(f"region:{r}" for r in REGIONS),
    ]


_FEATURE_NAMES: list[str] = _build_feature_names()


def evaluate_classifier(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    categorical_features: dict[str, np.ndarray] | None = None,
    alpha: float = 0.05,
) -> StatsReport:
    """Run all statistical evaluations on a classifier's predictions.

    Parameters
    ----------
    X
        Feature matrix (n_samples, n_features) used for the F-test.
    y_true
        Ground-truth class labels.
    y_pred
        Predicted class labels.
    y_proba
        Optional (n_samples, n_classes) probability matrix. When provided,
        the full probabilistic evaluation is computed: ROC curves + AUC,
        precision-recall curves + average precision, calibration / ECE,
        top-K accuracy, log loss, and Brier score. When ``None`` those fields
        are left unset (the F-test / confusion-matrix / chi-square still run).
    class_names
        Display names for the classes (confusion matrix).
    feature_names
        Display names for the feature columns (F-test). Defaults to the MLP
        feature names.
    categorical_features
        Mapping of ``feature_name -> 1-D categorical array`` for the chi-square
        independence tests. When ``None``, the region/laterality/pattern
        one-hot columns are decoded from ``X`` using the default feature layout.
    alpha
        Significance threshold for all tests.
    """
    from brainframe.classification.diseases import disease_names

    n_classes = int(max(y_true.max(), y_pred.max())) + 1 if len(y_true) else 1
    if class_names is None:
        names = disease_names()
        class_names = names[:n_classes] if len(names) >= n_classes else names + [f"class_{i}" for i in range(len(names), n_classes)]
    cm = confusion_matrix(y_true, y_pred, class_names)
    if feature_names is None:
        feature_names = _FEATURE_NAMES[: X.shape[1]] if X.ndim == 2 else _FEATURE_NAMES
    ft = f_test(X, y_true, feature_names, alpha=alpha)
    # Chi-square: independence of predicted class vs each categorical feature,
    # plus goodness-of-fit of the predicted-class distribution.
    chi_results: list[ChiSquareResult] = []
    if categorical_features is None:
        categorical_features = _decode_categorical(X)
    for fname, fvals in categorical_features.items():
        chi_results.append(chi_square_independence(y_pred, fvals, fname, alpha=alpha))
    pred_counts = np.bincount(np.asarray(y_pred).ravel().astype(int), minlength=n_classes)
    chi_results.append(chi_square_goodness_of_fit(pred_counts, n_classes, alpha=alpha))
    # Probabilistic metrics (only when a probability matrix is supplied).
    roc_res = pr_res = cal_res = topk_res = None
    ll = brier = None
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        roc_res = roc_curves(y_true, y_proba, class_names)
        pr_res = pr_curves(y_true, y_proba, class_names)
        cal_res = calibration_curve(y_true, y_proba)
        topk_res = top_k_accuracy(y_true, y_proba)
        ll = multiclass_log_loss(y_true, y_proba)
        brier = multiclass_brier_score(y_true, y_proba, n_classes)
    return StatsReport(
        confusion=cm, f_test=ft, chi_square=chi_results,
        n_samples=int(len(y_true)), n_classes=n_classes,
        roc=roc_res, pr=pr_res, calibration=cal_res, top_k=topk_res,
        log_loss=ll, brier_score=brier,
        _y_true=np.asarray(y_true), _y_proba=y_proba,
    )


def _decode_categorical(X: np.ndarray) -> dict[str, np.ndarray]:
    """Decode the pattern/laterality/region one-hot blocks of the MLP feature
    vector back into categorical labels for the chi-square test.

    The layout matches :func:`brainframe.classification.trained_model._encode_features`:
    ``[log(vol), n_regions, pattern*5, laterality*4, region*12]``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = X.shape[1]
    out: dict[str, np.ndarray] = {}
    from brainframe.classification.diseases import LATERALITIES, PATTERNS, REGIONS

    p_names = PATTERNS
    if n >= 2 + len(p_names):
        block = X[:, 2:2 + len(p_names)]
        out["pattern"] = np.array([
            p_names[int(np.argmax(row))] if row.sum() > 0 else "unknown" for row in block
        ])
    # laterality block: columns 2+len(PATTERNS) .. +len(LATERALITIES)
    l_names = LATERALITIES
    l_off = 2 + len(p_names)
    if n >= l_off + len(l_names):
        block = X[:, l_off:l_off + len(l_names)]
        out["laterality"] = np.array([
            l_names[int(np.argmax(row))] if row.sum() > 0 else "unknown" for row in block
        ])
    # region block: columns l_off+len(LATERALITIES) .. end
    r_names = REGIONS
    r_off = l_off + len(l_names)
    r_end = min(n, r_off + len(r_names))
    if r_end > r_off:
        block = X[:, r_off:r_end]
        out["dominant_region"] = np.array([
            r_names[int(np.argmax(row))] if row.sum() > 0 else "unknown" for row in block
        ])
    return out


def evaluate_trained_mlp(n_per_class: int = 200, seed: int = 42, alpha: float = 0.05) -> StatsReport:
    """Convenience: generate the signature-derived eval set, run the trained MLP
    on it, and return the full :class:`StatsReport`.

    This is the entry point used by the Model Evaluation page and the CLI.
    """
    import torch

    from brainframe.classification.trained_model import (
        _DEFAULT_WEIGHTS,
        _FEATURE_DIM,
        DiseaseMLP,
        generate_training_samples,
    )

    X, y = generate_training_samples(n_per_class=n_per_class, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.85 * len(X))
    va_X, va_y = X[idx[cut:]], y[idx[cut:]]
    if not _DEFAULT_WEIGHTS.exists():
        from scripts.train_disease_classifier import train

        train(epochs=300, n_per_class=600, seed=42)
    model = DiseaseMLP(_FEATURE_DIM)
    model.load_state_dict(torch.load(_DEFAULT_WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(va_X))
        y_proba = torch.softmax(logits, dim=1).numpy()
    y_pred = y_proba.argmax(1)
    return evaluate_classifier(va_X, va_y, y_pred, y_proba=y_proba, alpha=alpha)
