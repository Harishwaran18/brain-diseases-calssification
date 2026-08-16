"""Page 7 -- statistical model evaluation (comprehensive dashboard).

Runs the trained disease MLP on a held-out signature-derived evaluation set and
applies a full battery of statistical & probabilistic evaluation techniques:

* **Confusion matrix** + per-class precision/recall/F1/specificity + aggregate
  metrics (macro/weighted, Cohen's kappa, MCC, balanced accuracy).
* **F-test (one-way ANOVA)** -- which input features discriminate diseases.
* **Chi-square test** -- whether predictions depend on categorical features.
* **ROC curves** (one-vs-rest) + macro/weighted AUC.
* **Precision-Recall curves** + average precision.
* **Calibration** (reliability diagram + ECE) + confidence histogram.
* **Top-K accuracy** -- differential-diagnosis quality.

The page is organised into tabs so each family of metrics has its own view,
with a headline metric-tile grid at the top.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from neurocure_app.components import charts

st.set_page_config(page_title="Model Evaluation · NeuroCure", layout="wide")
st.title("📊 Statistical Model Evaluation")
st.caption(
    "A comprehensive, statistically rigorous characterisation of the trained "
    "21-disease classifier on a held-out signature-derived evaluation set: "
    "confusion matrix, F-test (ANOVA), chi-square, ROC, precision-recall, "
    "calibration, and top-K accuracy."
)

# ---- Controls ----
st.subheader("Evaluation set")
col_a, col_b, col_c = st.columns(3)
with col_a:
    n_per_class = st.slider(
        "Samples per disease class", 50, 400, 150, step=50,
        help="More samples give smoother statistics but take longer to evaluate.",
    )
with col_b:
    seed = st.number_input("Random seed", value=42, step=1)
with col_c:
    alpha = st.slider(
        "Significance level (α)", 0.01, 0.10, 0.05, step=0.01,
        help="Threshold below which a result is deemed statistically significant.",
    )

if st.button("▶ Run statistical evaluation", type="primary"):
    with st.spinner("Generating evaluation set, running the trained MLP, and computing all metrics…"):
        from brainframe.classification.stats import evaluate_trained_mlp

        report = evaluate_trained_mlp(n_per_class=int(n_per_class), seed=int(seed), alpha=float(alpha))
    st.session_state["stats_report"] = report.to_dict()
    st.session_state["stats_alpha"] = float(alpha)
    st.session_state["stats_y_true"] = report._y_true.tolist() if report._y_true is not None else []
    st.session_state["stats_y_proba"] = report._y_proba.tolist() if report._y_proba is not None else None
    st.rerun()

report = st.session_state.get("stats_report")
alpha = st.session_state.get("stats_alpha", 0.05)
y_true_raw = st.session_state.get("stats_y_true") or []
y_proba_raw = st.session_state.get("stats_y_proba")

if report is None:
    st.info("Press **Run statistical evaluation** to compute the full battery of statistics.")
    st.stop()

cm = report["confusion"]
ft = report["f_test"]
chi = report["chi_square"]
roc = report.get("roc")
pr = report.get("pr")
cal = report.get("calibration")
topk = report.get("top_k")
has_proba = roc is not None

# ---- Headline metric tiles ----
tiles: list[dict[str, object]] = [
    {"label": "Accuracy", "value": f"{cm['accuracy']:.1%}", "color": "#3fb950",
     "hint": f"{report['n_samples']} samples · {report['n_classes']} classes"},
    {"label": "Macro F1", "value": f"{cm['macro_f1']:.3f}", "color": "#58a6ff",
     "hint": f"weighted {cm['weighted_f1']:.3f}"},
    {"label": "Balanced acc", "value": f"{cm['balanced_accuracy']:.3f}", "color": "#bc8cff",
     "hint": "mean recall"},
    {"label": "Cohen's κ", "value": f"{cm['cohen_kappa']:.3f}", "color": "#d29922",
     "hint": "chance-corrected agreement"},
]
if has_proba:
    tiles += [
        {"label": "ROC-AUC", "value": f"{roc['macro_auc']:.3f}", "color": "#3fb950",
         "hint": f"weighted {roc['weighted_auc']:.3f}"},
        {"label": "Top-3 acc", "value": f"{topk['accuracies'][2]:.1%}", "color": "#56d364",
         "hint": "differential diagnosis"},
        {"label": "ECE", "value": f"{cal['ece']:.3f}", "color": "#f85149",
         "hint": "calibration error"},
        {"label": "Log loss", "value": f"{report['log_loss']:.3f}", "color": "#ff7b72",
         "hint": "cross-entropy"},
    ]
charts.metric_tiles(tiles)

# ---- Tabs ----
tab_overview, tab_conf, tab_roc, tab_cal, tab_feat = st.tabs([
    "📑 Overview", "🧩 Confusion Matrix", "📈 ROC & PR", "🎯 Calibration", "🔬 Feature Analysis",
])


def _confusion_details_table(cm: dict) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(cm["class_names"]):
        rows.append({
            "Disease": name, "Precision": cm["precision"][i], "Recall": cm["recall"][i],
            "F1": cm["f1"][i], "Specificity": cm["specificity"][i], "Support": cm["support"][i],
        })
    return pd.DataFrame(rows)


# ---- Overview tab ----
with tab_overview:
    st.subheader("Metric summary")
    summary = {
        "Accuracy": cm["accuracy"], "Balanced accuracy": cm["balanced_accuracy"],
        "Macro precision": cm["macro_precision"], "Macro recall": cm["macro_recall"],
        "Macro F1": cm["macro_f1"], "Macro specificity": cm["macro_specificity"],
        "Weighted precision": cm["weighted_precision"], "Weighted recall": cm["weighted_recall"],
        "Weighted F1": cm["weighted_f1"],
        "Cohen's Kappa": cm["cohen_kappa"], "MCC": cm["mcc"],
    }
    if has_proba:
        summary.update({
            "ROC-AUC (macro)": roc["macro_auc"], "ROC-AUC (weighted)": roc["weighted_auc"],
            "PR-AP (macro)": pr["macro_ap"], "PR-AP (weighted)": pr["weighted_ap"],
            "Log loss": report["log_loss"], "Brier score": report["brier_score"],
            "ECE": cal["ece"],
        })
    s_df = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    s_df["Value"] = s_df["Value"].map(lambda v: f"{v:.4f}")
    st.dataframe(s_df, use_container_width=True, hide_index=True)

    if has_proba:
        st.markdown("#### Top-K accuracy (differential-diagnosis quality)")
        st.caption(
            "With 21 diseases, top-3 accuracy is the clinically meaningful metric: "
            "the shortlist a radiologist would consider before ordering follow-up tests."
        )
        charts.render(charts.top_k_chart(topk))

    st.markdown("#### Top confused disease pairs")
    st.caption("The worst off-diagonal confusions — actionable targets for improvement.")
    pairs = charts.top_confused_pairs(cm, top_n=15)
    if pairs:
        pdf = pd.DataFrame(pairs)
        pdf["Rate"] = pdf["Rate"].map(lambda r: f"{r:.0%}")
        st.dataframe(pdf, use_container_width=True, hide_index=True)
    else:
        st.success("No off-diagonal confusions — the model classified every sample correctly!")


# ---- Confusion Matrix tab ----
with tab_conf:
    norm = st.toggle("Row-normalize (recall view)", value=False,
                     help="Normalize each row to sum to 1, so cell values = per-class recall.")
    if norm:
        charts.render(charts.normalized_confusion_matrix_chart(cm))
    else:
        charts.render(charts.confusion_matrix_chart(cm))
    with st.expander("Per-class precision / recall / F1 / specificity", expanded=True):
        st.dataframe(_confusion_details_table(cm), use_container_width=True, hide_index=True)
    charts.render(charts.per_class_metrics_chart(cm))


# ---- ROC & PR tab ----
with tab_roc:
    if not has_proba:
        st.warning("Probabilistic metrics require the trained MLP's softmax outputs.")
        st.stop()
    st.subheader("ROC curves (one-vs-rest)")
    st.caption(
        "Each curve shows the trade-off between true-positive and false-positive "
        "rate for one disease vs all others. AUC = 1.0 is perfect; 0.5 is chance. "
        "The worst classes (lowest AUC) are plotted first."
    )
    charts.render(charts.roc_curves_chart(roc))
    with st.expander("Per-class AUC table"):
        roc_df = pd.DataFrame({"Disease": roc["class_names"], "AUC": roc["auc"]})
        roc_df["AUC"] = roc_df["AUC"].map(lambda v: f"{v:.4f}")
        roc_df = roc_df.sort_values("AUC", ascending=False).reset_index(drop=True)
        st.dataframe(roc_df, use_container_width=True, hide_index=True)

    st.subheader("Precision-Recall curves (one-vs-rest)")
    st.caption(
        "More informative than ROC for imbalanced classes: the area under each "
        "curve is the average precision (AP) for that disease."
    )
    charts.render(charts.pr_curves_chart(pr))
    with st.expander("Per-class average-precision table"):
        pr_df = pd.DataFrame({"Disease": pr["class_names"], "AP": pr["average_precision"]})
        pr_df["AP"] = pr_df["AP"].map(lambda v: f"{v:.4f}")
        pr_df = pr_df.sort_values("AP", ascending=False).reset_index(drop=True)
        st.dataframe(pr_df, use_container_width=True, hide_index=True)


# ---- Calibration tab ----
with tab_cal:
    if not has_proba:
        st.warning("Calibration metrics require the trained MLP's softmax outputs.")
        st.stop()
    st.subheader("Reliability diagram")
    st.caption(
        "Compares the model's predicted confidence to its empirical accuracy. "
        "Points on the dashed diagonal are perfectly calibrated. A large gap "
        "(high ECE) means the model is over/under-confident."
    )
    charts.render(charts.calibration_chart(cal))
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Expected Calibration Error (ECE)", f"{cal['ece']:.4f}")
    with c2:
        st.metric("Max calibration error", f"{cal['max_calibration_error']:.4f}")
    st.subheader("Confidence distribution")
    st.caption(
        "Green = correct predictions, red = incorrect. A well-behaved model "
        "concentrates incorrect predictions at low confidence and correct ones "
        "at high confidence (good separation)."
    )
    if y_true_raw and y_proba_raw is not None:
        charts.render(charts.confidence_histogram_chart(y_true_raw, y_proba_raw))
    else:
        st.info("Confidence histogram requires the raw probability matrix (re-run the evaluation).")


# ---- Feature Analysis tab ----
with tab_feat:
    st.subheader("F-test (one-way ANOVA, feature significance)")
    st.caption(
        "For every input feature, tests whether its mean differs significantly "
        "across disease classes (H₀: equal means). A large F / small p (green) "
        "means the feature discriminates between diseases — it is informative."
    )
    charts.render(charts.f_test_chart(ft))
    n_sig = sum(ft["significant"])
    best_idx = ft["f_statistic"].index(max(ft["f_statistic"]))
    st.markdown(
        f"**{n_sig} / {len(ft['feature_names'])}** features are statistically "
        f"significant at α = {alpha}. The most discriminative feature is "
        f"**{ft['feature_names'][best_idx]}** (F = {max(ft['f_statistic']):.1f})."
    )

    st.subheader("Chi-square test of independence")
    st.caption(
        "Tests whether the predicted disease label is statistically independent "
        "of each categorical lesion feature. A significant result (red) rejects "
        "independence — the prediction genuinely depends on that feature. The "
        "final bar tests whether the predicted-class distribution differs from uniform."
    )
    charts.render(charts.chi_square_chart(chi))
    with st.expander("Chi-square details"):
        rows = [
            {
                "Test": c["name"], "χ²": round(c["chi2"], 2), "dof": c["dof"],
                "p-value": c["p_value"], "Significant": "yes" if c["significant"] else "no",
            }
            for c in chi
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
        ### Interpretation guide
        - **Confusion matrix**: bright off-diagonal cells are systematic
          misclassifications (often clinically related diseases sharing a lesion signature).
        - **F-test**: high-F features drive the classifier; non-significant features
          could be dropped (basis for feature selection).
        - **Chi-square**: significant dependence on region/laterality/pattern confirms
          the classifier uses the anatomical evidence it should; a *non*-significant
          goodness-of-fit means it does not over-predict any single disease.
        - **ROC / PR**: near-1.0 AUC/AP means confident, separable predictions.
        - **Calibration**: low ECE means the reported confidence is trustworthy.
        """
    )

# ---- Download report ----
st.divider()
st.download_button(
    "📥 Download full report (JSON)",
    data=json.dumps(report, indent=2, default=str),
    file_name="neurocure_model_evaluation.json",
    mime="application/json",
    help="Export every metric, curve, and table as a JSON file.",
)
