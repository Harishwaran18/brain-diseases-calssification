"""Page 3 -- disease prediction with confidence and explanation."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state
from neurocure_app.components import charts

st.title("🔬 Disease Prediction")
sess = state.require_session()

if sess.reconstruction is None:
    st.info("Reconstruction is needed before prediction. Visit the 3D Brain page first.")
    st.page_link("pages/2_3D_Brain.py", label="→ Go to 3D Brain", icon="🧠")
    st.stop()

if sess.classification is None:
    if st.button("▶ Run disease classification", type="primary"):
        state.run_step("Classifying disease (evidence engine)", sess.predict)
        st.rerun()
    st.stop()

clf = sess.classification
pred = clf["prediction"]
conf = clf.get("confidence", 0.0)
disease_name = clf.get("disease_name", f"Class {pred}")
short = clf.get("disease_short_name", "")
features = clf.get("features", {})
evidence = clf.get("evidence", {})
scores = evidence.get("scores", [])
differential = clf.get("differential", [])

# Confidence-driven colour: green when high, amber when moderate, red when low.
if conf >= 0.90:
    color, conf_label = "#3fb950", "HIGH confidence"
elif conf >= 0.70:
    color, conf_label = "#d29922", "MODERATE confidence"
else:
    color, conf_label = "#f85149", "LOW confidence (equivocal)"

st.markdown(
    f"""
    <div style="background:#161b22;border:2px solid {color};border-radius:12px;
    padding:24px;margin:16px 0">
      <div style="font-size:.8rem;color:#8b949e;text-transform:uppercase;
      letter-spacing:.04em">Predicted diagnosis</div>
      <div style="font-size:1.6rem;font-weight:700;color:{color};margin-top:4px">
        {disease_name}</div>
      <div style="color:#8b949e;margin-top:10px">
        <span style="background:{color};color:#fff;padding:4px 12px;border-radius:12px;
        font-size:.8rem;font-weight:700">{conf:.0%} · {conf_label}</span>
        &nbsp;·&nbsp; Disease class <b>{pred}</b> ({short})
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Evidence summary (transparent, human-readable).
st.subheader("How the prediction was reached")
st.info(clf.get("evidence_summary", ""))

# Feature summary.
feat_cols = st.columns(4)
feat_cols[0].metric("Lesion pattern", features.get("pattern", "—").replace("_", " "))
feat_cols[1].metric("Laterality", features.get("laterality", "—"))
feat_cols[2].metric("Dominant region", features.get("dominant_region", "—").replace("_", " "))
feat_cols[3].metric("Total volume", f"{features.get('total_volume_mm3', 0):.0f} mm³")

# Per-axis evidence breakdown (transparency).
st.subheader("Evidence breakdown (per disease, top 5)")
st.caption(
    "Each disease is scored on four independent axes — region, pattern, "
    "laterality, and size. Confidence is high only when all axes agree on the "
    "same disease. This is a transparent, auditable differential engine."
)
if scores:
    charts.render(charts.evidence_chart(scores))

# Top-3 differential diagnosis.
st.subheader("Differential diagnosis (top 3)")
if differential:
    charts.render(charts.differential_chart(differential))
    for d in differential:
        st.markdown(
            f"- **{d['name']}** — relative probability {d['probability']:.0%} "
            f"(agreement score {d['score']:.2f})"
        )

st.subheader("Per-class probabilities (all 10 diseases)")
charts.render(charts.disease_chart(clf.get("probabilities", []), pred))

st.markdown(
    """
    The prediction feeds directly into the **therapy recommendation** on the
    next page. Each disease maps to a curated, literature-referenced curing
    technique tailored to its lesion signature.
    """
)

st.page_link("pages/4_Therapy.py", label="→ Next: Therapy Recommendation", icon="💊")
