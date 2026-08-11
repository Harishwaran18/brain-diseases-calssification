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
        state.run_step("Classifying disease (3D CNN)", sess.predict)
        st.rerun()
    st.stop()

clf = sess.classification
pred = clf["prediction"]
probs = clf.get("probabilities", [])
labels = {
    0: ("Healthy / No neurodegeneration", "#3fb950", "Low"),
    1: ("Early-stage neurodegenerative changes", "#d29922", "Moderate"),
    2: ("Moderate neurodegeneration (Alzheimer's / Parkinson's spectrum)", "#db6d28", "High"),
    3: ("Advanced neurodegeneration (severe atrophy / lesion burden)", "#f85149", "Critical"),
}
label, color, risk = labels.get(pred, (f"Class {pred}", "#3aa6e6", "Unknown"))

st.markdown(
    f"""
    <div style="background:#161b22;border:2px solid {color};border-radius:12px;
    padding:24px;margin:16px 0">
      <div style="font-size:1.5rem;font-weight:700;color:{color}">{label}</div>
      <div style="color:#8b949e;margin-top:8px">
        <span style="background:{color};color:#fff;padding:3px 10px;border-radius:12px;
        font-size:.75rem;font-weight:600">{risk} risk</span>
        &nbsp;·&nbsp; Predicted class <b>{pred}</b>
        &nbsp;·&nbsp; Confidence <b>{max(probs):.1%}</b>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Per-class probabilities")
charts.render(charts.disease_chart(probs, pred))

st.subheader("How the prediction works")
st.markdown(
    """
    The classifier is a lightweight 3D convolutional network that operates on the
    normalized brain volume. It outputs a probability over four disease stages
    (healthy → early → moderate → advanced). The prediction feeds directly into
    the **therapy recommendation** on the next page.
    """
)

st.page_link("pages/4_Therapy.py", label="→ Next: Therapy Recommendation", icon="💊")
