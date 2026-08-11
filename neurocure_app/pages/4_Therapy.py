"""Page 4 -- recommended curing/therapy technique with rationale."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state

st.title("💊 Therapy Recommendation")
sess = state.require_session()

# Ensure evaluation (lesion analysis) is available for the recommender.
if sess.evaluation is None:
    if sess.label_volume is None:
        st.info("Segmentation is needed first. Visit the 3D Brain page.")
        st.page_link("pages/2_3D_Brain.py", label="→ Go to 3D Brain", icon="🧠")
        st.stop()
    if st.button("▶ Analyze lesions", type="primary"):
        state.run_step("Analyzing lesion regions", sess.evaluate)
        st.rerun()
    st.stop()

if sess.recommendation is None:
    state.run_step("Recommending therapy technique", sess.recommend)
    st.rerun()

rec = sess.recommendation
tech = rec.technique

# Show the disease this therapy targets + prediction confidence.
clf = sess.classification or {}
disease_name = clf.get("disease_name", f"class {rec.disease_class}")
conf = clf.get("confidence", 0.0)
st.markdown(
    f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:16px;margin:8px 0 16px">
      <div style="color:#8b949e;font-size:.8rem;text-transform:uppercase;
      letter-spacing:.04em">Targeting diagnosis</div>
      <div style="font-size:1.2rem;font-weight:700;color:#e6edf3;margin-top:2px">
        {disease_name}</div>
      <div style="color:#8b949e;font-size:.85rem;margin-top:4px">
        Prediction confidence {conf:.0%} · disease class {rec.disease_class}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;margin:16px 0">
      <div style="color:#3aa6e6;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em">
        Recommended technique</div>
      <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;margin-top:4px">{tech.name}</div>
      <div style="color:#8b949e;margin-top:8px">Mode: <b>{tech.mode}</b>
        &nbsp;·&nbsp; Dose: <b>{tech.dose}</b>
        &nbsp;·&nbsp; Radius: <b>{tech.radius_mm} mm</b>
        &nbsp;·&nbsp; Kernel: <b>{tech.kernel}</b> (σ={tech.sigma_mm} mm)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Rationale")
st.markdown(rec.rationale)

st.subheader("Expected effect")
st.info(tech.expected_effect)

if tech.references:
    st.subheader("References")
    for r in tech.references:
        st.markdown(f"- {r}")

st.subheader("Lesion summary")
lesion = sess.evaluation["lesion"].to_dict()
c1, c2 = st.columns(2)
c1.metric("Total lesion volume", f"{lesion.get('total_lesion_volume_mm3', 0):.0f} mm³")
c2.metric("Detected regions", lesion.get("n_regions", 0))

st.page_link("pages/5_Simulate.py", label="→ Next: Live Cure Simulation", icon="🎬")
