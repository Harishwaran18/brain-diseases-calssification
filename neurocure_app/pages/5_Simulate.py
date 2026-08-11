"""Page 5 -- LIVE 3D animation of the cure acting on the real brain."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state
from neurocure_app.components import charts
from neurocure_app.components.three_viewer import render_three_brain

st.title("🎬 Live Cure Simulation")
sess = state.require_session()

if sess.recommendation is None:
    st.info("A therapy recommendation is needed first. Visit the Therapy page.")
    st.page_link("pages/4_Therapy.py", label="→ Go to Therapy", icon="💊")
    st.stop()

if sess.simulation_override is None:
    if st.button("▶ Run the cure simulation", type="primary"):
        state.run_step(
            f"Simulating '{sess.recommendation.technique.name}'",
            sess.simulate,
        )
        st.rerun()
    st.stop()

sim = sess.simulation_override
before = sim["before_lesion_volume_mm3"]
after = sim["after_lesion_volume_mm3"]
recovery = (before - after) / before if before else 0.0
comp = sess.evaluation["compatibility"].to_dict()
risk = comp.get("risk", 0.5)

disease_name = sess.classification.get("disease_name") if sess.classification else None
technique_name = sess.recommendation.technique.name

st.markdown(
    f"""
    <div style="background:#161b22;border-radius:12px;padding:20px;margin:16px 0">
      <div style="color:#3aa6e6;font-size:.8rem;text-transform:uppercase">Treating disease</div>
      <div style="font-size:1.2rem;font-weight:700;color:#e6edf3">{disease_name or "—"}</div>
      <div style="color:#3fb950;font-size:.8rem;text-transform:uppercase;margin-top:10px">Curing technique</div>
      <div style="font-size:1.2rem;font-weight:700;color:#e6edf3">{technique_name}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Before", f"{before:.0f} mm³")
c2.metric("After", f"{after:.0f} mm³", f"-{(before - after):.0f} mm³")
c3.metric("Recovery", f"{recovery:.1%}")

st.subheader("Verdict")
charts.verdict_banner(recovery, risk)

st.subheader("LIVE 3D cure animation (WebGL)")
st.caption(
    "Genuine WebGL (Three.js) animation: press ▶ Play cure to watch the lesion "
    "mesh shrink toward its centroid as the therapy takes effect. The overlay "
    "names the disease being treated and the curing technique being applied."
)
recon = sess.reconstruction
if recon is not None:
    cortex = sess.load_real_cortex()
    cortex_mesh = cortex.meshes[0] if cortex and cortex.meshes else None
    tissue_meshes = list(recon["meshes"].meshes)
    lesion_mesh = next((m for m in tissue_meshes if m.label == "lesion"), None)
    render_three_brain(
        cortex_mesh=cortex_mesh,
        tissue_meshes=tissue_meshes,
        lesion_mesh=lesion_mesh,
        disease_name=disease_name,
        technique_name=technique_name,
        before_volume=float(before),
        after_volume=float(after),
    )
else:
    st.warning("Reconstruction not available; visit the 3D Brain page first.")

st.subheader("Lesion volume over time")
charts.render(charts.cure_timeline_chart(before, after))

st.subheader("Quantitative outcome")
oc1, oc2, oc3 = st.columns(3)
oc1.metric("Affected voxels", f"{sim.get('affected_voxels', 0):,}")
oc2.metric("Affected fraction", f"{sim.get('affected_fraction', 0):.1%}")
oc3.metric("Compatibility score", f"{comp.get('score', 0):.3f}")

st.page_link("pages/6_Report.py", label="→ Next: Download Report", icon="📄")
