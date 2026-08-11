"""Page 5 -- LIVE 3D animation of the cure acting on the real brain."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state
from neurocure_app.components import charts
from neurocure_app.components.viewer3d import render_brain_3d

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

st.markdown(
    f"""
    <div style="background:#161b22;border-radius:12px;padding:20px;margin:16px 0">
      <div style="color:#3aa6e6;font-size:.8rem;text-transform:uppercase">Recommended therapy</div>
      <div style="font-size:1.2rem;font-weight:700;color:#e6edf3">{sess.recommendation.technique.name}</div>
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

st.subheader("LIVE 3D cure animation")
st.caption(
    "Press ▶ Play to watch the lesion mesh shrink toward its centroid as the therapy "
    "takes effect over time. The colored volume reflects the lesion reversing."
)
recon = sess.reconstruction
if recon is not None:
    lesion_regions = sess.evaluation["lesion"].to_dict().get("regions", [])
    cortex = sess.load_real_cortex()
    cortex_mesh = cortex.meshes[0] if cortex and cortex.meshes else None
    render_brain_3d(
        recon["meshes"],
        label_volume=recon["label_volume"],
        spacing=recon["spacing"],
        lesion_regions=lesion_regions,
        simulation=sim,
        cortex_mesh=cortex_mesh,
        height=620,
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
