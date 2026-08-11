"""Page 2 -- show the REAL reconstructed 3D brain in an interactive viewer."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state
from neurocure_app.components.three_viewer import render_three_brain
from neurocure_app.components.viewer3d import render_axial_slice

st.title("🧠 3D Brain Reconstruction")
sess = state.require_session()

# Run segmentation + reconstruction if not yet done.
if sess.label_volume is None:
    if st.button("▶ Run segmentation + reconstruction", type="primary"):
        state.run_step("Segmenting (retraining-free SAM)", sess.segment)
        state.run_step("Reconstructing 3D meshes (Marching Cubes)", sess.reconstruct)
        st.rerun()
    st.stop()

if sess.reconstruction is None:
    state.run_step("Reconstructing 3D meshes (Marching Cubes)", sess.reconstruct)
    st.rerun()

recon = sess.reconstruction
metrics = recon["metrics"]

st.success(
    f"3D brain reconstructed: {len(recon['meshes'].meshes)} tissue structures, "
    f"total volume {metrics['total_volume_mm3']:.0f} mm³."
)

st.subheader("Interactive 3D viewer (WebGL)")
st.caption(
    "Genuine WebGL rendering (Three.js) of the real fsaverage cortex with PBR "
    "shading — the gold standard for in-browser 3D brain visualisation. Drag to "
    "rotate, scroll to zoom, right-drag to pan. The red lesion mesh highlights "
    "pathology."
)

sim = sess.simulation_override or (
    sess.evaluation["simulation"].to_dict() if sess.evaluation else None
)

# Real fsaverage cortex backdrop (genuine folded human brain surface).
cortex = sess.load_real_cortex()
cortex_mesh = cortex.meshes[0] if cortex and cortex.meshes else None

# Segmented tissue meshes + lesion mesh for the WebGL scene.
tissue_meshes = list(recon["meshes"].meshes)
lesion_mesh = next((m for m in tissue_meshes if m.label == "lesion"), None)
before_v = float(sim.get("before_lesion_volume_mm3", 0.0)) if sim else 0.0
after_v = float(sim.get("after_lesion_volume_mm3", 0.0)) if sim else 0.0

render_three_brain(
    cortex_mesh=cortex_mesh,
    tissue_meshes=tissue_meshes,
    lesion_mesh=lesion_mesh,
    disease_name=sess.classification.get("disease_name") if sess.classification else None,
    technique_name=(sess.recommendation.technique.name if sess.recommendation else None),
    before_volume=before_v,
    after_volume=after_v,
)

st.divider()
st.subheader("Tissue metrics")
per_label = metrics.get("per_label", {})
cols = st.columns(min(4, len(per_label)))
for col, (name, m) in zip(cols, per_label.items(), strict=False):
    col.metric(name, f"{m.get('volume_mm3', 0):.0f} mm³", f"{m.get('n_faces', 0)} faces")

st.subheader("Axial slice")
render_axial_slice(sess.volume, title="Raw MRI axial slice")

st.page_link("pages/3_Predict.py", label="→ Next: Disease Prediction", icon="🔬")
