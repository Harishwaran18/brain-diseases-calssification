"""Page 1 -- upload a brain MRI (NIfTI/DICOM) or load the demo brain."""

from __future__ import annotations

import streamlit as st

from neurocure_app import state
from neurocure_app.components.viewer3d import render_axial_slice

st.title("📤 Upload Brain MRI")
st.markdown("Load a brain MRI volume to begin the analysis. NIfTI (`.nii`/`.nii.gz`) is preferred.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Option A: Upload a file")
    uploaded = st.file_uploader(
        "Choose a NIfTI/DICOM file",
        type=["nii", "nii.gz", "dcm"],
        help="Upload a brain MRI. NIfTI volumes work best; DICOM series can be a single file.",
    )
with col2:
    st.subheader("Option B: Use the demo brain")
    st.caption("No download required -- a realistic synthetic brain phantom is bundled.")
    if st.button("🧠 Load demo brain", use_container_width=True):
        sess = state.init_session()
        state.run_step("Loading demo brain", sess.load_demo_brain)
        st.success("Demo brain loaded! Go to **3D Brain** to see it reconstructed.")
        st.switch_page("pages/2_3D_Brain.py")

if uploaded is not None:
    tmp = state.get_work_dir() / f"upload_{uploaded.name}"
    tmp.write_bytes(uploaded.read())
    sess = state.init_session()
    try:
        state.run_step(f"Loading {uploaded.name}", sess.ingest, tmp)
        st.success(f"Loaded **{uploaded.name}** (shape {sess.volume.shape}).")
        st.switch_page("pages/2_3D_Brain.py")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load file: {e}")

st.divider()

# Show a preview of whatever is currently loaded.
sess = state.get_session()
if sess is not None and sess.volume is not None:
    st.subheader("Preview")
    render_axial_slice(sess.volume, title=f"Axial slice (shape {sess.volume.shape})")
else:
    st.info("No scan loaded yet. Pick an option above.")
