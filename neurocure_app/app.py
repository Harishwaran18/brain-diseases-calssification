"""NeuroCure -- interactive web platform entry point.

Run with::

    streamlit run neurocure_app/app.py

Implements the end-to-end user workflow:
  upload MRI -> see the REAL 3D brain -> disease prediction ->
  recommended therapy -> LIVE 3D cure simulation -> downloadable report.

The heavy AI work is done by the ``brainframe`` engine
(:class:`~brainframe.session.Session`); this file is the thin presentation shell.
"""

from __future__ import annotations

import streamlit as st

from neurocure_app import state

st.set_page_config(
    page_title="NeuroCure — 3D Brain Reconstruction & Therapy Evaluation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0d1117; }
    .stApp, .stApp p, .stApp span, .stApp li { color: #e6edf3; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e6edf3; }
    [data-testid="stSidebar"] { background: #161b22; }
    .stButton > button, .stDownloadButton > button {
        background: #238636; color: #fff; border: none; border-radius: 6px;
    }
    .stButton > button:hover { background: #2ea043; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("🧠 NeuroCure")
    st.markdown(
        "### Retraining-Free AI Framework for 3D Neurodegenerative Brain "
        "Reconstruction & Computational Evaluation for Neuroregenerative Therapy"
    )
    st.divider()

    sess = state.get_session()
    if sess is None:
        st.info(
            "👋 Welcome to **NeuroCure**. Upload a brain MRI (NIfTI/DICOM) or load "
            "the bundled demo brain to begin."
        )
        st.page_link("pages/1_Upload.py", label="📤 Start → Upload a brain scan", icon="📤")
    else:
        st.success("✅ A scan is loaded. Use the sidebar to navigate the workflow.")
        summary = sess.summary()
        st.subheader("Session overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume shape", "×".join(str(d) for d in (summary.get("volume_shape") or [])))
        c2.metric("Stages done", len(summary.get("stages", [])))
        c3.metric(
            "Prediction",
            f"Class {summary['prediction']['prediction']}" if summary.get("prediction") else "—",
        )
        c4.metric(
            "Lesion volume",
            f"{summary.get('lesion_volume_mm3', 0):.0f} mm³"
            if summary.get("lesion_volume_mm3")
            else "—",
        )
        if summary.get("compatibility_score") is not None:
            st.metric("Therapy compatibility score", f"{summary['compatibility_score']:.3f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Workflow**")
    st.sidebar.page_link("pages/1_Upload.py", label="1. Upload / Demo brain", icon="📤")
    st.sidebar.page_link("pages/2_3D_Brain.py", label="2. 3D Brain", icon="🧠")
    st.sidebar.page_link("pages/3_Predict.py", label="3. Disease Prediction", icon="🔬")
    st.sidebar.page_link("pages/4_Therapy.py", label="4. Therapy Recommendation", icon="💊")
    st.sidebar.page_link("pages/5_Simulate.py", label="5. Live Cure Simulation", icon="🎬")
    st.sidebar.page_link("pages/6_Report.py", label="6. Download Report", icon="📄")


main()
