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

# ---------------------------------------------------------------------------
# FIX: Make the project root available to Python.
# This allows imports such as:
#     from neurocure_app import state
# to work correctly when Streamlit runs neurocure_app/app.py.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    /* ---- Base theme: dark clinical dashboard ---- */
    .stApp { background: #0d1117; }
    .stApp, .stApp p, .stApp span, .stApp li { color: #e6edf3; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e6edf3; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

    /* ---- Buttons: primary green, hover lighter ---- */
    .stButton > button, .stDownloadButton > button {
        background: #238636; color: #fff; border: none; border-radius: 6px;
        font-weight: 600; transition: background .15s, transform .08s;
    }
    .stButton > button:hover { background: #2ea043; transform: translateY(-1px); }
    .stButton > button:active { transform: translateY(0); }

    /* ---- Metric tiles: subtle card background ---- */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { color: #8b949e; font-size: 12px !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3; }

    /* ---- Section headers with accent underline ---- */
    .nc-section-header {
        color: #58a6ff;
        border-bottom: 2px solid #1f6feb;
        padding-bottom: 6px;
        margin-top: 8px;
    }

    /* ---- Pipeline section cards ---- */
    .nc-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .nc-card h3 { margin-top: 0; }

    /* ---- Status badges ---- */
    .nc-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600;
    }
    .nc-badge-done { background: #1a3a2a; color: #3fb950; }
    .nc-badge-pending { background: #3a2a1a; color: #f0a040; }
    .nc-badge-empty { background: #2a2a2a; color: #8b949e; }

    /* ---- Progress bar ---- */
    .stProgress > div > div { background: #238636; }

    /* ---- Info/success/warning boxes ---- */
    [data-testid="stAlert"] { border-radius: 8px; }

    /* ---- Expander headers ---- */
    .streamlit-expanderHeader {
        background: #161b22;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _badge(status: str) -> str:
    """Return an HTML status badge for the pipeline overview."""
    styles = {
        "done": "nc-badge nc-badge-done",
        "pending": "nc-badge nc-badge-pending",
        "empty": "nc-badge nc-badge-empty",
    }
    labels = {"done": "✓ Done", "pending": "⏳ Pending", "empty": "○ Not started"}
    return f'<span class="{styles.get(status, styles["empty"])}">{labels[status]}</span>'


def main() -> None:
    st.title("🧠 NeuroCure")
    st.markdown(
        "### Retraining-Free AI Framework for 3D Neurodegenerative Brain "
        "Reconstruction & Computational Evaluation for Neuroregenerative Therapy"
    )
    st.markdown(
        '<p style="color:#8b949e; margin-top:-8px;">36-disease classifier · '
        '3-member ensemble MLP with self-attention · 43 anatomical features · '
        "real fsaverage cortex + Harvard-Oxford deep nuclei</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    sess = state.get_session()

    # ---- Two-column dashboard: Diagnostic Pipeline | Therapeutic Simulation ----
    diag_col, therapy_col = st.columns(2)

    with diag_col:
        st.markdown(
            '<h3 class="nc-section-header">🔬 Diagnostic Pipeline</h3>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nc-card">', unsafe_allow_html=True)

        if sess is None:
            st.markdown(
                f"📤 Upload {_badge('empty')}",
                unsafe_allow_html=True,
            )
            st.markdown("No scan loaded yet.")
            st.page_link(
                "pages/1_Upload.py",
                label="→ Upload a brain scan",
                icon="📤",
            )
        else:
            summary = sess.summary()
            vol_shape = "×".join(
                str(d) for d in (summary.get("volume_shape") or [])
            )

            st.markdown(
                f"📤 Upload {_badge('done')}",
                unsafe_allow_html=True,
            )
            st.caption(f"Volume: {vol_shape}")

            has_recon = (
                summary.get("stages")
                and "reconstruct" in summary["stages"]
            )

            st.markdown(
                f"🧠 3D Reconstruction "
                f"{_badge('done' if has_recon else 'pending')}",
                unsafe_allow_html=True,
            )

            if has_recon:
                st.page_link(
                    "pages/2_3D_Brain.py",
                    label="→ View 3D brain",
                    icon="🧠",
                )

            has_pred = bool(summary.get("prediction"))

            st.markdown(
                f"🔬 Disease Prediction "
                f"{_badge('done' if has_pred else 'pending')}",
                unsafe_allow_html=True,
            )

            if has_pred:
                st.metric(
                    "Predicted disease",
                    summary["prediction"].get(
                        "disease_name",
                        f"Class {summary['prediction']['prediction']}",
                    ),
                )

                st.page_link(
                    "pages/3_Predict.py",
                    label="→ View prediction",
                    icon="🔬",
                )

        st.markdown('</div>', unsafe_allow_html=True)

    with therapy_col:
        st.markdown(
            '<h3 class="nc-section-header">💊 Therapeutic Simulation</h3>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nc-card">', unsafe_allow_html=True)

        if sess is None or not sess.recommendation:
            st.markdown(
                f"💊 Therapy Recommendation {_badge('empty')}",
                unsafe_allow_html=True,
            )
            st.caption("Run disease prediction first.")

            st.markdown(
                f"🎬 Cure Simulation {_badge('empty')}",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"📄 Report {_badge('empty')}",
                unsafe_allow_html=True,
            )

        else:
            summary = sess.summary() if sess else {}

            st.markdown(
                f"💊 Therapy Recommendation {_badge('done')}",
                unsafe_allow_html=True,
            )

            st.metric(
                "Recommended technique",
                sess.recommendation.technique.name,
            )

            st.page_link(
                "pages/4_Therapy.py",
                label="→ View therapy",
                icon="💊",
            )

            has_eval = bool(summary.get("compatibility_score"))

            st.markdown(
                f"🎬 Cure Simulation "
                f"{_badge('done' if has_eval else 'pending')}",
                unsafe_allow_html=True,
            )

            if has_eval:
                st.page_link(
                    "pages/5_Simulate.py",
                    label="→ View simulation",
                    icon="🎬",
                )

            st.markdown(
                f"📄 Report {_badge('done' if has_eval else 'pending')}",
                unsafe_allow_html=True,
            )

            if has_eval:
                st.page_link(
                    "pages/6_Report.py",
                    label="→ Download report",
                    icon="📄",
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Quick-start CTA when no session ----
    if sess is None:
        st.divider()

        st.info(
            "👋 Welcome to **NeuroCure**. Upload a brain MRI (NIfTI/DICOM) or load "
            "the bundled demo brain to begin the diagnostic + therapeutic workflow."
        )

        st.page_link(
            "pages/1_Upload.py",
            label="📤 Start → Upload a brain scan",
            icon="📤",
        )

    # ---- Sidebar navigation ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")

    st.sidebar.page_link(
        "pages/1_Upload.py",
        label="1. Upload / Demo brain",
        icon="📤",
    )

    st.sidebar.page_link(
        "pages/2_3D_Brain.py",
        label="2. 3D Brain",
        icon="🧠",
    )

    st.sidebar.page_link(
        "pages/3_Predict.py",
        label="3. Disease Prediction",
        icon="🔬",
    )

    st.sidebar.page_link(
        "pages/4_Therapy.py",
        label="4. Therapy Recommendation",
        icon="💊",
    )

    st.sidebar.page_link(
        "pages/5_Simulate.py",
        label="5. Live Cure Simulation",
        icon="🎬",
    )

    st.sidebar.page_link(
        "pages/6_Report.py",
        label="6. Download Report",
        icon="📄",
    )

    st.sidebar.page_link(
        "pages/7_Model_Evaluation.py",
        label="7. Statistical Evaluation",
        icon="📊",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("NeuroCure v2.0 · 36-disease ensemble MLP")


main()