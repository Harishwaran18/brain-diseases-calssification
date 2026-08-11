"""Page 6 -- generate and download the final report."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from neurocure_app import state

st.title("📄 Download Report")
sess = state.require_session()

stages = sess.summary().get("stages", [])
if len(stages) < 3:
    st.warning(
        "Run more of the workflow first (segment → reconstruct → predict → "
        "simulate) before generating the report."
    )
    st.page_link("pages/1_Upload.py", label="→ Go to Upload", icon="📤")
    st.stop()

if st.button("📄 Generate report (HTML + JSON)", type="primary"):
    report_path = state.run_step("Generating unified report", sess.generate_report)
    st.success(f"Report generated: `{report_path}`")
    st.session_state["neurocure_report_path"] = str(report_path)

report_path = st.session_state.get("neurocure_report_path")
if report_path and Path(report_path).exists():
    st.subheader("Download")
    data = Path(report_path).read_bytes()
    st.download_button(
        "⬇ Download HTML report", data, file_name="neurocure_report.html", mime="text/html"
    )
    manifest = Path(report_path).parent / "report_manifest.json"
    if manifest.exists():
        st.download_button(
            "⬇ Download JSON manifest",
            manifest.read_bytes(),
            file_name="neurocure_report_manifest.json",
            mime="application/json",
        )
    st.subheader("Summary")
    summary = sess.summary()
    c1, c2, c3, c4 = st.columns(4)
    if summary.get("prediction"):
        c1.metric("Prediction", f"Class {summary['prediction']['prediction']}")
    c2.metric("Lesion volume", f"{summary.get('lesion_volume_mm3', 0):.0f} mm³")
    if summary.get("before_volume_mm3") and summary.get("after_volume_mm3"):
        c3.metric(
            "Recovery",
            f"{(summary['before_volume_mm3'] - summary['after_volume_mm3']) / summary['before_volume_mm3']:.1%}",
        )
    c4.metric("Compatibility", f"{summary.get('compatibility_score', 0):.3f}")
    st.markdown(f"📍 Report location: `{report_path}`")
else:
    st.info("Click the button above to generate the report.")
